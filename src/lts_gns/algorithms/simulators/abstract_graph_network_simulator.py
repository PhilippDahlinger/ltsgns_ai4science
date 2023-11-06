from abc import ABC, abstractmethod
from typing import Optional, Dict

import torch
from multi_daft_vi.lnpdf import LNPDF
from torch import nn
from torch_geometric.data import Batch

from hmpn.abstract.abstract_message_passing_base import AbstractMessagePassingBase
from hmpn.get_hmpn import get_hmpn_from_graph
from lts_gns.architectures.decoder.decoder import Decoder
from lts_gns.architectures.decoder.decoder_factory import DecoderFactory
from lts_gns.envs.abstract_gns_environment import AbstractGNSEnvironment
from lts_gns.envs.task.task import Task
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict


class AbstractGraphNetworkSimulator(LNPDF, ABC):
    """
    Abstract base class for simulators.
    """

    def __init__(self, simulator_config: ConfigDict, example_input_batch: Batch,
                 env: AbstractGNSEnvironment,
                 device: str):
        # todo @Niklas why is there no super().__init__() for the LNPDF?
        self._simulator_config = simulator_config
        self._device = device

        # build the encode-process-decode structure
        # our processor also contains the encoder as a simple MLP for the nodes, edges and potentially globals
        gnn_config = simulator_config.gnn
        self._processor: AbstractMessagePassingBase = get_hmpn_from_graph(example_graph=example_input_batch,
                                                                          latent_dimension=gnn_config.latent_dimension,
                                                                          node_name=keys.MESH,
                                                                          unpack_output=False,  # return full graph
                                                                          base_config=gnn_config.base,
                                                                          device=device)

        self._decoder: nn.Module = self._build_decoder(example_input_batch)
        self._optimizer = self._build_optimizer(simulator_config.optimizer)

        if simulator_config.get("checkpoint", {}).get("load_checkpoint"):
            self.load_from_checkpoint(simulator_config.checkpoint)  # load in processor and decoder

        self._loss_function_name = simulator_config.loss

        self._batch: Batch | None = None
        self._task_belonging: torch.Tensor | None = None
        self._batch_y: torch.Tensor | None = None
        self._next_mesh_pos: torch.Tensor | None = None

        # functions to integrate the predictions and update batches with new information. Taken from the env
        if hasattr(env, "integrate_predictions_fn"):
            self._integrate_predictions_fn = env.integrate_predictions_fn
        if hasattr(env, "update_batch_fn"):
            self._update_batch_fn = env.update_batch_fn
        if hasattr(env, "model_to_world_normalizer"):
            self._model_to_world_normalizer = env.model_to_world_normalizer
        if hasattr(env, "graph_updater"):
            self._graph_updater = env.graph_updater

    def _build_decoder(self, example_input_batch: Batch) -> Decoder:
        decoder = DecoderFactory.build_decoder(decoder_config=self.simulator_config.decoder,
                                               input_dimensions=self._input_dimensions,
                                               action_dim=example_input_batch.y.shape[1],
                                               simulator_class=str(type(self)).split(".")[-1].split("'")[0],
                                               device=self._device)
        return decoder

    def update_batch_from_state(self, mesh_state: Dict[str, torch.Tensor],
                                batch_or_task: Batch | Task) -> Batch | Task:
        """
        In-place updates the provided graph with the simulator predictions.

        Args:
            mesh_state: The predictions of the simulator. Can e.g., be velocities.
                Dictionary of e.g., positions, velocities, pressures. Each entry has shape (n_vertices, d_world)
            batch_or_task: A batch or task object consisting of a batch of graphs.
                The batch is updated in-place using the simulator_predictions according to the integration function

        Returns:

        """
        if isinstance(batch_or_task, Batch):
            batch_or_task = self._update_batch_fn(batch_or_task, mesh_state)
        else:
            # move to batch for batched operations
            batch_or_task = Batch.from_data_list(batch_or_task.trajectory)
            batch_or_task = self._update_batch_fn(batch_or_task, mesh_state)
            batch_or_task = Task(batch_or_task.to_data_list())
        return batch_or_task

    @property
    def _input_dimensions(self) -> Dict[str, int]:
        raise NotImplementedError("This method needs to be implemented by the subclass.")

    def _build_optimizer(self, optimizer_config: ConfigDict) -> torch.optim.Optimizer:
        """
        Builds the optimizer for the simulator.
        Have 2 groups, one for the GNN and one for the Decoder
        Args:
            optimizer_config:

        Returns:

        """
        # TODO: make this more flexible, i.e. different optimizers, more hyperparameters
        # todo encoder
        optimizer = torch.optim.Adam([
            {"params": self.processor.parameters(), "lr": optimizer_config.gnn_learning_rate},
            {"params": self.decoder.parameters(), "lr": optimizer_config.decoder_learning_rate}
        ])
        return optimizer

    def set_train_mode(self, train: bool):
        """
        Sets the train mode for the simulator.
        Args:
            train: bool. True: train mode, False: eval mode

        Returns:

        """
        self._processor.train(train)
        self._decoder.train(train)

    @property
    def simulator_config(self) -> ConfigDict:
        return self._simulator_config

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            raise ValueError("Optimizer has not been set yet.")
        return self._optimizer

    @property
    def batch(self) -> Batch:
        if self._batch is None:
            raise ValueError("Batch has not been set yet.")
        return self._batch

    @property
    def processor(self) -> AbstractMessagePassingBase:
        return self._processor

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    @property
    def batch_y(self) -> torch.Tensor:
        if self._batch_y is None:
            raise ValueError("No data given. Call 'condition_on_data()' first.")
        return self._batch_y

    @property
    def next_mesh_pos(self) -> torch.Tensor:
        if self._next_mesh_pos is None:
            raise ValueError("No data given. Call 'condition_on_data()' first.")
        return self._next_mesh_pos

    @property
    def task_belonging(self) -> torch.Tensor:
        if self._task_belonging is None:
            raise ValueError("No data given. Call 'condition_on_data()' first.")
        return self._task_belonging

    def _apply_loss(self, loss: torch.Tensor):
        """
        Applies the loss to the parameters of the simulator.
        :param loss: The loss to be applied.
        """
        self._optimizer.zero_grad()
        loss.backward()
        # TODO: Add gradient clipping, might have to overwrite this method in the specific simulator class.
        self._optimizer.step()

    def loss(self, batch: Batch, **kwargs) -> torch.Tensor:
        """
        Computes the loss of the simulator.
        We update the parameters of the simulator with the derivative of this function.
        For simple GNS this is the MSE or Chamfer distance between the predicted next step and the actual next step.
        For the LTS-GNS, this is the MC approximation of the ELBO for either.
        This uses samples of z, which are passed as kwargs.
        Args:
            batch: The batch of graphs to be used for the loss computation.
            **kwargs: Additional arguments. For the LTS-GNS, this is the samples of z.
                Z has shape (n_samples, n_task_in_batch, d_z)
                and represents samples from the current posterior distribution of z.

        Returns: The loss. Shape (1,)

        """
        if self._loss_function_name == "mse":
            predictions = self._predict(batch=batch, **kwargs)  # normalized network outputs
            labels = self.batch_y
        elif self._loss_function_name == "chamfer":
            predictions = self.predict_denormalize_and_integrate(batch=batch, **kwargs).get(keys.PREDICTIONS)
            labels = self.next_mesh_pos

            # todo for simplicity, we assume that the number of nodes is the same for all graphs in the batch
            nodes_per_graph = sum(batch.node_type[:batch.ptr[1]] == 0)
            predictions = predictions.reshape(*predictions.shape[:-2], -1, nodes_per_graph, predictions.shape[-1])
            labels = labels.reshape(-1, nodes_per_graph, labels.shape[-1])
        else:
            raise ValueError(f"Loss function {self._loss_function_name} not implemented.")
        return self._loss(predictions, labels, loss_function_name=self._loss_function_name)

    @abstractmethod
    def _predict(self, batch: Batch, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _loss(self, predictions: torch.Tensor, labels: torch.Tensor, loss_function_name: str) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def predict_denormalize_and_integrate(self, batch: Batch, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def train_step(self, batch: Batch, **kwargs) -> torch.Tensor:
        loss = self.loss(batch=batch, **kwargs)
        self._apply_loss(loss)
        return loss

    @abstractmethod
    def evaluate_state(self, predicted_positions: torch.Tensor,
                       reference_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("This method has to be implemented by the specific simulator.")

    def condition_on_data(self, batch: Batch, task_belonging: ValueDict) -> bool:
        """
        Loads the batch of data into the simulator. The log density and the loss function will be evaluated on this batch.
        :param batch: The batch of data. Consists of several tasks
        :param task_belonging: A dictionary that maps the task id to the index of the task in the batch.
        :return: True if the simulator is ready to be used. False if a problem occurred.
        """
        # todo @Niklas: Why is this a bool?
        self._task_belonging = task_belonging
        self._batch_y = batch.y
        self._next_mesh_pos = batch.next_mesh_pos
        self._batch = batch
        return True

    def save_checkpoint(self, directory: str, iteration: int | str, is_initial_save: bool, is_final_save: bool = False):
        """
        Saves the state dict of the simulator to the specified directory.
        Args:
            directory:
            iteration: current iteration or a string describing this save. f
            is_initial_save:
            is_final_save:

        Returns:

        """
        save_dict = self._get_all_state_dicts()
        if is_final_save:
            file_name = f"{keys.STATEDICT}_final.pt"
        else:
            file_name = f"{keys.STATEDICT}t_{iteration}.pt"
        import os
        torch.save(save_dict, os.path.join(directory, file_name))

    def _get_all_state_dicts(self) -> ValueDict:
        save_dict = {"gnn_params": self.processor.state_dict(),
                     "decoder_params": self.decoder.state_dict(),
                     "optimizer_params": self.optimizer.state_dict()}
        return save_dict

    def load_from_checkpoint(self, checkpoint_config: ConfigDict) -> None:
        """
        Loads the algorithm state from the given checkpoint path/experiment configuration name.
        May be used at the start of the algorithm to resume training.
        Args:
            checkpoint_config: Dictionary containing the configuration of the checkpoint to load. Includes
                checkpoint_path: Path to a checkpoint folder of a previous execution of the same algorithm
                iteration: (Optional[int]) The iteration to load. If not provided, will load the last available iter
                repetition: (int) The algorithm repetition/seed to load. If not provided, will load the first repetition

        Returns:

        """
        import pathlib
        # get checkpoint path and iteration
        experiment_name: str = checkpoint_config.get("experiment_name")
        iteration: int = checkpoint_config.get("iteration")
        repetition: str = checkpoint_config.get("repetition")
        if repetition is None:
            repetition = "rep_00"  # default to first repetition

        checkpoint_path = pathlib.Path(experiment_name) / "log" / repetition / "checkpoints"
        assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist"

        # load state dict for network
        if iteration is None:
            file_name = f"{keys.STATEDICT}_{keys.FINAL}.pt"
            if not (checkpoint_path / file_name).exists():
                # if final.pkl does not exist, load the last iteration instead
                file_name = sorted(list(checkpoint_path.glob("*.pt")))[-1].name
        else:  # if iteration is given, load the corresponding file
            assert isinstance(iteration, int) or iteration == keys.FINAL, \
                f"Invalid iteration {iteration}. Must be an integer or the string 'final'."
            file_name = f"{keys.STATEDICT}_{iteration}.pt"
        state_dict_file = checkpoint_path / file_name
        state_dict = torch.load(state_dict_file, map_location=self._device)  # Load the state dictionary from the file

        self._insert_state_dict_params(state_dict)

        optimizer_params = state_dict.get('optimizer_params')
        if optimizer_params is not None:
            self.optimizer.load_state_dict(optimizer_params)

    def _insert_state_dict_params(self, state_dict):
        # Extract the parameters for processor and decoder
        processor_params = state_dict.get('gnn_params')
        decoder_params = state_dict.get('decoder_params')
        # Validate that the parameters were actually found in the loaded state_dict
        assert processor_params is not None, "gnn_params not found in the state_dict"
        assert decoder_params is not None, "decoder_params not found in the state_dict"
        # Load the parameters into self.processor and self.decoder
        self.processor.load_state_dict(processor_params)
        self.decoder.load_state_dict(decoder_params)

