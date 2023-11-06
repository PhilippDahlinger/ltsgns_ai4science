import copy
import time
from typing import Tuple, Union, Optional, Dict

import torch
from gmm_util.gmm import GMM
from hmpn.abstract.abstract_message_passing_base import AbstractMessagePassingBase
from multi_daft_vi.util_multi_daft import create_initial_gmm_parameters
from torch import nn
from torch_geometric.data import Batch

from lts_gns.algorithms.simulators.abstract_graph_network_simulator import AbstractGraphNetworkSimulator
from lts_gns.algorithms.simulators.simulator_util import unpack_node_features
from lts_gns.architectures.util.chamfer_distance import padded_chamfer_distance
from lts_gns.architectures.util.graph_dropout import GraphDropout
from lts_gns.envs.abstract_gns_environment import AbstractGNSEnvironment
from lts_gns.envs.cnp_environment import CNPEnvironment
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict
from lts_gns.util.util import polyak_update


class CNPSimulator(AbstractGraphNetworkSimulator):
    """
    Simulator with the Message Passing Network (MPN) architecture.
    """

    def __init__(self, simulator_config: ConfigDict, example_input_batch: Batch, d_r: int,
                 env: AbstractGNSEnvironment, device: str):
        """
        Builds the simulator. It predicts the next state of the environment, given the current state and a latent state.
        :param simulator_config:
        :param example_input_batch: To get the dimension of the input.
        """
        self._d_r = d_r
        # todo putting this so far up is a bit ugly, but we need the self._d_r to build the decoder in super().__init__
        super().__init__(simulator_config,
                         example_input_batch=example_input_batch,
                         env=env,
                         device=device)

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    @property
    def processor(self) -> AbstractMessagePassingBase:
        return self._processor

    @property
    def _input_dimensions(self) -> Dict[str, int]:
        return {keys.PROCESSOR_DIMENSION: self.simulator_config.gnn.latent_dimension,
                keys.R_DIMENSION: self._d_r}

    def _loss(self, predictions: torch.Tensor, labels: torch.Tensor, loss_function_name: str) -> torch.Tensor:
        """
        Computes the loss of the simulator. This is in this case the negative log likelihood part of the ELBO.
        computes output with gradients for the GNN loss.
        """
        # log likelihood of the model velocities (therefore normalized) or the positions in the chamfer case
        log_likelihood_per_time_step = self._log_likelihood_per_time_step(predictions=predictions,
                                                                          labels=labels,
                                                                          likelihood_std=1.0,
                                                                          loss_function_name=loss_function_name
                                                                          )
        # the Gradient of the ELBO is the same as the Gradient of the log likelihood
        # mean over all time steps to obtain the loss
        loss = -log_likelihood_per_time_step.mean()
        return loss

    def evaluate_state(self, predicted_mesh_state: Dict[str, torch.Tensor],
                       reference_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Computes various metrics for the predicted positions given the internal labels.
        Args:
            predicted_mesh_state: Arbitrary outputs of the simulator. Can e.g., be interpreted as node velocities
            depending on the task.
            reference_labels: If given, the metrics are computed with respect to these labels. Otherwise, the internal

        Returns: Dictionary with the metrics
            "mse": MSE over all samples and time steps.
                Shape: () (float)
            "log_likelihood": Log likelihood of the predicted positions given the labels.
                Shape: () (float)

        """
        # todo does the log likelihood make sense here?
        # todo does the mse make sense as a mean over the results for all latents?
        predictions = predicted_mesh_state[keys.PREDICTIONS]
        if reference_labels is None:
            reference_labels = self.next_mesh_pos
        return {keys.MSE: self.mse(predictions=predictions,
                                   reference_pos=reference_labels),
                keys.MLL: self.log_marginal_likelihood(predictions=predictions,
                                                       reference_labels=reference_labels),
                }

    def mse(self, predictions: torch.Tensor, reference_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean squared error of the predicted positions given the internal next mesh positions.
        """
        mse = torch.mean((predictions - reference_pos) ** 2)
        return mse

    def log_marginal_likelihood(self, predictions: torch.Tensor,
                                reference_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the log marginal likelihood of the data given the model parameters.
        Implementation of Equation (19) from Volpp. et. al. 2023
        Args:
            predictions: Tensor with shape (n_vertices, d_world)
            reference_labels: If given, the metrics are computed with respect to these labels. Otherwise, the internal
                labels are used.

        Returns:

        """
        log_likelihood_per_time_step = self._log_likelihood_per_time_step(predictions=predictions,
                                                                          labels=reference_labels
                                                                          )
        # sum over all time steps
        log_likelihood = torch.sum(log_likelihood_per_time_step, dim=0)
        return log_likelihood

    def _log_likelihood_per_time_step(self, predictions: torch.Tensor,
                                      labels: torch.Tensor,
                                      likelihood_std: float | None = None,
                                      loss_function_name: str | None = None) -> torch.Tensor:
        """
        Computes the log likelihood of the predicted positions given the labels.
        The labels are the next positions of the nodes.
        :param predictions: tensor with shape (n_vertices, d_world).
            Can e.g., be the predicted positions of the simulator, but could also be the model velocities for training.
        :param labels: If given, the metrics are computed with respect to these labels.
            Otherwise, the internal next_mesh positions are used.
        :param likelihood_std: standard deviation of the gaussian likelihood. If None, the std from the config is used.
        :return: log-likelihood of each time step. Shape: (num_samples, num_time_steps)
        """
        # log likelihood of a gaussian with mean extend_label and std given from config
        if likelihood_std is None:
            likelihood_std = self._simulator_config.likelihood.std
        # constant_term = - n_actions / 2 * np.log(2 * np.pi) - n_actions * np.log(likelihood_std)

        if loss_function_name is None:
            loss_function_name = self._loss_function_name

        if loss_function_name == "mse":
            log_likelihood_per_node = - 0.5 * torch.sum((predictions - labels) ** 2 / likelihood_std ** 2, dim=-1)
            # likelihood has shape (n_vertices/n_nodes)
            if self._simulator_config.likelihood.graph_aggregation == "sum":
                aggregator = torch.sum
            elif self._simulator_config.likelihood.graph_aggregation == "mean":
                aggregator = torch.mean
            else:
                raise ValueError(
                    f"Unknown graph aggregation method: {self._simulator_config.likelihood.graph_aggregation}")
            mesh_nodes_per_time_step = self.task_belonging[keys.MESH_NODES_PER_TIME]
            log_likelihood_per_time_step = torch.stack([aggregator(t, dim=0)
                                                        for t in torch.split(log_likelihood_per_node,
                                                                             mesh_nodes_per_time_step, dim=0)], dim=0)
            # now has shape (num_time_steps/num_graphs)

        elif loss_function_name == "chamfer":
            # TODO: Currently wrong with z
            # We use multiple z over batched graphs and only one label. Need to broadcast the label to all z samples
            z_shape = predictions.shape[0]
            predictions = predictions.reshape(predictions.shape[0], -1,
                                              self.task_belonging[keys.MESH_NODES_PER_TIME][0],
                                              predictions.shape[-1])
            labels = labels.reshape(-1, self.task_belonging[keys.MESH_NODES_PER_TIME][0], labels.shape[-1])
            labels = labels.repeat(predictions.shape[0], 1, 1, 1)
            labels = labels.reshape(-1, *labels.shape[2:])

            predictions = predictions.reshape(-1, *predictions.shape[2:])
            chamfer_loss = padded_chamfer_distance(predictions, labels,
                                                   density_aware=self._simulator_config.chamfer.density_aware,
                                                   forward_only=self._simulator_config.chamfer.forward_only,
                                                   point_reduction=self._simulator_config.likelihood.graph_aggregation)
            # todo equality check?
            chamfer_loss = - 0.5 * chamfer_loss / (likelihood_std ** 2)
            log_likelihood_per_time_step = chamfer_loss.reshape(z_shape, -1)
            # predictions have shape (num_samples, num_graphs, n_vertices, d_world)

        else:
            raise ValueError(f"Unknown loss function: {loss_function_name}")

        return log_likelihood_per_time_step

    def predict_denormalize_and_integrate(self, r: torch.Tensor | None, batch: Optional[Batch] = None,
                                          ) -> Dict[str, torch.Tensor]:
        """
        Predicts the quantities of the next time step, denormalizes the model velocities back to world space
        and integrates them with the current mesh
        Args:
            r:
            batch: Calculates the processor output from self._batch if no batch is provided,
              or from batch otherwise

        Returns: A dictionary with the integrated quantities. The keys are the quantities and the values are tensors
            with shape (n_vertices, d_world). Must always have a key PREDICTION to calculate losses/metrics

        """
        simulator_predictions = self._predict(r=r, batch=batch)

        # denormalize the velocities
        simulator_predictions = self._model_to_world_normalizer(simulator_predictions)
        if batch is None:
            batch = self._batch
        mesh_state = self._integrate_predictions_fn(quantities_to_integrate=simulator_predictions,
                                                    batch=batch)
        return mesh_state

    def _predict(self, r: torch.Tensor | None, batch: Optional[Batch] = None) -> torch.Tensor:
        """
        Predicts the positions of the next time step. Requires a latent r.
        Will use the internal batch if no batch is given. Since the processor calculation of this batch is cached, uses
        the cached processor output if possible.
        Args:
            r: Tensor with shape (num_tasks, d_r)
            batch: If given, the internal batch is replaced by this one.
        Returns: Simulator predictions of shape (num_nodes, action_dimension)

        """
        if batch is None:
            processed_batch = self.processor(self._batch)
            processor_output = unpack_node_features(processed_batch, node_type=keys.MESH)
        else:
            processed_batch = self.processor(batch)
            processor_output = unpack_node_features(processed_batch, node_type=keys.MESH)
        mesh_nodes_per_task = self.task_belonging[keys.MESH_NODES_PER_TASK]
        simulator_predictions = self.decoder(processor_output=processor_output,
                                             r=r,
                                             mesh_nodes_per_task=mesh_nodes_per_task)
        # sim_output has shape (n_vertices, d_world)
        # compare with labels, problem: the labels are the next positions, the sim_output are the velocities.
        return simulator_predictions

    def can_sample(self):
        return False

    def sample(self, n: int):
        raise NotImplementedError
