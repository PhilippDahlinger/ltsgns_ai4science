from typing import Optional, Dict

import torch
from torch_geometric.data import Batch

from lts_gns.architectures.decoder.decoder import Decoder
from lts_gns.architectures.decoder.decoder_factory import DecoderFactory
from lts_gns.algorithms.simulators.abstract_graph_network_simulator import AbstractGraphNetworkSimulator
from lts_gns.algorithms.simulators.simulator_util import unpack_node_features
from lts_gns.architectures.util.chamfer_distance import padded_chamfer_distance
from lts_gns.architectures.prodmp.prodmp import ProDMPPredictor
from lts_gns.util import keys


def mse(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean squared error of the predicted positions given the internal next mesh positions.
    """
    mse = torch.mean((predictions - labels) ** 2)
    return mse


class MGNProDMPSimulator(AbstractGraphNetworkSimulator):
    """
    Simulator with the Message Passing Network (MPN) architecture.
    """

    @property
    def _input_dimensions(self) -> Dict[str, int]:
        return {keys.PROCESSOR_DIMENSION: self.simulator_config.gnn.latent_dimension, }
    
    @property
    def mp_predictor(self):
        if not hasattr(self, "_mp_predictor"):
            raise ValueError("MP predictor not initialized yet.")
        return self._mp_predictor
    
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
            predictions = self.predict_denormalize_and_integrate(batch=batch, **kwargs).get(keys.PREDICTIONS)
            labels = self.next_mesh_pos # labels is conditioned on t
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

    def _loss(self, predictions: torch.Tensor, labels: torch.Tensor, loss_function_name) -> torch.Tensor:
        if loss_function_name == "mse":
            return mse(predictions=predictions, labels=labels)
        elif loss_function_name == "chamfer":
            # expects an input of shape (batch_size, num_nodes, action_dimension)
            loss = padded_chamfer_distance(predictions, labels,
                                           density_aware=self._simulator_config.chamfer.density_aware,
                                           forward_only=self._simulator_config.chamfer.forward_only,
                                           point_reduction="mean")

            return loss.mean()  # sums over all dimensions, i.e., returns a scalar
        else:
            raise ValueError(f"Unknown loss function {loss_function_name}")
        
    def _build_decoder(self, example_input_batch: Batch) -> Decoder:
        self._mp_predictor = ProDMPPredictor(
            num_dof=example_input_batch.y.shape[1], 
            mp_config=self.simulator_config.mp_config.get_raw_dict()
        )
        decoder = DecoderFactory.build_decoder(decoder_config=self.simulator_config.decoder,
                                               input_dimensions=self._input_dimensions,
                                               action_dim=self._mp_predictor.output_size,
                                               simulator_class=str(type(self)).split(".")[-1].split("'")[0],
                                               device=self._device)
        return decoder

    def predict_denormalize_and_integrate(self, batch: Optional[Batch] = None, **kwargs) -> Dict[str, torch.Tensor]:
        if batch is None:
            batch = self._batch
        basis_weights = self._predict(batch=batch)

        mesh_init_pos = batch.pos[batch.node_type == 0]
        mesh_init_vel = batch.next_mesh_pos - mesh_init_pos
        mesh_state = self.mp_predictor(mesh_init_pos, mesh_init_vel, basis_weights=basis_weights)
        return mesh_state

    def _predict(self, batch: Batch, **kwargs) -> torch.Tensor:
        """
        Predicts the positions of the next time step.
        Args:
            batch: If given, the internal batch is replaced by this one.

        Returns: Predicted positions of shape (num_nodes, action_dimension)

        """
        processed_batch = self.processor(batch)
        mesh_features = unpack_node_features(processed_batch, node_type=keys.MESH)
        decoded_batch = self.decoder(mesh_features)  # velocities of shape (num_nodes, action_dimension)
        return decoded_batch

    def evaluate_state(self, predicted_mesh_state: Dict[str, torch.Tensor],
                       reference_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Computes various metrics for the predicted positions given the internal labels.
        Args:
            predicted_mesh_state: A dictionary of the (inferred/simulated) state of the mesh.
                Includes e.g., the positions of the nodes, as well as their velocities, pressure fields etc.
            reference_labels: If given, the metrics are computed with respect to these labels. Otherwise, the internal
                labels are used.


        Returns: Dictionary with the metrics
            "mse": MSE over all samples and time steps.
                Shape: () (float)

        """
        predictions = predicted_mesh_state[keys.PREDICTIONS]
        if reference_labels is None:  # use positions of next mesh as reference
            reference_labels = self.next_mesh_pos
        return {keys.MSE: mse(predictions=predictions,
                              labels=reference_labels)
                }
