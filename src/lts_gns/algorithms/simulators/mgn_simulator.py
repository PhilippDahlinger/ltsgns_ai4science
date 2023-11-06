from typing import Optional, Dict

import torch
from torch_geometric.data import Batch

from lts_gns.algorithms.simulators.abstract_graph_network_simulator import AbstractGraphNetworkSimulator
from lts_gns.algorithms.simulators.simulator_util import unpack_node_features
from lts_gns.architectures.util.chamfer_distance import padded_chamfer_distance
from lts_gns.util import keys


def mse(predictions: torch.Tensor, labels: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the mean squared error of the predicted positions given the internal next mesh positions.
    """
    if ptr is None:
        mse = torch.mean((predictions - labels) ** 2)
    else:
        # use scatter mean to compute the mse for each batch element separately
        mse = torch.scatter(src=(predictions - labels) ** 2, dim=0, index=ptr, reduce="mean")
    return mse


class MGNSimulator(AbstractGraphNetworkSimulator):
    """
    Simulator with the Message Passing Network (MPN) architecture.
    """

    @property
    def _input_dimensions(self) -> Dict[str, int]:
        return {keys.PROCESSOR_DIMENSION: self.simulator_config.gnn.latent_dimension, }

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

    def predict_denormalize_and_integrate(self, batch: Optional[Batch] = None, **kwargs) -> Dict[str, torch.Tensor]:
        if batch is None:
            batch = self._batch
        simulator_predictions = self._predict(batch=batch)
        simulator_predictions = self._model_to_world_normalizer(simulator_predictions)
        mesh_state = self._integrate_predictions_fn(quantities_to_integrate=simulator_predictions,
                                                    batch=batch)
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
                       reference_labels: Optional[torch.Tensor] = None, ptr=None) -> Dict[str, torch.Tensor]:
        """
        Computes various metrics for the predicted positions given the internal labels.
        Args:
            predicted_mesh_state: A dictionary of the (inferred/simulated) state of the mesh.
                Includes e.g., the positions of the nodes, as well as their velocities, pressure fields etc.
            reference_labels: If given, the metrics are computed with respect to these labels. Otherwise, the internal
                labels are used.
            ptr: If given, the metrics are computed for each batch element separately.


        Returns: Dictionary with the metrics
            "mse": MSE over all samples and time steps.
                Shape: () (float) if ptr is None, otherwise shape (#batch_size,)

        """
        predictions = predicted_mesh_state[keys.PREDICTIONS]
        if reference_labels is None:  # use positions of next mesh as reference
            reference_labels = self.next_mesh_pos
        return {keys.MSE: mse(predictions=predictions,
                              labels=reference_labels,
                              ptr=ptr)
                }
