from typing import Dict

import torch
from torch_geometric.data import Batch

from lts_gns.util import keys
from lts_gns.util.util import node_type_mask


def integrate_predictions(quantities_to_integrate: torch.Tensor,
                          batch: Batch,
                          integration_order: int = 1,
                          d_t=1.0) -> Dict[str, torch.Tensor]:
    """
    Numerically integrate the provided quantities to obtain updated positions.
    This is done by a simple n-th order Euler integration, but might be changed to a more sophisticated method.
    We allow a batch dimension (num_samples,) for the values, as this is necessary for training.
    During training, integrates the quanitites on the noisy old positions. During inference, there is no noise added,
    so the integration is done on the true old positions.
    Args:
        quantities_to_integrate: The quantities to integrate. Shape (num_samples, num_mesh_nodes, d_world)
        integration_order: The order of the integration. 1 is velocity integration, 2 is acceleration integration.
        d_t: The time step size

    Returns: The updated positions

    """
    # todo right now this is our "integration" function. It returns new positions per node. Instead, we could
    #  return either a full graph object, or precisely the things that we need for the task in a nice interface
    agent_node_mask = node_type_mask(batch, keys.MESH)
    noisy_old_positions = batch.pos[agent_node_mask]
    # old_positions have shape (num_mesh_nodes, d_world),
    # here, we can use broadcasting to add the velocities for multiple samples
    # of shape (num_samples, num_mesh_nodes, d_world)
    if integration_order == 1:
        # velocity integration
        new_positions = noisy_old_positions + quantities_to_integrate * d_t
    else:
        # can use e.g., accelerations here
        raise ValueError(f"Integration order {integration_order} not supported.")
    return {keys.POSITIONS: new_positions,
            keys.PREDICTIONS: new_positions,
            keys.VELOCITIES: quantities_to_integrate,
            }


def unpack_node_features(graph: Batch, node_type: str = keys.MESH) -> torch.Tensor:
    """
    Unpacking the node features of the nodes with agent_node_type from a homogeneous graph.
    :param graph:
    :param node_type:
    :return: tensor of shape [num_agent_nodes, num_features]
    """
    node_features = graph.x
    node_types = graph.node_type
    agent_node_index = graph.node_type_description[0].index(node_type)
    agent_node_mask = node_types == agent_node_index
    result = node_features[agent_node_mask]
    return result
