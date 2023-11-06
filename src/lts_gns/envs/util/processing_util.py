from typing import Tuple

import torch
from torch_geometric import transforms
from torch_geometric.data import Batch, Data


def add_gaussian_noise(batch: Batch, node_type_description: str, sigma: float,
                       device: torch.device | str) -> Batch:
    """
    Adds gaussian noise to the position of the nodes of a certain type in a batch.
    Args:
        batch:
        node_type_description:
        sigma:
        device:

    Returns: The batch with the added noise on the nodes.

    """
    # todo maybe move this to a separate file
    node_type = batch.node_type_description[0].index(node_type_description)
    indices = torch.where(batch.node_type == node_type)[0]
    num_pos_features = batch.pos.shape[1]
    noise = torch.randn(indices.shape[0], num_pos_features).to(device) * sigma
    batch.pos[indices] += noise
    # GGNS also changed y_old, (which is the position of only the mesh nodes).
    # We do not do this here and get the next position by
    # accessing the position vector with a mask.
    return batch


def process_global_features(x: torch.Tensor, global_features: torch.Tensor,
                            input_selection: str | None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process the task properties and concatenate them to the input features, according to the task_properties_input_selection.
    Args:
        x: Node features shape (n_nodes, n_features)
        global_features: Task properties shape (n_task_properties,)
        input_selection: str, one of "global", "local"/"node" or None. Where to add the global features.
    Returns:
        x: Node features shape (n_nodes, n_features + n_task_properties) if task_properties_input_selection is "local"/"node" or None
           or with shape (n_nodes, n_features) if task_properties_input_selection is "global"
        u: Task properties shape (1, n_task_properties) if task_properties_input_selection is "global" or None otherwise

    """
    match input_selection:
        case "global":
            u = global_features.reshape(1, -1)
        case "local" | "node":
            x = torch.cat((x, global_features.repeat([x.shape[0], 1])), dim=1)
            u = None
        case None:
            # no input features from task properties
            u = None
        case _:
            raise ValueError(f"Unknown task_properties_input_selection: `{input_selection}`")
    return x, u


def add_distances_from_positions(data_or_batch: Batch | Data, add_euclidian_distance: bool) -> Batch | Data:
    """
    Transform the node positions to the edges as relative distance together with (if needed) Euclidean norm and add
    them to the edge features
    :param data_or_batch:
    :return:
    """
    if data_or_batch.edge_index is None or data_or_batch.edge_index.shape[1] == 0:
        # there are no edges, so we can't add edge features. Do nothing in this case
        return data_or_batch

    if hasattr(data_or_batch, "edge_type_description"):
        add_z_distance = data_or_batch.pos.shape[1] == 3
        if isinstance(data_or_batch, Batch):
            for edge_type_description in data_or_batch.edge_type_description:
                _update_edge_type_description(edge_type_description=edge_type_description,
                                              add_z_distance=add_z_distance,
                                              add_euclidian_distance=add_euclidian_distance)
        else:
            _update_edge_type_description(edge_type_description=data_or_batch.edge_type_description,
                                          add_z_distance=add_z_distance,
                                          add_euclidian_distance=add_euclidian_distance)

    if add_euclidian_distance:
        data_transform = transforms.Compose([transforms.Cartesian(norm=False, cat=True),
                                             transforms.Distance(norm=False, cat=True)])
    else:
        data_transform = transforms.Cartesian(norm=False, cat=True)
    out_batch = data_transform(data_or_batch)
    return out_batch


def _update_edge_type_description(edge_type_description, add_z_distance: bool, add_euclidian_distance: bool):
    edge_type_description.extend(["x_distance", "y_distance"])
    if add_z_distance:
        edge_type_description.append("z_distance")
    if add_euclidian_distance:
        edge_type_description.append("euclidian_distance")
