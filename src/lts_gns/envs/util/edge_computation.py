from typing import Optional, List, Dict

import torch
import torch_cluster
from torch_geometric.data import Data

from lts_gns.envs.util.processing_util import add_distances_from_positions
from lts_gns.envs.util.task_processor_util import get_one_hot_features_and_types
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict


def build_edges(edge_index_dict: Dict[str, torch.Tensor]):
    """
    Build edges from edge_index_dict. Returns a tuple of edge_attr, edge_index, edge_type, edge_type_description,
    where edge_attr is a one-hot encoding of the edges, edge_index is the edge index, edge_type is an integer of the
    edge types and edge_type_description is a list of strings in the same order as edge_type.

    Args:
        edge_index_dict: Dictionary of edge indices. Keys are edge types, values are edge indices shaped (2, num_edges)

    Returns:

    """
    num_edges = []
    for key, value in edge_index_dict.items():
        num_edges.append(value.shape[1])
    edge_attr, edge_type = get_one_hot_features_and_types(num_edges)
    edge_index = torch.cat(tuple(edge_index_dict.values()), dim=1).long()
    edge_type_description = list(edge_index_dict.keys())
    return edge_attr, edge_index, edge_type, edge_type_description


def create_radius_edges(radius: float, source_nodes: torch.Tensor, source_shift: int,
                        target_nodes: Optional[torch.Tensor] = None, target_shift: int = 0,
                        max_num_neighbors: int = 100) -> torch.Tensor:
    """
    Create edges between source and target nodes based on a radius.
    Args:
        radius: Radius to create edges within
        source_nodes: Source nodes to create edges from
        source_shift: Shift indices of source nodes by this amount
        target_nodes: Target nodes to create edges to. If None, source and target nodes are assumed to be the same
        target_shift: Shift indices of target nodes by this amount. Defaults to 0
        max_num_neighbors: Maximum number of neighbors/edges to consider per node. Defaults to 100. A very high radius
          combined with a low max_num_neighbors essentially corresponds to a k-nearest neighbor graph.

    Returns: Edge indices between source and target nodes as a torch tensor of shape (2, num_edges)

    """
    if target_nodes is not None:
        # both source and target nodes provided
        source_target_edges = torch_cluster.radius(x=target_nodes,
                                                   y=source_nodes,
                                                   r=radius,
                                                   max_num_neighbors=max_num_neighbors)
    else:
        # only source nodes provided, so assume that source and target nodes are the same
        source_target_edges = torch_cluster.radius_graph(source_nodes, r=radius,
                                                         max_num_neighbors=max_num_neighbors)
    # shift the edge indices to the correct node type, i.e., the first node type has indices 0 to num_nodes[0]-1
    source_target_edges[0, :] += source_shift
    source_target_edges[1, :] += target_shift
    return source_target_edges


def build_edges_from_data_dict(data_dict: ValueDict, pos_keys: List, num_nodes: List,
                               connectivity_setting: ConfigDict,
                               use_canonic_mesh_positions: bool) -> (torch.Tensor,
                                                                     torch.Tensor,
                                                                     torch.Tensor,
                                                                     List[str]):
    """
    Function to build the edge features from the data_dict
    :param data_dict: Dict containing all the data for a single timestep in torch tensor format.
    :param pos_keys: List of keys for the positions of the nodes
    :param num_nodes: List of numbers of nodes per type
    :param connectivity_setting: ConfigDict with the connectivity settings
    :param use_canonic_mesh_positions: bool: if relative mesh positions should be used as edge features
    :return: edge_attr, edge_type, edge_index, edge_type_description
    """

    edge_index_dict = _get_edge_indices(data_dict, pos_keys,
                                        num_nodes=num_nodes, connectivity_setting=connectivity_setting)

    # build edge_attr (one-hot)
    edge_attr, edge_index, edge_type, edge_type_description = build_edges(edge_index_dict)

    # add mesh_coordinates to mesh edges. This essentially encodes the distances between nodes in mesh space, meaning
    # that we do not need to differentiate between mesh and world edges in the model
    if use_canonic_mesh_positions:
        edge_attr = _add_canonic_mesh_positions(edge_index_dict, edge_attr, edge_type,
                                                input_mesh_edge_index=data_dict[keys.MESH_EDGE_INDEX],
                                                initial_mesh_positions=data_dict[keys.INITIAL_MESH_POSITIONS],
                                                include_euclidean_distance=True)
        edge_type_description.extend(["canonic_x_distance", "canonic_y_distance"])
        if data_dict["mesh"].shape[1] == 3:
            edge_type_description.append("canonic_z_distance")

        edge_type_description.append("canonic_euclidean_distance")

    return edge_attr, edge_type, edge_index, edge_type_description


def _get_edge_indices(data_dict, pos_keys, num_nodes, connectivity_setting):
    # save the offset for each node type. This is used to shift the edge indices to the correct node type
    index_shift_dict = {}
    for i, pos_key in enumerate(pos_keys):
        index_shift_dict[pos_key] = sum(num_nodes[0:i])
    edge_index_dict = {}
    # mesh edges: read from mesh_edge_index. Reverse edges for undirected graph
    mesh_edges = torch.cat((data_dict[keys.MESH_EDGE_INDEX], data_dict[keys.MESH_EDGE_INDEX][[1, 0]]), dim=1)
    mesh_edges += index_shift_dict[keys.MESH]
    edge_index_dict[keys.MESH_MESH] = mesh_edges

    if keys.COLLIDER in data_dict:
        if keys.COLLIDER_EDGE_INDEX in data_dict:  # add collider-collider edges: read from collider_edge_index
            collider_edges = torch.cat((data_dict[keys.COLLIDER_EDGE_INDEX],
                                        data_dict[keys.COLLIDER_EDGE_INDEX][[1, 0]]),
                                       dim=1)
            collider_edges += index_shift_dict[keys.COLLIDER]
            edge_index_dict[keys.COLLIDER_COLLIDER] = collider_edges
        # collider-mesh edges and mesh-collider edges

        # add collider-mesh edges and mesh-collider edges
        collider_mesh_edges = create_radius_edges(radius=connectivity_setting.collider_mesh_radius,
                                                  source_nodes=data_dict[keys.COLLIDER],
                                                  source_shift=index_shift_dict[keys.COLLIDER],
                                                  target_nodes=data_dict[keys.MESH],
                                                  target_shift=index_shift_dict[keys.MESH])
        edge_index_dict[keys.COLLIDER_MESH] = collider_mesh_edges
        mesh_collider_edges = torch.flip(collider_mesh_edges, dims=(0,))  # reverse edges
        edge_index_dict[keys.MESH_COLLIDER] = mesh_collider_edges

    if connectivity_setting.world_mesh_radius is not None:  # world-mesh edges: radius graph
        world_mesh_edges = create_radius_edges(radius=connectivity_setting.world_mesh_radius,
                                               source_nodes=data_dict[keys.MESH],
                                               source_shift=index_shift_dict[keys.MESH])
        remove_duplicates_with_mesh_edges(edge_index_dict[keys.MESH_MESH], world_mesh_edges)
        edge_index_dict[keys.WORLD_MESH] = world_mesh_edges
    return edge_index_dict


def _add_canonic_mesh_positions(edge_index_dict: ValueDict, edge_attr: torch.Tensor, edge_type: torch.Tensor,
                                input_mesh_edge_index: torch.Tensor,
                                initial_mesh_positions: torch.Tensor,
                                include_euclidean_distance: bool = True) -> torch.Tensor:
    """
    Adds the relative mesh positions to the mesh edges (in contrast to the world edges) and zero anywhere else.
    Refer to MGN by Pfaff et al. 2020 for more details.
    Args:
        edge_index_dict: Dictionary containing all different edges. Used to find out which are the mesh-mesh edges.
        edge_attr: Current edge features
        edge_type: Tensor containing the edges types
        input_mesh_edge_index: Mesh edge index tensor
        initial_mesh_positions: Initial positions of the mesh nodes "mesh coordinates"
        include_euclidean_distance: If true, the Euclidean distance is added to the mesh positions

    Returns:
        edge_attr: updated edge features
    """
    mesh_edge_type = list(edge_index_dict.keys()).index(keys.MESH_MESH)
    indices = torch.where(edge_type == mesh_edge_type)[0]  # only create edges between mesh nodes
    mesh_edge_index = torch.cat((input_mesh_edge_index, input_mesh_edge_index[[1, 0]]), dim=1).long()

    transformed_data = add_distances_from_positions(data_or_batch=Data(pos=initial_mesh_positions,
                                                                       edge_index=mesh_edge_index),
                                                    add_euclidian_distance=include_euclidean_distance)
    mesh_attr = transformed_data.edge_attr

    mesh_positions = torch.zeros(edge_attr.shape[0], mesh_attr.shape[1])  # fill other distances with 0
    mesh_positions[indices, :] = mesh_attr
    edge_attr = torch.cat((edge_attr, mesh_positions), dim=1)
    return edge_attr


def remove_duplicates_with_mesh_edges(mesh_edges: torch.Tensor, world_edges: torch.Tensor) -> torch.Tensor:
    """
    Removes the duplicates with the mesh edges have of the world edges that are created using a nearest neighbor search.
    (only MGN)
    To speed this up the adjacency matrices are used
    Args:
        mesh_edges: edge list of the mesh edges
        world_edges: edge list of the world edges

    Returns:
        new_world_edges: updated world edges without duplicates
    """
    import torch_geometric.utils as utils
    adj_mesh = utils.to_dense_adj(mesh_edges)
    if world_edges.shape[1] > 0:
        adj_world = utils.to_dense_adj(world_edges)
    else:
        adj_world = torch.zeros_like(adj_mesh)
    if adj_world.shape[1] < adj_mesh.shape[1]:
        padding_size = adj_mesh.shape[1] - adj_world.shape[1]
        padding_mask = torch.nn.ConstantPad2d((0, padding_size, 0, padding_size), 0)
        adj_world = padding_mask(adj_world)
    elif adj_world.shape[1] > adj_mesh.shape[1]:
        padding_size = adj_world.shape[1] - adj_mesh.shape[1]
        padding_mask = torch.nn.ConstantPad2d((0, padding_size, 0, padding_size), 0)
        adj_mesh = padding_mask(adj_mesh)
    new_adj = adj_world - adj_mesh
    new_adj[new_adj < 0] = 0
    new_world_edges = utils.dense_to_sparse(new_adj)[0]
    return new_world_edges
