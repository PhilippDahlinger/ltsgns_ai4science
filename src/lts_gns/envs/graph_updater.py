import warnings

import torch
from torch_geometric.data import Batch, Data

from lts_gns.envs.util.processing_util import add_gaussian_noise, add_distances_from_positions
from lts_gns.envs.util.edge_computation import build_edges, create_radius_edges, remove_duplicates_with_mesh_edges
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.util import node_type_mask, edge_type_mask


def recompute_mesh_collider_edges(graph: Data, preprocess_config: ConfigDict, device: torch.device | str) -> Data:
    """
    New function to recompute edges from data_dict, try to mimic the function from the env to get the edges
    in the first place.
    Args:
        graph: One timestep of the simulation
        preprocess_config: ConfigDict containing the connectivity setting
        device: Device to compute on
    Returns:

    """
    try:
        connectivity_setting = preprocess_config.connectivity_setting
    except ValueError:
        warnings.warn("No connectivity setting found. Skipping Edge recomputation.")
        return graph

    mesh_nodes = graph.pos[node_type_mask(graph, keys.MESH)]
    collider_nodes = graph.pos[node_type_mask(graph, keys.COLLIDER)]
    mesh_edge_mask = edge_type_mask(graph, keys.MESH_MESH)
    mesh_edges = graph.edge_index[:, mesh_edge_mask]
    # compute offsets for the small radius graph computations
    index_shift_dict = {keys.MESH: 0, keys.COLLIDER: mesh_nodes.shape[0]}
    edge_index_dict = {keys.MESH_MESH: mesh_edges}

    if keys.COLLIDER_COLLIDER in graph.edge_type_description:
        collider_edge_mask = edge_type_mask(graph, keys.COLLIDER_COLLIDER)
        collider_edges = graph.edge_index[:, collider_edge_mask]
        edge_index_dict |= {keys.COLLIDER_COLLIDER: collider_edges}

    if keys.MESH_COLLIDER in graph.edge_type_description:  # collider-mesh edges and mesh-collider edges
        collider_mesh_edges = create_radius_edges(radius=connectivity_setting.collider_mesh_radius,
                                                  source_nodes=collider_nodes,
                                                  source_shift=index_shift_dict[keys.COLLIDER],
                                                  target_nodes=mesh_nodes,
                                                  target_shift=index_shift_dict[keys.MESH])
        edge_index_dict[keys.COLLIDER_MESH] = collider_mesh_edges
        mesh_collider_edges = torch.flip(collider_mesh_edges, dims=(0,))  # reverse edges
        edge_index_dict[keys.MESH_COLLIDER] = mesh_collider_edges

    if keys.WORLD_MESH in graph.edge_type_description:  # mesh-mesh edges in world space
        world_mesh_edges = create_radius_edges(radius=connectivity_setting.world_mesh_radius,
                                               source_nodes=mesh_nodes,
                                               source_shift=index_shift_dict[keys.MESH])
        remove_duplicates_with_mesh_edges(edge_index_dict[keys.MESH_MESH], world_mesh_edges)
        # we do not need to add the reverse edges, as the edges in the mesh are undirected
        edge_index_dict[keys.WORLD_MESH] = world_mesh_edges

    # re-build all edges from the edge_index_dict.
    # This includes one-hot encodings and canonic (i.e., initial) distances

    # build edge_attr (one-hot)
    edge_attr, edge_index, edge_type, edge_type_description = build_edges(edge_index_dict)

    # add new attributes to device
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    edge_type = edge_type.to(device)

    if preprocess_config.use_canonic_mesh_positions:
        # reuse the canonic positions from the original graph, i.e., the initial distances between the mesh nodes
        mesh_edge_attr = graph.edge_attr[mesh_edge_mask]
        canonic_indices = [i for i, s in enumerate(graph.edge_type_description) if s.startswith("canonic")]
        original_canonic_mesh_positions = mesh_edge_attr[:, canonic_indices]
        # concat zeros to the remaining edges, as these are always the not mesh-mesh edges
        zero_vector = torch.zeros(edge_attr.shape[0] - original_canonic_mesh_positions.shape[0],
                                  len(canonic_indices), device=device)
        original_canonic_mesh_positions = torch.cat((original_canonic_mesh_positions, zero_vector)
                                                    , dim=0)
        edge_attr = torch.cat((edge_attr, original_canonic_mesh_positions), dim=1)

    graph.__setattr__("edge_attr", edge_attr)
    graph.__setattr__("edge_type", edge_type)
    graph.__setattr__("edge_index", edge_index)
    return graph


def _remove_edge_distances(batch: Batch) -> Batch:
    """
    remove the distances of the old edges if there are any. The distances are found via edge_type_description,
    which is a list of (identical) lists of strings over the graphs in the batch
    Args:
        batch:

    Returns:

    """

    canonic_indices = [i for i, s in enumerate(batch.edge_type_description[0])
                       if s in ["x_distance", "y_distance", "z_distance", "euclidian_distance"]]
    if canonic_indices:
        # Create a list of all indices
        all_indices = torch.arange(batch.edge_attr.shape[1])
        # Get the indices that are to be kept
        indices_to_keep = torch.tensor([index for index in all_indices if index not in canonic_indices])
        # Select only the desired indices
        batch.edge_attr = batch.edge_attr[:, indices_to_keep]
    return batch


class GraphUpdater:
    """
    Class that processes a batch of graphs by adding noise to the node positions, recomputing the edges, and
    transforming the node positions to edge features.
    """

    def __init__(self, env_config: ConfigDict, device: str | torch.device):
        self.postprocess_config = env_config.postprocess
        self.preprocess_config = env_config.preprocess
        self._device = device

    def process_batch(self, batch: Batch,
                      add_noise_to_node_positions: bool = True,
                      recompute_edges: bool = False) -> Batch:
        """
        Postprocesses the batch. This includes:
        - potentially include pointcloud data (# TODO)
        - convert to device
        - add noise to mesh and pointcloud positions
        - maybe recompute edges between the mesh and the collider/pointcloud/mesh in world space
        - transform positions to edge features
        Args:
            batch:
            add_noise_to_node_positions: Whether to add noise to node positions of the mesh
            recompute_edges: Whether to recompute the edges between the mesh and
                the collider/pointcloud/mesh in world space
        Returns: the updated batch of graphs

        """
        if self.preprocess_config.get("use_point_cloud", False):
            raise NotImplementedError("Point cloud data is not yet supported.")

        batch = batch.to(self._device)
        if add_noise_to_node_positions:
            batch = self._add_noise_to_node_positions(batch)
        if recompute_edges:
            # if all edges are recomputed, we don't need to remove the distances from the old edges
            batch = self._recompute_edges(batch=batch)
        else:
            batch = _remove_edge_distances(batch)

        batch = self._add_distances_from_positions(batch=batch)
        return batch

    def _recompute_edges(self, batch: Batch) -> Batch:
        if keys.COLLIDER in batch.node_type_description[0]:
            # recompute the edges between the mesh and the collider in world space
            data_list = batch.to_data_list()
            new_data_list = []
            for graph in data_list:
                recomputed_graph = recompute_mesh_collider_edges(graph=graph,
                                                                 preprocess_config=self.preprocess_config,
                                                                 device=self._device)
                new_data_list.append(recomputed_graph)
            batch = Batch.from_data_list(new_data_list)
        else:
            # no collider, so we can't recompute the edges. Instead, we remove the distances from the old edges
            batch = _remove_edge_distances(batch)
        return batch

    def _add_distances_from_positions(self, batch: Batch) -> Batch:
        return add_distances_from_positions(batch,
                                            add_euclidian_distance=self.postprocess_config.euclidian_distance_feature)

    def _add_noise_to_node_positions(self, batch: Batch) -> Batch:
        """
        Adds noise to the batch. This includes:
        - noise to mesh positions
        - noise to pointcloud positions
        :param batch: the batch to add noise to
        :return: the batch with added noise
        """
        if self.postprocess_config.input_mesh_noise > 0:
            batch = add_gaussian_noise(batch=batch,
                                       node_type_description=keys.MESH,
                                       sigma=self.postprocess_config.input_mesh_noise,
                                       device=self._device)

        if self.postprocess_config.input_point_cloud_noise > 0:
            raise NotImplementedError("Point cloud data is not yet supported.")
            # batch = _add_noise(batch, "point_cloud_positions", self.postprocess_config.input_point_cloud_noise)
        return batch
