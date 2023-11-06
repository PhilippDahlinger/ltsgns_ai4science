import os
import pickle

import torch
import torch_cluster
import torch_geometric.transforms as T
from tqdm import tqdm

from lts_gns.util.own_types import *


def get_connectivity_setting(dataset: str) -> Tuple:
    """
    Outputs the corresponding properties: edge radii, input time step and euclidean_distance for the used dataset
    Dataset here refers to the method how the graph is build from the input data which comes from SOFA
    Args:
        dataset: Name of that specifies the dataset
    Returns: Tuple.
        edge_radius_dict: Resulting edge radius dict for dataset
        input_timestep: Use point cloud of time step 't' or 't+1'
        euclidian_distance: Is a Euclidean distance as edges feature used
        tissue_task: Is this a 3D dataset
    """
    # 2D Deformable Plate connectivity settings
    if dataset == "coarse_meshgraphnet_t":
        edge_radius = [0.0, 0.08, None, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_meshgraphnet_world_t":
        edge_radius = [0.0, 0.08, 0.35, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3]  # 0.25 smallest mesh edge
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_full_graph_t":
        edge_radius = [0.1, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_no_pcd_edges_t":
        edge_radius = [0.0, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_no_pcd_edges_no_col_t":
        edge_radius = [0.0, 0.0, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_reduced_graph_t":
        edge_radius = [0.0, 0.08, None, 0.08, 0.08, 0.0, 0.08, 0.08, 0.0]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_reduced_graph_larger_radius_t":
        edge_radius = [0.0, 0.08, None, 0.18, 0.2, 0.0, 0.18, 0.2, 0.0]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_equal_edges_t":
        edge_radius = [0.2, 0.08, None, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        input_timestep = "t"
        euclidian_distance = True

    # 3D Tisse Manipulation connectivity settings
    elif dataset == "tissue_meshgraphnet_t":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_meshgraphnet_world_t":
        edge_radius = [0.0, 0.0, 0.2, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4]  # 0.068 # smalles mesh edge
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_no_pcd_edges_t":
        edge_radius = [0.0, 0.0, None, 0.16, 0.06, 0.4, 0.16, 0.06, 0.4]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_full_graph_t":
        edge_radius = [0.07, 0.0, None, 0.16, 0.06, 0.4, 0.16, 0.06, 0.4]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_reduced_pcd_edges_t":
        edge_radius = [0.05, 0.0, None, 0.16, 0.06, 0.4, 0.16, 0.06, 0.4]
        input_timestep = "t"
        euclidian_distance = True

    # 3D Cavity Grasping connectivity settings
    elif dataset == "cavity_meshgraphnet":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.05, 0.0, 0.0, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "cavity_meshgraphnet_world":
        edge_radius = [0.0, 0.0, 0.07, 0.0, 0.0, 0.05, 0.0, 0.0, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "cavity_no_pcd_edges":
        edge_radius = [0.0, 0.0, None, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "cavity_no_pcd_edges_nc":
        edge_radius = [0.0, 0.0, None, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "cavity_full_graph":
        edge_radius = [0.05, 0.0, None, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "cavity_no_collider":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        input_timestep = "t"
        euclidian_distance = True

    else:
        raise ValueError(f"Dataset {dataset} does currently not exist. Consider adding it to get_connectivity_setting")

    edge_radius_dict = get_radius_dict(edge_radius)

    if "tissue" in dataset:
        tissue_task = True
    elif "cavity" in dataset:
        tissue_task = True
    else:
        tissue_task = False

    return edge_radius_dict, input_timestep, euclidian_distance, tissue_task


def get_radius_dict(edge_radius: list) -> Dict:
    """
    Build an edge radius dict from a list of edge radii
    Args:
        edge_radius: List of the used edge radii
    Returns:
        edge_radius_dict: Dict containing the edge radii with their names
    """
    edge_radius_keys = [('grid', '0', 'grid'),
                        ('collider', '1', 'collider'),
                        ('mesh', '2', 'mesh'),
                        ('collider', '3', 'grid'),
                        ('mesh', '4', 'grid'),
                        ('mesh', '5', 'collider'),
                        ('grid', '6', 'collider'),
                        ('grid', '7', 'mesh'),
                        ('collider', '8', 'mesh')]

    edge_radius_dict = dict(zip(edge_radius_keys, edge_radius))
    return edge_radius_dict


def prepare_data_from_sofa(mesh_rollout_dict: Dict, use_color: bool = False, use_poisson: bool = False) -> Tuple:
    """
    Prepares the dict from SOFA for 2D data, normalizes it and outputs a tuple with important data
    Args:
        mesh_rollout_dict: Dict from SOFA
        use_color: True if color textured gradient is used
        use_poisson: True if the Poisson's ratio is used

    Returns: Tuple of data for point cloud, collider and mesh
    """
    nodes_grid = mesh_rollout_dict["nodes_grid"]
    edge_index_grid = mesh_rollout_dict["edge_index_grid"]
    nodes_collider = mesh_rollout_dict["nodes_collider"]
    pcd_positions = mesh_rollout_dict["pcd_points"]
    pcd_colors = mesh_rollout_dict["pcd_colors"]
    if use_poisson:
        poisson_ratio = mesh_rollout_dict["poisson_ratio"]
        poisson_ratio = (poisson_ratio + 0.205) * (200 / 139)  # normalize to -1,1
    else:
        poisson_ratio = None

    if not use_color:
        pcd_colors = None

    data = (pcd_positions, nodes_collider, nodes_grid, edge_index_grid, pcd_colors, poisson_ratio)
    return data


def prepare_data_for_trajectory(data: Tuple, timestep: int, input_timestep: str = 't+1', use_color: bool = False,
                                tissue_task: bool = False) -> Tuple:
    """
    Function to get the correct data and convert to tensors from a single time step of a trajectory of the prepared data output from SOFA
    Args:
        data: Tuple of the data from the prepare_from_sofa function
        timestep: timestep in trajectory
        input_timestep: Use collider and point cloud from time step 't' or 't+1'
        use_color:
        tissue_task:

    Returns: Tuple
        grid_positions: Tensor containing point cloud positions
        collider_positions: Tensor containing collider positions
        mesh_positions: Tensor containing mesh positions
        input_mesh_edge_index: Tensor containing mesh edge indices
        label: Tensor containing mesh positions of the next time step
        grid_colors: Tensor containing colors for point cloud
        initial_mesh_positions: Tensor containing the initial mesh positions for mesh-edge generation (see MGN)
        next_collider_positions (only 3D): Tensor containing the position of the collider in the next time step (to calculate velocity)
        poisson (only 2D): Tensor containing poisson ratio of the current data sample
    """
    if tissue_task:
        pcd_positions_grid, nodes_collider, nodes_grid, edge_index_grid, pcd_colors, poisson_ratio = data
    else:
        pcd_positions_grid, nodes_collider, nodes_grid, edge_index_grid, pcd_colors, poisson_ratio = data
    if poisson_ratio is not None:
        poisson_ratio = torch.tensor(poisson_ratio)

    if input_timestep == 't+1':
        grid_positions = torch.tensor(pcd_positions_grid[timestep + 1])
        collider_positions = torch.tensor(nodes_collider[timestep + 1])
        mesh_positions = torch.tensor(nodes_grid[timestep])
        mesh_edge_index = torch.tensor(edge_index_grid.T).long()
        label = torch.tensor(nodes_grid[timestep + 1])
        next_collider_positions = torch.tensor(nodes_collider[timestep + 2])
        initial_mesh_positions = torch.tensor(nodes_grid[0])
        if use_color:
            grid_colors = torch.tensor(pcd_colors[timestep + 1])
        else:
            grid_colors = None
    elif input_timestep == 't':
        grid_positions = torch.tensor(pcd_positions_grid[timestep])
        collider_positions = torch.tensor(nodes_collider[timestep])
        mesh_positions = torch.tensor(nodes_grid[timestep])
        mesh_edge_index = torch.tensor(edge_index_grid.T).long()
        label = torch.tensor(nodes_grid[timestep + 1])
        next_collider_positions = torch.tensor(nodes_collider[timestep + 1])
        initial_mesh_positions = torch.tensor(nodes_grid[0])
        if use_color:
            grid_colors = torch.tensor(pcd_colors[timestep])
        else:
            grid_colors = None
    else:
        raise ValueError("input_timestep can only be t or t+1")

    if tissue_task:
        data = grid_positions, collider_positions, mesh_positions, mesh_edge_index, label, grid_colors, initial_mesh_positions.float(), next_collider_positions, poisson_ratio
    else:
        data = grid_positions, collider_positions, mesh_positions, mesh_edge_index, label, grid_colors, initial_mesh_positions.float(), poisson_ratio

    return data


def build_one_hot_features(num_per_type: list) -> Tensor:
    """
    Builds one-hot feature tensor indicating the edge/node type from numbers per type
    Args:
        num_per_type: List of numbers of nodes per type

    Returns:
        features: One-hot features Tensor
    """
    total_num = sum(num_per_type)
    features = torch.zeros(total_num, len(num_per_type))
    for typ in range(len(num_per_type)):
        features[sum(num_per_type[0:typ]): sum(num_per_type[0:typ + 1]), typ] = 1
    return features


def build_type(num_per_type: list) -> Tensor:
    """
    Build node or edge type tensor from list of numbers per type
    Args:
        num_per_type: list of numbers per type

    Returns:
        features: Tensor containing the type as number
    """
    total_num = sum(num_per_type)
    features = torch.zeros(total_num)
    for typ in range(len(num_per_type)):
        features[sum(num_per_type[0:typ]): sum(num_per_type[0:typ + 1])] = typ
    return features


def remove_duplicates_with_mesh_edges(mesh_edges: Tensor, world_edges: Tensor) -> Tensor:
    """
    Removes the duplicates with the mesh edges have of the world edges that are created using a nearset neighbor search. (only MGN)
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


def get_relative_mesh_positions(mesh_edge_index: Tensor, mesh_positions: Tensor) -> Tensor:
    """
    Transform the positions of the mesh into a relative position encoding along with the Euclidean distance in the edges
    Args:
        mesh_edge_index: Tensor containing the mesh edge indices
        mesh_positions: Tensor containing mesh positions

    Returns:
        edge_attr: Tensor containing the batched edge features
    """
    data = Data(pos=mesh_positions,
                edge_index=mesh_edge_index)
    transforms = T.Compose([T.Cartesian(norm=False, cat=True), T.Distance(norm=False, cat=True)])
    data = transforms(data)
    return data.edge_attr


def add_relative_mesh_positions(edge_attr: Tensor, edge_type: Tensor, input_mesh_edge_index: Tensor,
                                initial_mesh_positions: Tensor) -> Tensor:
    """
    Adds the relative mesh positions to the mesh edges (in contrast to the world edges) and zero anywhere else.
    Refer to MGN by Pfaff et al. 2020 for more details.
    Args:
        edge_attr: Current edge features
        edge_type: Tensor containing the edges types
        input_mesh_edge_index: Mesh edge index tensor
        initial_mesh_positions: Initial positions of the mesh nodes "mesh coordinates"

    Returns:
        edge_attr: updated edge features
    """
    indices = torch.where(edge_type == 2)[0]  # type 2: mesh edges
    mesh_edge_index = torch.cat((input_mesh_edge_index, input_mesh_edge_index[[1, 0]]), dim=1).long()
    mesh_attr = get_relative_mesh_positions(mesh_edge_index, initial_mesh_positions)
    mesh_positions = torch.zeros(edge_attr.shape[0], mesh_attr.shape[1])
    mesh_positions[indices, :] = mesh_attr
    edge_attr = torch.cat((edge_attr, mesh_positions), dim=1)
    return edge_attr


def create_homo_graph_from_raw(input_data,
                               edge_radius_dict: Dict,
                               output_device,
                               use_mesh_coordinates: bool = False,
                               tissue_task: bool = False) -> Data:
    """
    Creates a homogeneous graph from the raw data (point cloud, collider, mesh) given the connectivity of the edge radius dict
    Args:
        input_data: Tuple containing the data for the time step
        edge_radius_dict: Edge radius dict describing the connectivity setting
        output_device: Working device for output data, either cpu or cuda
        use_mesh_coordinates: enables message passing also in mesh coordinate space
        tissue_task: True if 3D data is used

    Returns:
        data: Data element containing the built graph
    """
    if tissue_task:
        grid_positions, collider_positions, mesh_positions, input_mesh_edge_index, label, grid_colors, initial_mesh_positions, next_collider_positions, poisson_ratio = input_data
    else:
        grid_positions, collider_positions, mesh_positions, input_mesh_edge_index, label, grid_colors, initial_mesh_positions, poisson_ratio = input_data

    # dictionary for positions
    pos_dict = {'grid': grid_positions,
                'collider': collider_positions,
                'mesh': mesh_positions}

    # build nodes features (one hot)
    num_nodes = []
    for values in pos_dict.values():
        num_nodes.append(values.shape[0])
    x = build_one_hot_features(num_nodes)

    # add colors and collider velocity to point cloud (only 2D)
    if grid_colors is not None:
        x_colors = torch.zeros_like(x)
        x_colors[0:num_nodes[0], :] = grid_colors
        x_colors[num_nodes[0]:num_nodes[0] + num_nodes[1], 2] = torch.ones(num_nodes[1]) * (-200.0 / 100 * 0.01)
        x = torch.cat((x, x_colors), dim=1)
    node_type = build_type(num_nodes)

    # # used if poisson ratio needed as input feature, but atm incompatible with Imputation training
    if poisson_ratio is not None:
        poisson_ratio = poisson_ratio.float()
        x_poisson = torch.ones_like(x[:, 0])
        x_poisson = x_poisson * poisson_ratio
        x = torch.cat((x, x_poisson.unsqueeze(1)), dim=1)
    else:
        poisson_ratio = torch.tensor([1.0])

    # for 3D data add collider velocity and static mesh node information to node features, poisson only used for 2D
    if tissue_task:
        collider_velocities = (next_collider_positions - collider_positions).squeeze()
        collider_velocities = torch.nn.functional.normalize(collider_velocities, dim=0)
        x_velocities = torch.zeros((x.shape[0], 3))
        x_velocities[num_nodes[0]:num_nodes[0] + num_nodes[1], :] = collider_velocities
        x = torch.cat((x, x_velocities), dim=1)
        x = add_static_tissue_info(x, num_nodes)

    # index shift dict for edge index matrix
    index_shift_dict = {'grid': 0,
                        'collider': num_nodes[0],
                        'mesh': num_nodes[0] + num_nodes[1]}
    # get gripper edges for tube task todo/quick and dirty solution
    # tube_task = False
    # if num_nodes[2] > 400:
    #     tube_task = True
    #     collider_edge_index = torch.tensor(get_tube_collider_triangles())

    # create edge_index dict with the same keys as edge_radius
    edge_index_dict = {}
    for key in edge_radius_dict.keys():
        if key[0] == key[2]:

            # use mesh connectivity instead of nearest neighbor
            if key[0] == 'mesh':
                edge_index_dict[key] = torch.cat((input_mesh_edge_index, input_mesh_edge_index[[1, 0]]), dim=1)
                edge_index_dict[key][0, :] += index_shift_dict[key[0]]
                edge_index_dict[key][1, :] += index_shift_dict[key[2]]
            # elif key[0] == 'collider':
            #     if tube_task:
            #         edge_index_dict[key] = torch.cat((collider_edge_index, collider_edge_index[[1, 0]]), dim=1)
            #         edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            #         edge_index_dict[key][1, :] += index_shift_dict[key[2]]
            #     else:
            #         edge_index_dict[key] = torch_cluster.radius_graph(pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
            #         edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            #         edge_index_dict[key][1, :] += index_shift_dict[key[2]]
            # use radius graph for edges between nodes of the same type
            else:
                edge_index_dict[key] = torch_cluster.radius_graph(pos_dict[key[0]], r=edge_radius_dict[key],
                                                                  max_num_neighbors=100)
                edge_index_dict[key][0, :] += index_shift_dict[key[0]]
                edge_index_dict[key][1, :] += index_shift_dict[key[2]]

        # use radius for edges between different sender and receiver nodes
        else:
            edge_index_dict[key] = torch_cluster.radius(pos_dict[key[2]], pos_dict[key[0]], r=edge_radius_dict[key],
                                                        max_num_neighbors=100)
            edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            edge_index_dict[key][1, :] += index_shift_dict[key[2]]

    # add world edges if edge radius for mesh is not none
    mesh_key = ('mesh', '2', 'mesh')
    world_key = ('mesh', '9', 'mesh')
    if edge_radius_dict[mesh_key] is not None:
        edge_index_dict[world_key] = torch_cluster.radius_graph(pos_dict['mesh'], r=edge_radius_dict[mesh_key],
                                                                max_num_neighbors=100)
        edge_index_dict[world_key][0, :] += index_shift_dict['mesh']
        edge_index_dict[world_key][1, :] += index_shift_dict['mesh']
        edge_index_dict[world_key] = remove_duplicates_with_mesh_edges(edge_index_dict[mesh_key],
                                                                       edge_index_dict[world_key])

    # build edge_attr (one-hot)
    num_edges = []
    for value in edge_index_dict.values():
        num_edges.append(value.shape[1])
    edge_attr = build_one_hot_features(num_edges)
    edge_type = build_type(num_edges)

    # add mesh_coordinates to mesh edges if used
    if use_mesh_coordinates:
        edge_attr = add_relative_mesh_positions(edge_attr, edge_type, input_mesh_edge_index, initial_mesh_positions)

    # create node positions tensor and edge_index from dicts
    pos = torch.cat(tuple(pos_dict.values()), dim=0)
    edge_index = torch.cat(tuple(edge_index_dict.values()), dim=1)

    # create data object for torch
    data = Data(x=x.float(),
                u=poisson_ratio,
                pos=pos.float(),
                edge_index=edge_index.long(),
                edge_attr=edge_attr.float(),
                y=label.float(),
                y_old=mesh_positions.float(),
                node_type=node_type,
                edge_type=edge_type,
                poisson_ratio=poisson_ratio).to(output_device)
    return data


def get_feature_info_from_data(data_point: Data, device, hetero: bool, use_color: bool, tissue_task: bool,
                               use_world_edges: bool, use_mesh_coordinates: bool, mgn_hetero: bool):
    """
    Retrieves the features dimensions from the generated data
    Args:
        data_point: Single PyG data element from dataset
        device: Working device, either cpu or cuda
        hetero: Does nothing if False
        use_color: Color gradient texture for point cloud
        tissue_task: True if 3D data is used
        use_world_edges: Use of explicit world edges (see MGN)
        use_mesh_coordinates: Encode mesh coordinate into mesh edges
        mgn_hetero: Use small hetero GNN like in MGN

    Returns: Tuple.
        in_node_features: Dictionary of node types
        in_edge_features: Dictionary of edge types
        num_node_features: Feature dimension of node features
        num_edge_features: Feature dimension of edge features

    """
    if hetero:
        data_point.batch = torch.zeros_like(data_point.x)
        data_point.ptr = torch.tensor([0, data_point.batch.shape[0]])
        data_point = convert_to_hetero_data(data_point, hetero, use_color, device, tissue_task, use_world_edges,
                                            use_mesh_coordinates, mgn_hetero)
        if mgn_hetero:
            num_node_features = data_point.node_stores[0].num_features
            num_edge_features = data_point.edge_stores[0].num_edge_features
            in_node_features = {'mesh': data_point.node_stores[0].num_features}
            in_edge_features = {('mesh', '0', 'mesh'): data_point.edge_stores[0].num_edge_features,
                                ('mesh', '1', 'mesh'): data_point.edge_stores[1].num_edge_features}
        else:
            num_node_features = data_point.node_stores[0].num_features
            num_edge_features = data_point.edge_stores[0].num_edge_features
            in_node_features = {'grid': data_point.node_stores[0].num_features,
                                'collider': data_point.node_stores[1].num_features,
                                'mesh': data_point.node_stores[2].num_features}
            in_edge_features = {('grid', '0', 'grid'): data_point.edge_stores[0].num_edge_features,
                                ('collider', '1', 'collider'): data_point.edge_stores[1].num_edge_features,
                                ('mesh', '2', 'mesh'): data_point.edge_stores[2].num_edge_features,
                                ('collider', '3', 'grid'): data_point.edge_stores[3].num_edge_features,
                                ('mesh', '4', 'grid'): data_point.edge_stores[4].num_edge_features,
                                ('mesh', '5', 'collider'): data_point.edge_stores[5].num_edge_features,
                                ('grid', '6', 'collider'): data_point.edge_stores[6].num_edge_features,
                                ('grid', '7', 'mesh'): data_point.edge_stores[7].num_edge_features,
                                ('collider', '8', 'mesh'): data_point.edge_stores[8].num_edge_features}
            if len(data_point.edge_stores) > 9:
                in_edge_features[('mesh', '9', 'mesh')] = data_point.edge_stores[9].num_edge_features
                in_edge_features[('collider', '10', 'collider')] = data_point.edge_stores[0].num_edge_features
                in_edge_features[('grid', '11', 'grid')] = data_point.edge_stores[0].num_edge_features
    else:
        in_node_features = {"0": data_point.x.shape[1]}
        in_edge_features = {"0": data_point.edge_attr.shape[1]}
        num_node_features = in_node_features['0']
        num_edge_features = in_edge_features['0']

    return in_node_features, in_edge_features, num_node_features, num_edge_features


def transform_position_to_edges(data: Data, euclidian_distance: bool) -> Data:
    """
    Transform the node positions in a homogeneous data element to the edges as relative distance together with (if needed) Euclidean norm
    Args:
        data: Data element
        euclidian_distance: True if Euclidean norm included as feature

    Returns:
        out_data: Transformed data object
    """
    if euclidian_distance:
        data_transform = T.Compose([T.Cartesian(norm=False, cat=True), T.Distance(norm=False, cat=True)])
    else:
        data_transform = T.Compose([T.Cartesian(norm=False, cat=True)])
    out_data = data_transform(data)
    return out_data


def build_2d_dataset_for_split(input_dataset: str,
                               path: str,
                               split: str,
                               edge_radius_dict: Dict,
                               device,
                               input_timestep: str,
                               use_mesh_coordinates: bool,
                               hetero: bool = False,
                               raw: bool = False,
                               use_color: bool = False,
                               use_poisson: bool = False) -> list:
    print(f"Generating {split} data")
    with open(os.path.join(path, input_dataset, input_dataset + "_" + split + ".pkl"), "rb") as file:
        rollout_data = pickle.load(file)
    trajectory_list = []

    for index, trajectory in enumerate(tqdm(rollout_data)):
        rollout_length = len(trajectory["nodes_grid"])
        data_list = []
        trajectory = prepare_data_from_sofa(trajectory, use_color, use_poisson)

        for timestep in (range(rollout_length - 2)):
            if raw:
                data = create_raw_graph(trajectory, timestep, use_color=use_color)
            else:

                # get trajectory data for current timestep
                data_timestep = prepare_data_for_trajectory(trajectory, timestep, input_timestep=input_timestep,
                                                            use_color=use_color)

                # create nearest neighbor graph with the given radius dict
                if hetero:
                    data = create_hetero_graph_from_raw(data_timestep, edge_radius_dict=edge_radius_dict,
                                                        output_device=device,
                                                        use_mesh_coordinates=use_mesh_coordinates)
                else:
                    data = create_homo_graph_from_raw(data_timestep, edge_radius_dict=edge_radius_dict,
                                                      output_device=device,
                                                      use_mesh_coordinates=use_mesh_coordinates)

            data_list.append(data)  # append object for timestep t to data_list
        trajectory_list.append(data_list)  # create list of trajectories with each trajectory being a list itself

    return trajectory_list


if __name__ == "__main__":
    connectivity_str = "coarse_full_graph_t"
    edge_radius_dict, input_timestep, euclidian_distance, tissue_task = get_connectivity_setting(connectivity_str)

    split = "test"
    input_dataset = "deformable_plate"
    path = "/home/philipp/Documents/projects/datasets/lts_gns"
    use_color = False
    use_poisson = False
    raw = False
    hetero = False
    use_mesh_coordinates = True
    device = "cpu"
    use_world_edges = False
    mgn_hetero = False

    trajectory_list_train = build_2d_dataset_for_split(input_dataset=input_dataset,
                                                       path=path,
                                                       split=split,
                                                       edge_radius_dict=edge_radius_dict,
                                                       device=device,
                                                       input_timestep=input_timestep,
                                                       use_mesh_coordinates=use_mesh_coordinates,
                                                       hetero=hetero,
                                                       raw=raw,
                                                       use_color=use_color,
                                                       use_poisson=use_poisson)
    print("stop")
    data_list = trajectory_list_train[0]
    data_point = transform_position_to_edges(copy.deepcopy(data_list[0]).to(device), euclidian_distance)
    in_node_features, in_edge_features, num_node_features, num_edge_features = get_feature_info_from_data(data_point,
                                                                                                          device,
                                                                                                          hetero,
                                                                                                          use_color,
                                                                                                          tissue_task,
                                                                                                          use_world_edges,
                                                                                                          use_mesh_coordinates,
                                                                                                          mgn_hetero)
    out_node_features = data_point.y.shape[1]  # 2 for 2D case
    in_global_features = 1
    len_trajectory = len(data_list)
