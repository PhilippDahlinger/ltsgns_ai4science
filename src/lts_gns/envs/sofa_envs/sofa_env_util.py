import os
import pickle
from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from lts_gns.architectures.normalizers.world_to_model_normalizer import WorldToModelNormalizer
from lts_gns.envs.util.processing_util import process_global_features
from lts_gns.envs.util.edge_computation import build_edges_from_data_dict
from lts_gns.envs.util.task_processor_util import compute_model_velocities, get_one_hot_features_and_types
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict
from lts_gns.util.util import node_type_mask


def load_raw_data(path_to_datasets: str, dataset_name: str, split: str, load_pc2mesh: bool = False) -> List[DataLoader]:
    if load_pc2mesh:
        with open(os.path.join(path_to_datasets, dataset_name, "pc2mesh" + "_" + split + ".pkl"), "rb") as file:
            rollout_data = pickle.load(file)
    else:
        with open(os.path.join(path_to_datasets, dataset_name, dataset_name + "_" + split + ".pkl"), "rb") as file:
            rollout_data = pickle.load(file)
    return rollout_data


def select_and_normalize_attributes(raw_task: ValueDict,
                                    use_point_cloud: bool,
                                    use_poisson_ratio: bool) -> dict:
    """
    Function to remove unnecessary attributes from the raw data and normalize the data if necessary.

    This function handles the deformable plate and the tissue manipulation task. For the latter, we *only* have
    mesh information about the tissue and the 3d-position of the gripper for the train and validation sets.
    We do *not* have the gripper mesh or the positions of the gripper mesh.
    We also do not have the liver mesh or the liver mesh positions.
    These are only available for the test set, and as such should be handled as if there were no collider edges.

    :param raw_task:  ValueDict with all the data for all timesteps
    :param use_point_cloud: bool: if point cloud should be used
    :param use_poisson_ratio: bool: if poisson ratio should be used as a feature
    :return: dict with all the data for all timesteps
    """
    if "nodes_grid" in raw_task.keys():  # deformable plate
        task: ValueDict = {keys.MESH: raw_task["nodes_grid"],
                           keys.MESH_EDGE_INDEX: raw_task["edge_index_grid"],
                           keys.MESH_FACES: raw_task["triangles_grid"],
                           keys.COLLIDER: raw_task["nodes_collider"],
                           keys.COLLIDER_EDGE_INDEX: raw_task["edge_index_collider"],
                           keys.COLLIDER_FACES: raw_task["triangles_collider"],
                           }
    elif "tissue_mesh_positions" in raw_task.keys():  # tissue manipulation
        task: ValueDict = {keys.MESH: raw_task["tissue_mesh_positions"],
                           keys.MESH_EDGE_INDEX: raw_task["tissue_mesh_edges"],
                           keys.MESH_FACES: raw_task["tissue_mesh_triangles"],
                           keys.COLLIDER: raw_task["gripper_position"],
                           }
        if "gripper_mesh_positions" in raw_task.keys():
            # visual information for the test set
            task[keys.VISUAL_COLLDER] = raw_task["gripper_mesh_positions"]
            task[keys.VISUAL_COLLIDER_FACES] = raw_task["gripper_mesh_triangles"]
    else:
        raise ValueError("Unknown task type")

    if use_point_cloud:
        if "pcd_points" in raw_task.keys():
            # deformable plate
            task[keys.POINT_CLOUD_POSITIONS] = raw_task["pcd_points"]
            task[keys.POINT_CLOUD_COLORS] = raw_task["pcd_colors"]
        elif "tissue_pcd_points" in raw_task.keys():
            # tissue manipulation
            task[keys.POINT_CLOUD_POSITIONS] = raw_task["tissue_pcd_points"]
        else:
            raise ValueError("Unknown task type/point clouds not supported!")

    if use_poisson_ratio:
        # always normalize poisson ratio and add it to the task, although it will usually not be used
        poisson_ratio = raw_task[keys.POISSON_RATIO]
        # normalize to -1,1
        poisson_ratio = (poisson_ratio + 0.205) * (200 / 139)
        task[keys.POISSON_RATIO] = poisson_ratio

    return task


def build_data_dict(task: ValueDict, timestep: int, world_to_model_normalizer: WorldToModelNormalizer) -> ValueDict:
    """
    Function to get the correct data and convert to tensors from a single time step of a trajectory of the prepared data
    output from SOFA
    :return: Dict containing all the data for a single timestep in torch tensor format.
    :param task: ValueDict with all the data for all timesteps
    :param timestep: timestep to get the data for
    :param world_to_model_normalizer: WorldToModelNormalizer to normalize the data
    """
    data_dict = {keys.MESH: torch.tensor(task[keys.MESH][timestep], dtype=torch.float32),
                 keys.NEXT_MESH_POS: torch.tensor(task[keys.MESH][timestep + 1], dtype=torch.float32),
                 keys.INITIAL_MESH_POSITIONS: torch.tensor(task[keys.MESH][0], dtype=torch.float32)}
    # the pc2mesh meshes have a mesh_edge_index for every step, catch that
    if isinstance(task[keys.MESH_EDGE_INDEX], list) and len(task[keys.MESH_EDGE_INDEX]) == len(task[keys.MESH]):
        data_dict[keys.MESH_EDGE_INDEX] = torch.tensor(task[keys.MESH_EDGE_INDEX][timestep].T, dtype=torch.long)
        data_dict[keys.MESH_FACES] = torch.tensor(task[keys.MESH_FACES][timestep], dtype=torch.long)
    else:
        data_dict[keys.MESH_EDGE_INDEX] = torch.tensor(task[keys.MESH_EDGE_INDEX].T, dtype=torch.long)
        data_dict[keys.MESH_FACES] = torch.tensor(task[keys.MESH_FACES], dtype=torch.long)

    if keys.COLLIDER in task.keys():  # add information about the collider mesh
        data_dict |= {keys.COLLIDER: torch.tensor(task[keys.COLLIDER][timestep], dtype=torch.float32)
                      }
        data_dict |= {keys.COLLIDER_VELOCITY: torch.tensor(task[keys.COLLIDER][timestep + 1] - task[keys.COLLIDER][timestep], dtype=torch.float32)}
    if keys.COLLIDER_EDGE_INDEX in task.keys():  # Collider has more than a single node
        data_dict |= {
                      keys.COLLIDER_EDGE_INDEX: torch.tensor(task[keys.COLLIDER_EDGE_INDEX].T, dtype=torch.long),
                      keys.COLLIDER_FACES: torch.tensor(task[keys.COLLIDER_FACES], dtype=torch.long), }

    if keys.VISUAL_COLLDER in task.keys():
        # add visual information about the collider mesh
        data_dict |= {keys.VISUAL_COLLDER: torch.tensor(task[keys.VISUAL_COLLDER][timestep], dtype=torch.float32),
                      keys.VISUAL_COLLIDER_FACES: torch.tensor(task[keys.VISUAL_COLLIDER_FACES], dtype=torch.long), }

    # have the normalized velocities (e.g. in model space) as labels
    model_velocities = compute_model_velocities(data_dict, world_to_model_normalizer=world_to_model_normalizer, dt=1.0)
    data_dict[keys.LABEL] = model_velocities

    if keys.POISSON_RATIO in task:
        data_dict[keys.POISSON_RATIO] = torch.tensor(task[keys.POISSON_RATIO], dtype=torch.float32)

    if keys.POINT_CLOUD_POSITIONS in task:
        data_dict[keys.POINT_CLOUD_POSITIONS] = torch.tensor(task[keys.POINT_CLOUD_POSITIONS][timestep], dtype=torch.float32)
    if keys.POINT_CLOUD_COLORS in task:
        data_dict[keys.POINT_CLOUD_COLORS] = torch.tensor(task[keys.POINT_CLOUD_COLORS][timestep], dtype=torch.float32)

    return data_dict


def build_graph(data_dict: ValueDict, connectivity_setting: ConfigDict, use_canonic_mesh_positions,
                task_properties_input_selection: str,
                use_collider_velocities: bool) -> Data:
    """
    Function to build the graph from the data_dict
    :param data_dict: Dict containing all the data for a single timestep in torch tensor format.
    :param connectivity_setting: ConfigDict with the connectivity settings
    :param use_canonic_mesh_positions: bool: if the canonic mesh positions should be used
    :param task_properties_input_selection: str: which task properties should be used
    :param use_collider_velocities: bool: if the collider velocities should be used as part of the x features
    :return: Graph created from the data_dict in torch_geometric format
    """
    # build nodes features (one hot node type)
    pos_keys = [keys.MESH]
    if keys.COLLIDER in data_dict:
        pos_keys.append(keys.COLLIDER)

    num_nodes = [data_dict[pos_key].shape[0] for pos_key in pos_keys]

    x, node_type = get_one_hot_features_and_types(input_list=num_nodes)
    if use_collider_velocities:
        collider_vel = data_dict[keys.COLLIDER_VELOCITY]
        padded_collider_vel = torch.zeros(size=(x.shape[0], collider_vel.shape[1]))
        padded_collider_vel[node_type == pos_keys.index(keys.COLLIDER)] = collider_vel
        x = torch.cat((x, padded_collider_vel), dim=1)

    pos = torch.cat(tuple(data_dict[pos_key] for pos_key in pos_keys), dim=0)

    # point cloud data
    if keys.POINT_CLOUD_POSITIONS in data_dict:
        point_cloud_positions = data_dict[keys.POINT_CLOUD_POSITIONS]
    else:
        point_cloud_positions = None

    # we save the poisson ratio as task property for the Task Properties Posterior Learner and directly
    # use it as node feature
    if keys.POISSON_RATIO in data_dict:
        task_properties = torch.tensor([data_dict[keys.POISSON_RATIO]]).reshape(1, -1)
        task_properties_description = ["poisson_ratio (normalized)"]
    else:
        task_properties = torch.zeros(size=(1, 0))
        task_properties_description = []

    x, u = process_global_features(x, task_properties, task_properties_input_selection)

    if keys.COLLIDER_FACES in data_dict:
        # collider exists. Move the indices of the collider faces to the correct position for proper visualization
        collider_faces = data_dict[keys.COLLIDER_FACES] + data_dict[keys.MESH].shape[0]
        collider_vertices = None  # given in "x"
    elif keys.VISUAL_COLLIDER_FACES in data_dict:
        # no collider exists, but a collider can be visualized. Indices do not need to be moved, but positions stored
        collider_faces = data_dict[keys.VISUAL_COLLIDER_FACES]
        collider_vertices = data_dict[keys.VISUAL_COLLDER]
    else:
        collider_faces = None
        collider_vertices = None

    data = Data(x=x,
                u=u,
                pos=pos,
                next_mesh_pos=data_dict[keys.NEXT_MESH_POS],
                y=data_dict[keys.LABEL],
                node_type=node_type,
                node_type_description=pos_keys,
                task_properties=task_properties,
                task_properties_description=task_properties_description,
                mesh_faces=data_dict[keys.MESH_FACES],
                collider_faces=collider_faces,
                collider_vertices=collider_vertices,
                point_cloud_positions=point_cloud_positions,
                )

    # data = _add_edges(data,
    #                   connectivity_setting,
    #                   use_canonic_mesh_positions)

    # edge features: edge_attr is just one-hot edge type, all other features are created after preprocessing
    edge_attr, edge_type, edge_index, edge_type_description = build_edges_from_data_dict(data_dict,
                                                                                         pos_keys,
                                                                                         num_nodes,
                                                                                         connectivity_setting,
                                                                                         use_canonic_mesh_positions)

    data.__setattr__("edge_attr", edge_attr)
    data.__setattr__("edge_type", edge_type)
    data.__setattr__("edge_index", edge_index)
    data.__setattr__("edge_type_description", edge_type_description)

    return data
