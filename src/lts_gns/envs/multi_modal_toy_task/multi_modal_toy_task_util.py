import os
from typing import List

import einops
import numpy as np
import torch
from torch_geometric.data import Data

from lts_gns.architectures.normalizers.world_to_model_normalizer import WorldToModelNormalizer
from lts_gns.envs.util.processing_util import process_global_features
from lts_gns.envs.util.task_processor_util import compute_model_velocities
from lts_gns.util import keys
from lts_gns.util.own_types import ValueDict


def load_raw_data(path_to_datasets: str, dataset_name: str, split: str) -> List[ValueDict]:
    path = os.path.join(path_to_datasets, dataset_name, dataset_name + "_" + split + ".npy")
    raw_data = np.load(path, allow_pickle=True)
    return raw_data


def select_and_normalize_attributes(raw_task: ValueDict) -> dict:
    """
    Function to remove unnecessary attributes from the raw data and normalize the data if necessary.
    :param raw_task:  ValueDict with all the data for all timesteps
    :param use_point_cloud: bool: if point cloud should be used
    :return: dict with all the data for all timesteps
    """
    task: ValueDict = {keys.MESH: raw_task["pos"],
                       "vel": raw_task["vel"],
                       "k_p": raw_task["k_p"],
                       "t_d": raw_task["t_d"],
                       "mass": raw_task["mass"],
                       "goal_pos": raw_task["goal_pos"],
                       }

    return task


def build_data_dict(task: ValueDict, timestep: int, world_to_model_normalizer: WorldToModelNormalizer) -> ValueDict:
    """
    Function to get the correct data and convert to tensors from a single time step of a trajectory of the prepared data output from SOFA
    :return: Dict containing all the data for a single timestep in torch tensor format.
    :param task: ValueDict with all the data for all timesteps
    :param timestep: timestep to get the data for
    :param world_to_model_normalizer: WorldToModelNormalizer object to normalize the velocities
    """
    data_dict = {keys.MESH: torch.tensor(task[keys.MESH][timestep], dtype=torch.float32),
                 "vel": torch.tensor(task["vel"][timestep], dtype=torch.float32),
                 keys.NEXT_MESH_POS: torch.tensor(task[keys.MESH][timestep + 1], dtype=torch.float32),
                 "k_p": torch.tensor([task["k_p"]], dtype=torch.float32),
                 "t_d": torch.tensor([task["t_d"]], dtype=torch.float32),
                 "mass": torch.tensor([task["mass"]], dtype=torch.float32),
                 "goal_pos": torch.tensor(task["goal_pos"], dtype=torch.float32),
                 "time_step": torch.tensor([timestep], dtype=torch.float32),
                    }

    # have the normalized velocities (e.g. in model space) as labels
    # dt is 0.1 here
    model_velocities = compute_model_velocities(data_dict, world_to_model_normalizer=world_to_model_normalizer, dt=0.1)
    data_dict[keys.LABEL] = model_velocities

    return data_dict


def build_graph(data_dict: ValueDict, task_properties_input_selection: str, use_vel_features: bool) -> Data:
    """
    Function to build the graph from the data_dict
    :param data_dict: Dict containing all the data for a single timestep in torch tensor format.
    :return: Graph created from the data_dict in torch_geometric format
    """
    # build nodes features (position and velocity)
    pos = einops.rearrange(data_dict[keys.MESH], 'd_world -> 1 d_world')

    if use_vel_features:
        vel = einops.rearrange(data_dict["vel"], 'd_world -> 1 d_world')
        x = torch.cat((pos, vel), dim=1)
    else:
        x = pos
    next_mesh_pos = einops.rearrange(data_dict[keys.NEXT_MESH_POS], 'd_world -> 1 d_world')
    y = einops.rearrange(data_dict[keys.LABEL], 'd_world -> 1 d_world')

    # we save the goal pos, the mass and the controller parameters as task properties
    task_properties = torch.cat([data_dict["goal_pos"], data_dict["mass"], data_dict["k_p"], data_dict["t_d"]]).reshape(
        1, -1)
    task_properties_description = ["goal_pos_x", "goal_pos_y", "mass", "k_p", "t_d"]
    x, u = process_global_features(x, task_properties, task_properties_input_selection)

    data = Data(x=x,
                pos=pos,
                next_mesh_pos=next_mesh_pos,
                y=y,
                u=u,
                node_type=torch.tensor([0.]),
                node_type_description=[keys.MESH],
                task_properties=task_properties,
                task_properties_description=task_properties_description,
                goal_pos=data_dict["goal_pos"],
                )

    # no edges, so empty tensors
    data.__setattr__("edge_attr", torch.zeros((0, 1)))
    data.__setattr__("edge_type", torch.zeros((0,)))
    data.__setattr__("edge_index", torch.zeros((2, 0), dtype=torch.long))
    data.__setattr__("edge_type_description", [])
    return data
