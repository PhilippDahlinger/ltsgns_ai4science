from typing import List, Dict, Callable

import torch
from torch_geometric.data import Data, Batch

from lts_gns.algorithms.simulators.simulator_util import integrate_predictions
from lts_gns.envs.abstract_processor import AbstractDataLoaderProcessor
from lts_gns.envs.sofa_envs.sofa_env_util import load_raw_data, select_and_normalize_attributes, build_data_dict, \
    build_graph
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict
from lts_gns.util.util import wrapped_partial, node_type_mask


class SofaDataLoaderProcessor(AbstractDataLoaderProcessor):

    def __init__(self, env_config: ConfigDict, task_name: str):
        super().__init__(env_config, task_name)
        self.load_pc2mesh = False


    ###########################################
    ####### Interfaces for data loading #######
    ###########################################

    def _get_rollout_length(self, raw_task: ValueDict) -> int:
        """
        Returns the rollout length of the task. This is the number of timesteps in the rollout.
        We have a -1 here because the initial for loop went to rollout_length -2 here and -1 for all other tasks
        Args:
            raw_task:

        Returns:

        """
        if "nodes_grid" in raw_task.keys():
            # Deforming plate task
            return len(raw_task["nodes_grid"]) - 1
        elif "tissue_mesh_positions" in raw_task.keys():
            # Tissue manipulation task
            return len(raw_task["tissue_mesh_positions"]) - 1
        else:
            raise ValueError("Unknown dataset format. Cannot determine rollout")

    def _load_raw_data(self, split: str) -> List[ValueDict]:
        return load_raw_data(self.preprocess_config.save_load.path_to_datasets,
                             self.task_name, split, load_pc2mesh=self.load_pc2mesh)

    def _select_and_normalize_attributes(self, raw_task: ValueDict) -> ValueDict:
        return select_and_normalize_attributes(raw_task,
                                               use_point_cloud=self.preprocess_config.use_point_cloud,
                                               use_poisson_ratio=self.preprocess_config.use_poisson_ratio,
                                               )

    def _build_data_dict(self, raw_task: ValueDict, timestep: int) -> ValueDict:
        return build_data_dict(raw_task, timestep, self.world_to_model_normalizer)

    def _build_graph(self, data_dict: ValueDict) -> Data:
        return build_graph(data_dict,
                           connectivity_setting=self.preprocess_config.connectivity_setting,
                           use_canonic_mesh_positions=self.preprocess_config.use_canonic_mesh_positions,
                           task_properties_input_selection=self.preprocess_config.task_properties_input_selection,
                           use_collider_velocities=self.preprocess_config.use_collider_velocities,)

    ###########################################
    ####### Functions for the processor #######
    ###########################################

    def get_integrate_predictions_fn(self) -> Callable[[torch.Tensor, Batch], Dict[str, torch.Tensor]]:
        return wrapped_partial(integrate_predictions, integration_order=1, d_t=1.0)

    def get_update_batch_fn(self) -> Callable[[Batch, Dict[str, torch.Tensor]], Batch]:
        def update_batch(batch: Batch, mesh_state: Dict[str, torch.Tensor]) -> Batch:
            assert keys.POSITIONS in mesh_state.keys(), "The mesh_state must contain the positions of the nodes"
            if len(mesh_state[keys.POSITIONS].shape) == 2:
                new_positions = mesh_state[keys.POSITIONS]
            elif len(mesh_state[keys.POSITIONS].shape) == 3:
                assert mesh_state[keys.POSITIONS].shape[0] == 1, "The mesh_state must contain only one z sample"
                new_positions = mesh_state[keys.POSITIONS][0]
            else:
                raise ValueError("The mesh_state must contain either 2D or 3D positions")
            mesh_nodes_mask = node_type_mask(graph_or_batch=batch, key=keys.MESH)
            batch.pos[mesh_nodes_mask] = new_positions
            return batch

        return update_batch
