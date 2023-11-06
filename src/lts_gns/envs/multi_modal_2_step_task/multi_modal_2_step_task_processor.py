from typing import List, Dict, Callable

import torch
from torch_geometric.data import Data, Batch

from lts_gns.algorithms.simulators.simulator_util import integrate_predictions
from lts_gns.envs.abstract_processor import AbstractDataLoaderProcessor
from lts_gns.envs.multi_modal_2_step_task.multi_modal_2_step_task_util import load_raw_data, \
    select_and_normalize_attributes, \
    build_data_dict, \
    build_graph
from lts_gns.util import keys
from lts_gns.util.own_types import ValueDict
from lts_gns.util.util import wrapped_partial


class MultiModal2StepTaskProcessor(AbstractDataLoaderProcessor):

    ###########################################
    ####### Interfaces for data loading #######
    ###########################################

    def _get_rollout_length(self, raw_task: ValueDict) -> int:
        return len(raw_task["pos"])

    def _load_raw_data(self, split: str) -> List[ValueDict]:
        return load_raw_data(self.preprocess_config.save_load.path_to_datasets, self.task_name, split)

    def _select_and_normalize_attributes(self, raw_task: ValueDict) -> ValueDict:
        return select_and_normalize_attributes(raw_task)

    def _build_data_dict(self, raw_task: ValueDict, timestep: int) -> ValueDict:
        return build_data_dict(raw_task, timestep, self.world_to_model_normalizer)

    def _build_graph(self, data_dict: ValueDict) -> Data:
        return build_graph(data_dict, self.preprocess_config.task_properties_input_selection)

    ###########################################
    ####### Functions for the processor #######
    ###########################################

    def get_integrate_predictions_fn(self) -> Callable[[torch.Tensor, Batch], Dict[str, torch.Tensor]]:
        return wrapped_partial(integrate_predictions, integration_order=1, d_t=1.0)

    def get_update_batch_fn(self) -> Callable[[Batch, Dict[str, torch.Tensor]], Batch]:
        def update_batch(batch: Batch, mesh_state: Dict[str, torch.Tensor]) -> Batch:
            assert keys.POSITIONS in mesh_state.keys(), "The mesh_state must contain the positions of the nodes"

            # the new pos of the mesh state has shape (num_samples, num_nodes, 2)
            # or (num_nodes, 2)
            if len(mesh_state[keys.POSITIONS].shape) == 2:
                new_positions = mesh_state[keys.POSITIONS]
            elif len(mesh_state[keys.POSITIONS].shape) == 3:
                assert mesh_state[keys.POSITIONS].shape[0] == 1, "The mesh_state must contain only one z sample"
                new_positions = mesh_state[keys.POSITIONS][0]
            else:
                raise ValueError("The Positions must have shape (num_samples, num_nodes, 2) or (num_nodes, 2)")
            batch.pos = new_positions
            # update also the x features: x = [pos, possible_task_properties]
            batch.x = torch.cat([batch.pos, batch.x[:, 2:]], dim=1)
            return batch

        return update_batch
