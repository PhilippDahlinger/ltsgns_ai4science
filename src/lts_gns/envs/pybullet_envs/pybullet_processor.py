from typing import List, Dict, Callable

import torch
from torch_geometric.data import Data, Batch

from lts_gns.algorithms.simulators.simulator_util import integrate_predictions
from lts_gns.envs.abstract_processor import AbstractDataLoaderProcessor
from lts_gns.envs.sofa_envs.sofa_env_util import load_raw_data, build_edges_from_data_dict
from lts_gns.envs.util.processing_util import process_global_features
from lts_gns.envs.util.task_processor_util import compute_model_velocities, get_one_hot_features_and_types
from lts_gns.util import keys
from lts_gns.util.own_types import ValueDict
from lts_gns.util.util import wrapped_partial, node_type_mask


class PyBulletDataLoaderProcessor(AbstractDataLoaderProcessor):

    ###########################################
    ####### Interfaces for data loading #######
    ###########################################

    def _get_rollout_length(self, raw_task: ValueDict) -> int:
        """
        Returns the rollout length of the task. This is the number of timesteps in the rollout.
        We have - 2 because we remove one time step for velocity information. Also the last time step is the final goal.
        Args:
            raw_task:

        Returns:

        """
        return len(raw_task["deformable_mesh"]) - 2

    def _load_raw_data(self, split: str) -> List[ValueDict]:
        path_to_datasets = self.preprocess_config.save_load.path_to_datasets
        dataset_name = self.task_name
        if self.preprocess_config.save_load.version is not None:
            dataset_name += "_" + self.preprocess_config.save_load.version
        return load_raw_data(path_to_datasets, dataset_name, split)

    def _select_and_normalize_attributes(self, raw_task: ValueDict) -> ValueDict:
        task: ValueDict = {keys.MESH: raw_task["deformable_mesh"],
                           keys.MESH_EDGE_INDEX: raw_task["deformable_edges"],
                           keys.MESH_FACES: raw_task["deformable_faces"],
                           keys.COLLIDER: raw_task["stick_mesh"],
                           keys.COLLIDER_EDGE_INDEX: raw_task["stick_edges"],
                           keys.COLLIDER_FACES: raw_task["stick_faces"],
                           "spring_stiffness": raw_task["spring_elastic_stiffness"],
                           }
        # normalize the spring stiffness
        task["spring_stiffness"] = (task["spring_stiffness"] - 100.421) / (998.633 - 100.421)

        if "fixed_mesh_nodes" in raw_task:
            task[keys.FIXED_MESH_INDICES] = raw_task["fixed_mesh_nodes"]
        return task

    def _build_data_dict(self, raw_task: ValueDict, timestep: int) -> ValueDict:
        data_dict = {keys.MESH: torch.tensor(raw_task[keys.MESH][timestep], dtype=torch.float32),
                     keys.MESH_EDGE_INDEX: torch.tensor(raw_task[keys.MESH_EDGE_INDEX].T, dtype=torch.long),
                     keys.MESH_FACES: torch.tensor(raw_task[keys.MESH_FACES], dtype=torch.long),
                     keys.COLLIDER: torch.tensor(raw_task[keys.COLLIDER][timestep], dtype=torch.float32),
                     keys.COLLIDER_EDGE_INDEX: torch.tensor(raw_task[keys.COLLIDER_EDGE_INDEX].T, dtype=torch.long),
                     keys.COLLIDER_FACES: torch.tensor(raw_task[keys.COLLIDER_FACES], dtype=torch.long),
                     keys.NEXT_MESH_POS: torch.tensor(raw_task[keys.MESH][timestep + 1], dtype=torch.float32),
                     keys.INITIAL_MESH_POSITIONS: torch.tensor(raw_task[keys.MESH][0], dtype=torch.float32)}

        # have the normalized velocities (e.g. in model space) as labels
        model_velocities = compute_model_velocities(data_dict, world_to_model_normalizer=self.world_to_model_normalizer,
                                                    dt=1.0)
        data_dict[keys.LABEL] = model_velocities

        data_dict["spring_stiffness"] = torch.tensor(raw_task["spring_stiffness"], dtype=torch.float32)

        if keys.FIXED_MESH_INDICES in raw_task:
            data_dict[keys.FIXED_MESH_INDICES] = torch.tensor(raw_task[keys.FIXED_MESH_INDICES], dtype=torch.long)
        return data_dict

    def _build_graph(self, data_dict: ValueDict) -> Data:
        """
           Function to build the graph from the data_dict
           :param data_dict: Dict containing all the data for a single timestep in torch tensor format.
           :return: Graph created from the data_dict in torch_geometric format
           """
        # build nodes features (one hot node type)
        node_type_description = [keys.MESH, keys.COLLIDER]
        num_nodes = []
        for pos_key in node_type_description:
            num_nodes.append(data_dict[pos_key].shape[0])
        x, node_type = get_one_hot_features_and_types(num_nodes)
        pos = torch.cat(tuple(data_dict[pos_key] for pos_key in node_type_description), dim=0)

        # we save the poisson ratio as task property for the Task Properties Posterior Learner and directly
        # use it as node feature
        if "spring_stiffness" in data_dict:
            task_properties = torch.tensor([data_dict["spring_stiffness"]]).reshape(1, -1)
            task_properties_description = ["spring_stiffness (normalized)"]
        else:
            task_properties = torch.zeros(size=(1, 0))
            task_properties_description = []

        if keys.FIXED_MESH_INDICES in data_dict:
            # give every node that is fixed a "1" as another node feature, and all others a "0". Add this to the node
            # type description, and overwrite the node type of the fixed nodes
            fixed_indices = data_dict[keys.FIXED_MESH_INDICES]
            fixed_index_mask = torch.zeros(size=(x.shape[0], 1), dtype=torch.float32)
            fixed_index_mask[fixed_indices] = 1
            x = torch.cat([x, fixed_index_mask], dim=1)  # add a 1-hot feature for the fixed node
            node_type_description.append(keys.FIXED_MESH)

            # set the node type of the fixed nodes to the last entry in the node type description, which is
            # a novel node type since we just added an element to the node type description
            node_type[fixed_indices] = len(node_type_description) - 1

            # remove labels for fixed nodes by removing them from the label tensor
            rows_to_keep = torch.ones(size=(data_dict[keys.MESH].shape[0],), dtype=torch.bool)
            rows_to_keep[fixed_indices] = 0
            labels = data_dict[keys.LABEL]
            labels = labels[rows_to_keep]
            next_mesh_pos = data_dict[keys.NEXT_MESH_POS][rows_to_keep]
        else:
            labels = data_dict[keys.LABEL]
            next_mesh_pos = data_dict[keys.NEXT_MESH_POS]

        x, u = process_global_features(x,
                                       global_features=task_properties,
                                       input_selection=self.preprocess_config.task_properties_input_selection)

        # move the indices of the collider faces to the correct position
        collider_faces = data_dict[keys.COLLIDER_FACES] + data_dict[keys.MESH].shape[0]

        data = Data(x=x,
                    u=u,
                    pos=pos,
                    next_mesh_pos=next_mesh_pos,
                    y=labels,
                    node_type=node_type,
                    node_type_description=node_type_description,
                    task_properties=task_properties,
                    task_properties_description=task_properties_description,
                    mesh_faces=data_dict[keys.MESH_FACES],
                    collider_faces=collider_faces,
                    )

        # data = _add_edges(data,
        #                   connectivity_setting,
        #                   use_canonic_mesh_positions)

        # edge features: edge_attr is just one-hot edge type, all other features are created after preprocessing
        edge_tuple = build_edges_from_data_dict(data_dict,
                                                node_type_description,
                                                num_nodes,
                                                self.preprocess_config.connectivity_setting,
                                                self.preprocess_config.use_canonic_mesh_positions)
        edge_attr, edge_type, edge_index, edge_type_description = edge_tuple

        data.__setattr__("edge_attr", edge_attr)
        data.__setattr__("edge_type", edge_type)
        data.__setattr__("edge_index", edge_index)
        data.__setattr__("edge_type_description", edge_type_description)

        return data

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
