import copy
import warnings
from collections import defaultdict
from typing import Tuple, Iterator

import numpy as np
from torch_geometric.data import Batch

from lts_gns.envs.abstract_gns_environment import AbstractGNSEnvironment
from lts_gns.envs.abstract_processor import AbstractDataLoaderProcessor
from lts_gns.envs.graph_updater import GraphUpdater
from lts_gns.envs.task.task import Task
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict


class MGNEnvironment(AbstractGNSEnvironment):
    def __init__(self, env_config: ConfigDict,
                 data_loader_processor: AbstractDataLoaderProcessor,
                 graph_updater: GraphUpdater,
                 device: str):
        super().__init__(env_config,
                         data_loader_processor=data_loader_processor,
                         graph_updater=graph_updater,
                         device=device)

        self.cumulative_train_lengths = np.cumsum([len(x) for x in self.task_dict[keys.TRAIN]])
        self.num_train_samples = self.cumulative_train_lengths[-1]  # total number of training graphs

    def get_next_train_task_batch(self, add_training_noise: bool) -> Tuple[Batch, ValueDict]:
        """
        Returns the next task batch. The task batch is a batch of tasks.
        It has a maximum time steps of postprocess_config.batch.max_train_batch_size.
        Size refers here as the number of graphs in the batch, not the number of tasks.

        Args:
            add_training_noise: whether to add noise to the training data

        Returns: Batch of tasks and a dictionary containing the following information:
            - task indices: which tasks are in this batch
            - mesh nodes per task: how many mesh nodes are in each task
            - mesh nodes per time step: how many mesh nodes are in each time step (i.e. graph)
            - time steps per task: how many time steps are in each task, i.e. how many graphs are in each task
        """
        max_batch_size = self.postprocess_config.batch.max_train_batch_size
        indices = self._get_next_indices(max_batch_size)

        # index goes over all data points. We instead want to get the task index for each data point, i.e.,
        # the trajectory that this data point belongs to, and the index of the data point within the trajectory.
        task_indices = np.searchsorted(self.cumulative_train_lengths, indices, side='right')
        indices = indices - np.concatenate(([0], self.cumulative_train_lengths[:-1]))[task_indices]

        if self._include_pc2mesh:
            if self.get_probability_for_real_mesh_dataset() > np.random.uniform():
                train_tasks = self.task_dict[keys.TRAIN]
            else:
                train_tasks = self.pc2mesh_task_dict[keys.TRAIN]
        else:
            train_tasks = self.task_dict[keys.TRAIN]
        batch_list = [train_tasks[task_index][index] for index, task_index in zip(indices, task_indices)]

        batch = Batch.from_data_list(batch_list[:max_batch_size])
        batch = self.postprocess_batch(batch, add_noise_to_node_positions=add_training_noise)
        task_belonging: ValueDict = {keys.MESH_NODES_PER_TIME: [batch_list[0].y.shape[0]] * len(batch_list)}
        return batch, task_belonging

    def _get_next_indices(self, batch_size: int):
        # todo this currently draws with replacement. We may want to change this to without replacement
        indices = np.random.permutation(self.num_train_samples)  # get random permutation of data
        indices = indices[:batch_size]  # get first max_batch_size indices
        indices.sort()  # sort indices to get the correct order
        return indices

    def build_eval_task_batch(self, current_task: Task, task_idx: int,
                              recompute_graph_edges: bool = True) -> Tuple[Batch, ValueDict]:
        """
        Builds a batch out of a single test/val task.
        Args:
            current_task: The task to build the batch from
            task_idx: The index of the task
            recompute_graph_edges: Whether to recompute the graph edges based on updated mesh positions or not.
                If False, the graph edges are kept as is

        Returns: A batch and a dictionary containing the following information:
            - task indices: which tasks are in this batch
            - mesh nodes per task: how many mesh nodes are in each task
            - mesh nodes per time step: how many mesh nodes are in each time step (i.e. graph)
            - time steps per task: how many time steps are in each task, i.e. how many graphs are in each task

        """
        batch = Batch.from_data_list(current_task.trajectory)
        batch = self.postprocess_batch(batch,
                                       add_noise_to_node_positions=False,
                                       recompute_edges=recompute_graph_edges)
        task_belonging: ValueDict = {
            keys.MESH_NODES_PER_TIME: [current_task.trajectory[0].y.shape[0]] * len(current_task.trajectory)}
        return batch, task_belonging  # task_belonging

    def context_val_batches(self) -> Iterator[Tuple[Batch, ValueDict]]:
        """
        Generates batches for the validation context tasks.
        :return: Iterator over (batch, task_belonging)
        """
        max_batch_size = self.postprocess_config.batch.max_eval_batch_size
        batch_list = []
        for task_idx, task in enumerate(self.val_tasks):
            batch_list += task.trajectory
            if len(batch_list) >= max_batch_size:
                batch = Batch.from_data_list(batch_list)
                batch = self.postprocess_batch(batch, add_noise_to_node_positions=False)
                task_belonging: ValueDict = {
                    keys.MESH_NODES_PER_TIME: [batch_list[0].y.shape[0]] * len(batch_list)}
                yield batch, task_belonging
                batch_list = []
        if len(batch_list) > 0:
            batch = Batch.from_data_list(batch_list)
            batch = self.postprocess_batch(batch, add_noise_to_node_positions=False)
            task_belonging: ValueDict = {
                keys.MESH_NODES_PER_TIME: [batch_list[0].y.shape[0]] * len(batch_list)}
            yield batch, task_belonging
