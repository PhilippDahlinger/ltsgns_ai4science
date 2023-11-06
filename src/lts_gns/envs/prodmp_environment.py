from collections import defaultdict
from typing import List, Iterator

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from lts_gns.envs.abstract_processor import AbstractDataLoaderProcessor
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.util import node_type_mask


class ProDMPEnvironment:
    def __init__(self, env_config: ConfigDict, data_loader_processor: AbstractDataLoaderProcessor, device: str,):
        # set internal configs
        self._env_config: ConfigDict = env_config
        self._use_point_cloud: bool = self._env_config.preprocess.use_point_cloud
        self._device: str = device

        # create train/val/test trajectories. These are just for visualization purposes to plot the ground truth
        # and to build the dataloaders from.
        self.trajectories = data_loader_processor.process()
        if self._use_point_cloud:
            padding_size = self.compute_point_cloud_padding_size()

        # TODO create config for this
        context_start = False
        if context_start:
            print("CONTEXT START ENABLED!!")

        # create train/val/test tasks.
        self.tasks = {keys.TRAIN: defaultdict(lambda: []), keys.VAL: defaultdict(lambda: []), keys.TEST: defaultdict(lambda: [])}
        self.tasks_list = {keys.TRAIN: [], keys.VAL: [], keys.TEST: []}
        for split in [keys.TRAIN, keys.VAL, keys.TEST]:
            if split == keys.TRAIN:
                aux_tasks_config = self._env_config.postprocess.auxiliary_tasks
            else:
                aux_tasks_config = self._env_config.postprocess.context_eval_tasks
            task_idx = 0
            aux_tasks = aux_tasks_config.auxiliary_tasks_per_task
            for traj_idx, traj in enumerate(self.trajectories[split].task_list):
                # create subtasks for each trajectory
                # we want to give at least one context point, hence the -1
                for anchor_idx in range(min(len(traj) - 1, aux_tasks)):
                    # TODO: is there a better way to choose the anchor indices?
                    # create Data object
                    anchor_graph = traj[anchor_idx].clone()
                    # sample indices for the context split

                    future_task_length = len(traj) - anchor_idx - 1
                    context_size = np.random.randint(aux_tasks_config.min_task_length,
                                                     min(aux_tasks_config.max_task_length, future_task_length))
                    # right now, we don't support slices of the trajectory
                    if context_start:
                        context_indices = torch.arange(context_size)
                    else:
                        context_indices = torch.tensor(np.sort(np.random.choice(future_task_length, context_size, replace=False)))
                    context_graphs = [traj[idx + anchor_idx] for idx in context_indices]
                    # get mesh node positions
                    node_positions = torch.stack([graph[keys.POSITIONS][node_type_mask(graph, key=keys.MESH)]
                                               for graph in context_graphs]) # shape (n_context, n_mesh_nodes, dim)
                    if keys.POINT_CLOUD_POSITIONS in context_graphs[0]:
                        point_cloud_positions = [graph[keys.POINT_CLOUD_POSITIONS] for graph in context_graphs]
                        # pad them to the same size
                        padded_point_cloud_positions = torch.full((len(point_cloud_positions), padding_size, point_cloud_positions[0].shape[-1]),  float("nan"))
                        for idx, point_cloud in enumerate(point_cloud_positions):
                            padded_point_cloud_positions[idx, :point_cloud.shape[0]] = point_cloud
                        point_cloud_positions = padded_point_cloud_positions
                    else:
                        point_cloud_positions = None
                    # add attributes to data object
                    anchor_graph[keys.CONTEXT_SIZES] = context_size
                    anchor_graph[keys.CONTEXT_INDICES] = context_indices
                    anchor_graph[keys.CONTEXT_NODE_POSITIONS] = node_positions
                    anchor_graph[keys.TRAJECTORY_INDICES] = traj_idx
                    anchor_graph[keys.ANCHOR_INDICES] = anchor_idx
                    anchor_graph[keys.TASK_INDICES] = task_idx
                    anchor_graph[keys.POINT_CLOUD_POSITIONS] = point_cloud_positions
                    # add to task
                    self.tasks[split][context_size].append(anchor_graph)
                    self.tasks_list[split].append(anchor_graph)
                    task_idx += 1

    def get_next_train_task_batch(self, **kwargs) -> Batch:
        batch_config = self.postprocess_config.batch
        train_tasks = self.tasks[keys.TRAIN]
        # sample a context size
        context_sizes = list(train_tasks.keys())
        context_size = np.random.choice(context_sizes)
        # sample a batch of size max_batch_size
        same_context_size_list = train_tasks[context_size]
        if batch_config.use_adaptive_batch_size:
            adapt_config = batch_config.adaptive_batch_size
            max_batch_size = int(adapt_config.batch_cost // ((
                                                                         1 - adapt_config.factor_per_context_point) + context_size * adapt_config.factor_per_context_point))
        else:
            max_batch_size = self.postprocess_config.batch.max_train_batch_size
        batch_size = min(max_batch_size, len(same_context_size_list))
        batch_indices = np.random.choice(len(same_context_size_list), batch_size, replace=False)
        batch = Batch.from_data_list([same_context_size_list[idx] for idx in batch_indices])
        return batch.to(self._device)

    def context_val_batches(self, use_validation_data=True) -> Iterator[Batch]:
        max_batch_size = self.postprocess_config.batch.max_eval_batch_size
        if use_validation_data:
            val_tasks = self.tasks[keys.VAL]
        else:
            val_tasks = self.tasks[keys.TEST]
        # loop over val_tasks, and yield batches of size max_batch_size
        batch = []
        for context_size in val_tasks:
            for task in val_tasks[context_size]:
                batch.append(task)
                if len(batch) == max_batch_size:
                    yield Batch.from_data_list(batch).to(self._device)
                    batch = []
            if len(batch) > 0:
                yield Batch.from_data_list(batch).to(self._device)
                batch = []

    def compute_point_cloud_padding_size(self):
        padding_size = 0
        for split in self.trajectories:
            for traj in self.trajectories[split].task_list:
                for graph in traj:
                    padding_size = max(padding_size, graph[keys.POINT_CLOUD_POSITIONS].shape[0])
        return padding_size

    @property
    def real_mesh_auxiliary_train_tasks(self) -> List[Data]:
        return self.tasks_list[keys.TRAIN]

    @property
    def val_tasks(self) -> List[Data]:
        return self.tasks_list[keys.VAL]

    @property
    def test_tasks(self) -> List[Data]:
        return self.tasks_list[keys.TEST]

    @property
    def postprocess_config(self) -> ConfigDict:
        """
        # postprocess config is used to create the auxiliary task Dataloader, and to manage the context split for the test/val tasks
        :return: Postprocess ConfigDict
        """
        return self._env_config.postprocess