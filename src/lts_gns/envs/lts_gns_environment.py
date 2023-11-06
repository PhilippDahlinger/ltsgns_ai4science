import copy
import warnings
from collections import defaultdict
from typing import List, Tuple, Iterator

import numpy as np
import torch
from torch_geometric.data import Batch

from lts_gns.envs.abstract_gns_environment import AbstractGNSEnvironment
from lts_gns.envs.abstract_processor import AbstractDataLoaderProcessor
from lts_gns.envs.graph_updater import GraphUpdater
from lts_gns.envs.task.task import Task
from lts_gns.envs.task.task_collection import TaskCollection
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict


class LTSGNSEnvironment(AbstractGNSEnvironment):
    # todo everything here should probably be a trainer, as this is the task splits and the batching

    def __init__(self, env_config: ConfigDict,
                 data_loader_processor: AbstractDataLoaderProcessor,
                 graph_updater: GraphUpdater,
                 device: str):
        super().__init__(env_config,
                         data_loader_processor=data_loader_processor,
                         graph_updater=graph_updater,
                         device=device)

        # create auxiliary tasks, validation and test subtask splits
        self.create_train_subtasks(task_dict=self.task_dict)
        self.create_val_test_subtasks(task_dict=self.task_dict)

        self._include_pc2mesh = self._env_config.preprocess.include_pc2mesh

        # incorporate pc2mesh data
        if self._include_pc2mesh:
            self.create_train_subtasks(task_dict=self.pc2mesh_task_dict)
            self.create_val_test_subtasks(task_dict=self.pc2mesh_task_dict)

    def create_train_subtasks(self, task_dict: ValueDict) -> None:
        """
        Creates auxiliary tasks for training.
        They can either be sliced or uniformly sampled. The length is uniform between a min and a max value.
        """
        auxiliary_task_config = self.postprocess_config.auxiliary_tasks

        train_tasks = task_dict[keys.TRAIN]
        train_tasks.create_subtasks(subtasks_per_task=auxiliary_task_config.auxiliary_tasks_per_task,
                                    min_subtask_length=auxiliary_task_config.min_task_length,
                                    max_subtask_length=auxiliary_task_config.max_task_length,
                                    sampling_type=auxiliary_task_config.sampling_type)

    def create_val_test_subtasks(self, task_dict: ValueDict):
        """
        Creates the context split tasks. This is used for the val and test tasks.
        """
        # Todo @niklas: Why do we only draw 1 context split task per task?
        context_eval_tasks = self.postprocess_config.context_eval_tasks

        for split in [keys.VAL, keys.TEST]:
            task_collection = task_dict[split]
            task_collection.create_subtasks(subtasks_per_task=1,
                                            min_subtask_length=context_eval_tasks.min_task_length,
                                            max_subtask_length=context_eval_tasks.max_task_length,
                                            sampling_type=context_eval_tasks.sampling_type)

    @property
    def real_mesh_auxiliary_train_tasks(self) -> List[Task]:
        # returns the subtask list of the train tasks with real meshes,
        # i.e., slices of the full training trajectories that are interpreted as individual tasks for meta learning
        assert len(self.task_dict[keys.TRAIN].subtask_list) > 0, "Auxiliary train tasks are not created yet."
        return self.task_dict[keys.TRAIN].subtask_list

    @property
    def pc2mesh_auxiliary_train_tasks(self) -> List[Task]:
        # returns the subtask list of the train tasks with meshes generated from the pointcloud,
        # i.e., slices of the full training trajectories that are interpreted as individual tasks for meta learning
        assert len(self.pc2mesh_task_dict[keys.TRAIN].subtask_list) > 0, "Auxiliary train tasks are not created yet."
        return self.pc2mesh_task_dict[keys.TRAIN].subtask_list

    def get_next_train_task_batch(self, add_training_noise: bool) -> Tuple[Batch, ValueDict]:
        """
        Returns the next task batch. The task batch is a batch of tasks.
        It has a maximum time steps of postprocess_config.batch.max_train_batch_size.
        Size refers here as the number of graphs in the batch, not the number of tasks.
        The tasks are sampled from the auxiliary train tasks.

        Args:
            add_training_noise: whether to add noise to the training data

        Returns:
            Batch of tasks and a dictionary containing the following information:
            - task indices: which tasks are in this batch
            - mesh nodes per task: how many mesh nodes are in each task
            - mesh nodes per time step: how many mesh nodes are in each time step (i.e. graph)
            - time steps per task: how many time steps are in each task, i.e. how many graphs are in each task
        """
        # todo could think about making this an iterator and changing the interfaces accordingly
        # todo @Niklas does it make sense to have sequences of length>1 in the same batch for the GNS training?
        #  Or should we train the GNS independent of the posterior with sequences of length 1?

        # get the dataset for this batch, either the real mesh or the pc2mesh
        # start_task_idx is used to distinguish between the different tasks in the posterior learner
        if self._include_pc2mesh:
            if self.get_probability_for_real_mesh_dataset() > np.random.uniform():
                auxiliary_train_tasks = self.real_mesh_auxiliary_train_tasks
                start_task_idx = 0
            else:
                auxiliary_train_tasks = self.pc2mesh_auxiliary_train_tasks
                start_task_idx = len(self.real_mesh_auxiliary_train_tasks)
        else:
            auxiliary_train_tasks = self.real_mesh_auxiliary_train_tasks
            start_task_idx = 0

        max_batch_size = self.postprocess_config.batch.max_train_batch_size
        batch_list = []
        task_belonging: ValueDict = defaultdict(list)
        if len(auxiliary_train_tasks[-1]) > max_batch_size:
            warnings.warn("The last task in the auxiliary train tasks is larger than the maximum batch size. "
                          "All these too-large tasks will be ignored.")

        indices = np.random.permutation(len(auxiliary_train_tasks))
        for task_index in indices:
            current_task = copy.deepcopy(auxiliary_train_tasks[task_index])

            batch_list += current_task.trajectory
            task_belonging = self._update_task_belonging(task_index + start_task_idx, current_task, task_belonging)

            if len(batch_list) > max_batch_size:
                break

        batch = Batch.from_data_list(batch_list)
        batch = self.postprocess_batch(batch, add_noise_to_node_positions=add_training_noise)
        task_belonging = self._finalize_task_belonging(task_belonging)
        return batch, task_belonging

    def _update_task_belonging(self, current_task_index, current_task, task_belonging):
        task_belonging[keys.TASK_INDICES].append(current_task_index)
        task_belonging[keys.MESH_NODES_PER_TASK].append(current_task[0].y.shape[0] * len(current_task))
        task_belonging[keys.MESH_NODES_PER_TIME].extend([current_task[0].y.shape[0]] * len(current_task))  # extend list
        task_belonging[keys.TIME_STEPS_PER_TASK].append(len(current_task))
        return task_belonging

    def _finalize_task_belonging(self, task_belonging):
        task_belonging[keys.TASK_INDICES] = torch.tensor(task_belonging[keys.TASK_INDICES], device=self._device, )
        task_belonging[keys.MESH_NODES_PER_TASK] = torch.tensor(task_belonging[keys.MESH_NODES_PER_TASK],
                                                                device=self._device, )
        task_belonging[keys.MESH_NODES_PER_TIME] = tuple(task_belonging[keys.MESH_NODES_PER_TIME])
        # for torch.split they want a tuple
        task_belonging[keys.TIME_STEPS_PER_TASK] = tuple(task_belonging[keys.TIME_STEPS_PER_TASK])
        # for torch.split they want a tuple
        return task_belonging

    def build_eval_task_batch(self, current_task: Task, task_idx: int,
                              recompute_graph_edges: bool = True,
                              add_noise_to_node_positions: bool = False) -> Tuple[Batch, ValueDict]:
        """
        Builds a batch out of a single test/val task.
        Args:
            current_task: The task to build the batch from
            task_idx: The index of the task
            recompute_graph_edges: Whether to recompute the graph edges based on updated mesh positions or not.
                If False, the graph edges are kept as is
            add_noise_to_node_positions: Whether to add noise to the node positions or not. usually always False

        Returns: A batch and a dictionary containing the following information:
            - task indices: which tasks are in this batch
            - mesh nodes per task: how many mesh nodes are in each task
            - mesh nodes per time step: how many mesh nodes are in each time step (i.e. graph)
            - time steps per task: how many time steps are in each task, i.e. how many graphs are in each task

        """
        # todo extract this?
        mesh_nodes_per_task = [current_task[0].y.shape[0] * len(current_task)]
        mesh_nodes_per_time_step = [current_task[0].y.shape[0]] * len(current_task)
        time_steps_per_task = [len(current_task)]
        task_belonging = {
            keys.TASK_INDICES: [task_idx],
            keys.MESH_NODES_PER_TASK: mesh_nodes_per_task,
            keys.MESH_NODES_PER_TIME: mesh_nodes_per_time_step,
            keys.TIME_STEPS_PER_TASK: time_steps_per_task
        }
        self._finalize_task_belonging(task_belonging)

        batch = Batch.from_data_list(current_task.trajectory)
        batch = self.postprocess_batch(batch,
                                       add_noise_to_node_positions=add_noise_to_node_positions,
                                       recompute_edges=recompute_graph_edges)
        return batch, task_belonging

    def context_val_batches(self) -> Iterator[Tuple[Batch, ValueDict]]:
        """
        Generates batches for the validation context tasks.
        :return: Iterator over (batch, task_belonging)
        """

        def _finalize_batch(batch_list, task_belonging):
            batch = Batch.from_data_list(batch_list)
            # no noise in the val data
            batch = self.postprocess_batch(batch, add_noise_to_node_positions=False)
            task_belonging = self._finalize_task_belonging(task_belonging)
            return batch, task_belonging

        max_batch_size = self.postprocess_config.batch.max_eval_batch_size

        batch_list = []
        task_belonging: ValueDict = defaultdict(list)
        current_batch_size = 0

        subtasks = self.val_tasks.subtask_list
        for task_index, subtask in enumerate(subtasks):
            if len(subtask) + current_batch_size >= max_batch_size:
                # if the batch size is exceeded, yield the current batch and start a new one
                yield _finalize_batch(batch_list, task_belonging)
                batch_list = []
                task_belonging: ValueDict = defaultdict(list)
                current_batch_size = 0
            # build upon the current batch
            task_belonging = self._update_task_belonging(task_index, subtask, task_belonging)
            batch_list += subtask.trajectory
            current_batch_size += len(subtask)

        # yield the last batch if it is not empty
        if len(batch_list) > 0:
            yield _finalize_batch(batch_list, task_belonging)
