import abc
from typing import Tuple, Iterator

from torch_geometric.data import Batch

from lts_gns.architectures.normalizers.model_to_world_normalizer import ModelToWorldNormalizer
from lts_gns.envs.abstract_processor import AbstractDataLoaderProcessor
from lts_gns.envs.graph_updater import GraphUpdater
from lts_gns.envs.task.task import Task
from lts_gns.envs.task.task_collection import TaskCollection
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict


class AbstractGNSEnvironment(abc.ABC):

    def __init__(self, env_config: ConfigDict,
                 data_loader_processor: AbstractDataLoaderProcessor,
                 graph_updater: GraphUpdater,
                 device: str):
        # set internal configs
        self._env_config: ConfigDict = env_config
        self._device: str = device

        # contains the values for normalization
        statistics_config = env_config.statistics
        self.model_to_world_normalizer = ModelToWorldNormalizer(statistics_config.use_output_normalizer,
                                                                world_mean=statistics_config.vel_mean,
                                                                world_std=statistics_config.vel_std).to(device)

        self.task_dict = data_loader_processor.process()  # preprocess data into a dictionary {split: [Task]},
        # where split is one of "train", "val" or "test".

        # get functions from data_loader_processor that are used to update the state of the environment
        # based on the concrete task
        self.integrate_predictions_fn = data_loader_processor.get_integrate_predictions_fn()
        self.update_batch_fn = data_loader_processor.get_update_batch_fn()

        self._graph_updater = graph_updater

        self._current_iteration = 0  # relevant for pc2mesh datasets or other iteration dependent stuff in env
        self.max_iterations = 1  # will be set by the algorithm

        # pc2mesh stuff
        self._include_pc2mesh = self._env_config.preprocess.include_pc2mesh
        if self._include_pc2mesh:
            # enable pc2mesh data flag and load data into noisy task dict
            data_loader_processor.load_pc2mesh = True
            self.pc2mesh_task_dict = data_loader_processor.process()

    @property
    def postprocess_config(self) -> ConfigDict:
        """
        # postprocess config is used to create the auxiliary task Dataloader, and to manage the context split for the test/val tasks
        :return: Postprocess ConfigDict
        """
        return self._env_config.postprocess

    @property
    def graph_updater(self) -> GraphUpdater:
        return self._graph_updater

    @property
    def val_tasks(self) -> TaskCollection:
        """
                Returns the context split val tasks. These are the tasks that are used for evaluation.
                In case with noisy pc2mesh meshes, we will always use these as the val tasks.
                Returns: TaskCollection
                """
        if self._include_pc2mesh:
            val_tasks = self.pc2mesh_task_dict[keys.VAL]
        else:
            val_tasks = self.task_dict[keys.VAL]
        assert len(val_tasks) > 0, "Context split val tasks are not created yet."
        return val_tasks

    @property
    def test_tasks(self) -> TaskCollection:
        """
               Returns the context split test tasks. These are the tasks that are used for evaluation.
               In case with noisy pc2mesh meshes, we will always use these as the test tasks.
               Returns: TaskCollection
               """
        if self._include_pc2mesh:
            test_tasks = self.pc2mesh_task_dict[keys.TEST]
        else:
            test_tasks = self.task_dict[keys.TEST]
        assert len(test_tasks) > 0, "Context split test tasks are not created yet."
        return test_tasks

    @abc.abstractmethod
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
        raise NotImplementedError("Abstract method 'get_next_train_task_batch' must be implemented in subclass.")

    def get_probability_for_real_mesh_dataset(self) -> float:
        """
        If we use the pc2mesh dataset, there should be a random chance to either select the real meshes or the noisy ones.
        This probability maybe should change to favor the pc2mesh dataset more over time.
        Returns: probability to select the real meshes
        """
        return max(1. - self.current_iteration / self.max_iterations, 0.0)

    @abc.abstractmethod
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
        raise NotImplementedError("Abstract method 'build_eval_task_batch' must be implemented in subclass.")

    @abc.abstractmethod
    def context_val_batches(self) -> Iterator[Tuple[Batch, ValueDict]]:
        """
        Generates batches for the validation context tasks.
        :return: Iterator over (batch, task_belonging)
        """
        raise NotImplementedError("Abstract method 'context_val_batches' must be implemented in subclass.")

    def postprocess_batch(self, batch: Batch,
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

        Returns: the postprocessed batch

        """
        return self.graph_updater.process_batch(batch=batch,
                                                add_noise_to_node_positions=add_noise_to_node_positions,
                                                recompute_edges=recompute_edges)

    @property
    def current_iteration(self):
        return self._current_iteration

    @current_iteration.setter
    def current_iteration(self, value):
        self._current_iteration = value

    @property
    def include_pc2mesh(self):
        return self._include_pc2mesh
