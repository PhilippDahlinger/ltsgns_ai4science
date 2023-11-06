import abc
import os
import pickle
import warnings
from typing import List, Dict

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from lts_gns.architectures.normalizers.world_to_model_normalizer import WorldToModelNormalizer
from lts_gns.envs.task.task import Task
from lts_gns.envs.task.task_collection import TaskCollection
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict


class AbstractDataLoaderProcessor(abc.ABC):

    def __init__(self, env_config: ConfigDict, task_name: str):

        self.debug_config = env_config.debug

        self.task_name = task_name
        self.preprocess_config = env_config.preprocess
        self.save_load_config = self.preprocess_config.save_load
        # contains the values for normalization

        statistics_config = env_config.statistics
        self.world_to_model_normalizer = WorldToModelNormalizer(statistics_config.use_output_normalizer,
                                                                world_mean=statistics_config.vel_mean,
                                                                world_std=statistics_config.vel_std)

    def __call__(self):
        return self.process()

    def process(self) -> Dict[str, TaskCollection]:
        """
        Create or load the preprocessed graphs. Returns a dictionary {split: [Task]}, where split is one of
        "train", "val" or "test".
        Args:

        Returns:

        """
        # load data or create it
        # todo general naming for this

        task_dict = None  # todo make this nicer
        preprocess_graphs = not self.save_load_config.load_preprocessed_graph
        if self.save_load_config.load_preprocessed_graph:
            try:
                task_dict = self._load_from_files()
            except FileNotFoundError as e:
                print(f"Could not load env '{self.preprocess_config.save_load.file_name}'. "
                      f"Falling back to preprocessing..")
                preprocess_graphs = True
        if preprocess_graphs:
            task_dict = self._preprocess()
            if self.save_load_config.save_preprocessed_graph:
                self._save_to_file(task_dict=task_dict)
        return task_dict

    def _preprocess(self) -> Dict[str, TaskCollection]:
        task_dict = {mode: TaskCollection(self._preprocess_split(mode))
                     for mode in self.split_modes}
        return task_dict

    def _preprocess_split(self, split: str) -> List[Task]:
        # load raw data
        rollout_data: List[ValueDict] = self._load_raw_data(split=split)

        task_list: List[Task] = []
        for index, raw_task in enumerate(tqdm(rollout_data, desc=f"Loading {split.title()} Data...")):
            # if we don't want to load all tasks (e.g. for debugging)
            if self.debug_config.max_tasks_per_split is not None and index >= self.debug_config.max_tasks_per_split:
                break

            if self.debug_config.get("max_rollout_length") is not None:
                rollout_length = self.debug_config.max_rollout_length
            else:
                rollout_length: int = self._get_rollout_length(raw_task=raw_task)
            raw_task: ValueDict = self._select_and_normalize_attributes(raw_task=raw_task)
            trajectory: List[Data] = []

            start_index = self.preprocess_config.start_index
            for timestep in range(start_index, rollout_length):
                data_dict: ValueDict = self._build_data_dict(raw_task=raw_task, timestep=timestep)
                self._add_train_label_noise(data_dict=data_dict)
                data: Data = self._build_graph(data_dict=data_dict)
                trajectory.append(data)

            # create a Task object from inner Data Object List
            task_list.append(Task(trajectory))
        return task_list

    def _add_train_label_noise(self, data_dict: ValueDict):
        true_labels = data_dict[keys.LABEL]
        if self.preprocess_config.train_label_noise > 0.0:
            noise = torch.randn_like(true_labels) * self.preprocess_config.train_label_noise
            data_dict[keys.LABEL] = true_labels + noise


    def _load_from_files(self) -> Dict[str, TaskCollection]:
        """
        Loads the task/simulations from the files specified in the config. Expects the graphs to be built and saved.
        The result should be accessed in the property train_tasks, test_tasks and val_tasks.
        The expected file path is

        preprocess_config.path_to_datasets/env_config.name/"preprocessed_graphs"/
        preprocessing_config.save_load.file_name_<train|val|test>.pkl

        """
        root_path = self._get_root_path()

        task_dict = {}
        for split in self.split_modes:
            with open(os.path.join(root_path, self.save_load_config.file_name + f"_{split}.pkl"), "rb") as f:
                pickled_tasks = pickle.load(f)
                task_dict[split] = TaskCollection([Task(pickled_task) for pickled_task in pickled_tasks])
        print(f"Loaded environment data from {os.path.join(root_path, self.preprocess_config.save_load.file_name)}.")
        return task_dict

    def _save_to_file(self, task_dict: Dict[str, TaskCollection]) -> bool:
        """
        Saves the preprocessed tasks to a file.
        The total file path will be

        preprocess_config.path_to_datasets/env_config.name/"preprocessed_graphs"/preprocessing_config.save_load.file_name_<train|val|test>.pkl

        :return: True if the save was successful.
        """
        root_path = self._get_root_path()
        os.makedirs(root_path, exist_ok=True)
        for split in self.split_modes:
            try:
                with open(os.path.join(root_path,
                                       self.save_load_config.file_name + f"_{split}.pkl"), "wb") as f:
                    trajectory_list = [pickled_task.trajectory for pickled_task in task_dict[split]]
                    pickle.dump(trajectory_list, f)
            except AssertionError as e:
                warnings.warn(f"Could not save the preprocessed tasks of mode {split} to a file. Error: {e}")
                return False

        print(f"Saved environment data to {os.path.join(root_path, self.save_load_config.file_name)}.")
        return True

    def _get_root_path(self):
        return os.path.join(self.save_load_config.path_to_datasets,
                            self.task_name,
                            keys.PREPROCESSED_GRAPHS)

    @property
    def split_modes(self):
        return [keys.TRAIN, keys.VAL, keys.TEST]

    ###########################################
    ####### Interfaces for data loading #######
    ###########################################

    @abc.abstractmethod
    def _get_rollout_length(self, raw_task: ValueDict) -> int:
        raise NotImplementedError("AbstractPreprocessor does not implement _get_rollout_length method")

    @abc.abstractmethod
    def _load_raw_data(self, split: str) -> List[ValueDict]:
        raise NotImplementedError("AbstractPreprocessor does not implement _load_raw_data method")

    @abc.abstractmethod
    def _select_and_normalize_attributes(self, raw_task: ValueDict) -> ValueDict:
        """
        Removes unused attributes such as point cloud or poisson values. Also normalizes stuff if necessary (task level)
        Args:
            raw_task:

        Returns:

        """
        raise NotImplementedError("AbstractPreprocessor does not implement _select_and_normalize_attributes method")

    @abc.abstractmethod
    def _build_data_dict(self, raw_task: ValueDict, timestep: int) -> ValueDict:
        """
        Load for one timestep the correct tensors into data dict (indexed with timestep)
        Args:
            raw_task:
            timestep:

        Returns:

        """
        raise NotImplementedError("AbstractPreprocessor does not implement _build_data_dict method")

    @abc.abstractmethod
    def _build_graph(self, data_dict: ValueDict) -> Data:
        """
        Build Data object from data dict, i.e., actually build a graph from a dictionary of tensors.
        Args:
            data_dict:

        Returns:

        """
        raise NotImplementedError("AbstractPreprocessor does not implement _build_graph method")

    ###########################################
    ####### Functions for the processor #######
    ###########################################

    @abc.abstractmethod
    def get_integrate_predictions_fn(self):  # -> Callable[[torch.Tensor, Batch], Dict[str, torch.Tensor]]:
        """
        An abstract function that gets the outputs of the GNS and a batch of data and returns the
         the mesh_state dictionary that contains the updated mesh positions, pressure fields, etc.
         The dictionary must always contain a key PREDICTIONS, which is used for the loss calculation and other metrics
        Returns:

        """
        raise NotImplementedError("AbstractPreprocessor does not implement get_integrate_predictions_fn method")

    @abc.abstractmethod
    def get_update_batch_fn(self):  # -> Callable[[Batch, Dict[str, torch.Tensor]], Batch]:
        """
        An abstract function that gets a batch of data and the mesh_state dictionary and returns an updated batch.
        For example, the batch could have "original" mesh positions, which can be replaced by the predicted ones in
        the mesh_state dictionary.
        Similarly, the state dict could contain pressure fields or other quantities that are not part of the batch,
        but can be used to update the batch.
        Returns:

        """
        raise NotImplementedError("AbstractPreprocessor does not implement get_update_batch_fn method")
