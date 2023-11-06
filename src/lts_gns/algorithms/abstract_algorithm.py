from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterator

import numpy as np
import torch
from tqdm import tqdm

from lts_gns.algorithms.simulators.abstract_graph_network_simulator import AbstractGraphNetworkSimulator
from lts_gns.envs.abstract_gns_environment import AbstractGNSEnvironment
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.dict_util import deep_update
from lts_gns.util.own_types import ValueDict


class AbstractAlgorithm(ABC):
    # full algorithm, containing the simulator
    def __init__(self, algorithm_config: ConfigDict, simulator: AbstractGraphNetworkSimulator,
                 env: AbstractGNSEnvironment, device: str):
        self._algorithm_config: ConfigDict = algorithm_config
        self._simulator = simulator
        self._env: AbstractGNSEnvironment = env
        self._device: str = device

        # how to do train the GNS
        self._add_training_noise = self.algorithm_config.training.add_training_noise

        self._best_eval_loss = np.inf
        self._save_checkpoint_this_epoch = False

    def train_step(self, n: int) -> ValueDict:
        self.simulator.set_train_mode(True)
        return self._train_step(n)


    def _train_step(self, n: int) -> ValueDict:
        """
        Performs one epoch of training. Computes the mean train loss and returns it as a ValueDict.
        Returns: The training metrics as a ValueDict.

        """
        # set the current iteration in the environment. This is used to determine the dataset to load.
        # (Only applied in cases with pc2mesh datasets)
        self.env.current_iteration = n
        scalar_metrics = defaultdict(list)
        for batch in tqdm(range(self.algorithm_config.training.batches_per_epoch), desc="Training Epoch"):
            step_loss = self._single_train_step()
            scalar_metrics[keys.TOTAL_LOSS].append(step_loss)

        # take the average of the training metrics over the training step
        training_metrics = {keys.SCALARS: {keys.TRAIN: {keys.ALL_TRAIN_TASKS: {key: np.mean(value)
                                                                               for key, value in
                                                                               scalar_metrics.items()}}}}
        return training_metrics

    @abstractmethod
    def _single_train_step(self) -> torch.Tensor:
        """
        Performs a single training step.
        Returns: The loss of the training step.

        """
        raise NotImplementedError

    @property
    def simulator(self) -> AbstractGraphNetworkSimulator:
        if self._simulator is None:
            raise ValueError("Simulator not set")
        return self._simulator

    @property
    def algorithm_config(self) -> ConfigDict:
        return self._algorithm_config

    @property
    def env(self) -> AbstractGNSEnvironment:
        if self._env is None:
            raise ValueError("Env not set")
        return self._env

    def eval_step(self, current_step: int = 0, use_validation_data: bool = True,
                  force_large_update: bool = False) -> ValueDict:
        """
        Performs one evaluation step.
        Args:
            current_step: The current step of the training loop. Used for determining whether to perform a large
            evaluation step.
            use_validation_data: Whether to use the validation data (True) or the test data (False).
            force_large_update: Whether to force a large evaluation step. Effectively overwrites the current_step

        Returns:

        """
        self.simulator.set_train_mode(False)
        small_eval_metrics = self._small_eval_step(data_iterator=self.env.context_val_batches())
        self.update_best_eval_loss(small_eval_metrics)
        initial_large_eval = self.algorithm_config.evaluation.large.initial_eval and current_step == 0
        do_large_update = current_step > 0 and current_step % self.algorithm_config.evaluation.large.frequency == 0
        # only evaluate large eval in the beginning if initial_eval is set to True
        if force_large_update or initial_large_eval or do_large_update:
            large_eval_metrics = self._large_eval_step(use_validation_data=use_validation_data)
            return deep_update(small_eval_metrics, large_eval_metrics)
        else:
            return small_eval_metrics

    @abstractmethod
    def save_checkpoint(self, directory: str, iteration: int, is_initial_save: bool, is_final_save: bool = False):
        raise NotImplementedError

    @abstractmethod
    def _small_eval_step(self, data_iterator: Iterator) -> ValueDict:
        """
        Performs one small evaluation step after every epoch, i.e., computing the mean validation loss of the data
        in the provided iterator.

        Args:
            data_iterator: An iterator over the validation or test data.
        """
        raise NotImplementedError

    @abstractmethod
    def _large_eval_step(self, use_validation_data: bool = True) -> ValueDict:
        """
        Performs a big evaluation step after N number of epochs,
        i.e. loading all val tasks and evaluating the k-step prediction,
        the log-marginal likelihood, the MSE and rendering a video of one task.
        """
        raise NotImplementedError

    @property
    def save_checkpoint_this_epoch(self) -> bool:
        return self._save_checkpoint_this_epoch

    @save_checkpoint_this_epoch.setter
    def save_checkpoint_this_epoch(self, value: bool):
        self._save_checkpoint_this_epoch = value

    def update_best_eval_loss(self, small_eval_metrics):
        current_eval_loss = small_eval_metrics[keys.SCALARS][keys.SMALL_EVAL][keys.ALL_EVAL_TASKS][keys.TOTAL_LOSS]
        if current_eval_loss < self._best_eval_loss:
            self._best_eval_loss = current_eval_loss
            self.save_checkpoint_this_epoch = True

