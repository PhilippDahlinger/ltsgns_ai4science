from abc import ABC, abstractmethod
from typing import List

import torch
from multi_daft_vi.lnpdf import LNPDF

from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict


class AbstractPosteriorLearner(ABC):

    def __init__(self, posterior_learner_config: ConfigDict, device: str):
        self._posterior_learner_config = posterior_learner_config
        self._device = device

    @abstractmethod
    def sample(self, n_samples: int, task_indices: torch.Tensor) -> torch.Tensor:
        """
        :param n_samples: Number of samples of z to be drawn.
        :param task_indices: shape (n_tasks,). The task ids for which the latent samples should be drawn. Usually a subset of all tasks
        :return: shape (n_samples, n_tasks, d_z)
        """
        raise NotImplementedError

    def fit(self, n_steps: int, task_indices: torch.Tensor, lnpdf: LNPDF, logging: bool = False) -> ValueDict:
        """
        Perform n steps of training on the posterior distribution of z.
        Args:
            n_steps: How many steps the posterior learner should do.
            task_indices: Marked posteriors to update.
            lnpdf: The likelihood function to fit the posterior to.
            logging: Whether to log the training process.

        Returns:
        Logging Dict of visualizations and metrics.
        """
        raise NotImplementedError

    @property
    def d_z(self):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, directory: str, iteration: int, is_initial_save: bool, is_final_save: bool = False):
        raise NotImplementedError
