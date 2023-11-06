from collections import defaultdict
from typing import Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
from einops import einops

from gmm_util.gmm import GMM
from matplotlib import pyplot as plt
from multi_daft_vi.recording.util import plot2d_plotly
from multi_daft_vi.util_multi_daft import create_initial_gmm_parameters
from multi_daft_vi.multi_daft import MultiDaft
import torch
from gmm_util.gmm import GMM
from multi_daft_vi.lnpdf import LNPDF
from multi_daft_vi.multi_daft import MultiDaft
from multi_daft_vi.util_multi_daft import create_initial_gmm_parameters
from plotly.graph_objs import Figure

from lts_gns.algorithms.posterior_learners.abstract_posterior_learner import AbstractPosteriorLearner
from lts_gns.algorithms.simulators.abstract_graph_network_simulator import AbstractGraphNetworkSimulator
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict


class TaskPropertiesLearner(AbstractPosteriorLearner):

    def __init__(self, posterior_learner_config: ConfigDict, device: str):
        """
        :param posterior_learner_config: ConfigDict containing the config for the posterior learner.
        :param n_all_train_tasks: Number of all training tasks.
        :param n_all_eval_tasks: Number of all evaluation tasks.
        :param device: Device to use.
        """
        super().__init__(posterior_learner_config, device=device)
        self._last_task_properties: torch.Tensor | None = None

    def sample(self, n_samples: int, task_indices: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marked posteriors.
        Args:
            n_samples: Number of samples to draw per task.
            task_indices: Indices of the marked posteriors to sample from. The indices are the same as in the GNS.

        Returns: Samples of shape (num_samples, num_tasks, d_z)

        """
        assert self._last_task_properties is not None, "No task properties available. Call fit first."
        return einops.repeat(self._last_task_properties, "t d -> s t d", s=n_samples)

    def fit(self, n_steps: int, task_indices: torch.Tensor, lnpdf: LNPDF, logging: bool = False) -> ValueDict:
        """
        Uses the GradientVIPS algorithm to update the marked posteriors.
        Args:
            n_steps: How many steps the posterior learner should do.
            task_indices: Marked posteriors to update. The indices are the same as in the GNS.
            lnpdf: Simulator class that is used to compute the log probabilities of the samples.
            logging:

        Returns: None
        """
        assert isinstance(lnpdf, AbstractGraphNetworkSimulator), "The lnpdf must be a GraphNetworkSimulator."
        lnpdf: AbstractGraphNetworkSimulator
        batch = lnpdf._batch
        self._last_task_properties = []
        try:
            time_steps_per_task = lnpdf.task_belonging["time_steps_per_task"]
            # Autoregressive case
            time_idx = 0
            for num_time_steps in time_steps_per_task:
                self._last_task_properties.append(batch.task_properties[time_idx])
                time_idx += num_time_steps
            self._last_task_properties = torch.stack(self._last_task_properties, dim=0)
        except ValueError:
            # Trajectory approach
            # here, every data object is one task with its corresponding task properties
            self._last_task_properties = batch.task_properties
        return {}


    def save_checkpoint(self, directory: str, iteration: int, is_initial_save: bool, is_final_save: bool = False):
        # TODO: Implement
        pass
