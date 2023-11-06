import torch
from multi_daft_vi.lnpdf import LNPDF

from lts_gns.algorithms.posterior_learners.abstract_posterior_learner import AbstractPosteriorLearner
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict


class ConstantPosteriorLearner(AbstractPosteriorLearner):

    def __init__(self, posterior_learner_config: ConfigDict, n_all_tasks: int, device: str):
        """
        :param n_all_tasks: Number of different trainng tasks. Needed for getting the correct shape of the latent samples
        """
        super().__init__(posterior_learner_config, device=device)
        self._d_z = posterior_learner_config.d_z
        self._n_all_tasks = n_all_tasks

    def sample(self, n_samples: int, task_indices: torch.Tensor) -> torch.Tensor:
        n_tasks = task_indices.shape[0]
        # return a tensor of shape (n_samples, n_tasks, d_z) with all zeros, as the posterior is constant
        return torch.zeros(n_samples, n_tasks, self.d_z).to(self._device)

    def fit(self, n_steps: int, task_indices: torch.Tensor, lnpdf: LNPDF, logging: bool = False) -> ValueDict:
        # nothing to fit, so just pass this
        # also no logging, so just return an empty dict
        return {}

    @property
    def n_all_tasks(self):
        return self._n_all_tasks

    @property
    def d_z(self):
        return self._d_z

    def save_checkpoint(self, directory: str, iteration: int, is_initial_save: bool, is_final_save: bool = False):
        print("Constant Posterior Learner does not have params to save.")
