from collections import defaultdict
from typing import Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px

from gmm_util.gmm import GMM
from matplotlib import pyplot as plt
from multi_daft_vi.recording.util import plot2d_plotly, plot1d_plotly
from multi_daft_vi.util_multi_daft import create_initial_gmm_parameters
from multi_daft_vi.multi_daft import MultiDaft
import torch
from gmm_util.gmm import GMM
from multi_daft_vi.lnpdf import LNPDF
from multi_daft_vi.multi_daft import MultiDaft
from multi_daft_vi.util_multi_daft import create_initial_gmm_parameters
from plotly.graph_objs import Figure

from lts_gns.algorithms.posterior_learners.abstract_posterior_learner import AbstractPosteriorLearner
from lts_gns.algorithms.simulators.ltsgns_simulator import LTSGNSSimulator
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict


class MultiDaftPosteriorLearner(AbstractPosteriorLearner):

    def __init__(self, posterior_learner_config: ConfigDict, n_all_train_tasks: int, n_all_eval_tasks: int,
                 device: str):
        """
        :param posterior_learner_config: ConfigDict containing the config for the posterior learner.
        :param n_all_train_tasks: Number of all training tasks.
        :param n_all_eval_tasks: Number of all evaluation tasks.
        :param device: Device to use.
        """
        super().__init__(posterior_learner_config, device=device)
        self._d_z = posterior_learner_config.d_z
        self._n_all_train_tasks = n_all_train_tasks
        self._n_all_eval_tasks = n_all_eval_tasks
        self._n_components = posterior_learner_config.n_components

        self._train_log_weights, self._train_means, self._train_precs = self._init_gmm_params(self._n_all_train_tasks)
        self._eval_log_weights, self._eval_means, self._eval_precs = self._init_gmm_params(self._n_all_eval_tasks)

        # if mode is eval_from_prior, these are the params that are used for sampling
        self._eval_from_prior_log_weights, self._eval_from_prior_means, self._eval_from_prior_precs = None, None, None

        # start always in train mode
        self._mode = "train"
        self.max_min_visualizations = 7
        self._sample_at_mean = posterior_learner_config.sample_at_mean
        if self._sample_at_mean:
            assert self._n_components == 1, "If sample_at_mean is True, n_components must be 1."

    def sample(self, n_samples: int, task_indices: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marked posteriors.
        Args:
            n_samples: Number of samples to draw per task.
            task_indices: Indices of the marked posteriors to sample from. The indices are the same as in the GNS.

        Returns: Samples of shape (num_samples, num_tasks, d_z)

        """
        log_weights, means, precs = self._get_gmm_params(task_indices)
        # only in evaluation, consider sampling at the mean
        if self.mode == "eval_from_prior" and self._sample_at_mean:
            # first component
            means = means[:, 0, :]
            z = means.repeat(n_samples, 1, 1)
        else:
            gmm = GMM(
                log_w=log_weights,
                mean=means,
                prec=precs,
            )
            z = gmm.sample(n_samples)
        return z


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
        log_weights, means, precs = self._get_gmm_params(task_indices)

        multi_daft = MultiDaft(algorithm_config=self._posterior_learner_config.get_raw_dict(),  # algo expects raw dict
                               target_dist=lnpdf,
                               log_w_init=log_weights,
                               mean_init=means,
                               prec_init=precs,
                               device=self._device,
                               )

        logging_dict = defaultdict(list)
        for i in range(n_steps):
            if logging and i % self._posterior_learner_config.logging_frequency == 0:
                logging_step = multi_daft.step(logging=True)
                for key, value in logging_step.items():
                    if key == "total_samples" and len(logging_dict[key]) > 0:
                        # add the previous total samples to the current one
                        logging_dict[key].append(value + logging_dict[key][-1])
                    else:
                        logging_dict[key].append(value)
                # add step to logging dict
                logging_dict["step"].append(i)
            else:
                multi_daft.step(logging=False)

        # save the new params
        self._set_gmm_params(task_indices, multi_daft.model.log_w, multi_daft.model.mean, multi_daft.model.prec)

        if logging:
            results = {}
            data = pd.DataFrame.from_dict(logging_dict)
            # make plotly plots for the logging
            if isinstance(logging_dict["elbo"][0], np.float32):
                fig = px.line(data, x="step", y="elbo", title="ELBO over time")
                results["elbo_plot"] = fig
            if self.d_z == 2:
                fig = plot2d_plotly(target_dist=multi_daft.target_dist, model=multi_daft.model,
                                    mini_batch_size=self._posterior_learner_config.mini_batch_size_for_target_density,
                                    use_log_space=False,
                                    device=self._device,
                                    min_x=-self.max_min_visualizations,
                                    max_x=self.max_min_visualizations,
                                    min_y=-self.max_min_visualizations,
                                    max_y=self.max_min_visualizations,
                                    )
                results["2d_vis"] = fig
            elif self.d_z == 1:
                fig = plot1d_plotly(target_dist=multi_daft.target_dist, model=multi_daft.model,
                                    mini_batch_size=self._posterior_learner_config.mini_batch_size_for_target_density,
                                    use_log_space=False,
                                    device=self._device,
                                    min_x=-self.max_min_visualizations,
                                    max_x=self.max_min_visualizations,
                                    )
                results["2d_vis"] = fig
            # print("Success", logging_dict["success"])
            # print("Eta", logging_dict["eta"])
            return results
        else:
            return {}

    def visualize_train_posteriors(self, task_indices: List[int], simulator: LTSGNSSimulator) -> Tuple[Figure, Figure, Figure]:
        self.mode = "train"
        log_weights, means, precs = self._get_gmm_params(torch.tensor(task_indices))
        model = GMM(
            log_w=log_weights,
            mean=means,
            prec=precs,
        )
        # build plotly figure
        fig_posterior = plot2d_plotly(target_dist=simulator, model=model,
                                      mini_batch_size=self._posterior_learner_config.mini_batch_size_for_target_density,
                                      device=self._device,
                                      min_x=-self.max_min_visualizations,
                                      max_x=self.max_min_visualizations,
                                      min_y=-self.max_min_visualizations,
                                      max_y=self.max_min_visualizations, )
        fig_prior = plot2d_plotly(target_dist=simulator._log_prior_density, model=model,
                                  mini_batch_size=self._posterior_learner_config.mini_batch_size_for_target_density,
                                  device=self._device,
                                  min_x=-self.max_min_visualizations,
                                  max_x=self.max_min_visualizations,
                                  min_y=-self.max_min_visualizations,
                                  max_y=self.max_min_visualizations, )
        fig_likelihood = plot2d_plotly(target_dist=simulator._log_likelihood, model=model,
                                       mini_batch_size=self._posterior_learner_config.mini_batch_size_for_target_density,
                                       device=self._device,
                                       min_x=-self.max_min_visualizations,
                                       max_x=self.max_min_visualizations,
                                       min_y=-self.max_min_visualizations,
                                       max_y=self.max_min_visualizations, )
        return fig_posterior, fig_prior, fig_likelihood

    def save_checkpoint(self, directory: str, iteration: int, is_initial_save: bool, is_final_save: bool = False):
        # TODO: Implement
        pass

    def _get_gmm_params(self, task_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the parameters of the GMM for the given task indices. If the mode is "train", the train parameters are
        returned. If the mode is "eval_from_checkpoint", the eval parameters are returned. If the mode is "eval_from_prior",
        the eval_from_prior parameters are returned. However, if they are None, we create new ones (with the length of the task_indices).
        Args:
            task_indices: torch.Tensor with shape (num_tasks,)

        Returns: log_weights, means, precs of the GMM with shape
                    log_weights: (num_tasks, n_components)
                    means: (num_tasks, n_components, d_z)
                    precs: (num_tasks, n_components, d_z, d_z)

        """

        def get_task_indices_params(log_weights, means, precs):
            # scatter to get the correct subset of the parameters
            log_weights = log_weights[task_indices]
            means = means[task_indices]
            precs = precs[task_indices]
            return log_weights, means, precs

        if self.mode == "train":
            log_weights = self._train_log_weights
            means = self._train_means
            precs = self._train_precs
            log_weights, means, precs = get_task_indices_params(log_weights, means, precs)
        elif self.mode == "eval_from_checkpoint":
            log_weights = self._eval_log_weights
            means = self._eval_means
            precs = self._eval_precs
            log_weights, means, precs = get_task_indices_params(log_weights, means, precs)
        elif self.mode == "eval_from_prior":
            if self._eval_from_prior_log_weights is None:
                # create new params
                log_weights, means, precs = self._init_gmm_params(len(task_indices))
            else:
                # use the params that were saved in the eval_from_prior params
                log_weights = self._eval_from_prior_log_weights
                means = self._eval_from_prior_means
                precs = self._eval_from_prior_precs
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return log_weights, means, precs

    def _set_gmm_params(self, task_indices: torch.Tensor, log_weights: torch.Tensor, means: torch.Tensor,
                        precs: torch.Tensor):
        if self.mode == "train":
            self._train_means[task_indices] = means.detach().clone()
            self._train_precs[task_indices] = precs.detach().clone()
            self._train_log_weights[task_indices] = log_weights.detach().clone()
        elif self.mode == "eval_from_checkpoint":
            self._eval_means[task_indices] = means.detach().clone()
            self._eval_precs[task_indices] = precs.detach().clone()
            self._eval_log_weights[task_indices] = log_weights.detach().clone()
        elif self.mode == "eval_from_prior":
            self._eval_from_prior_log_weights = log_weights.detach().clone()
            self._eval_from_prior_means = means.detach().clone()
            self._eval_from_prior_precs = precs.detach().clone()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def reset_eval_from_prior(self):
        """
        Resets the eval_from_prior params to None. This needs to be called
        if a new tasks needs to be evaluated from scratch.
        """
        self._eval_from_prior_log_weights = None
        self._eval_from_prior_means = None
        self._eval_from_prior_precs = None

    def _init_gmm_params(self, n_tasks: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param n_all_tasks: Number of different trainng tasks. Needed for getting the correct shape of the latent GMMs
        :return: (log_weights, means, precs) with shapes (n_tasks, n_components),
                                                         (n_tasks, n_components, d_z) and
                                                         (n_tasks, n_components, d_z, d_z)
        """
        # for now, use the MultiDAFT util function to create the initial parameters, might adapt it to a GNS init later
        log_weights, means, precs = create_initial_gmm_parameters(d_z=self.d_z,
                                                                  n_tasks=n_tasks,
                                                                  n_components=self.n_components,
                                                                  prior_scale=self._posterior_learner_config.prior_scale,
                                                                  initial_var=self._posterior_learner_config.initial_var,
                                                                  )
        return log_weights.to(self._device), means.to(self._device), precs.to(self._device)

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        """
        Sets the mode of the posterior learner. The mode can be "train", "eval_from_checkpoint" or "eval_from_prior".
        train: Use the train parameters.
        eval_from_checkpoint: Use the parameters from the last checkpoint.
        eval_from_prior: Create new parameters and train the GMM completely from scratch. The params are saved in the
                         eval_from_prior params. They are overwritten when the next step is called.
        Args:
            mode: str, the mode to set.
        """
        if mode not in ["train", "eval_from_checkpoint", "eval_from_prior"]:
            raise ValueError(f"Unknown mode: {mode}")
        self._mode = mode

    @property
    def n_all_tasks(self):
        raise AttributeError("n_all_tasks is not implemented for MultiDAFT. "
                             "Use n_all_train_tasks or n_all_eval_tasks instead.")

    @property
    def n_all_train_tasks(self) -> int:
        return self._n_all_train_tasks

    @property
    def n_all_eval_tasks(self) -> int:
        return self._n_all_eval_tasks

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def d_z(self):
        return self._d_z
