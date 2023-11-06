import copy
import time
import warnings
from collections import defaultdict
from typing import List, Tuple, Iterator, Optional, Dict

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from lts_gns.algorithms.abstract_algorithm import AbstractAlgorithm
from lts_gns.algorithms.algorithm_util import list_from_config
from lts_gns.algorithms.posterior_learners.abstract_posterior_learner import AbstractPosteriorLearner
from lts_gns.algorithms.posterior_learners.get_posterior_learner import get_posterior_learner
from lts_gns.algorithms.posterior_learners.multi_daft_posterior_learner import MultiDaftPosteriorLearner
from lts_gns.algorithms.simulators.ltsgns_prodmp_simulator import LTSGNSProDMPSimulator
from lts_gns.envs.prodmp_environment import ProDMPEnvironment
from lts_gns.envs.task.task import Task
from lts_gns.envs.task.task_collection import TaskCollection
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.dict_util import deep_update
from lts_gns.util.own_types import ValueDict
from lts_gns.algorithms.simulators.abstract_graph_network_simulator import AbstractGraphNetworkSimulator
from lts_gns.util.util import node_type_mask, to_numpy, prefix_dict


class LTSGNS_ProDMP(AbstractAlgorithm):

    def __init__(self, algorithm_config: ConfigDict, simulator: AbstractGraphNetworkSimulator,
                 env: ProDMPEnvironment, device: str):
        super().__init__(algorithm_config, simulator=simulator, env=env, device=device)

        # create the posterior learner
        self._posterior_learner = get_posterior_learner(self.algorithm_config.posterior_learner, self.env,
                                                        device=device)

    def _single_train_step(self) -> torch.Tensor:
        """
        Performs a single training step and returns the training loss as a detached item.
        For lts_gns, this training step consists of
        1) Loading the data
        2) Updating the posterior learner
        3) Updating the simulator

        Returns: The training loss as a detached item, i.e., as a scalar without a computational graph.

        """
        # set the simulator loss method to mse if pointclouds are used
        if self.env._use_point_cloud:
            self.simulator._loss_function_name = "mse"

        # 1) load a batch of data to train on
        batch = self.env.get_next_train_task_batch()
        # 2) update the posterior learner and sample z
        self.simulator.mode = "posterior_step"
        num_posterior_fit_steps = self.algorithm_config.training.num_posterior_learner_steps
        num_posterior_samples = self.algorithm_config.training.num_z_samples_for_elbo_estimate

        z, _ = self._condition_model_and_get_posterior_samples(batch=batch,
                                                               num_posterior_fit_steps=num_posterior_fit_steps,
                                                               num_posterior_samples=num_posterior_samples,
                                                               mode="train",
                                                               logging=False)
        # 3) update the simulator using the same training data as for the posterior learner
        self.simulator.mode = "gnn_step"
        loss = self.simulator.train_step(batch=batch, z=z)

        return loss.item()

    def _small_eval_step(self, data_iterator: Iterator) -> ValueDict:
        """
        Performs one small evaluation step after every epoch, i.e., computing the mean validation loss of the data
        in the provided iterator.

        Args:
            data_iterator: An iterator over the validation or test data.
        """
        # TODO what about eval mode of the models?
        losses = []
        num_posterior_fit_steps = self.algorithm_config.evaluation.small.num_posterior_learner_step
        num_posterior_samples = self.algorithm_config.training.num_z_samples_for_elbo_estimate
        for batch in tqdm(data_iterator, desc="small eval"):
            z, _ = self._condition_model_and_get_posterior_samples(batch=batch,
                                                                   num_posterior_fit_steps=num_posterior_fit_steps,
                                                                   num_posterior_samples=num_posterior_samples,
                                                                   mode="eval_from_checkpoint",
                                                                   logging=False)
            with torch.no_grad():
                losses.append(self.simulator.loss(batch=batch, z=z).item())
        return {
            keys.SCALARS: {
                keys.SMALL_EVAL: {keys.ALL_EVAL_TASKS: {keys.TOTAL_LOSS: torch.mean(torch.tensor(losses)).item()}}}}

    def _large_eval_step(self, use_validation_data: bool = True) -> ValueDict:
        """
        Performs a big evaluation step after N number of epochs,
        i.e. loading all val tasks and evaluating the k-step prediction,
        the log-marginal likelihood, the MSE and rendering a video of one task.
        """
        if use_validation_data:
            tasks = self.env.val_tasks
            data_iterator = self.env.context_val_batches
            trajectories = self.env.trajectories[keys.VAL]
        else:
            tasks = self.env.test_tasks
            data_iterator = lambda: self.env.context_val_batches(use_validation_data=False)
            trajectories = self.env.trajectories[keys.TEST]

        large_eval_results = {}

        # set the simulator loss method to chamfer if pointclouds are used
        # This is currenlty the way to enforce that the evaluation is done with pointclouds
        if self.env._use_point_cloud:
            self.simulator._loss_function_name = "chamfer"

        k_step_results = self._k_step_evaluations(data_iterator(), trajectories)
        k_step_results = {keys.SCALARS: {keys.LARGE_EVAL: {keys.ALL_EVAL_TASKS: k_step_results}}}

        if self.algorithm_config.evaluation.large.mse_ggns_style:
            ggns_mse_result = self._rollout_mse_ggns_style(data_iterator(), trajectories)
            k_step_results[keys.SCALARS][keys.LARGE_EVAL][keys.ALL_EVAL_TASKS]["ggns_mse"] = ggns_mse_result

        eval_config = self.algorithm_config.evaluation.large
        animation_task_indices = list_from_config(eval_config.animation_task_indices)

        full_rollout_results = {keys.VISUALIZATIONS: {keys.LARGE_EVAL: {}},
                                keys.SCALARS: {keys.LARGE_EVAL: {}}}
        for task_idx in tqdm(animation_task_indices, desc="Full Rollouts"):
            # this is the auxiliary subtask
            context_task = tasks[task_idx]
            eval_task = trajectories[context_task[keys.TRAJECTORY_INDICES]]
            # the dictionary "to_visualize" will be passed to the GraphVisualizer. It makes GIFs out of trajectories.
            rollout_data_dict, rollout_metrics = self.full_rollout(context_task=context_task,
                                                                   traj=eval_task,
                                                                   )
            posterior_logs = rollout_metrics[keys.POSTERIOR_LOGGING]

            full_rollout_results[keys.VISUALIZATIONS][keys.LARGE_EVAL][f"Task_{task_idx}"] = {"to_visualize": {}}

            # predicted trajectory
            for rollout_name, rollout_data in rollout_data_dict.items():
                full_rollout_results[keys.VISUALIZATIONS][keys.LARGE_EVAL][f"Task_{task_idx}"]["to_visualize"][
                    f"full_rollout_{rollout_name}_data"] = rollout_data
            # ground truth trajectory
            gth_data = eval_task.get_subtask(context_task[keys.ANCHOR_INDICES], len(eval_task), deepcopy=True, device="cpu")
            full_rollout_results[keys.VISUALIZATIONS][keys.LARGE_EVAL][f"Task_{task_idx}"]["to_visualize"][
                "ground_truth_data"] = gth_data
            # TODO: How to visualize random context?
            # # context trajectory
            # full_rollout_results[keys.VISUALIZATIONS][keys.LARGE_EVAL][f"Task_{task_idx}"]["to_visualize"][
            #     "context_data"] = context_task.trajectory
            for key, item in posterior_logs.items():
                full_rollout_results[keys.VISUALIZATIONS][keys.LARGE_EVAL][f"Task_{task_idx}"][key] = item
            full_rollout_results[keys.SCALARS][keys.LARGE_EVAL][f"Task_{task_idx}_scalars"] = rollout_metrics[
                keys.SCALARS]
            # add context size to the results
            full_rollout_results[keys.SCALARS][keys.LARGE_EVAL][f"Task_{task_idx}_scalars"]["context_size"] = context_task[keys.CONTEXT_SIZES]

        large_eval_results = deep_update(large_eval_results, k_step_results)
        large_eval_results = deep_update(large_eval_results, full_rollout_results)

        # TODO: Training GMMs is not supported yet
        # if isinstance(self.posterior_learner, MultiDaftPosteriorLearner):
        #     train_figures: ValueDict = self._plot_train_gmm(list_from_config(eval_config.training_gmm_indices))
        #     correctly_name_train_figures = {keys.VISUALIZATIONS: {keys.TRAIN: {}}}
        #     for density_type in train_figures.keys():
        #         for task_idx, figure in train_figures[density_type].items():
        #             correctly_name_train_figures[keys.VISUALIZATIONS][keys.TRAIN][f"Task_{task_idx}_{density_type}"] = \
        #                 figure
        #     large_eval_results = deep_update(large_eval_results, correctly_name_train_figures)
        return large_eval_results

    def full_rollout(self, context_task: Data, traj: Task) -> Tuple[Dict[str, List[Data]], ValueDict]:
        """
        Performs a full rollout of the simulator on the eval task for a *single* trajectory
        Args:
            context_task:
            eval_task:
            task_idx:

        Returns: A list of Data objects containing the mesh positions of the rollout and a dictionary of metrics.


        """
        eval_config = self.algorithm_config.evaluation.large

        # output trajectory is deepcopy of current eval task. Here we will save the newly simulated positions
        full_metrics = {}

        context_batch = Batch.from_data_list([context_task]).to(self._device)

        z, posterior_results = self._condition_model_and_get_posterior_samples(batch=context_batch,
                                                                                            num_posterior_fit_steps=eval_config.num_posterior_learner_steps,
                                                                                            num_posterior_samples=eval_config.num_z_samples_per_animation_task,
                                                                                            mode="eval_from_prior",
                                                                                            logging=True)
        with torch.no_grad():
            predictions = self.simulator._predict(z=z, batch=context_batch, predict_context_timesteps=False, use_cached_processor_output=False)
            last_time_step = len(traj) - 1 - context_task[keys.ANCHOR_INDICES]
            last_predictions = predictions[:, :, last_time_step, :, :]
            # compute metrics
            last_gth = traj.trajectory[-1][keys.POSITIONS].to(predictions.device)
            last_gth = last_gth[node_type_mask(traj.trajectory[-1], key=keys.MESH)]
            last_gth = last_gth[None, ...]
            trajectory_metrics = self.simulator.evaluate_state(predictions=last_predictions,
                                                               reference_labels=last_gth)
            full_metrics |= prefix_dict(to_numpy(trajectory_metrics),
                                        prefix=f"combined_z")
            # detach from graph, prefix and append to dict
        output_trajectories = {}
        for z_idx, prediction in enumerate(predictions):
            # the task dimension is always 1
            prediction = prediction[0]
            output_trajectory = copy.deepcopy(traj.trajectory)
            # skip to the anchor index
            output_trajectory = output_trajectory[context_task[keys.ANCHOR_INDICES]:]
            # replace the node positions with the predicted ones
            for time_step, graph in enumerate(output_trajectory):
                graph[keys.POSITIONS][node_type_mask(graph, key=keys.MESH)] = prediction[time_step, :, :].detach().cpu()
            output_trajectories["z_" + str(z_idx)] = output_trajectory
        logging_results = {keys.SCALARS: full_metrics, keys.POSTERIOR_LOGGING: posterior_results}
        return output_trajectories, logging_results

    def _rollout_mse_ggns_style(self, data_iterator, trajectories):
        eval_config = self.algorithm_config.evaluation.large
        # iterating over all val/test trajectories
        trajectory_metrics = []
        for batch in tqdm(data_iterator, desc="Rollout MSE GGNS Style"):
            # get corresponding trajectories
            traj_indices = batch[keys.TRAJECTORY_INDICES]
            # relevant trajectories for this batch
            batch_trajs = [trajectories.task_list[traj_index] for traj_index in traj_indices]
            # we only support one trajectory per batch in order to get the right mean averaging
            if len(batch_trajs) > 1:
                # TODO: support multiple trajectories per batch
                return -1.0
            assert batch[keys.ANCHOR_INDICES].item() == 0

            # get the posterior samples for this batch
            num_posterior_fit_steps = eval_config.num_posterior_learner_steps
            z, _ = self._condition_model_and_get_posterior_samples(batch=batch,
                                                                   num_posterior_fit_steps=num_posterior_fit_steps,
                                                                   num_posterior_samples=1,
                                                                   mode="eval_from_prior",
                                                                   logging=False)

            with torch.no_grad():
                predictions = self.simulator._predict(z=z, batch=batch, predict_context_timesteps=False,
                                                      use_cached_processor_output=False)
                # get the targets
                traj = batch_trajs[0]
                labels = []
                for graph in traj.trajectory:
                    labels.append(graph[keys.POSITIONS][node_type_mask(graph, key=keys.MESH)])
                labels = torch.stack(labels)
                # add num samples and tasks dimensions
                labels = labels[None, None, ...]
                labels = labels.to(self._device)

                # compute metrics
                trajectory_metrics.append(self.simulator.mse(predictions=predictions, reference_pos=labels))
                # full_metrics |= prefix_dict(to_numpy(trajectory_metrics), prefix=f"k_{k}")
        final_result = torch.mean(torch.tensor(trajectory_metrics)).item()
        return final_result

    def _k_step_evaluations(self, data_iterator: Iterator[Batch], trajectories: TaskCollection) -> ValueDict:
        eval_config = self.algorithm_config.evaluation.large
        multi_step_evaluations = torch.tensor(eval_config.multi_step_evaluations, device=self._device)
        if multi_step_evaluations is None or len(multi_step_evaluations) == 0:
            return {}
        max_k = max(multi_step_evaluations)  # max k-step prediction to evaluate

        full_metrics = defaultdict(list)
        # iterating over all val/test trajectories
        for batch in tqdm(data_iterator, desc="Large k-step evaluation"):
            # get corresponding trajectories
            traj_indices = batch[keys.TRAJECTORY_INDICES]
            # relevant trajectories for this batch
            batch_trajs = [trajectories.task_list[traj_index] for traj_index in traj_indices]
            traj_length = len(batch_trajs[0])
            # check that the anchor index is not too far in the future
            assert torch.all(batch[keys.ANCHOR_INDICES] + max_k < traj_length), "Anchor index is too far in the future."

            # get the posterior samples for this batch
            num_posterior_fit_steps = eval_config.num_posterior_learner_steps
            z, _ = self._condition_model_and_get_posterior_samples(batch=batch,
                                                                   num_posterior_fit_steps=num_posterior_fit_steps,
                                                                   num_posterior_samples=1,
                                                                   mode="eval_from_prior",
                                                                   logging=False)
            with torch.no_grad():
                predictions = self.simulator._predict(z=z, batch=batch, predict_context_timesteps=False, use_cached_processor_output=False)
                for k in multi_step_evaluations:
                    # get the targets
                    absolute_context_indices = batch[keys.ANCHOR_INDICES] + k
                    labels = []
                    for idx, traj in enumerate(batch_trajs):
                        label_graph = traj[absolute_context_indices[idx]]

                        labels.append(label_graph.pos[node_type_mask(label_graph, keys.MESH)])
                    labels = torch.stack(labels)
                    labels = labels.to(self._device)
                    # compute metrics
                    trajectory_metrics = self.simulator.evaluate_state(predictions=predictions[:, :, k, :, :],
                                                                       reference_labels=labels)
                    full_metrics |= prefix_dict(to_numpy(trajectory_metrics), prefix=f"k_{k}")

        # gather the results by averaging over all tasks
        return_metrics = {key: np.mean(value) for key, value in full_metrics.items()}
        return return_metrics

    def _condition_model_and_get_posterior_samples(self, *,
                                                   batch: Batch,
                                                   num_posterior_fit_steps: int,
                                                   num_posterior_samples: int,
                                                   mode: Optional[str] = None,
                                                   logging: bool = False) -> Tuple[torch.Tensor, ValueDict]:
        """
        Condition the model on the given batch and task belonging and return posterior samples. Logs the fitting of
        the posterior learner.
        Args:
            batch: Data to condition the model on
            task_belonging: Task belonging of the batch
            num_posterior_fit_steps:
            num_posterior_samples: How many z should be drawn from the posterior
            mode: Mode for the Posterior Learner, currently only in use for MultiDaftPosteriorLearner
            logging: Whether to log the fitting of the posterior learner

        Returns: z samples from the posterior and logging of the posterior learner (empty dict if no logging enabled)

        """

        # load the batch into the simulator
        self.simulator.condition_on_data(batch)
        task_indices = batch[keys.TASK_INDICES]
        # adapt the posterior learner
        if isinstance(self.posterior_learner, MultiDaftPosteriorLearner):
            # set eval_from_checkpoint mode. This is the same as train, but it uses the eval GMM parameters
            if mode is None:
                raise ValueError("Mode may not be None for MultiDaftPosteriorLearner")
            self._posterior_learner: MultiDaftPosteriorLearner
            self._posterior_learner.mode = mode
            # reset the current weights in the posterior logger if mode is eval_from_prior
            if mode == "eval_from_prior":
                self._posterior_learner.reset_eval_from_prior()
        posterior_learner_logging_results = self.posterior_learner.fit(n_steps=num_posterior_fit_steps,
                                                                       task_indices=task_indices,
                                                                       lnpdf=self.simulator,
                                                                       logging=logging, )
        # Get the samples from the posterior
        z = self.posterior_learner.sample(n_samples=num_posterior_samples,
                                          task_indices=task_indices
                                          )
        return z, posterior_learner_logging_results

    def save_checkpoint(self, directory: str, iteration: int, is_initial_save: bool, is_final_save: bool = False):
        self.simulator.save_checkpoint(directory, iteration, is_initial_save, is_final_save)
        self.posterior_learner.save_checkpoint(directory, iteration, is_initial_save, is_final_save)

    @property
    def posterior_learner(self) -> AbstractPosteriorLearner:
        if self._posterior_learner is None:
            raise ValueError("Posterior learner not set")
        return self._posterior_learner

    @property
    def simulator(self) -> LTSGNSProDMPSimulator:
        if self._simulator is None:
            raise ValueError("Simulator not set")
        self._simulator: LTSGNSProDMPSimulator
        return self._simulator

    @property
    def env(self) -> ProDMPEnvironment:
        env: ProDMPEnvironment = super().env
        return env

    def _plot_train_gmm(self, train_task_indices: List[int]) -> ValueDict:
        """
        Plot the GMM of the training data for the given task indices
        Args:
            train_task_indices: Task indices to plot the GMM for

        Returns: List of figures

        """
        figures = {"posterior": {}, "prior": {}, "likelihood": {}}
        # get the data for the given task indices
        for train_task_idx in tqdm(train_task_indices, desc="Plotting train GMMs"):
            train_task = self.env.real_mesh_auxiliary_train_tasks[train_task_idx]
            train_batch, train_task_belonging = self.env.build_eval_task_batch(train_task, train_task_idx,
                                                                               recompute_graph_edges=False)
            # condition on data
            self.simulator.condition_on_data(train_batch, train_task_belonging)
            # get the GMM params
            self._posterior_learner: MultiDaftPosteriorLearner
            fig_posterior, fig_prior, fig_likelihood = self._posterior_learner.visualize_train_posteriors(
                [train_task_idx],
                self.simulator)
            figures["posterior"][train_task_idx] = fig_posterior
            figures["prior"][train_task_idx] = fig_prior
            figures["likelihood"][train_task_idx] = fig_likelihood
        return figures
