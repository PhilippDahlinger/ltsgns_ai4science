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
from lts_gns.algorithms.simulators.ltsgns_simulator import LTSGNSSimulator
from lts_gns.algorithms.simulators.mgn_simulator import mse
from lts_gns.envs.abstract_gns_environment import AbstractGNSEnvironment
from lts_gns.envs.lts_gns_environment import LTSGNSEnvironment
from lts_gns.envs.task.task import Task
from lts_gns.envs.task.task_collection import TaskCollection
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.dict_util import deep_update
from lts_gns.util.own_types import ValueDict
from lts_gns.util.util import to_numpy, prefix_dict, node_type_mask
from lts_gns.algorithms.simulators.abstract_graph_network_simulator import AbstractGraphNetworkSimulator


class LTSGNS(AbstractAlgorithm):

    def __init__(self, algorithm_config: ConfigDict, simulator: AbstractGraphNetworkSimulator,
                 env: AbstractGNSEnvironment, device: str):
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
        # 1) load a batch of data to train on
        batch, task_belonging = self.env.get_next_train_task_batch(add_training_noise=self._add_training_noise)
        task_indices = task_belonging[keys.TASK_INDICES]

        # 2) update the posterior learner
        if isinstance(self.posterior_learner, MultiDaftPosteriorLearner):
            self._posterior_learner: MultiDaftPosteriorLearner
            self._posterior_learner.mode = "train"  # set train mode
        self.simulator.mode = "posterior_step"
        self.simulator.condition_on_data(batch, task_belonging)  # load the batch into the simulator
        self.posterior_learner.fit(n_steps=self.algorithm_config.training.num_posterior_learner_steps,
                                   task_indices=task_indices, lnpdf=self.simulator)

        # 3) update the simulator using the same training data as for the posterior learner
        self.simulator.mode = "gnn_step"
        z = self.posterior_learner.sample(n_samples=self.algorithm_config.training.num_z_samples_for_elbo_estimate,
                                          task_indices=task_indices)  # shape (n_samples, n_tasks, n_latent_dims)
        loss = self.simulator.train_step(batch=batch, z=z)

        return loss.item()

    def _small_eval_step(self, data_iterator: Iterator) -> ValueDict:
        """
        Performs one small evaluation step after every epoch, i.e., computing the mean validation loss of the data
        in the provided iterator.

        Args:
            data_iterator: An iterator over the validation or test data.
        """
        losses = []
        num_posterior_fit_steps = self.algorithm_config.evaluation.small.num_posterior_learner_step
        num_posterior_samples = self.algorithm_config.training.num_z_samples_for_elbo_estimate
        for batch, task_belonging in data_iterator:
            z, _ = self._condition_model_and_get_posterior_samples(batch=batch,
                                                                   task_belonging=task_belonging,
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
        else:
            tasks = self.env.test_tasks

        large_eval_results = {}

        k_step_results = self._k_step_evaluations(tasks)
        k_step_results = {keys.SCALARS: {keys.LARGE_EVAL: {keys.ALL_EVAL_TASKS: k_step_results}}}

        if self.algorithm_config.evaluation.large.mse_ggns_style:
            ggns_mse_result = self._rollout_mse_ggns_style(tasks)
            k_step_results[keys.SCALARS][keys.LARGE_EVAL][keys.ALL_EVAL_TASKS]["ggns_mse"] = ggns_mse_result

        eval_config = self.algorithm_config.evaluation.large
        animation_task_indices = list_from_config(eval_config.animation_task_indices)

        full_rollout_results = {keys.VISUALIZATIONS: {keys.LARGE_EVAL: {}},
                                keys.SCALARS: {keys.LARGE_EVAL: {}}}
        for task_idx in animation_task_indices:
            context_task = tasks.subtask_list[task_idx]
            eval_task = tasks.map_subtasks_to_tasks(task_idx)

            # the dictionary "to_visualize" will be passed to the GraphVisualizer. It makes GIFs out of trajectories.

            rollout_data_dict, rollout_metrics = self.full_rollout(context_task=context_task,
                                                                   eval_task=eval_task,
                                                                   task_idx=task_idx)
            posterior_logs = rollout_metrics[keys.POSTERIOR_LOGGING]

            full_rollout_results[keys.VISUALIZATIONS][keys.LARGE_EVAL][f"Task_{task_idx}"] = {"to_visualize": {}}

            # predicted trajectory
            for rollout_name, rollout_data in rollout_data_dict.items():
                full_rollout_results[keys.VISUALIZATIONS][keys.LARGE_EVAL][f"Task_{task_idx}"]["to_visualize"][
                    f"full_rollout_{rollout_name}_data"] = rollout_data
            # ground truth trajectory
            if len(eval_task.trajectory) == 1 and len(rollout_data) == 2:
                gth_data = [eval_task.trajectory[0], eval_task.trajectory[0]]
            else:
                gth_data = eval_task.trajectory
            full_rollout_results[keys.VISUALIZATIONS][keys.LARGE_EVAL][f"Task_{task_idx}"]["to_visualize"][
                "ground_truth_data"] = gth_data
            # context trajectory
            full_rollout_results[keys.VISUALIZATIONS][keys.LARGE_EVAL][f"Task_{task_idx}"]["to_visualize"][
                "context_data"] = context_task.trajectory
            for key, item in posterior_logs.items():
                full_rollout_results[keys.VISUALIZATIONS][keys.LARGE_EVAL][f"Task_{task_idx}"][key] = item
            full_rollout_results[keys.SCALARS][keys.LARGE_EVAL][f"Task_{task_idx}_scalars"] = \
                (rollout_metrics)[ keys.SCALARS]
            # add context size to the results
            full_rollout_results[keys.SCALARS][keys.LARGE_EVAL][f"Task_{task_idx}_scalars"]["context_size"] = len(
                context_task.trajectory)

        large_eval_results = deep_update(large_eval_results, k_step_results)
        large_eval_results = deep_update(large_eval_results, full_rollout_results)

        if isinstance(self.posterior_learner, MultiDaftPosteriorLearner):
            train_figures: ValueDict = self._plot_train_gmm(list_from_config(eval_config.training_gmm_indices))
            correctly_name_train_figures = {keys.VISUALIZATIONS: {keys.TRAIN: {}}}
            for density_type in train_figures.keys():
                for task_idx, figure in train_figures[density_type].items():
                    correctly_name_train_figures[keys.VISUALIZATIONS][keys.TRAIN][f"Task_{task_idx}_{density_type}"] = \
                        figure
            large_eval_results = deep_update(large_eval_results, correctly_name_train_figures)
        return large_eval_results

    def full_rollout(self, context_task, eval_task, task_idx) -> Tuple[Dict[str, List[Data]], ValueDict]:
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
        recompute_edge_list = list_from_config(eval_config.recompute_edges)
        full_metrics = {}
        output_trajectories = {}
        posterior_results = {}
        for recompute_edges in recompute_edge_list:
            recompute_edge_str = f"recomputed_edges" if recompute_edges else "fixed_edges"

            context_batch, task_belonging = self.env.build_eval_task_batch(context_task, task_idx,
                                                                           recompute_graph_edges=False)
            # do not need to recompute edges at step 0

            # TODO: think about drawing multiple samples
            all_zs, posterior_logging_results = self._condition_model_and_get_posterior_samples(batch=context_batch,
                                                                                                task_belonging=task_belonging,
                                                                                                num_posterior_fit_steps=eval_config.num_posterior_learner_steps,
                                                                                                num_posterior_samples=eval_config.num_z_samples_per_animation_task,
                                                                                                mode="eval_from_prior",
                                                                                                logging=True)
            for current_z_idx in range(eval_config.num_z_samples_per_animation_task):
                # loop for multiple simulations with different z samples
                z = all_zs[current_z_idx:current_z_idx + 1]
                output_trajectory = [eval_task.trajectory[0].clone()]
                with torch.no_grad():
                    # build the batch for this subtask, which is a single simulation state.
                    # Includes processing the graph features and edge connectivity
                    current_simulation_step = eval_task.get_subtask(0, 1, deepcopy=True)
                    batched_step, task_belonging = self.env.build_eval_task_batch(current_simulation_step, task_idx,
                                                                                  recompute_graph_edges=False)
                    # todo should this be on cpu or gpu?
                    if len(current_simulation_step) > 1:
                        raise NotImplementedError("Multiple subtasks not implemented yet.")

                    # hack to make 2 step task working
                    if len(eval_task.trajectory) == 1:
                        start_idx = 0
                    else:
                        start_idx = 1

                    for step in range(start_idx, len(eval_task.trajectory)):
                        # index from 1 here as we are interested in the step
                        # that we are predicting. I.e., for step 0, we are predicting the positions/features of step 1.

                        # evaluate the simulator on this subtask
                        self.simulator.condition_on_data(batched_step, task_belonging)

                        # predict quantities for the next step and go back to cpu, since the rest of the eval is done there
                        updated_mesh_state = self.simulator.predict_denormalize_and_integrate(z=z,
                                                                                              use_cached_processor_output=True)
                        # @Niklas Do not take the first z sample, we expect the mesh state to have the z dimension!
                        updated_mesh_state = {k: v.cpu()
                                              for k, v in updated_mesh_state.items()}  # move to cpu,

                        # each entry has shape (1, num_mesh_nodes, task_dimension). The 1 is needed for the eval loop below

                        # get the **next** known system state (e.g., collider positions) and
                        # set the positions of the mesh nodes to the predicted positions for the **next step**
                        # Doing this here keeps the code closer together, and also leads to nicer indices
                        current_simulation_step = eval_task.get_subtask(step, step + 1, deepcopy=True, device="cpu")
                        current_simulation_step = self.simulator.update_batch_from_state(mesh_state=updated_mesh_state,
                                                                                         batch_or_task=current_simulation_step)
                        batched_step, task_belonging = self.env.build_eval_task_batch(current_simulation_step,
                                                                                      task_idx,
                                                                                      recompute_graph_edges=recompute_edges)

                        data = batched_step.to_data_list()[0].cpu().clone()
                        output_trajectory.append(data)

                    # compute metrics
                    trajectory_metrics = self.simulator.evaluate_state(predicted_mesh_state=updated_mesh_state,
                                                                       reference_labels=eval_task.trajectory[
                                                                           -1].next_mesh_pos)
                    full_metrics |= prefix_dict(to_numpy(trajectory_metrics),
                                                prefix=f"{recompute_edge_str}_z_{current_z_idx}")
                    # detach from graph, prefix and append to dict
                output_trajectories[f"{recompute_edge_str}_z_{current_z_idx}"] = output_trajectory

            posterior_results |= prefix_dict(posterior_logging_results, prefix=f"{recompute_edge_str}")

        logging_results = {keys.SCALARS: full_metrics, keys.POSTERIOR_LOGGING: posterior_results}
        return output_trajectories, logging_results

    def _rollout_mse_ggns_style(self, task_collection: TaskCollection):
        eval_config = self.algorithm_config.evaluation.large
        task_ggns_mse = []
        for task_idx, eval_task in tqdm(enumerate(task_collection),
                                           desc="Rollout MSE GGNS Style",
                                           total=len(task_collection)):

            # this will break once there are more than 1 subtask per task
            context_task = task_collection.subtask_list[task_idx]
            context_batch, task_belonging = self.env.build_eval_task_batch(context_task, task_idx,
                                                                           recompute_graph_edges=False)
            all_zs, posterior_logging_results = self._condition_model_and_get_posterior_samples(batch=context_batch,
                                                                                                task_belonging=task_belonging,
                                                                                                num_posterior_fit_steps=eval_config.num_posterior_learner_steps,
                                                                                                num_posterior_samples=1,
                                                                                                mode="eval_from_prior",
                                                                                                logging=True)
            z = all_zs[0:1]
            with torch.no_grad():
                # build the batch for this subtask, which is a single simulation state.
                # Includes processing the graph features and edge connectivity
                current_simulation_step = eval_task.get_subtask(0, 1, deepcopy=True)
                batched_step, task_belonging = self.env.build_eval_task_batch(current_simulation_step, task_idx,
                                                                              recompute_graph_edges=False)
                traj_ggns_mse = []
                if len(current_simulation_step) > 1:
                    raise NotImplementedError("Multiple subtasks not implemented yet.")
                for step in range(1, len(eval_task.trajectory)):
                    # index from 1 here as we are interested in the step
                    # that we are predicting. I.e., for step 0, we are predicting the positions/features of step 1.

                    # evaluate the simulator on this subtask
                    self.simulator.condition_on_data(batched_step, task_belonging)
                    # predict quantities for the next step and go back to cpu, since the rest of the eval is done there
                    updated_mesh_state = self.simulator.predict_denormalize_and_integrate(z=z,
                                                                                          use_cached_processor_output=True)
                    updated_mesh_state = {k: v.cpu() for k, v in updated_mesh_state.items()}  # move to cpu
                    # each entry has shape (1, num_mesh_nodes, task_dimension). The 1 is needed for the eval loop below
                    ref_graph = eval_task.get_subtask(step, step + 1)[0]
                    labels = ref_graph[keys.POSITIONS][node_type_mask(ref_graph, key=keys.MESH)]
                    prediction = updated_mesh_state["pos"]
                    traj_ggns_mse.append(mse(prediction, labels))
                    # get the **next** known system state (e.g., collider positions) and
                    # set the positions of the mesh nodes to the predicted positions for the **next step**
                    # Doing this here keeps the code closer together, and also leads to nicer indices
                    current_simulation_step = eval_task.get_subtask(step, step + 1, deepcopy=True, device="cpu")
                    current_simulation_step = self.simulator.update_batch_from_state(mesh_state=updated_mesh_state,
                                                                                     batch_or_task=current_simulation_step)
                    batched_step, task_belonging = self.env.build_eval_task_batch(current_simulation_step,
                                                                                  task_idx,
                                                                                  recompute_graph_edges=True)
                # print("len of traj_ggns_mse", len(traj_ggns_mse))
                task_ggns_mse.append(torch.stack(traj_ggns_mse).mean())

        # gather the results by averaging over all tasks
        task_ggns_mse = torch.stack(task_ggns_mse).mean().item()
        print("task_ggns_mse", task_ggns_mse)
        return task_ggns_mse


    def _k_step_evaluations(self, task_collection: TaskCollection) -> ValueDict:
        eval_config = self.algorithm_config.evaluation.large
        # iterating over all val/test trajectories

        multi_step_evaluations = eval_config.multi_step_evaluations
        if multi_step_evaluations is None or len(multi_step_evaluations) == 0:
            return {}
        max_k = max(multi_step_evaluations)  # max k-step prediction to evaluate

        full_metrics = defaultdict(list)

        subtasks = task_collection.subtask_list
        recompute_edge_list = list_from_config(eval_config.recompute_edges)
        for first_step_only in [True, False]:
            for recompute_edges in recompute_edge_list:
                # todo simplify the above
                recompute_edge_str = f"recomputed_edges" if recompute_edges else "fixed_edges"
                for task_idx, context_task in tqdm(enumerate(subtasks),
                                                   desc="Large k-step evaluation",
                                                   total=len(subtasks)):
                    # evaluate the simulator on this task with different k-step-predictions. We use a context task
                    #  to fit a **new** posterior for each evaluation task, and evaluate it on the full trajectory/task
                    eval_task = task_collection.map_subtasks_to_tasks(task_idx)
                    if max_k >= len(eval_task.trajectory) - 1:
                        warnings.warn("max_k is larger than the length of the trajectory. Skipping this task.")
                        continue
                    # todo @Niklas there is a way to refactor this with MGN and everything, but that's a bit more work
                    #  and I don't think it's worth it right now
                    self._evaluate_k_step_task(context_task, eval_task, full_metrics, max_k,
                                               multi_step_evaluations, recompute_edge_str, recompute_edges, task_idx,
                                               first_step_only=first_step_only)

        # gather the results by averaging over all tasks
        return_metrics = {key: np.mean(value) for key, value in full_metrics.items()}
        return return_metrics

    def _evaluate_k_step_task(self, context_task, eval_task, full_metrics, max_k, multi_step_evaluations,
                              recompute_edge_str, recompute_edges, task_idx, first_step_only: bool):
        if first_step_only:
            num_parallel_evaluations = 1
            dict_prefix = ""
        else:
            # We evaluate on all possible k-step predictions.
            # For T time steps, we can do T-k+1 k-step predictions
            num_parallel_evaluations = len(eval_task.trajectory) - max_k + 1
            dict_prefix = "Full "

        # fit the posterior on the context task
        context_batch, task_belonging = self.env.build_eval_task_batch(context_task, task_idx,
                                                                       recompute_graph_edges=False)
        num_posterior_fit_steps = self.algorithm_config.evaluation.large.num_posterior_learner_steps
        z, _ = self._condition_model_and_get_posterior_samples(batch=context_batch,
                                                               task_belonging=task_belonging,
                                                               num_posterior_fit_steps=num_posterior_fit_steps,
                                                               num_posterior_samples=1,
                                                               mode="eval_from_prior",
                                                               logging=False)
        with torch.no_grad():  # no gradient needed during evaluation
            # We evaluate on all possible k-step predictions.
            # For T time steps, we can do T-k+1 k-step predictions
            current_sub_task = eval_task.get_subtask(0, num_parallel_evaluations,
                                                     deepcopy=True, device=self._device)
            # update the batch for this subtask,
            # i.e., use the updated mesh positions to update the distances encoded in the edge features
            subtask_batch, task_belonging = self.env.build_eval_task_batch(current_sub_task, task_idx,
                                                                           recompute_graph_edges=False)
            for k in range(1, max_k + 1):
                # Calculate the k-step predictions by doing a single step max(k) times and
                # updating the ground truth mesh positions with the predicted ones in each step

                # condition the simulator on this new step
                # TODO @niklas I don't really like that a bunch of things happen in secret here, as this sets new labels etc.
                #   In general, the simulator should not hold the data, but only the model. The data should probably be its own
                #   object, which is then passed to the simulator.
                #       Comment from Philipp: We kinda need the data to be inside the simulator, at least the log_density() call
                #   has to be able to access the data indirectly without having to pass it explicitly. But I agree that the
                #   simulator should not hold the data, but only the model.
                self.simulator.condition_on_data(subtask_batch, task_belonging)

                # predict the next step
                mesh_state = self.simulator.predict_denormalize_and_integrate(z=z,
                                                                              use_cached_processor_output=True)
                if k in multi_step_evaluations:  # gather metrics of the k-step prediction
                    evaluations = to_numpy(self.simulator.evaluate_state(mesh_state))
                    evaluations = prefix_dict(evaluations, prefix=f"{recompute_edge_str}")
                    for key, value in evaluations.items():
                        # full_metrics is a dict of lists. Each list contains the results for all tasks
                        full_metrics[f"{dict_prefix}{k}-step {key}"].append(value)
                if k < max_k:  # prepare **next step**, except for at the last step
                    current_sub_task = eval_task.get_subtask(k, k + num_parallel_evaluations,
                                                             deepcopy=True, device=self._device)

                    current_sub_task = self.simulator.update_batch_from_state(mesh_state=mesh_state,
                                                                              batch_or_task=current_sub_task)
                    subtask_batch, task_belonging = self.env.build_eval_task_batch(current_sub_task, task_idx,
                                                                                   recompute_graph_edges=recompute_edges)

    def _condition_model_and_get_posterior_samples(self, *,
                                                   batch: Batch,
                                                   task_belonging: ValueDict,
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
        self.simulator.condition_on_data(batch, task_belonging)
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
                                                                       task_indices=task_belonging[keys.TASK_INDICES],
                                                                       lnpdf=self.simulator,
                                                                       logging=logging, )
        # Get the samples from the posterior
        z = self.posterior_learner.sample(n_samples=num_posterior_samples,
                                          task_indices=task_belonging[keys.TASK_INDICES]
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
    def simulator(self) -> LTSGNSSimulator:
        if self._simulator is None:
            raise ValueError("Simulator not set")
        self._simulator: LTSGNSSimulator
        return self._simulator

    @property
    def env(self) -> LTSGNSEnvironment:
        env: LTSGNSEnvironment = super().env
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
