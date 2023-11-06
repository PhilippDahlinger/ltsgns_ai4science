import warnings
from collections import defaultdict
from typing import List, Tuple, Iterator, Dict

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from lts_gns.algorithms.abstract_algorithm import AbstractAlgorithm
from lts_gns.algorithms.algorithm_util import list_from_config
from lts_gns.algorithms.simulators.mgn_simulator import MGNSimulator
from lts_gns.envs.mgn_environment import MGNEnvironment
from lts_gns.envs.task.task_collection import TaskCollection
from lts_gns.util import keys
from lts_gns.util.dict_util import deep_update
from lts_gns.util.own_types import ValueDict
from lts_gns.util.util import to_numpy, prefix_dict


class MGNProDMP(AbstractAlgorithm):
    def _single_train_step(self) -> torch.Tensor:
        """
        Performs a single training step and returns the training loss as a detached item.

        Returns: The training loss as a detached item.

        """
        batch, task_belonging = self.env.get_next_train_task_batch(add_training_noise=self._add_training_noise)

        self.simulator.condition_on_data(batch, task_belonging)
        loss = self.simulator.train_step(batch=batch)
        return loss.item()

    def _small_eval_step(self, data_iterator: Iterator) -> ValueDict:
        """
        Performs one small evaluation step after every epoch, i.e., computing the mean validation loss of the data
        in the provided iterator.

        Args:
            data_iterator: An iterator over the validation or test data.
        """
        losses = []
        for batch, task_belonging in data_iterator:
            # load the batch into the simulator
            self.simulator.condition_on_data(batch, task_belonging)
            with torch.no_grad():
                losses.append(self.simulator.loss(batch=batch).item())
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
            task_list = self.env.val_tasks
        else:
            task_list = self.env.test_tasks

        large_eval_results = {}

        k_step_results = self._k_step_evaluations(task_list)
        k_step_results = {keys.SCALARS: {keys.LARGE_EVAL: {keys.ALL_EVAL_TASKS: k_step_results}}}

        eval_config = self.algorithm_config.evaluation.large
        task_indices = list_from_config(eval_config.animation_task_indices)

        full_rollout_results = {keys.VISUALIZATIONS: {keys.LARGE_EVAL: {}},
                                keys.SCALARS: {keys.LARGE_EVAL: {}}}
        for task_idx in task_indices:
            eval_task = task_list[task_idx]
            # the dictionary "to_visualize" will be passed to the GraphVisualizer. It makes GIFs out of trajectories.

            rollout_data_dict, rollout_metrics = self.full_rollout(eval_task=eval_task,
                                                                   task_idx=task_idx)

            full_rollout_results[keys.VISUALIZATIONS][keys.LARGE_EVAL][f"Task_{task_idx}"] = {"to_visualize": {}}

            for rollout_name, rollout_data in rollout_data_dict.items():
                full_rollout_results[keys.VISUALIZATIONS][keys.LARGE_EVAL][f"Task_{task_idx}"]["to_visualize"][
                    f"full_rollout_{rollout_name}_data"] = rollout_data
            full_rollout_results[keys.VISUALIZATIONS][keys.LARGE_EVAL][f"Task_{task_idx}"]["to_visualize"][
                "ground_truth_data"] = eval_task.trajectory
            full_rollout_results[keys.SCALARS][keys.LARGE_EVAL][f"Task_{task_idx}"] = rollout_metrics[keys.SCALARS]

        large_eval_results = deep_update(large_eval_results, k_step_results)
        large_eval_results = deep_update(large_eval_results, full_rollout_results)
        return large_eval_results

    def full_rollout(self, eval_task, task_idx) -> Tuple[Dict[str, List[Data]], ValueDict]:
        """
        Performs a full rollout of the simulator on the eval task for a *single* trajectory
        Args:
            eval_task:
            task_idx:

        Returns: A list of Data objects containing the mesh positions of the rollout and a dictionary of metrics.


        """
        eval_config = self.algorithm_config.evaluation.large

        # output trajectory is deepcopy of current eval task. Here we will save the newly simulated positions
        full_metrics = {}
        output_trajectories = {}
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
            
            # evaluate the simulator on this subtask
            self.simulator.condition_on_data(batched_step, task_belonging)
            
            updated_mesh_trajectories = self.simulator.predict_denormalize_and_integrate()
            
            for updated_mesh_state in updated_mesh_trajectories:
                data = output_trajectory[0].clone()
                data.x[data.node_type == keys.MESH] = updated_mesh_state
                output_trajectory.append(data)

            # for step in range(1, len(eval_task.trajectory)):
            #     # index from 1 here as we are interested in the step
            #     # that we are predicting. I.e., for step 0, we are predicting the positions/features of step 1.

            #     # evaluate the simulator on this subtask
            #     self.simulator.condition_on_data(batched_step, task_belonging)

            #     # predict quantities for the next step and go back to cpu, since the rest of the eval is done there
            #     updated_mesh_state = self.simulator.predict_denormalize_and_integrate()
            #     # todo merge this with lts gns eval as its the same thing
            #     updated_mesh_state = {k: v.cpu() for k, v in updated_mesh_state.items()}  # move to cpu
            #     # each entry has shape (1, num_mesh_nodes, task_dimension). The 1 is needed for the eval loop below

            #     # get the **next** known system state (e.g., collider positions) and
            #     # set the positions of the mesh nodes to the predicted positions for the **next step**
            #     # Doing this here keeps the code closer together, and also leads to nicer indices
            #     current_simulation_step = eval_task.get_subtask(step, step + 1, deepcopy=True, device="cpu")
            #     current_simulation_step = self.simulator.update_batch_from_state(mesh_state=updated_mesh_state,
            #                                                                      batch_or_task=current_simulation_step)
            #     batched_step, task_belonging = self.env.build_eval_task_batch(current_simulation_step,
            #                                                                   task_idx,
            #                                                                   recompute_graph_edges=recompute_edges)

            #     data = batched_step.to_data_list()[0].cpu().clone()
            #     output_trajectory.append(data)

            # compute metrics
            trajectory_metrics = self.simulator.evaluate_state(predicted_mesh_state=updated_mesh_state,
                                                                reference_labels=eval_task.trajectory[-1].next_mesh_pos)
            full_metrics |= prefix_dict(to_numpy(trajectory_metrics), prefix=f"prodmp")
            # detach from graph, prefix and append to dict
        # output_trajectories[recompute_edge_str] = output_trajectory
        logging_results = {keys.SCALARS: full_metrics}
        return output_trajectory, logging_results

    def _k_step_evaluations(self, task_list: TaskCollection) -> ValueDict:
        # todo @niklas: This is super slow on GPU for whatever reason
        eval_config = self.algorithm_config.evaluation.large
        # iterating over all val/test trajectories

        multi_step_evaluations = eval_config.multi_step_evaluations
        if multi_step_evaluations is None or len(multi_step_evaluations) == 0:
            return {}
        max_k = max(multi_step_evaluations)  # max k-step prediction to evaluate

        full_metrics = defaultdict(list)

        recompute_edge_list = list_from_config(eval_config.recompute_edges)
        for recompute_edges in recompute_edge_list:
            recompute_edge_str = f"recomputed_edges" if recompute_edges else "fixed_edges"
            for task_idx, eval_task in tqdm(enumerate(task_list),
                                            desc="Large k-step evaluation",
                                            total=len(task_list)):

                # evaluate the simulator on this task with different k-step-predictions
                if max_k >= len(eval_task.trajectory) - 1:
                    warnings.warn("max_k is larger than the length of the trajectory. Skipping this task.")
                    continue

                with torch.no_grad():  # no gradient needed during evaluation
                    # We evaluate on all possible k-step predictions.
                    # For T time steps, we can do T-k+1 k-step predictions
                    num_parallel_evaluations = len(eval_task.trajectory) - max_k + 1
                    current_sub_task = eval_task.get_subtask(0, num_parallel_evaluations,
                                                             deepcopy=True, device=self._device)
                    # update the batch for this subtask,
                    # i.e., use the updated mesh positions to update the distances encoded in the edge features
                    subtask_batch, task_belonging = self.env.build_eval_task_batch(current_sub_task, task_idx,
                                                                                   recompute_graph_edges=False)

                    for k in range(1, max_k + 1):
                        # Calculate the k-step predictions by doing a single step max(k) times and
                        # updating the ground truth mesh positions with the predicted ones in each step
                        self.simulator.condition_on_data(subtask_batch, task_belonging)

                        # predict the next step
                        mesh_state = self.simulator.predict_denormalize_and_integrate()

                        if k in multi_step_evaluations:  # gather metrics of the k-step prediction
                            evaluations = to_numpy(self.simulator.evaluate_state(mesh_state))
                            evaluations = prefix_dict(evaluations, prefix=f"{recompute_edge_str}")
                            for key, value in evaluations.items():
                                # full_metrics is a dict of lists. Each list contains the results for all tasks
                                full_metrics[f"{k}-step {key}"].append(value)

                        if k < max_k:  # prepare **next step**, except for at the last step
                            current_sub_task = eval_task.get_subtask(k, k + num_parallel_evaluations,
                                                                     deepcopy=True, device=self._device)

                            current_sub_task = self.simulator.update_batch_from_state(mesh_state=mesh_state,
                                                                                      batch_or_task=current_sub_task)
                            subtask_batch, task_belonging = self.env.build_eval_task_batch(current_sub_task, task_idx,
                                                                                           recompute_graph_edges=recompute_edges)

        # gather the results by averaging over all tasks
        return_metrics = {key: np.mean(value) for key, value in full_metrics.items()}
        return return_metrics

    def save_checkpoint(self, directory: str, iteration: int, is_initial_save: bool, is_final_save: bool = False):
        self.simulator.save_checkpoint(directory, iteration, is_initial_save, is_final_save)

    @property
    def simulator(self) -> MGNSimulator:
        if self._simulator is None:
            raise ValueError("Simulator not set")
        self._simulator: MGNSimulator
        return self._simulator

    @property
    def env(self) -> MGNEnvironment:
        env: MGNEnvironment = super().env
        return env
