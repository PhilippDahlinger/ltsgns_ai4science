import os
from pathlib import Path

from lts_gns.algorithms.abstract_algorithm import AbstractAlgorithm
from lts_gns.recording.loggers.abstract_logger import AbstractLogger
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.dict_util import deep_update_weird_custom_config_dicts
from lts_gns.util.own_types import *
from lts_gns.visualizations.graph_visualizer import GraphVisualizer


def visualize_trajectories(recorded_values, graph_visualizer: GraphVisualizer,
                           iteration: int, save_animation: bool = False,
                           save_path: str | Path = None):
    if keys.VISUALIZATIONS in recorded_values:
        vis_dict = recorded_values[keys.VISUALIZATIONS]
        for eval_key, all_tasks_dict in vis_dict.items():
            for task_name, task_dict in all_tasks_dict.items():
                if "to_visualize" not in task_dict:
                    continue
                data_dict = task_dict["to_visualize"]

                # find ground truth trajectory
                ground_truth_traj = None
                for traj_name, traj_data in data_dict.items():
                    if "ground_truth" in traj_name:
                        ground_truth_traj = traj_data
                        break

                for traj_name, traj_data in data_dict.items():
                    if "ground_truth" in traj_name:  # ground truth is plotted together with predicted trajectory
                        continue
                    if traj_name.endswith("_data"):   # remove _data suffix
                        traj_name = traj_name[:-5]

                    animation_name = f"{task_name.lower()}__{traj_name}"

                    if "context" in traj_name:
                        filename = f"{animation_name}.gif"
                        animation = graph_visualizer.visualize_trajectory(traj_data,
                                                                          context=True)
                    else:
                        filename = f"{animation_name}__it_{iteration}.gif"
                        animation = graph_visualizer.visualize_trajectory(traj_data,
                                                                          ground_truth_trajectory=ground_truth_traj)

                    if save_animation:
                        graph_visualizer.save_animation(animation, save_path, filename)

                    task_dict[animation_name] = animation

                # delete to_visualize key as this is not needed anymore and contains large data
                del task_dict["to_visualize"]


class VisualizationLogger(AbstractLogger):
    """
    Creates checkpoints of the algorithm at a given frequency (in iterations)
    """

    def __init__(self, config: ConfigDict, algorithm: AbstractAlgorithm):
        super().__init__(config=config, algorithm=algorithm)

        env_config = config.env
        env_name = env_config.name
        specific_env_config = env_config.common
        specific_env_config = deep_update_weird_custom_config_dicts(specific_env_config,
                                                                    env_config[env_name])

        self.graph_visualizer = GraphVisualizer(visualization_config=specific_env_config.visualization)

    def log_iteration(self, recorded_values: ValueDict, iteration: int
                      ) -> None:
        """
        Calls the internal save_checkpoint() method of the algorithm with the current iteration
        Args:
            recorded_values: A dictionary of previously recorded values
            iteration: The current iteration of the algorithm
        Returns:

        """
        self._writer.info(f"Logging visualizations for iteration '{iteration}'")
        graph_visualizer = self.graph_visualizer
        visualize_trajectories(recorded_values, graph_visualizer, iteration, save_animation=False)

    def finalize(self) -> None:
        pass
