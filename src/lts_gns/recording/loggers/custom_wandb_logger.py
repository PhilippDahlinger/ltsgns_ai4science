import matplotlib.pyplot as plt

from lts_gns.recording.loggers.logger_util.wandb_util import reset_wandb_env, get_job_type, wandbfy
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import *
from lts_gns.recording.loggers.abstract_logger import AbstractLogger
from lts_gns.algorithms.abstract_algorithm import AbstractAlgorithm
import wandb
from lts_gns.util import keys as Keys
import os





class CustomWAndBLogger(AbstractLogger):
    """
    Logs (some) recorded results using wandb.ai.
    """

    def __init__(self, config: ConfigDict, algorithm: AbstractAlgorithm):
        super().__init__(config=config, algorithm=algorithm)
        reset_wandb_env()

        wandb_params = config.recording.wandb
        project_name = wandb_params.get("project_name")
        environment_name = config.env.name

        if wandb_params.get("task_name") is not None:
            project_name = project_name + "_" + wandb_params.get("task_name")
        elif environment_name is not None:
            project_name = project_name + "_" + environment_name
        else:
            # no further specification of the project, just use the initial project_name
            project_name = project_name

        recording_structure = config.get("_recording_structure")
        groupname = recording_structure.get("_groupname")[-127:]
        runname = recording_structure.get("_runname")[-127:]
        recording_dir = recording_structure.get("_recording_dir")
        job_name = recording_structure.get("_job_name")

        tags = wandb_params.get("tags", [])
        if tags is None:
            tags = []
        if config.get("algorithm").get("name") is not None:
            tags.append(config.get("algorithm").get("name"))

        entity = wandb_params.get("entity")

        start_method = wandb_params.get("start_method")
        settings = wandb.Settings(start_method=start_method) if start_method is not None else None

        self.wandb_logger = wandb.init(project=project_name,  # name of the whole project
                                       tags=tags,  # tags to search the runs by. Currently, contains algorithm name
                                       job_type=job_name,  # name of your experiment
                                       group=groupname,  # group of identical hyperparameters for different seeds
                                       name=runname,  # individual repetitions
                                       dir=recording_dir,  # local directory for wandb recording
                                       config=config.get_raw_dict(),  # full file config
                                       reinit=False,
                                       entity=entity,
                                       settings=settings
                                       )
        # look at decoder weights
        wandb.watch(algorithm.simulator.decoder, log='all')
        # wandb.watch(algorithm.simulator.processor, log='all')

    def log_iteration(self, recorded_values: ValueDict, iteration: int) -> None:
        """
        Parses and logs the given dict of recorder metrics to wandb.
        Args:
            recorded_values: A dictionary of previously recorded things
            iteration: The current iteration of the algorithm
        Returns:

        """
        wandb_log_dict = {}
        # log_scalars
        if Keys.SCALARS in recorded_values:
            scalars_dict = recorded_values[Keys.SCALARS]
            # category is train, small_eval, large_eval, not so relevant
            for category_name, category_metrics in scalars_dict.items():
                for task_quantity_name, task_quantity_metrics in category_metrics.items():
                    if isinstance(task_quantity_metrics, dict):
                        for metric_name, metric_value in task_quantity_metrics.items():
                            wandb_log_dict[task_quantity_name + "/" + metric_name] = metric_value
                    else:
                        # if it is not nested, just log the metric
                        wandb_log_dict[category_name + "/" + task_quantity_name] = task_quantity_metrics
            wandb_log_dict['default/iteration'] = iteration

        # log visualizations and animations
        if Keys.VISUALIZATIONS in recorded_values:
            large_eval_vis_dict = recorded_values[Keys.VISUALIZATIONS][Keys.LARGE_EVAL]
            for task_name, task_figures in large_eval_vis_dict.items():
                for vis_name, vis_figure in task_figures.items():
                    vis_figure = wandbfy(vis_figure)
                    wandb_log_dict[f"{task_name}/{vis_name}"] = vis_figure

            if Keys.TRAIN in recorded_values[Keys.VISUALIZATIONS]:
                train_vis_dict = recorded_values[Keys.VISUALIZATIONS][Keys.TRAIN]
                for vis_name, vis_figure in train_vis_dict.items():
                    vis_figure = wandbfy(vis_figure)
                    wandb_log_dict[f"train/{vis_name}"] = vis_figure

        if wandb_log_dict:  # logging dictionary is not empty
            wandb.log(wandb_log_dict, step=iteration)

    def finalize(self) -> None:
        """
        Properly close the wandb logger
        Returns:

        """
        wandb.finish()
