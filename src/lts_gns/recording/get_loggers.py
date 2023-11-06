from typing import List

from lts_gns.algorithms.abstract_algorithm import AbstractAlgorithm
from lts_gns.recording.loggers.abstract_logger import AbstractLogger
from lts_gns.util.config_dict import ConfigDict


def get_loggers(config: ConfigDict,
                algorithm: AbstractAlgorithm) -> List[AbstractLogger]:
    """
    Create a list of all loggers used for the current run. The order of the loggers may matter, since loggers can pass
    computed values to subsequent ones.
    Args:
        config: A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
            used by cw2 for the current run.
        algorithm: An instance of the algorithm to run.

    Returns: A list of loggers to use.

    """
    recording_dict = config.recording


    logger_classes = []

    # The order matters here, as scalars and visualizations logger write something to the logging dict which
    # the wandb logger than sends to wandb
    if recording_dict.config:
        from lts_gns.recording.loggers.config_logger import ConfigLogger
        logger_classes.append(ConfigLogger)
    if recording_dict.scalars:
        from lts_gns.recording.loggers.scalars_logger import ScalarsLogger
        logger_classes.append(ScalarsLogger)
    if recording_dict.visualizations:
        from lts_gns.recording.loggers.visualization_logger import VisualizationLogger
        logger_classes.append(VisualizationLogger)
    if recording_dict.wandb.enabled:
        from lts_gns.recording.loggers.custom_wandb_logger import CustomWAndBLogger
        logger_classes.append(CustomWAndBLogger)
    if recording_dict.checkpoint:
        from lts_gns.recording.loggers.checkpoint_logger import CheckpointLogger
        logger_classes.append(CheckpointLogger)


    loggers = [logger(config=config, algorithm=algorithm)
               for logger in logger_classes]
    return loggers
