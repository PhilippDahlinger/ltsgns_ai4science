import os

from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import *
from lts_gns.recording.loggers.abstract_logger import AbstractLogger
from lts_gns.algorithms.abstract_algorithm import AbstractAlgorithm


class CheckpointLogger(AbstractLogger):
    """
    Creates checkpoints of the algorithm at a given frequency (in iterations)
    """

    def __init__(self, config: ConfigDict, algorithm: AbstractAlgorithm):
        super().__init__(config=config, algorithm=algorithm)
        self._is_initial_save = True
        self.checkpoint_directory = os.path.join(self._recording_directory, "checkpoints")
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        self._checkpoint_frequency: int = config.recording.checkpoint_frequency

    def log_iteration(self, recorded_values: ValueDict, iteration: int) -> None:
        """
        Calls the internal save_checkpoint() method of the algorithm with the current iteration
        Args:
            recorded_values: A dictionary of previously recorded values
            iteration: The current iteration of the algorithm
        Returns:

        """
        if self._checkpoint_frequency > 0 and iteration % self._checkpoint_frequency == 0:
            self._writer.info(msg="Checkpointing algorithm")
            self._algorithm.save_checkpoint(directory=self.checkpoint_directory,
                                            iteration=iteration,
                                            is_initial_save=self._is_initial_save,
                                            is_final_save=False)
            self._is_initial_save = False

        if self._algorithm.save_checkpoint_this_epoch:
            self._writer.info(msg=f"Saving best validation checkpoint, current iteration: {iteration}")
            self._algorithm.save_checkpoint(directory=self.checkpoint_directory,
                                            iteration="best_validation",
                                            is_initial_save=False,
                                            is_final_save=False)
            self._algorithm.save_checkpoint_this_epoch = False

    def finalize(self) -> None:
        """
        Makes a final checkpoint of the algorithm
        Returns:

        """
        self._algorithm.save_checkpoint(directory=self.checkpoint_directory, iteration=-1,
                                        is_initial_save=self._is_initial_save,
                                        is_final_save=True)
