from abc import ABC, abstractmethod

from lts_gns.algorithms.abstract_algorithm import AbstractAlgorithm
from lts_gns.recording.loggers.logger_util.get_logging_writer import get_logging_writer
from lts_gns.recording.loggers.logger_util.logger_util import process_logger_name
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import *


class AbstractLogger(ABC):
    def __init__(self, config: ConfigDict, algorithm: AbstractAlgorithm):
        self._config = config
        self._algorithm = algorithm
        self._recording_directory: str = config._recording_structure._recording_dir
        self._writer = get_logging_writer(writer_name=self.processed_name,
                                          recording_directory=self._recording_directory)

    @abstractmethod
    def log_iteration(self, recorded_values: ValueDict,
                      iteration: int) -> None:
        """
        Log the current training iteration of the algorithm instance.
        Args:
            recorded_values: Metrics and other information that was computed by previous loggers
            iteration: The current algorithm iteration. Is provided for internal consistency, since we may not want to
              record every algorithm iteration

        Returns:

        """
        raise NotImplementedError

    def finalize(self) -> None:
        """
        Finalizes the recording, e.g., by saving certain things to disk or by postpressing the results in one way or
        another.
        Returns:

        """
        raise NotImplementedError

    def remove_writer(self) -> None:
        self._writer.handlers = []
        del self._writer

    @property
    def processed_name(self) -> str:
        return process_logger_name(self.__class__.__name__)
