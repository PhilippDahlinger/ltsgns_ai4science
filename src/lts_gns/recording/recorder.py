from lts_gns.util.own_types import *

from lts_gns.algorithms.abstract_algorithm import AbstractAlgorithm
from lts_gns.recording.loggers.abstract_logger import AbstractLogger
from lts_gns.recording.loggers.logger_util.get_logging_writer import get_logging_writer
from lts_gns.recording.get_loggers import get_loggers
from lts_gns.util import keys as Keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict


class Recorder:
    """
    Records the algorithm whenever called by computing common recording values and then delegating the
    recording itself to different recorders.
    """

    def __init__(self, config: ConfigDict,
                 algorithm: AbstractAlgorithm):
        """
        Initialize the recorder, which itself is an assortment of loggers
        Args:
            config:
            algorithm:
        """
        self._loggers: List[AbstractLogger] = get_loggers(config=config, algorithm=algorithm)
        self._algorithm = algorithm
        self._writer = get_logging_writer(self.__class__.__name__,
                                          recording_directory=config._recording_structure._recording_dir)

    def record_iteration(self, iteration: int, recorded_values: ValueDict) -> ValueDict:
        self._writer.info("Recording iteration {}".format(iteration))

        for logger in self._loggers:
            try:
                logger.log_iteration(recorded_values=recorded_values,
                                     iteration=iteration)
            except Exception as e:
                self._writer.error("Error with logger '{}': {}".format(logger.__class__.__name__, repr(e)))
        self._writer.info("Finished recording iteration {}\n".format(iteration))
        scalars = recorded_values.get(Keys.SCALARS, {})
        return scalars

    def finalize(self) -> None:
        """
        Finalizes the recording by asking all loggers to finalize. The loggers may individually save things to disk or
        postprocess results.
        Returns:

        """
        self._writer.info("Finished experiment! Finalizing recording")
        for logger in self._loggers:
            logger.finalize()
            logger.remove_writer()

        self._writer.info("Finalized recording.")
        self._writer.handlers = []
        del self._writer
