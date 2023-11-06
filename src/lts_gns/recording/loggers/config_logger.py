import datetime

from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import *
from lts_gns.recording.loggers.abstract_logger import AbstractLogger
from lts_gns.algorithms.abstract_algorithm import AbstractAlgorithm
from lts_gns.recording.loggers.logger_util.logger_util import save_to_yaml
from pprint import pformat


class ConfigLogger(AbstractLogger):
    """
    A very basic logger that prints the config file as an output at the start of the experiment. Also saves the config
    as a .yaml in the experiment's directory.
    """

    def __init__(self, config: ConfigDict, algorithm: AbstractAlgorithm):
        super().__init__(config=config, algorithm=algorithm)
        save_to_yaml(dictionary=config.get_raw_dict(), save_name=self.processed_name,
                     recording_directory=config._recording_structure._recording_dir)
        self._writer.info(f"Start time: {datetime.datetime.now()}")
        self._writer.info("\n" + pformat(object=config, indent=2))

    def log_iteration(self, recorded_values: ValueDict,
                      iteration: int) -> None:
        pass

    def finalize(self) -> None:
        pass
