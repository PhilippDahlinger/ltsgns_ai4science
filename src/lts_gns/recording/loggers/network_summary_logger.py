from lts_gns.algorithms.abstract_algorithm import AbstractAlgorithm
from lts_gns.recording.loggers.abstract_logger import AbstractLogger
from lts_gns.util.own_types import *


class NetworkSummaryLogger(AbstractLogger):
    """
    A very basic logger that prints the config file as an output at the start of the experiment. Also saves the config
    as a .yaml in the experiment's directory.
    """

    def log_iteration(self, recorded_values: ValueDict, iteration: int) -> None:
        if iteration == 0:
            assert isinstance(self._algorithm, AbstractAlgorithm), f"Algorithm must be some GNS, " \
                                                             f"provided {type(self._algorithm)} " \
                                                             f"instead"
            self._writer.info("Started NetworkSummaryLogger.")
            # TODO: Update
            # self._writer.info(self._algorithm.policy)
            # total_network_parameters = sum(p.numel() for p in self._algorithm.policy.parameters())
            # trainable_parameters = sum(p.numel() for p in self._algorithm.policy.parameters() if p.requires_grad)
            # self._writer.info(f"Total parameters: {total_network_parameters}")
            # self._writer.info(f"Trainable parameters: {trainable_parameters}")

    def finalize(self) -> None:
        pass
