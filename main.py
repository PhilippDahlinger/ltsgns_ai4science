import os

import numpy as np
import torch
import yaml
from cw2 import experiment, cluster_work
from cw2.cw_data import cw_logging
from cw2.cw_error import ExperimentSurrender

from lts_gns.algorithms.algorithm_factory import AlgorithmFactory
from lts_gns.envs.gns_environment_factory import GNSEnvironmentFactory
from lts_gns.recording.recorder import Recorder
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.dict_util import deep_update, check_against_default
from lts_gns.util.dict_util import prefix_keys
from lts_gns.util.initialize_config import _insert_recording_structure
from lts_gns.visualizations.graph_visualizer import GraphVisualizer


class LTSGNSExperiment(experiment.AbstractIterativeExperiment):
    def initialize(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # if debug mode is enabled, the config is checked against the default params.
        # This is done to ensure that all parameters are set correctly and there is no typo in the specific exp config
        if cw_config["_debug"]:
            self._debug_initialization(cw_config)
        # config
        self.initialize_config(cw_config, rep)
        self.initialize_seed(rep)

        # infer device
        if self.config.device == "cpu":
            self._device = "cpu"
        elif self.config.device == "cuda":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            print("Unknown device: ", self.config.device)
            print("Using cpu instead...")
            self._device = "cpu"

        print("Using device: ", self._device)
        # deterministic setup for reproducibility
        if self._device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # environment
        self._env = GNSEnvironmentFactory()(self.config.env, algorithm_name=self.config.algorithm.name, device=self._device)

        # algorithm
        self.algorithm = AlgorithmFactory()(self.config.algorithm, self.env, device=self._device)

        # add max iterations to env of algorithm
        self.algorithm.env.max_iterations = cw_config["iterations"]

        # TODO: correct implementation of visualization
        # ax = self._graph_visualizer.visualize_single_graph(data, ax=None)
        # self._graph_visualizer.visualize_trajectory(task.trajectory, save_animation=True)
        # plt.show()
        self._recorder = Recorder(self.config, algorithm=self.algorithm)

    def _debug_initialization(self, cw_config):
        # disable wandb logging
        print("Debug mode enabled, disabling wandb logging...")
        cw_config["params"]["recording"]["wandb"]["enabled"] = False
        # load debug config
        debug_configs = list(yaml.safe_load_all(open("configs/default.yml", "r")))
        debug_config = None
        for debug_config in debug_configs:
            if debug_config.get("params") is not None:
                debug_config = debug_config["params"]
                break
        assert debug_config is not None, "No debug config found"
        # check if all parameters are set
        print("Checking config against default config")
        print("------------------------------------------------")
        same_keys = check_against_default(cw_config["params"], debug_config, allowed_exceptions=[""])
        if same_keys:
            print("All keys are set correctly")
        print("-----------------------------------------------")
        if not same_keys:
            print("Some keys are not set correctly, aborting")
            raise ExperimentSurrender

    def initialize_config(self, cw_config, rep: int):
        _insert_recording_structure(cw_config, rep)
        self._config = ConfigDict.from_python_dict(cw_config["params"])
        # try to insert the slurm id
        try:
            self.config.slurm_array_job_id = os.environ["SLURM_ARRAY_JOB_ID"]
            self.config.slurm_job_id = os.environ["SLURM_JOB_ID"]
        except KeyError:
            pass
        # freeze all editing to the config. This guarantees that the logged config is the one which is actually used
        self.config.finalize_adding()
        self.config.finalize_modifying()

    def initialize_seed(self, rep: int):
        # seed is altered depending on the current rep
        numpy_seed = self.config.random_seeds.numpy + rep
        pytorch_seed = self.config.random_seeds.pytorch + rep
        torch.manual_seed(pytorch_seed)
        np.random.seed(numpy_seed)

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        training_metrics = self.algorithm.train_step(n)

        eval_metrics = self.algorithm.eval_step(current_step=n, use_validation_data=True)

        recorded_values = deep_update(training_metrics, eval_metrics)
        scalars = self._recorder.record_iteration(iteration=n, recorded_values=recorded_values)
        return scalars

    def finalize(self, surrender: ExperimentSurrender = None, crash: bool = False):
        if self._recorder is not None:
            try:
                self._recorder.finalize()
            except Exception as e:
                print("Failed finalizing recorder: {}".format(e))

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass

    @property
    def config(self):
        return self._config

    @property
    def env(self):
        return self._env


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(LTSGNSExperiment)
    cw.run()
