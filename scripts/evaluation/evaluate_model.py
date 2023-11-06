import os
import sys
from pathlib import Path

import torch
import yaml

from lts_gns.algorithms.algorithm_factory import AlgorithmFactory
from lts_gns.envs.gns_environment_factory import GNSEnvironmentFactory
from lts_gns.recording.loggers.visualization_logger import visualize_trajectories
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.dict_util import deep_update_weird_custom_config_dicts
from lts_gns.visualizations.graph_visualizer import GraphVisualizer

# path hacking for scripts from top level/level this file is called from
current_directory = os.getcwd()
sys.path.append(current_directory)


class CustomPrinter:
    def __init__(self, verbosity: int):
        self._verbosity = verbosity

    def print(self, string: str, threshold: int = 0):
        if threshold > self._verbosity:
            print("  " * threshold + string)

    def __call__(self, string: str, threshold: int = 0):
        self.print(string, threshold)


class Evaluator:

    def __init__(self, root_path: Path | str | None = Path("output/cw2_data"),
                 evaluation_config: Path | str | ConfigDict | None = None,
                 checkpoint_iteration: int | str | None = None,
                 eval_name: str = "default",
                 first_rep_only: bool = False,
                 verbosity: int = 0):
        """
        Evaluates the results of an experiment. The experiment is assumed to be structured as follows:
        root_path
        ├── experiment
        │   ├── experiment_sub1
        │   │   ├── log
        │   │   │   ├── rep_00
        │   │   │   │   ├── checkpoints
        │   │   │   │   │   ├── checkpoint.ckpt
        │   │   │   │   ├── config.yaml
        │   │   │   │   ├── ...
        │   │   │   ├── rep_01
        │   │   │   │   ├── checkpoints
        │   │   │   │   │   ├── checkpoint.ckpt
        │   │   │   │   ├── config.yaml
        │   │   │   │   ├── ...
        │   │   │   ├── ...
        │   ├── experiment_sub2
        │   │   ├── log
        │   │   │   ├── ...
        │   │   ├── ...
        │   ├── ...

        Args:
            root_path:
            evaluation_config: [Optional] Path to an evaluation config file or a ConfigDict containing the evaluation
                config. If provided, will overwrite the "evaluation" section of the config.yaml file in the root_path,
                allowing for different and custom evaluation configs for the same experiment.
            checkpoint_iteration: [Optional] Iteration to evaluate. If not provided, the last iteration is evaluated
            eval_name: [Optional] Name of the evaluation. Used to create a folder in the root_path to store the results.
            verbosity: Verbosity level. 0: no output, 1: minimal output, 2: more output, 3: full output
        """
        if root_path is None:
            root_path = Path("output/cw2_data")
        elif isinstance(root_path, str):
            root_path = Path(root_path)
        self._root_path = root_path

        if isinstance(evaluation_config, str):
            evaluation_config = Path(evaluation_config)
        if isinstance(evaluation_config, Path):
            assert evaluation_config.exists(), f"Path '{evaluation_config}' not found."
            with open(evaluation_config) as file:
                evaluation_config = yaml.safe_load(file)
        if evaluation_config is not None:
            self._evaluation_config = ConfigDict.from_python_dict(evaluation_config)
        else:
            self._evaluation_config = None

        self._eval_name = eval_name
        self._first_rep_only = first_rep_only
        self._checkpoint_iteration = int(checkpoint_iteration) if checkpoint_iteration is not None else None
        self.printer = CustomPrinter(verbosity)

    def evaluate_experiment(self, experiment_name: str):
        experiment_path = self._root_path / experiment_name  # Path overloads the / operator to join paths
        if not experiment_path.exists():
            raise FileNotFoundError(f"Path '{experiment_path}' not found.")
        subexperiment_paths = [x for x in experiment_path.iterdir() if x.is_dir()]

        if len(subexperiment_paths) == 1 and subexperiment_paths[0].name == "log":
            # if there is only one subexperiment and it is the log folder, we evaluate the whole experiment
            self._evaluate_subexperiment(experiment_path)
        else:
            self.printer(f"Experiment '{experiment_name}' contains {len(subexperiment_paths)} subexperiments.", 0)
            # multiple subexperiments, evaluate each one separately
            # todo test?
            for subexperiment_path in subexperiment_paths:
                self.printer(f"Evaluating subexperiment '{subexperiment_path.name}'...", 1)
                assert experiment_name in subexperiment_path.name, \
                    f"Experiment name '{experiment_name}' not found in subexperiment path '{subexperiment_path}'."
                self._evaluate_subexperiment(subexperiment_path)

    def _evaluate_subexperiment(self, subexperiment_path: Path):
        """
        Evalautes a single subexperiment (i.e., one list/grid entry from the experiment grid) for all repetitions.
        Args:
            subexperiment_path: Path to the subexperiment folder

        Returns:

        """
        subexperiment_path = subexperiment_path / "log"  # append log folder
        assert subexperiment_path.exists(), f"Path '{subexperiment_path}' not found."

        repetitions = sorted([x for x in subexperiment_path.iterdir() if x.is_dir()])
        if self._first_rep_only:
            for repetition in repetitions[:1]:
                self._evaluate_repetition(repetition)
        else:
            for repetition in repetitions:
                self._evaluate_repetition(repetition)


    def _evaluate_repetition(self, repetition: Path):
        """
        Evaluates a single repetition of a subexperiment (i.e., one run of the experiment with a specific seed).
        Args:
            repetition: Path to the repetition folder. Should contain a config.yaml file and a checkpoints folder.

        Returns:

        """
        config = self._get_config(repetition)
        if config["device"] == "cpu":
            device = "cpu"
        elif config["device"] == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.printer(f"Unknown device: {config['device']} for repetition {repetition}. Using CPU", 3)
            device = "cpu"

        gns_env = GNSEnvironmentFactory()(config.env, algorithm_name=config.algorithm.name, device=device)
        algorithm = AlgorithmFactory()(config.algorithm, gns_env, device=device)
        # eval on the test set!
        eval_metrics = algorithm.eval_step(use_validation_data=False, force_large_update=True)

        env_config = config.env
        specific_env_config = env_config.common
        specific_env_config = deep_update_weird_custom_config_dicts(specific_env_config,
                                                                    env_config[env_config.name])
        graph_visualizer = GraphVisualizer(visualization_config=specific_env_config.visualization)

        save_path = f"output/evaluations/{config['_recording_structure']['_job_name']}/{self._eval_name}/{repetition.name}"
        # create path if it does not exist
        Path(save_path).mkdir(parents=True, exist_ok=True)

        visualize_trajectories(recorded_values=eval_metrics, graph_visualizer=graph_visualizer,
                               iteration=self._checkpoint_iteration, save_animation=True,
                               save_path=save_path)
        scalars = eval_metrics["scalars"]
        # save dict using pickle
        import pickle
        with open(os.path.join(save_path, "scalars.pkl"), "wb") as file:
            pickle.dump(scalars, file)

    def _get_config(self, repetition):
        # load config.yaml
        with open(repetition / "config.yaml") as file:
            current_config = yaml.safe_load(file)
        current_config["algorithm"]["common"]["simulator"]["checkpoint"] = {
            "load_checkpoint": True,
            "experiment_name": repetition.parent.parent,
            "iteration": self._checkpoint_iteration,
            "repetition": repetition.name,
        }
        current_config = ConfigDict.from_python_dict(current_config)

        if isinstance(self._evaluation_config, ConfigDict):
            current_config = deep_update_weird_custom_config_dicts(current_config,
                                                                   self._evaluation_config)
        self.printer(f"Current config: {current_config}", 2)
        return current_config


def _get_args():
    import argparse
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", help="Name of the experiment config to evaluate.")
    parser.add_argument("-eval", "--evaluation_config",
                        help="Path to an evaluation config file. If provided, will overwrite the 'evaluation' section "
                             "of the config.yaml file in the root_path, allowing for different and "
                             "custom evaluation configs for the same experiment.")
    parser.add_argument("-name", "--eval_name", default="default", help="Name of the evaluation Used for saving the "
                                                                        "results. Defaults to 'default'.")
    parser.add_argument("-root", "--root_path",
                        help="Path to the root folder of the experiment. Defaults to 'output/cw2_data'.")
    parser.add_argument("-iter", "--iteration", type=int,
                        help="Iteration to evaluate. If not provided, the last iteration is evaluated")
    parser.add_argument("-v", "--verbosity", type=int, default=0,
                        help="Verbosity level. 0: no output, 1: minimal output, 2: more output, 3: full output")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _get_args()
    evaluator = Evaluator(root_path=args.root_path,
                          evaluation_config=args.evaluation_config,
                          checkpoint_iteration=args.iteration,
                          eval_name=args.eval_name)
    evaluator.evaluate_experiment(experiment_name=args.experiment_name)
