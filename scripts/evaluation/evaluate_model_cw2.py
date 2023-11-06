from cw2 import experiment, cluster_work
from cw2.cw_data import cw_logging
from cw2.cw_error import ExperimentSurrender

from scripts.evaluation.evaluate_model import Evaluator


class EvaluateModel(experiment.AbstractExperiment):
    def initialize(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        self.eval_config = cw_config["params"]["evaluation"]
        self.evaluator = Evaluator(root_path=self.eval_config["root_path"],
                              evaluation_config=cw_config["params"],
                              checkpoint_iteration=self.eval_config["iteration"],
                              eval_name=self.eval_config["eval_name"],
                              first_rep_only=self.eval_config.get("first_rep_only", False),)

    def run(self, cw_config: dict, rep: int, n: int) -> dict:
        self.evaluator.evaluate_experiment(experiment_name=self.eval_config["experiment_name"])
        return {}

    def finalize(self, surrender: ExperimentSurrender = None, crash: bool = False):
        pass

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass

if __name__ == "__main__":
    cw = cluster_work.ClusterWork(EvaluateModel)
    cw.run()
