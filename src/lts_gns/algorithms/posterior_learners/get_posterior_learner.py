from lts_gns.algorithms.posterior_learners.abstract_posterior_learner import AbstractPosteriorLearner
from lts_gns.envs.abstract_gns_environment import AbstractGNSEnvironment
from lts_gns.envs.lts_gns_environment import LTSGNSEnvironment
from lts_gns.util.config_dict import ConfigDict


def get_posterior_learner(posterior_learner_config: ConfigDict, env: LTSGNSEnvironment,
                          device: str) -> AbstractPosteriorLearner:
    name = posterior_learner_config.name
    if name == "constant_posterior_learner":
        from lts_gns.algorithms.posterior_learners.constant_posterior_learner import ConstantPosteriorLearner
        n_all_tasks = len(env.real_mesh_auxiliary_train_tasks)
        return ConstantPosteriorLearner(posterior_learner_config.constant_posterior_learner, n_all_tasks, device=device)
    elif name == "multi_daft_posterior_learner":
        from lts_gns.algorithms.posterior_learners.multi_daft_posterior_learner import MultiDaftPosteriorLearner
        if hasattr(env, "include_pc2mesh") and env.include_pc2mesh:
            n_all_train_tasks = len(env.real_mesh_auxiliary_train_tasks) + len(env.pc2mesh_auxiliary_train_tasks)
        else:
            n_all_train_tasks = len(env.real_mesh_auxiliary_train_tasks)
        n_all_eval_tasks = len(env.val_tasks)
        return MultiDaftPosteriorLearner(posterior_learner_config.multi_daft_posterior_learner,
                                         n_all_train_tasks=n_all_train_tasks,
                                         n_all_eval_tasks=n_all_eval_tasks, device=device)
    elif name == "task_properties_learner":
        from lts_gns.algorithms.posterior_learners.task_properties_learner import TaskPropertiesLearner
        return TaskPropertiesLearner(posterior_learner_config.task_properties_learner, device=device)
    else:
        raise ValueError(f"Unknown posterior learner name {name}")
