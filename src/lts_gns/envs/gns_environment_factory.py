from lts_gns.envs.abstract_gns_environment import AbstractGNSEnvironment
from lts_gns.envs.multi_decision_toy_task.multi_decision_toy_task_processor import MultiDecisionToyTaskProcessor
from lts_gns.envs.multi_modal_2_step_task.multi_modal_2_step_task_processor import MultiModal2StepTaskProcessor
from lts_gns.envs.multi_modal_toy_task.multi_modal_toy_task_processor import MultiModalToyTaskProcessor
from lts_gns.envs.graph_updater import GraphUpdater
from lts_gns.envs.pybullet_envs.pybullet_processor import PyBulletDataLoaderProcessor
from lts_gns.envs.sofa_envs.sofa_processor import SofaDataLoaderProcessor
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.dict_util import deep_update_weird_custom_config_dicts


class GNSEnvironmentFactory:
    def __call__(self, *args, **kwargs):
        return self.build_environment(*args, **kwargs)

    @staticmethod
    def build_environment(env_config: ConfigDict, algorithm_name: str, device: str) -> AbstractGNSEnvironment:

        env_name = env_config.name
        specific_env_config = env_config.common
        specific_env_config = deep_update_weird_custom_config_dicts(specific_env_config,
                                                                    env_config[env_name])

        match env_name:
            case "deformable_plate":
                data_loader_processor = SofaDataLoaderProcessor
            case "tissue_manipulation":
                data_loader_processor = SofaDataLoaderProcessor
            case "cavity_grasping":
                data_loader_processor = SofaDataLoaderProcessor
            case "multi_modal_toy_task":
                data_loader_processor = MultiModalToyTaskProcessor
            case "multi_modal_2_step_task":
                data_loader_processor = MultiModal2StepTaskProcessor
            case "multi_decision_toy_task":
                data_loader_processor = MultiDecisionToyTaskProcessor
            case "pybullet_uniform":
                data_loader_processor = PyBulletDataLoaderProcessor
            case "pybullet_square_cloth":
                data_loader_processor = PyBulletDataLoaderProcessor
            case _:
                raise ValueError(f"Environment {env_name} unknown.")
        data_loader_processor = data_loader_processor(env_config=specific_env_config, task_name=env_name)

        graph_updater = GraphUpdater(specific_env_config, device=device)

        match algorithm_name:
            case "lts_gns":
                from lts_gns.envs.lts_gns_environment import LTSGNSEnvironment
                environment = LTSGNSEnvironment(env_config=specific_env_config,
                                                data_loader_processor=data_loader_processor,
                                                graph_updater=graph_updater,
                                                device=device)
            case "mgn" | "mgn_prodmp":
                from lts_gns.envs.mgn_environment import MGNEnvironment
                environment = MGNEnvironment(env_config=specific_env_config,
                                             data_loader_processor=data_loader_processor,
                                             graph_updater=graph_updater,
                                             device=device)
            case "cnp":
                from lts_gns.envs.cnp_environment import CNPEnvironment
                environment = CNPEnvironment(env_config=specific_env_config,
                                             data_loader_processor=data_loader_processor,
                                             graph_updater=graph_updater,
                                             device=device)
            case "lts_gns_prodmp":
                from lts_gns.envs.prodmp_environment import ProDMPEnvironment
                environment = ProDMPEnvironment(env_config=specific_env_config,
                                                data_loader_processor=data_loader_processor,
                                                device=device)
            case _:
                raise ValueError(f"Unknown algorithm {algorithm_name}")
        return environment
