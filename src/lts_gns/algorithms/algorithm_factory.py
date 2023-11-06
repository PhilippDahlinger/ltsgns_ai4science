from lts_gns.algorithms.simulators.ltsgns_prodmp_simulator import LTSGNSProDMPSimulator
from lts_gns.envs.abstract_gns_environment import AbstractGNSEnvironment
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.dict_util import deep_update_weird_custom_config_dicts


class AlgorithmFactory:
    def __call__(self, *args, **kwargs):
        return self.build_algorithm(*args, **kwargs)

    @staticmethod
    def build_algorithm(algorithm_config: ConfigDict, env: AbstractGNSEnvironment, device: str):
        algorithm_name = algorithm_config.name
        specific_algorithm_config = algorithm_config.common

        # calculate an example batch to get the sizes of the network inputs and outputs
        example_batch = env.get_next_train_task_batch(add_training_noise=False)[0]

        match algorithm_name:
            case "lts_gns":
                from lts_gns.algorithms.lts_gns import LTSGNS
                from lts_gns.algorithms.simulators.ltsgns_simulator import LTSGNSSimulator
                algorithm_class = LTSGNS
                specific_algorithm_config = deep_update_weird_custom_config_dicts(specific_algorithm_config,
                                                                                  algorithm_config.lts_gns)
                simulator_config = specific_algorithm_config.simulator

                # get the latent dimension of the posterior learner, this is needed as input to the decoder
                if specific_algorithm_config.posterior_learner.name == "task_properties_learner":
                    d_z = example_batch["task_properties"].shape[-1]
                else:
                    d_z = specific_algorithm_config.posterior_learner[
                        specific_algorithm_config.posterior_learner.name].d_z
                simulator = LTSGNSSimulator(simulator_config,
                                            example_input_batch=example_batch,
                                            d_z=d_z,
                                            env=env,
                                            device=device)
            case "mgn":
                from lts_gns.algorithms.mgn import MGN
                from lts_gns.algorithms.simulators.mgn_simulator import MGNSimulator
                algorithm_class = MGN
                specific_algorithm_config = deep_update_weird_custom_config_dicts(specific_algorithm_config,
                                                                                  algorithm_config.mgn)
                simulator_config = specific_algorithm_config.simulator
                simulator = MGNSimulator(simulator_config,
                                         example_input_batch=example_batch,
                                         env=env,
                                         device=device)
            case "mgn_prodmp":
                from lts_gns.algorithms.mgn_prodmp import MGNProDMP
                from lts_gns.algorithms.simulators.mgn_prodmp_simulator import MGNProDMPSimulator
                algorithm_class = MGNProDMP
                specific_algorithm_config = deep_update_weird_custom_config_dicts(specific_algorithm_config,
                                                                                  algorithm_config.mgn)
                simulator_config = specific_algorithm_config.simulator
                simulator = MGNProDMPSimulator(simulator_config,
                                               example_input_batch=example_batch,
                                               env=env,
                                               device=device)
            case "cnp":
                from lts_gns.algorithms.cnp import CNP
                from lts_gns.algorithms.simulators.cnp_simulator import CNPSimulator
                algorithm_class = CNP
                specific_algorithm_config = deep_update_weird_custom_config_dicts(specific_algorithm_config,
                                                                                  algorithm_config.cnp)
                simulator_config = specific_algorithm_config.simulator
                d_r = specific_algorithm_config.encoder.d_r
                simulator = CNPSimulator(simulator_config, example_input_batch=example_batch, d_r=d_r,
                                         env=env, device=device)
            case "lts_gns_prodmp":
                from lts_gns.algorithms.lts_gns_prodmp import LTSGNS_ProDMP
                algorithm_class = LTSGNS_ProDMP
                specific_algorithm_config = deep_update_weird_custom_config_dicts(specific_algorithm_config,
                                                                                  algorithm_config.lts_gns_prodmp)

                simulator_config = specific_algorithm_config.simulator
                # get the latent dimension of the posterior learner, this is needed as input to the decoder
                if specific_algorithm_config.posterior_learner.name == "task_properties_learner":
                    d_z = example_batch["task_properties"].shape[-1]
                else:
                    d_z = specific_algorithm_config.posterior_learner[
                        specific_algorithm_config.posterior_learner.name].d_z

                simulator = LTSGNSProDMPSimulator(simulator_config, example_input_batch=example_batch,
                                                  d_z=d_z,
                                                  env=env, device=device)
            case _:
                raise ValueError(f"Unknown algorithm {algorithm_name}")

        algorithm = algorithm_class(specific_algorithm_config, simulator, env, device)

        return algorithm
