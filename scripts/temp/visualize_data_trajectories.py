# load default yaml where all tasks are defined
import yaml

from lts_gns.envs.gns_environment_factory import GNSEnvironmentFactory
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.dict_util import deep_update_weird_custom_config_dicts
from lts_gns.visualizations.graph_visualizer import GraphVisualizer

default_yaml = list(yaml.safe_load_all(open("configs/default.yml", "r")))
# get th env config
config = default_yaml[1]["params"]
env_config = config["env"]
# change the task name
env_name = "cavity_grasping"
env_config["name"] = env_name
env_config["common"]["debug"]["max_tasks_per_split"] = 5
# env_config["common"]["visualization"]["num_frames"] = -1
config_dict = ConfigDict.from_python_dict(config)
env_config = config_dict.env



env = GNSEnvironmentFactory()(env_config, algorithm_name="lts_gns", device="cpu")

specific_env_config = deep_update_weird_custom_config_dicts(env_config.common,
                                                            env_config[env_name])
train_data = env.task_dict[keys.TRAIN]

visualizer = GraphVisualizer(specific_env_config["visualization"])
for task in train_data:
    fig = visualizer.visualize_trajectory(task.trajectory, context=True)
    fig.show()
