from typing import List

import torch
import yaml
from tqdm import tqdm

from lts_gns.envs.gns_environment_factory import GNSEnvironmentFactory
from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.util import node_type_mask


def main(tasks: List[str]):
    # load default yaml where all tasks are defined
    torch.set_printoptions(precision=10)
    default_yaml = list(yaml.safe_load_all(open("configs/default.yml", "r")))

    # get the env config
    env_config = default_yaml[1]["params"]["env"]

    for task_name in tasks:
        env_config["name"] = task_name
        config_dict = ConfigDict.from_python_dict(env_config)
        env = GNSEnvironmentFactory()(config_dict, algorithm_name="lts_gns", device="cpu")
        train_data = env.task_dict[keys.TRAIN]
        all_vels = []
        for task in tqdm(train_data):
            for graph in task.trajectory:
                mesh_mask = node_type_mask(graph, keys.MESH)
                mesh_pos = graph.pos[mesh_mask]
                next_mesh_pos_pos = graph.next_mesh_pos
                # TODO incorporate dt in the config
                if task_name == "multi_modal_toy_task":
                    dt = 0.1
                elif task_name == "deformable_plate":
                    dt = 1.0
                else:
                    dt = 1.0
                vel = (next_mesh_pos_pos - mesh_pos) / dt
                all_vels.append(vel)
        all_vels = torch.cat(all_vels, dim=0)
        print("All vels shape:", all_vels.shape)
        mean = all_vels.mean(dim=0)
        std = all_vels.std(dim=0)
        print(f"task: {task_name}")
        print(f"mean: {mean}")
        print(f"std: {std}")
        print("---------------")


if __name__ == '__main__':
    # change the task name if you have a new environment. Currently, we have the following tasks:
    # "multi_modal_toy_task", "deformable_plate"
    # "tissue_manipulation", "pybullet_uniform"
    tasks = ["pybullet_square_cloth"]  # ["deformable_plate", "tissue_manipulation"]
    main(tasks=tasks)
