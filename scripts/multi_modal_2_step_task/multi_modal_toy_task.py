import os

import numpy as np
import torch
from gmm_util.gmm import GMM
from matplotlib import pyplot as plt


def build_trajectory(start_gmm, goal_gmm):
    start = start_gmm.sample(1).numpy()
    goal = goal_gmm.sample(1).numpy()
    return {"pos": [start, goal], "goal_pos": goal.squeeze()}


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    prec = 300 * torch.stack([torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2)])
    goal_gmm = GMM(log_w=torch.log(torch.tensor([0.25, 0.25, 0.25, 0.25])),
                   mean=torch.tensor([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]]),
                   prec=prec)
    start_gmm = GMM(log_w=torch.tensor([0.0]), mean=torch.tensor([[0.0, 0.0]]), prec=300 * torch.eye(2).unsqueeze(0))
    num_trajectories = 1000

    all_trajectories = []

    for i in range(num_trajectories):

        all_trajectories.append(build_trajectory(start_gmm, goal_gmm, ))
        if i % 100 == 0:
            print(all_trajectories[-1])

    train_tasks = all_trajectories[:700]
    val_tasks = all_trajectories[700:850]
    test_tasks = all_trajectories[850:]

    root_path = "/home/philipp/projects/datasets/lts_gns/multi_modal_2_step_task"
    os.makedirs(root_path, exist_ok=True)
    np.save(os.path.join(root_path, "multi_modal_2_step_task_train.npy"), train_tasks)
    np.save(os.path.join(root_path, "multi_modal_2_step_task_val.npy"), val_tasks)
    np.save(os.path.join(root_path, "multi_modal_2_step_task_test.npy"), test_tasks)
    print("Saved Data")


if __name__ == "__main__":
    main()
