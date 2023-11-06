import os

import numpy as np
from matplotlib import pyplot as plt


class Node:
    def __init__(self, inital_pos: np.array, mass: float):
        self.pos = inital_pos
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)
        self.mass = mass

    def update(self, force: np.array, dt: float):
        self.acc = force / self.mass
        self.vel += self.acc * dt
        self.pos += self.vel * dt

    def pd_control(self, goal, k_p, t_d):
        force = k_p * (goal.pos - self.pos - t_d * self.vel)
        return force


def plot(fig, axes, node: Node, goal: Node, i: float):
    axes.clear()
    axes.set_xlim(-1.5, 1.5)
    axes.set_ylim(-1.5, 1.5)
    axes.scatter(node.pos[0], node.pos[1], color="blue", marker="o", s=100)
    axes.scatter(goal.pos[0], goal.pos[1], color="red", marker="x", s=100)
    axes.set_title(f"i={i}")


def simulate_trajectory(k_p, t_d, mass=1.0, dt=0.1, visualize=True, length=50):
    goal_pos_idx = np.random.randint(0, 4, size=1)[0]
    goal_pos = [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])][goal_pos_idx]
    goal = Node(goal_pos, 1)
    node = Node(np.array([np.random.randn() / 20, np.random.randn() / 20]), mass=mass)

    trajectory = {"pos": [np.copy(node.pos)],
                  "vel": [np.copy(node.vel)],
                  "acc": [np.copy(node.acc)],
                  "mass": mass,
                  "goal_pos": goal_pos,
                  "k_p": k_p,
                  "t_d": t_d,
                  "dt": dt, }

    if visualize:
        plt.close()
        fig, axes = plt.subplots(1, 1)
    for i in range(length):
        force = node.pd_control(goal, k_p, t_d)
        node.update(force, dt)
        trajectory["pos"].append(np.copy(node.pos))
        trajectory["vel"].append(np.copy(node.vel))
        trajectory["acc"].append(np.copy(node.acc))
        if visualize:
            plot(fig, axes, node, goal, i)
            plt.pause(dt / 10)
    return trajectory


def main():
    np.random.seed(42)
    k_ps = [1.2, 1.5, 1.8]
    t_ds = [1.5, 1.8, 2.0]

    num_trajectories = 1000

    all_trajectories = []

    for i in range(num_trajectories):
        k_p = np.random.choice(k_ps)
        t_d = np.random.choice(t_ds)
        all_trajectories.append(simulate_trajectory(k_p, t_d, visualize=False, length=50))
        print(f"Trajectory {i}: Goal Pos: {all_trajectories[-1]['goal_pos']}")

    train_tasks = all_trajectories[:700]
    val_tasks = all_trajectories[700:850]
    test_tasks = all_trajectories[850:]

    root_path = "/home/philipp/projects/datasets/lts_gns/multi_modal_toy_task_direct_start"
    os.makedirs(root_path, exist_ok=True)
    np.save(os.path.join(root_path, "multi_modal_toy_task_direct_start_train.npy"), train_tasks)
    np.save(os.path.join(root_path, "multi_modal_toy_task_direct_start_val.npy"), val_tasks)
    np.save(os.path.join(root_path, "multi_modal_toy_task_direct_start_test.npy"), test_tasks)

    print("Saved Data")


if __name__ == "__main__":
    plt.ion()
    main()
