import pickle

import numpy as np
import torch
from hmpn.common.hmpn_util import make_batch
from matplotlib import pyplot as plt
from tqdm import tqdm

from playground.concave_meshing import ball_pivoting_reconstruction
from playground.pc2mesh import get_assignment, ifp
from scripts.supervised_pc2mesh.supervised_pc2mesh_train import SupervisedGNN, build_graph_from_pc


def convert_to_pc2mesh_dataset(dataset, split, total_nodes=81):
    use_distance_model = False
    tasks = []
    for trajectory in dataset:
        for pc in trajectory["pcd_points"]:
            data = build_graph_from_pc(torch.tensor(pc, dtype=torch.float32), None, None, include_mesh=False)
            batch = make_batch(data)
            break
        break
    if use_distance_model:
        model = SupervisedGNN(batch, latent_dimension=64, output_dimension=1)
        weights = torch.load("output/checkpoints/final_models/kluster_distances_deep_and_thick_2655.pt")
        model.load_state_dict(weights)
        model.eval()
        model.to("cuda")
    else:
        model = SupervisedGNN(batch, latent_dimension=64, output_dimension=2)
        weights = torch.load("output/checkpoints/final_models/kluster_displacement_deep_and_thick_2580.pt")
        model.load_state_dict(weights)
        model.eval()
        model.to("cuda")
    for trajectory in tqdm(dataset):
        nodes_grid = []
        edge_index_grid = []
        triangles_grid = []
        for i, pc in enumerate(trajectory["pcd_points"]):
            pc = torch.tensor(pc, dtype=torch.float32)
            data = build_graph_from_pc(pc, None, None, include_mesh=False)
            data = data.to("cuda")
            with torch.no_grad():
                output = model(data)

            if use_distance_model:
                raise NotImplementedError("Currently not supported in this version of the script")
                pc = pc.cpu().numpy()
                distances = output.detach().cpu().numpy().reshape(-1)
                old_nodes = subsample_nodes(pc, distances, prev_nodes, total_nodes=total_nodes, use_ifp=False)
            else:
                pc = pc.cpu().numpy() - output.detach().cpu().numpy()
                distances = np.linalg.norm(output.cpu().numpy(), axis=1)
                old_nodes = subsample_nodes(pc, distances, None, total_nodes=total_nodes, use_ifp=True)

            # create edges
            tri = ball_pivoting_reconstruction(old_nodes, radii=[0.1, 0.3])
            nodes = np.asarray(tri.vertices[:, :2])
            assert nodes.shape == (total_nodes, 2)
            if len(nodes) != len(old_nodes):
                print("stop")
            edges = np.asarray(tri.edges)
            faces = np.asarray(tri.faces)
            nodes_grid.append(np.asarray(nodes))
            edge_index_grid.append(np.asarray(edges))
            triangles_grid.append(np.asarray(faces))
            # # plot
            # plt.ion()
            # plt.cla()
            # plt.scatter(pc[:, 0], pc[:, 1], c=distances)
            # plt.scatter(nodes[:, 0], nodes[:, 1], c="red")
            # plt.triplot(nodes[:, 0], nodes[:, 1], faces)
            # plt.show()
            # plt.pause(0.01)
        new_trajectory = {
            "nodes_grid": nodes_grid,
            "edge_index_grid": edge_index_grid,
            "triangles_grid": triangles_grid,
            "nodes_collider": trajectory["nodes_collider"],
            "triangles_collider": trajectory["triangles_collider"],
            "edge_index_collider": trajectory["edge_index_collider"],
            "poisson_ratio": trajectory["poisson_ratio"],

        }
        tasks.append(new_trajectory)
    # save
    with open(f"../datasets/lts_gns/deformable_plate/pc2mesh_{split}.pkl", "wb") as f:
        pickle.dump(tasks, f)


def subsample_nodes(pc, distances, prev_nodes, sigma=0.1, total_nodes=81, use_ifp=False):
    plt.ion()
    nodes = []
    # fig = plt.figure()
    # plt.gca()
    # ax = fig.add_subplot(111)
    # decrease distances around prev nodes

    if use_ifp:
        return ifp(pc, init_convex_hull=False, target_points=total_nodes)[0]
    else:
        if prev_nodes is not None:
            for prev_node in prev_nodes:
                distances = distances - 0.01 * np.exp(-np.linalg.norm(pc - prev_node, axis=1) ** 2 / sigma ** 2)
        while len(nodes) < total_nodes:
            # find the node with the lowest distance
            min_index = np.argmin(distances)
            nodes.append(pc[min_index])
            # update distances with a gaussian kernel
            # distances = distances + distances[min_index] * np.exp(-np.linalg.norm(pc - pc[min_index], axis=1) ** 2 / sigma ** 2)
            distances = distances + 0.1 * np.exp(-np.linalg.norm(pc - pc[min_index], axis=1) ** 2 / sigma ** 2)
            # 3d matplotlib plot
            # ax.cla()
            # ax.scatter(pc[:, 0], pc[:, 1], c=distances)
            # ax.scatter(nodes[-1][0], nodes[-1][1], c="red")
            # plt.show()
            # plt.pause(.1)
        return np.array(nodes)


def open_dataset():
    rollouts = []
    for split in ["train", "val", "test"]:
        with open(f"../datasets/lts_gns/deformable_plate/deformable_plate_{split}.pkl", "rb") as file:
            rollouts.append(pickle.load(file))
    return rollouts[0], rollouts[1], rollouts[2]


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = open_dataset()
    convert_to_pc2mesh_dataset(train_dataset, "train")
    convert_to_pc2mesh_dataset(val_dataset, "val")
    convert_to_pc2mesh_dataset(test_dataset, "test")
