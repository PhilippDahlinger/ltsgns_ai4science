import os
import pickle

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lts_gns.util.own_types import ValueDict


def visualize(mesh: np.array, edges: np.array, faces: np.array, stick: np.array, plot_edges=False,):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Generate random data for demonstration


    # Create the figure and 3D axes
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()

    for t in range(0, len(mesh), 1):
        ax.clear()
        # plot mesh:
        pos = mesh[t]
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        # Plot the scatter mesh points
        ax.scatter(x, y, z, c='b', marker='o', s=2)
        ax.add_collection3d(Poly3DCollection(pos[faces]))
        # Plot the lines connecting selected points
        if plot_edges:
            line_indices = edges
            for idx1, idx2 in line_indices:
                ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], [z[idx1], z[idx2]], c='black')
        # plot collider
        stick_pos = stick[t]
        x, y, z = stick_pos[:, 0], stick_pos[:, 1], stick_pos[:, 2]
        # Plot the scatter mesh points
        ax.scatter(x, y, z, c='r', marker='o', s=5)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Deformable Mesh Visualization for t={t}')
        # ax.set_ylim3d(0.3, 1.0)
        # ax.set_xlim3d(-0, 1.0)
        # Show the plot
        plt.pause(0.1)



if __name__ == "__main__":
    root_path = "../datasets/lts_gns/pybullet_envs/deformable_dataset_3"

    # with open(os.path.join(root_path, "deformable_dataset_16-05-2023_10-41-24.pkl"), "rb") as f:
    #     dataset = pickle.load(f)

    mesh = np.load(os.path.join(root_path, "deformable_mesh.npy"), allow_pickle=True)
    faces = np.load(os.path.join(root_path, "deformable_faces.npy"), allow_pickle=True)
    edges = np.load(os.path.join(root_path, "deformable_edges.npy"), allow_pickle=True)
    collider = np.load(os.path.join(root_path, "collider.npy"), allow_pickle=True)
    stick_mesh = np.load(os.path.join(root_path, "stick_mesh.npy"), allow_pickle=True)
    start_pos_stick_xz = np.load(os.path.join(root_path, "start_pos_stick_xz.npy"), allow_pickle=True)
    visualize(mesh, edges, faces, stick_mesh, plot_edges=False)



