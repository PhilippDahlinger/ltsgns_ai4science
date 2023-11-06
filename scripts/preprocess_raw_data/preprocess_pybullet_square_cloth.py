import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

dataset_folder = "../datasets/lts_gns/pybullet_envs"

# name = "pybullet_uniform_1_raw"
file_name = "deformable_dataset_29-08-2023_15-24-01-871086869"
file_path = os.path.join(dataset_folder, file_name)  # name, file_name)
save_name = "pybullet_square_cloth_convex_stick_debug"
output_path = os.path.join(dataset_folder, save_name)

raw_data = {}

for file in os.listdir(file_path):
    raw_data[os.path.splitext(file)[0]] = np.load(os.path.join(file_path, file))

stick_edges = np.array([[0, 1], [1, 4], [4, 3], [3, 0], [0, 4],
                        [1, 5], [5, 2], [2, 0], [1, 2],
                        [5, 7], [7, 6], [6, 2], [5, 6],
                        [6, 3], [4, 7], [3, 7],
                        [3, 2],
                        [4, 5]])

stick_faces = np.array([[0, 1, 4], [0, 4, 3], [1, 5, 2], [1, 2, 0], [5, 7, 6], [5, 6, 2], [7, 3, 6],
                        [7, 4, 3], [3, 2, 6], [2, 3, 0], [4, 5, 1], [4, 7, 5]])

# find out the normalizing factors for spring stiffness
spring_stiffness = raw_data["spring_elastic_stiffness"]
min_s = np.min(spring_stiffness)
max_s = np.max(spring_stiffness)
print(f"min_s: {min_s}, max_s: {max_s}")

train_data = []
val_data = []
test_data = []
for element_property in range(3):
    for i in tqdm(range(300), desc="Processing data"):
        graph_dict = {}
        for file in raw_data:
            print(f"{raw_data[file].shape=}")
            raw_data_tensor = raw_data[file][element_property]  # tensor with all the data for one element property
            if len(raw_data_tensor) == 300:
                graph_dict[file] = raw_data[file][element_property, i]
            elif len(raw_data_tensor) == 1:
                graph_dict[file] = raw_data[file][element_property, 0]
            else:
                raise ValueError(f"Unexpected shape of raw data: {raw_data_tensor.shape}")
        graph_dict["stick_edges"] = stick_edges
        graph_dict["stick_faces"] = stick_faces
        graph_dict["fixed_mesh_nodes"] = np.array([0, 372])
        if i < 220:
            train_data.append(graph_dict)
        elif i < 260:
            val_data.append(graph_dict)
        else:
            test_data.append(graph_dict)


def plot_mesh_at_timestep(nodes, edges, faces, timestep):
    # Get the nodes for the given timestep
    nodes_t = nodes[timestep]

    # Plot the nodes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.scatter(nodes_t[:, 0], nodes_t[:, 1], nodes_t[:, 2], c='b')
    print("@@@", faces.shape)
    # Plot the edges
    for edge in edges:
        ax.plot3D(*nodes_t[edge, :].T, c='k')

    # # Plot the faces
    # for face in faces:
    #     ax.add_collection3d(Poly3DCollection([nodes_t[face]], alpha=0.5))

    # Label the nodes with their indices
    for i, node in enumerate(nodes_t):
        ax.text(node[0], node[1], node[2], str(i), fontsize=9)

    plt.pause(1)

#
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for t in range(50):
    ax.clear()
    plot_mesh_at_timestep(faces=raw_data["deformable_faces"][0, 3],
                          nodes=raw_data["deformable_mesh"][0, 3],
                          edges=raw_data["deformable_edges"][0, 3],
                          timestep=t)


def save():
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving training data to {output_path}")
    with open(os.path.join(output_path, f"{save_name}_train.pkl"), "wb") as f:
        pickle.dump(train_data, f)

    print(f"Saving validation data to {output_path}")
    with open(os.path.join(output_path, f"{save_name}_val.pkl"), "wb") as f:
        pickle.dump(val_data, f)

    print(f"Saving test data to {output_path}")
    with open(os.path.join(output_path, f"{save_name}_test.pkl"), "wb") as f:
        pickle.dump(test_data, f)


# save()


def plot_stick():
    import matplotlib.pyplot as plt
    stick = raw_data["stick_mesh"][0][40]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(stick[:, 0], stick[:, 1], stick[:, 2])
    # have id at each point labeled
    for i in range(len(stick)):
        ax.text(stick[i, 0], stick[i, 1], stick[i, 2], str(i))
    # # plot edges
    # for edge in stick_edges:
    #     ax.plot(stick[edge, 0], stick[edge, 1], stick[edge, 2], color='blue')
    # plot faces as polygons
    for face in stick_faces:
        ax.plot(stick[face, 0], stick[face, 1], stick[face, 2], color='red')
    # have face id in the middle of each face
    for face in stick_faces:
        ax.text(np.mean(stick[face, 0]), np.mean(stick[face, 1]), np.mean(stick[face, 2]), str(face))
    plt.show()
