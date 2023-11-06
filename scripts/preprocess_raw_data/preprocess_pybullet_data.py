import os
import pickle
from tqdm import tqdm

import numpy as np

dataset_folder = "../datasets/lts_gns"

name = "pybullet_uniform_1_raw"
file_name = "dataset1000uniform"
file_path = os.path.join(dataset_folder, name, file_name)
output_path = os.path.join(dataset_folder, "pybullet_uniform_1")

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
for i in tqdm(range(1000), desc="Processing data"):
    graph_dict = {}
    for file in raw_data:
        graph_dict[file] = raw_data[file][i]
    graph_dict["stick_edges"] = stick_edges
    graph_dict["stick_faces"] = stick_faces
    if i < 700:
        train_data.append(graph_dict)
    elif i < 850:
        val_data.append(graph_dict)
    else:
        test_data.append(graph_dict)

with open(os.path.join(output_path, f"pybullet_uniform_1_train.pkl"), "wb") as f:
    pickle.dump(train_data, f)
with open(os.path.join(output_path, f"pybullet_uniform_1_val.pkl"), "wb") as f:
    pickle.dump(val_data, f)
with open(os.path.join(output_path, f"pybullet_uniform_1_test.pkl"), "wb") as f:
    pickle.dump(test_data, f)




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