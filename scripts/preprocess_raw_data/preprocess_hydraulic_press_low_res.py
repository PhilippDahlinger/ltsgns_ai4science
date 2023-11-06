import os
import pickle
from tqdm import tqdm
import meshio
import numpy as np
import json


def plot_mesh(mesh_nodes, mesh_edges, collider_nodes, collider_edges, mesh_temperature):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # plot mesh nodes with colorscale of temperature and plot a colorbar
    ax.scatter(mesh_nodes[:, 0], mesh_nodes[:, 1], mesh_nodes[:, 2], c=mesh_temperature)
    plt.colorbar(ax.scatter(mesh_nodes[:, 0], mesh_nodes[:, 1], mesh_nodes[:, 2], c=mesh_temperature))

    ax.scatter(collider_nodes[:, 0], collider_nodes[:, 1], collider_nodes[:, 2], c="g")
    # plot edges
    # for edge in mesh_edges:
    #     ax.plot(mesh_nodes[edge, 0], mesh_nodes[edge, 1], mesh_nodes[edge, 2], c="b")
    # for edge in collider_edges:
    #     ax.plot(collider_nodes[edge, 0], collider_nodes[edge, 1], collider_nodes[edge, 2], c="b")


    plt.show()



dataset_folder = "../datasets/lts_gns"

file_name = "hydraulic_press_low_res_raw"
file_path = os.path.join(dataset_folder, file_name)
output_path = os.path.join(dataset_folder, "hydraulic_press_low_res")

all_sims = []

for sim_folder in sorted(os.listdir(file_path)):
    print(sim_folder)
    mesh_file = "PLY1"
    lower_collider_file = "U-1"
    upper_collider_file = "S_01-1"
    num_time_steps = 21
    mesh_nodes = []
    mesh_edges = []
    mesh_faces = []
    mesh_temperature = []
    collider_nodes = []
    collider_edges = []
    collider_faces = []
    for t in range(num_time_steps):
        vtk_file = os.path.join(file_path, sim_folder, mesh_file, f"{mesh_file}-{t}.vtk")
        mesh = meshio.read(vtk_file)

        vtk_file = os.path.join(file_path, sim_folder, lower_collider_file, f"{lower_collider_file}-{t}.vtk")
        lower_collider = meshio.read(vtk_file)

        vtk_file = os.path.join(file_path, sim_folder, upper_collider_file, f"{upper_collider_file}-{t}.vtk")
        upper_collider = meshio.read(vtk_file)

        # MESH
        # normalized node positions
        mesh_nodes.append(mesh.points / 140)
        mesh_faces.append(mesh.cells_dict["triangle"])
        # compute the edges from the faces
        current_mesh_edges = []
        for face in mesh_faces[-1]:
            for i in range(3):
                current_mesh_edges.append(np.array([face[i], face[(i+1)%3]]))
        # remove duplicates
        current_mesh_edges = np.unique(np.array(current_mesh_edges), axis=0)
        mesh_edges.append(current_mesh_edges)
        # temperature
        mesh_temperature.append(mesh.point_data["NT11"])

        # COLLIDER
        collider_nodes.append(np.concatenate([lower_collider.points, upper_collider.points]) / 140)
        upper_faces = upper_collider.cells_dict["triangle"] + len(lower_collider.points)
        collider_faces.append(np.concatenate([lower_collider.cells_dict["triangle"], upper_faces]))
        # compute the edges from the faces
        current_collider_edges = []
        for face in collider_faces[-1]:
            for i in range(3):
                current_collider_edges.append(np.array([face[i], face[(i+1)%3]]))
        # remove duplicates
        current_collider_edges = np.unique(np.array(current_collider_edges), axis=0)
        collider_edges.append(current_collider_edges)

        plot_mesh(mesh_nodes[-1], mesh_edges[-1], collider_nodes[-1], collider_edges[-1], mesh_temperature[-1])

    # get the parameters
    # load json file "Parameter.json"
    json_file = os.path.join(file_path, sim_folder, "Parameter.json")
    with open(json_file) as f:
        parameters = json.load(f)
    current_sim = {
        "mesh_nodes": np.array(mesh_nodes),
        "mesh_edges": np.array(mesh_edges),
        "mesh_faces": np.array(mesh_faces),
        "mesh_temperature": np.array(mesh_temperature),
        "collider_nodes": np.array(collider_nodes),
        "collider_edges": np.array(collider_edges),
        "collider_faces": np.array(collider_faces),
        "thickness": parameters["Laminat_Thickness"],
        "orientation": parameters["Laminat_Orientation"],
        "temp_ply": parameters["TPly"],
        "temp_collider": parameters["TTool"],
    }
    all_sims.append(current_sim)


train_data = all_sims[:700]
val_data = all_sims[700:850]
test_data = all_sims[850:]

# with open(os.path.join(output_path, f"hydraulic_press_low_res_train.pkl"), "wb") as f:
#     pickle.dump(train_data, f)
# with open(os.path.join(output_path, f"hydraulic_press_low_res_val.pkl"), "wb") as f:
#     pickle.dump(val_data, f)
# with open(os.path.join(output_path, f"hydraulic_press_low_res_test.pkl"), "wb") as f:
#     pickle.dump(test_data, f)




