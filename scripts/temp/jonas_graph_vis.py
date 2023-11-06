import matplotlib.pyplot as plt

import src.utils.get_connectivity_setting
from src.utils import dataset_utils, graph_utils
from tqdm import tqdm
import torch
import matplotlib.animation as animation
import numpy as np

def visualize_points(pos, edge_index=None, node_type=None, edge_type=None, edge=1, color_list=None):
    plt.rcParams.update({'figure.constrained_layout.use': True})
    plt.rcParams.update({"font.family": "serif"})
    #fig, ax = plt.figure(figsize=(4, 4))
    edge_keys = [('grid', '0', 'grid'),
    ('collider', '1', 'collider'),
    ('mesh', '2', 'mesh'),
    ('collider', '3', 'grid'),
    ('mesh', '4', 'grid'),
    ('mesh', '5', 'collider'),
    ('grid', '6', 'collider'),
    ('grid', '7', 'mesh'),
    ('collider', '8', 'mesh')]
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    # if edge_index is not None:
        # index_edge=torch.where(edge_type == edge)[0]
        # print("number of edges displayed: ", len(index_edge))
        # if index_edge is not None:
        #     mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        #     mask[index_edge] = True
        #     edge_index = edge_index[:,mask]
    for (src, dst) in edge_index.t().tolist():
         src = pos[src].tolist()
         dst = pos[dst].tolist()
         plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1.0, color='black')
    index2 = torch.where(node_type == 2)[0]
    index1 = torch.where(node_type == 1)[0]
    index0 = torch.where(node_type == 0)[0]
    mask0 = torch.zeros(pos.size(0), dtype=torch.bool)
    mask1 = torch.zeros(pos.size(0), dtype=torch.bool)
    mask2 = torch.zeros(pos.size(0), dtype=torch.bool)
    mask0[index0] = True
    mask1[index1] = True
    mask2[index2] = True
    plt.scatter(pos[mask0, 0], pos[mask0, 1], s=80, c='gold', zorder=1000,
                label='Point Cloud') #color_list
    plt.scatter(pos[mask1, 0], pos[mask1, 1], s=80, color='blue', zorder=2000,
                label='Collider Points')
    plt.scatter(pos[mask2, 0], pos[mask2, 1], s=80, zorder=1000,
                label='Mesh Points', color='green')
    ax.set_box_aspect(1)
    ax.legend(fontsize=26)
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.10, 1.40))
    plt.axis('off')
    #plt.title(f'Scene with edge type: {edge_keys[edge]}')

    #plt.savefig(f'Figure_{edge_keys[edge]}.pdf')
    #plt.savefig('figure.pdf')
    plt.show()



def convert_timestep_str_to_bool(input_timestep):
    if input_timestep == "t":
        old_timestep = True
    elif input_timestep == "t+1":
        old_timestep = False
    else:
        raise ValueError(f"Timestep {input_timestep} is neither t nor t+1. Please choose one of these")
    return old_timestep


def animate_rollout(pos, edge_index=None, node_type=None, edge_type=None, color_list=None, edge=0, save_animation=False, stride=1):
    #(rollout_predicted, rollout_target, rollout_collider, fig_number=0, loss="", stride=1, save_animation=False):
    num_frames = int(len(pos))

    # create figure and axes objects
    plt.rcParams.update({'figure.constrained_layout.use': True})
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    def animate(i):
        ax.clear()
        index=torch.where(edge_type[i*stride] == edge)[0]
        mask = torch.zeros(edge_index[i*stride].size(1), dtype=torch.bool)
        mask[index] = True
        edge_index_new = edge_index[i*stride][:,mask]
        for (src, dst) in edge_index_new.t().tolist():
             src = pos[i*stride][src].tolist()
             dst = pos[i*stride][dst].tolist()
             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=0.5, color='black')
        index2 = torch.where(node_type[i*stride] == 2)[0]
        index1 = torch.where(node_type[i*stride] == 1)[0]
        index0 = torch.where(node_type[i * stride] == 0)[0]
        mask0 = torch.zeros(pos[i*stride].size(0), dtype=torch.bool)
        mask1 = torch.zeros(pos[i * stride].size(0), dtype=torch.bool)
        mask2 = torch.zeros(pos[i * stride].size(0), dtype=torch.bool)
        mask0[index0] = True
        mask1[index1] = True
        mask2[index2] = True
        #plt.scatter(pos[i*stride][mask0, 0], pos[i*stride][mask0, 1], s=10, c='orange', zorder=1000, label='pointcloud') #color_list[i*stride]
        plt.scatter(pos[i*stride][mask1, 0], pos[i*stride][mask1, 1], s=10, color='darkgreen', zorder=1000, label='collider points')
        plt.scatter(pos[i*stride][mask2, 0], pos[i*stride][mask2, 1], marker='+', s=100, zorder=1000, label='mesh points')
        ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
        ax.set_box_aspect(1)
        plt.axis('off')

    # call the animation
    ani = animation.FuncAnimation(fig, animate, frames=int(num_frames/stride), interval=50)
    if save_animation:
        ani.save('animations/animation_rollout_graph_edge_type' + str(edge) + '.gif', writer='pillow', fps=10)
    else:
        # show the plot
        plt.show()

def add_pointcloud_dropout(data, pointcloud_dropout, hetero):
    if hetero:
        raise NotImplementedError("Not implemented yet")
    else:
        x = np.random.rand(1)
        if x < pointcloud_dropout:
            node_types = [1, 2] #[0]
            node_indices = []
            for node_type in node_types:
                node_indices.append(torch.where(data.node_type == node_type)[0])
            node_indices = torch.cat(node_indices, dim=0)

            edge_types = [1, 2, 5, 8] #[0, 3, 4, 6, 7]
            edge_indices = []
            for edge_type in edge_types:
                edge_indices.append(torch.where(data.edge_type == edge_type)[0])
            edge_indices = torch.cat(edge_indices, dim=0)
            num_node_type = []
            num_node_type_0 = []
            for batch in range(int(torch.max(data.batch) + 1)):
                batch_data = data.node_type[data.batch == batch]
                num_node_type_0.append(len(batch_data[batch_data == 0]))
                num_node_type.append(len(batch_data))
            num_node_type_0 = list(np.cumsum(num_node_type_0))
            num_node_type = list(np.cumsum(num_node_type))
            num_node_type = [0]+num_node_type

            new_pos = data.pos[node_indices]
            new_x = data.x[node_indices]
            new_batch = data.batch[node_indices]
            new_node_type = data.node_type[node_indices]
            new_edge_attr = data.edge_attr[edge_indices]
            new_edge_index = data.edge_index[:,edge_indices]
            new_edge_type = data.edge_type[edge_indices]

            # index shift for new_edge_attr:
            for index in range(len(num_node_type_0)):
                new_edge_index = torch.where(torch.logical_and(new_edge_index > num_node_type[index],  new_edge_index < num_node_type[index+1]), new_edge_index - num_node_type_0[index], new_edge_index)
            data.pos = new_pos
            data.x = new_x
            data.batch = new_batch
            data.node_type = new_node_type
            data.edge_attr = new_edge_attr
            data.edge_index = new_edge_index
            data.edge_type = new_edge_type

    return data

def main():
    dataset = "coarse_full_graph_t"
    build_from = "trapez_materials_contact_voxel"
    eval_mode = "eval"
    use_mesh_coordinates = False

    edge_radius_dict, input_timestep, euclidian_distance, tissue_task = src.utils.get_connectivity_setting.get_connectivity_setting(dataset)
    old_timestep = convert_timestep_str_to_bool(input_timestep)
    timestep = 48

    trajectory_list_raw = dataset_utils.build_dataset_for_split(build_from, "/home/jlinki/Documents/Repos/Physical-Simulation-from-Observation/", eval_mode, edge_radius_dict, 'cpu', input_timestep, euclidian_distance, hetero=False, raw=True, use_color=False)


    #for i in tqdm(range(len(trajectory_list_raw))):
    i=0
    pos_list = []
    edge_index_list = []
    node_type_list = []
    edge_type_list = []
    color_list = []
    for index, data in enumerate(tqdm(trajectory_list_raw[i])):
        if old_timestep:
            data_timestep = data.grid_positions_old, data.collider_positions_old, data.mesh_positions, data.mesh_edge_index, data.y, None, data.initial_mesh_positions, None
            #color_list.append(data.grid_colors_old)
        else:
            data_timestep = data.grid_positions, data.collider_positions, data.mesh_positions, data.mesh_edge_index, data.y, None, data.initial_mesh_positions, None
            #color_list.append(data.grid_colors)
        data = graph_utils.create_graph_from_raw(data_timestep, edge_radius_dict=edge_radius_dict, output_device='cpu', use_mesh_coordinates=use_mesh_coordinates)
        data.batch = torch.zeros_like(data.node_type)
        data = add_pointcloud_dropout(data, 0, False)
        pos_list.append(data.pos)
        edge_index_list.append(data.edge_index)
        node_type_list.append(data.node_type)
        edge_type_list.append(data.edge_type)
    print("Maximum number of edges: ", len(edge_type_list[-1]))
    #for edge in range(6):
    #animate_rollout(pos_list, edge_index_list, node_type=node_type_list, edge_type=edge_type_list, color_list=color_list, edge=9, save_animation=False, stride=1)
    visualize_points(pos_list[timestep], edge_index_list[timestep], node_type=node_type_list[timestep], edge_type=edge_type_list[timestep], edge=9, color_list=None)

if __name__ == '__main__':
    main()