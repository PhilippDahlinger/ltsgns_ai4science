from typing import List

import torch
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from torch_geometric.data import Data

from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict


def visualize_trajectory(trajectory: List[Data], visualization_config: ConfigDict,
                         limits: ConfigDict, fps: int,
                         ground_truth_trajectory: Data | None = None,
                         context: bool = False) -> animation.FuncAnimation:
    """
    Visualize a trajectory of graphs using matplotlib.
    Args:
        trajectory: List of Data objects forming the trajectory. Assumes that the first graph is the initial graph
            and that the topology of the graphs does not change over time.
        visualization_config: ConfigDict containing the animation/visualization configuration
        limits: ConfigDict containing the limits of the plot.
        fps: Frames per second of the animation.
        ground_truth_trajectory: List of Data objects forming the ground truth trajectory. If None, no ground truth
            is plotted.
        context: Whether the current plot is that of a context trajectory

    Returns: None

    """
    animation_config = visualization_config.matplotlib

    def animate(i):
        ax.clear()
        if ground_truth_trajectory is None:
            gth_data = None
        else:
            gth_data = ground_truth_trajectory[i]
        visualize_single_graph(trajectory[i], ground_truth_data=gth_data,
                               ax=ax, limits=limits,
                               show_legend=animation_config.show_legend,
                               context=context)

    fig, ax = plt.subplots()
    fig.set_size_inches(animation_config.fig_size, animation_config.fig_size)
    plt.tight_layout()

    ani = animation.FuncAnimation(fig, animate, frames=len(trajectory), interval=int(1000/fps))
    return ani


def visualize_single_graph(data: Data,
                           ground_truth_data: Data,
                           limits: ConfigDict,
                           ax: Axis | None = None,
                           show_legend: bool = True,
                           context: bool = False) -> Axis:
    """
    Visualize a single graph.
    Args:
        data: Data object. Has to contain the following keys:
            - pos
            - node_type
            - node_type_description
            - edge_index
            - edge_type
            - edge_type_description
        ground_truth_data: Data object containing the ground truth or None
        ax: matplotlib axis to plot on. If None: a new figure is created.
        limits: Dictionary of {xlim: [min, max], ylim: [min, max]}
        show_legend: Whether to show the legend
        context:

    Returns:

    """
    try:
        pos: torch.Tensor = data.pos
        node_type: torch.Tensor = data.node_type
        node_type_description: List[str] = data.node_type_description
        edge_index: torch.Tensor = data.edge_index
        edge_type: torch.Tensor = data.edge_type
        edge_type_description: List[str] = data.edge_type_description
    except AttributeError as e:
        raise AttributeError("Data object has to contain the following keys: pos, node_type, "
                             "node_type_description, edge_index, edge_type, edge_type_description") from e

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)

    # lims
    if limits.xlim is not None:
        ax.set_xlim(limits.xlim)
    if limits.ylim is not None:
        ax.set_ylim(limits.ylim)

    mesh_colors = ["turquoise", "orange"]

    # MESH
    if hasattr(data, keys.MESH_FACES):
        # plot polygon of faces
        for current_data, color in zip([ground_truth_data, data], mesh_colors):
            if current_data is None:
                continue
            faces = current_data[keys.MESH_FACES]
            current_mesh_collection = [Polygon(face, closed=True) for face in current_data.pos[faces]]
            current_mesh_collection = PatchCollection(current_mesh_collection, alpha=0.4, ec='black', fc=color)
            ax.add_collection(current_mesh_collection)
    else:
        for current_data in [ground_truth_data, data]:
            if current_data is None:
                continue
            # plot every mesh edge
            # if there are no edges: skip
            if current_data.edge_index is None or current_data.edge_index.shape[1] == 0:
                continue
            mesh_type_index = current_data.edge_type_description.index(keys.MESH_MESH)
            mesh_indices = current_data.edge_type == mesh_type_index
            mesh_edge_index_0 = current_data.edge_index[0, mesh_indices]
            mesh_edge_index_1 = current_data.edge_index[1, mesh_indices]
            x = torch.stack((current_data.pos[mesh_edge_index_0][:, 0], current_data.pos[mesh_edge_index_1][:, 0]))
            y = torch.stack((current_data.pos[mesh_edge_index_0][:, 1], current_data.pos[mesh_edge_index_1][:, 1]))
            ax.plot(x, y, linewidth=1.0, color='black')

    # COLLIDER
    if hasattr(data, keys.COLLIDER_FACES):
        # only need data, since ground truth is the same
        faces = data[keys.COLLIDER_FACES]
        collider_collection = [Polygon(face, closed=True) for face in pos[faces]]
        collider_collection = PatchCollection(collider_collection, alpha=0.3, linewidths=0, ec='blue', fc='blue')
        ax.add_collection(collider_collection)

    # Collider Mesh edges: right now they are irritating
    # if data.edge_index.shape[1] != 0:  # has edges
    #     mesh_collider_type_index = edge_type_description.index(keys.MESH_COLLIDER)
    #     mesh_collider_indices = edge_type == mesh_collider_type_index
    #     mesh_collider_edge_index_0 = edge_index[0, mesh_collider_indices]
    #     mesh_collider_edge_index_1 = edge_index[1, mesh_collider_indices]
    #     x = torch.stack((pos[mesh_collider_edge_index_0][:, 0], pos[mesh_collider_edge_index_1][:, 0]))
    #     y = torch.stack((pos[mesh_collider_edge_index_0][:, 1], pos[mesh_collider_edge_index_1][:, 1]))
    #     ax.plot(x, y, linestyle="-", linewidth=0.5, color='gray')

    for current_data, color, type in zip([ground_truth_data, data], mesh_colors, ["ground truth", "prediction"]):
        if current_data is None:
            continue
        indices = {}
        masks = {}
        colors = {}
        labels = {}
        # figure out which nodes are of which type
        for i, description in enumerate(current_data.node_type_description):
            indices[description] = torch.where(current_data.node_type == i)[0]
            masks[description] = torch.zeros(current_data.pos.size(0), dtype=torch.bool)
            masks[description][indices[description]] = True
            if description == keys.COLLIDER:
                colors[description] = "red"
                labels[description] = None
            else:
                colors[description] = color
                if context:
                    labels[description] = "context"
                else:
                    labels[description] = type
        # plot the nodes
        # for description, mask in masks.items():
        #     if type == "prediction" and description == keys.COLLIDER:
        #         # only draw collider once
        #         continue
        #     ax.scatter(current_data.pos[mask, 0], current_data.pos[mask, 1], s=80, c=colors[description], zorder=1000,
        #                label=labels[description])

    # plot goal pos if available
    if hasattr(data, "goal_pos"):
        if len(data.goal_pos.shape) == 1:
            # only one goal
            ax.scatter(data.goal_pos[0], data.goal_pos[1], s=80, c="green", zorder=1000, label="goal")
        else:
            for sub_goal in data.goal_pos:
                ax.scatter(sub_goal[0], sub_goal[1], s=80, c="green", zorder=1000, label="goal")

    if show_legend:
        ax.legend(fontsize=26)
    ax.set_axis_off()

    return ax
