from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
from torch_geometric.data import Data

from lts_gns.util import keys
from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.util import node_type_mask, to_numpy, edge_type_mask


def visualize_trajectory(trajectory: List[Data], visualization_config: ConfigDict,
                         limits: ConfigDict, fps: int,
                         ground_truth_trajectory: List[Data] | None = None,
                         context: bool = False) -> go.Figure:
    """
    Visualize a trajectory of graphs using plotly.
    Args:
        trajectory: List of Data objects forming the trajectory. Assumes that the first graph is the initial graph
            and that the topology of the graphs does not change over time.
        visualization_config: ConfigDict containing the animation configuration
        limits: ConfigDict containing the limits of the plot.
        fps: Frames per second of the animation.
        ground_truth_trajectory: List of Data objects forming the ground truth trajectory. If None, no ground truth
            is plotted.
        context: Whether the current plot is that of a context trajectory

    Returns: None
    """
    animation_config = visualization_config.plotly
    mesh_vertices, mesh_faces, collider_vertices, collider_faces = get_meshes(trajectory)

    if ground_truth_trajectory is not None:
        reference_vertices, _, _, _ = get_meshes(ground_truth_trajectory)
    else:
        reference_vertices = None

    mesh_collider_edges = _get_edges(trajectory, key=keys.MESH_COLLIDER)
    world_mesh_edges = _get_edges(trajectory, key=keys.WORLD_MESH)

    fig = get_figure(predicted_positions=mesh_vertices,
                     mesh_faces=mesh_faces,
                     collider_positions=collider_vertices,
                     collider_faces=collider_faces,
                     reference_positions=reference_vertices,
                     mesh_collider_edges=mesh_collider_edges,
                     world_mesh_edges=world_mesh_edges,
                     limits=limits,
                     animation_config=animation_config,
                     frame_duration=int(1000 / fps))

    return fig


def get_meshes(trajectory: List[Data]) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Retrieve the mesh and collider data from the trajectory. We assume that the mesh topology does not change over time,
    so we can retrieve the faces from the first graph in the trajectory.
    The positions can change however, so we build a list over them given the provided stride.
    Args:
        trajectory: List of Data objects forming the trajectory.

    Returns: List of (mesh vertices, mesh faces, collider vertices, collider faces)

    """
    first_step = trajectory[0]

    # information about the mesh
    if hasattr(first_step, keys.MESH_FACES):
        mesh_faces = first_step[keys.MESH_FACES]
        vertex_keys = [keys.MESH]
        if keys.FIXED_MESH in first_step.node_type_description:
            vertex_keys.append(keys.FIXED_MESH)
        mesh_vertices = _get_vertices(trajectory, keys=vertex_keys)
    else:
        raise NotImplementedError(f"Data object does not contain key: {keys.MESH_FACES}")

    collider_vertices = None
    collider_faces = None

    if hasattr(first_step, "collider_vertices"):
        # for some tasks, visual collider vertex positions are stored externally.
        # These then do not affect the simulation of the mesh, but are still nice to visualize
        collider_vertices = np.array([to_numpy(step.collider_vertices) for step in trajectory])
    elif keys.COLLIDER in first_step.node_type_description:
        collider_vertices = _get_vertices(trajectory, keys=keys.COLLIDER)

    # information about the collider faces
    if hasattr(first_step, keys.COLLIDER_FACES):
        collider_faces = first_step[keys.COLLIDER_FACES]

    return mesh_vertices, mesh_faces, collider_vertices, collider_faces


def _get_vertices(trajectory, keys: str | List[str]) -> np.ndarray:
    """
    Retrieve the vertices of the given type or types from the trajectory.
    Args:
        trajectory:
        keys:

    Returns:

    """
    if isinstance(keys, str):
        keys = [keys]
    mesh_vertices = []
    for step in range(len(trajectory)):
        data = trajectory[step]
        node_mask = [node_type_mask(data, key) for key in keys]  # list of masks for each key
        node_mask = torch.stack(node_mask).any(dim=0)  # combine the masks, check if any of them is true

        mesh_vertices.append(to_numpy(data.pos[node_mask]))
    mesh_vertices = np.array(mesh_vertices)
    return mesh_vertices


def _get_edges(trajectory: List[Data], key: str) -> List[np.array] | None:
    """
    Retrieve the edges of the given type from the trajectory. Returns None if the edges are not present.
    Args:
        trajectory: List of Data objects forming the trajectory.
        key: The edge type to retrieve

    Returns: List of edges per step if the edges exist, None otherwise.

    """
    if key not in trajectory[0].edge_type_description:
        return None
    mesh_collider_edges = []
    for step in range(len(trajectory)):
        data = trajectory[step]
        mesh_collider_edges.append(to_numpy(data.edge_index[:, edge_type_mask(data, key)]))
    return mesh_collider_edges


def get_figure(predicted_positions: np.ndarray, mesh_faces: np.ndarray | torch.Tensor,
               collider_positions: np.ndarray | None, collider_faces: np.ndarray | torch.Tensor | None,
               reference_positions: np.ndarray | None, mesh_collider_edges: List[np.ndarray] | None,
               world_mesh_edges: List[np.ndarray] | None,
               limits: ConfigDict, animation_config: ConfigDict,
               frame_duration: int = 100) -> go.Figure:
    """
    Create a plotly figure for the given data. If some of the data is None, it is not plotted.
    Args:
        predicted_positions: List of predicted node/vertex of the positions of the predicted mesh for each timestep.
            Has shape (num_timesteps, num_nodes, 3)
        mesh_faces: List of faces of the mesh. Has shape (num_faces, 3). Does not change over time.
        collider_positions: List of collider node/vertex positions for each timestep.
            Has shape (num_timesteps, num_collider_nodes, 3)
        collider_faces: List of faces of the collider. Has shape (num_faces, 3). Does not change over time.
        reference_positions: List of reference node/vertex positions for each timestep.
            Has shape (num_timesteps, num_nodes, 3). Acts as a ground truth. Has the mesh topology of the predictions.
        mesh_collider_edges: Edges between the mesh and the collider. Is a list with entries of shape (num_edges, 2).
            Does change over time, and may have different number of edges for each timestep.
        world_mesh_edges: Mesh-mesh edges in world space. Is a list with entries of shape (num_edges, 2).
            Does change over time, and may have different number of edges for each timestep.
        limits: ConfigDict containing the limits of the plot.
            Is a dictionary {xlim: [min, max], ylim: [min, max], zlim: [min, max]}
        animation_config: ConfigDict containing the animation configuration
        frame_duration: Duration of each frame in milliseconds

    Returns: Plotly figure

    """
    # Create the 3D figure
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Step: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": frame_duration,
                       "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []}

    frames = []
    for timestep in range(len(predicted_positions)):
        timestep_mesh_vertices = predicted_positions[timestep]

        vertex_trace = _get_vertex_trace(vertex_positions=timestep_mesh_vertices, name="Vertices")
        edge_trace = _get_edge_trace_from_faces(vertex_positions=timestep_mesh_vertices, faces=mesh_faces, name="Edges")
        face_trace = _get_face_trace(vertex_positions=timestep_mesh_vertices, faces=mesh_faces, opacity=0.4,
                                     color="orange", name="Faces")

        traces = [vertex_trace, edge_trace, face_trace]

        if collider_positions is not None:  # add traces for the collider
            timestep_collider_vertices = collider_positions[timestep]
            # collider_vertex_trace = _get_vertex_trace(vertex_positions=timestep_collider_vertices,
            #                                           name="Collider Vertices", color="red")
            # traces.extend([collider_vertex_trace])
            if collider_faces is not None:
                collider_edge_trace = _get_edge_trace_from_faces(vertex_positions=timestep_collider_vertices,
                                                                 faces=collider_faces, name="Collider Edges")
                collider_face_trace = _get_face_trace(vertex_positions=timestep_collider_vertices, faces=collider_faces,
                                                      color="gray", opacity=1.0, name="Collider")
                traces.extend([collider_edge_trace, collider_face_trace])
            if mesh_collider_edges is not None:
                timestep_mesh_collider_edges = mesh_collider_edges[timestep]

                # Don't render Mesh collider edges
                # mesh_collider_edge_trace = _get_edge_trace_from_edges(in_vertices=timestep_mesh_vertices,
                #                                                       out_vertices=timestep_collider_vertices,
                #                                                       edges=timestep_mesh_collider_edges,
                #                                                       name="Mesh Collider Edges", color="darksalmon")
                # traces.extend([mesh_collider_edge_trace])

            if world_mesh_edges is not None:
                timestep_world_mesh_edges = world_mesh_edges[timestep]
                world_mesh_edge_trace = _get_edge_trace_from_edges(in_vertices=timestep_mesh_vertices,
                                                                   out_vertices=timestep_mesh_vertices,
                                                                   edges=timestep_world_mesh_edges,
                                                                   name="World Mesh Edges", color="darkslategray")
                traces.extend([world_mesh_edge_trace])


        if reference_positions is not None:  # add a trace for the reference
            timestep_reference_positions = reference_positions[timestep]
            reference_face_trace = _get_face_trace(vertex_positions=timestep_reference_positions, faces=mesh_faces,
                                                   color="turquoise", opacity=0.4, name="Ground Truth")
            traces.extend([reference_face_trace])

        frame_name = f"Frame {timestep}"
        frame = go.Frame(data=traces, name=frame_name)
        frames.append(frame)

        slider_step = {"args": [
            [frame_name],  # need to have a correspondence here to tell which frames to animate
            {"frame": {"frame_duration": frame_duration,
                       "redraw": True  # must be set to True to update the plot
                       },
             "mode": "immediate",
             "transition": {"frame_duration": frame_duration}}
        ],
            "label": timestep,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig = _build_figure(frames, sliders_dict, frame_duration=frame_duration,
                        limits=limits, animation_config=animation_config)

    return fig


def _build_figure(frames, sliders_dict, frame_duration: int, limits: ConfigDict, animation_config: ConfigDict):
    """
    Builds the figure for the animation.
    Args:
        frames: 
        sliders_dict: 
        frame_duration: frame_duration of the animation in milliseconds. Converts to fps as 1000/frame_duration.
        limits: Dictionary of {xlim: [min, max], ylim: [min, max], zlim: [min, max]}
        animation_config: ConfigDict containing the animation configuration

    Returns:

    """
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(range=limits.xlim, showgrid = False,showticklabels = False),
                yaxis=dict(range=limits.ylim, showgrid = False,showticklabels = False),
                zaxis=dict(range=limits.zlim, showgrid = False,showticklabels = False),
                aspectmode='cube'  # equal aspect ratio for all axes
            ),
            sliders=[sliders_dict],
            updatemenus=[dict(
                type="buttons",
                x=0,  # position of the "Play" button
                y=0,  # position of the "Play" button
                buttons=[dict(label=animation_config.button_name,
                              method="animate",
                              args=[None, {"frame": {"frame_duration": frame_duration,
                                                     "redraw": True  # must be true
                                                     },
                                           "fromcurrent": True,
                                           "transition": {"frame_duration": frame_duration}}],
                              )])]),
        frames=frames
    )
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    return fig


def _get_vertex_trace(vertex_positions, showlegend=True, name="Vertices", color="black"):
    vertex_trace = go.Scatter3d(x=vertex_positions[:, 0],
                                y=vertex_positions[:, 1],
                                z=vertex_positions[:, 2],
                                mode="markers",
                                marker={"size": 2, "color": color},
                                name=name,
                                showlegend=showlegend)
    return vertex_trace


def _get_edge_trace_from_faces(vertex_positions, faces, showlegend=True, name="Edges", color: str = "black"):
    """
    Returns a trace for the edges of the mesh faces
    Args:
        vertex_positions:
        faces:
        showlegend:
        name:
        color:

    Returns:

    """
    faces = _rescale_indices(index_array=faces)
    num_faces = faces.shape[0]
    edge_x_positions = np.full(shape=4 * num_faces, fill_value=None)
    edge_y_positions = np.full(shape=4 * num_faces, fill_value=None)
    edge_z_positions = np.full(shape=4 * num_faces, fill_value=None)
    edge_x_positions[0::4] = vertex_positions[faces[:, 0], 0]
    edge_x_positions[1::4] = vertex_positions[faces[:, 1], 0]
    edge_x_positions[2::4] = vertex_positions[faces[:, 2], 0]
    edge_y_positions[0::4] = vertex_positions[faces[:, 0], 1]
    edge_y_positions[1::4] = vertex_positions[faces[:, 1], 1]
    edge_y_positions[2::4] = vertex_positions[faces[:, 2], 1]
    edge_z_positions[0::4] = vertex_positions[faces[:, 0], 2]
    edge_z_positions[1::4] = vertex_positions[faces[:, 1], 2]
    edge_z_positions[2::4] = vertex_positions[faces[:, 2], 2]

    edge_trace = go.Scatter3d(x=edge_x_positions,
                              y=edge_y_positions,
                              z=edge_z_positions,
                              mode="lines",
                              line=dict(color=color, width=1),
                              name=name,
                              showlegend=showlegend)
    return edge_trace


def _get_edge_trace_from_edges(in_vertices, out_vertices, edges, name: str, color: str = "black"):
    """
    Returns a trace for the edges of the mesh faces from the edge indices, which are applied to the in and out vertices.
    Args:
        in_vertices: List of vertex positions for the in vertices
        out_vertices: List of vertex positions for the out vertices
        edges:
        name:
        color:

    Returns:

    """
    num_edges = edges.shape[1]
    edge_x_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_y_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_z_positions = np.full(shape=3 * num_edges, fill_value=None)
    edge_x_positions[0::3] = in_vertices[edges[0], 0]
    edge_y_positions[0::3] = in_vertices[edges[0], 1]
    edge_z_positions[0::3] = in_vertices[edges[0], 2]
    adjusted_collider_edges = _rescale_indices(index_array=edges[1])
    edge_x_positions[1::3] = out_vertices[adjusted_collider_edges, 0]
    edge_y_positions[1::3] = out_vertices[adjusted_collider_edges, 1]
    edge_z_positions[1::3] = out_vertices[adjusted_collider_edges, 2]
    mesh_collider_edge_trace = go.Scatter3d(x=edge_x_positions,
                                            y=edge_y_positions,
                                            z=edge_z_positions,
                                            mode="lines",
                                            line=dict(color=color, width=1),
                                            name=name,
                                            showlegend=True)
    return mesh_collider_edge_trace


def _get_face_trace(vertex_positions, faces, showlegend=True, color: str = "lightblue", opacity: float = 0.5,
                    name="Faces"):
    faces = _rescale_indices(index_array=faces)
    face_trace = go.Mesh3d(
        x=vertex_positions[:, 0],
        y=vertex_positions[:, 1],
        z=vertex_positions[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        showlegend=showlegend,
        name=name
    )
    return face_trace


def _rescale_indices(index_array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Adjusts the indices of the index array such that the smallest index is 1.
    Indices will be in range to be in the range [0, index_entities - 1].
    Args:
        index_array: An array of integer indices of arbitrary shape

    Returns:

    """
    if np.prod(index_array.shape) == 0:  # works for np and torch
        # empty index :(
        return index_array
    # check that the indices of the faces are not shifted
    min_index = index_array.min()
    if min_index > 0:
        index_array = index_array - min_index
    return index_array
