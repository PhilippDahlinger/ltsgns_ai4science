import pickle

import numpy as np
import plotly
import plotly.graph_objects as go

import wandb


def get_mesh(stride: int):
    with open("../../datasets/lts_gns/tissue_manipulation/tissue_manipulation_test.pkl", "rb") as file:
        rollout_data = pickle.load(file)
    # Define the vertices of the tetrahedron (you might need to change these to match your specific case)
    first_trajectory = rollout_data[0]

    mesh_vertices = np.array(first_trajectory["tissue_mesh_positions"])[::stride]
    mesh_faces = first_trajectory["tissue_mesh_triangles"]  # connectivity does not change over time
    gripper_vertices = np.array(first_trajectory["gripper_mesh_positions"])[::stride]
    gripper_faces = first_trajectory["gripper_mesh_triangles"]  # connectivity does not change over time
    return mesh_vertices, mesh_faces, gripper_vertices, gripper_faces


def get_figure(predicted_positions, reference_positions, mesh_faces, gripper_positions, gripper_faces):
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
        "transition": {"duration": 100, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []}
    frames = []
    for timestep, (timestep_reference_positions, timestep_mesh_vertices, timestep_gripper_vertices) in \
            enumerate(zip(reference_positions, predicted_positions, gripper_positions)):
        noisy_face_trace = get_face_trace(vertex_positions=timestep_reference_positions, faces=mesh_faces,
                                          color="lightblue", opacity=0.3, name="Ground Truth")

        vertex_trace = get_vertex_trace(vertex_positions=timestep_mesh_vertices, name="Vertices")
        edge_trace = get_edge_trace(vertex_positions=timestep_mesh_vertices, faces=mesh_faces, name="Edges")
        face_trace = get_face_trace(vertex_positions=timestep_mesh_vertices, faces=mesh_faces,
                                    color="orange", name="Faces")

        gripper_trace = get_face_trace(vertex_positions=timestep_gripper_vertices, faces=gripper_faces,
                                       color="gray", opacity=1.0, name="Gripper")
        frame_name = f"Frame {timestep}"
        frame = go.Frame(data=[face_trace, edge_trace, vertex_trace,
                               noisy_face_trace,
                               gripper_trace], name=frame_name)
        frames.append(frame)
        slider_step = {"args": [
            [frame_name],  # which frames to animate
            {"frame": {"duration": 100,
                       "redraw": True  # must be set to True to update the plot
                       },
             "mode": "immediate",
             "transition": {"duration": 100}}
        ],
            "label": timestep,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(range=[-1, 1]),  # adjust these values to fit your data
                yaxis=dict(range=[-1, 1]),  # adjust these values to fit your data
                zaxis=dict(range=[-1.5, 2]),  # adjust these values to fit your data
                aspectmode='cube'
            ),
            sliders=[sliders_dict],
            updatemenus=[dict(
                type="buttons",
                x=0,  # position of the "Play" button
                y=0,  # position of the "Play" button
                buttons=[dict(label="Yoinks",
                              method="animate",
                              args=[None, {"frame": {"duration": 100,
                                                     "redraw": True},
                                           "fromcurrent": True,
                                           "transition": {"duration": 100}}],
                              )])]),
        frames=frames
    )

    return fig


def get_face_trace(vertex_positions, faces, showlegend=True, color: str = "lightblue", opacity: float = 0.5,
                   name="Faces"):
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


def get_vertex_trace(vertex_positions, showlegend=True, name="Vertices"):
    vertex_trace = go.Scatter3d(x=vertex_positions[:, 0],
                                y=vertex_positions[:, 1],
                                z=vertex_positions[:, 2],
                                mode="markers",
                                marker={"size": 2, "color": "black"},
                                name=name,
                                showlegend=showlegend)
    return vertex_trace


def get_edge_trace(vertex_positions, faces, showlegend=True, name="Edges"):
    num_edges = faces.shape[0]
    edge_x_positions = np.full(shape=4 * num_edges, fill_value=None)
    edge_y_positions = np.full(shape=4 * num_edges, fill_value=None)
    edge_z_positions = np.full(shape=4 * num_edges, fill_value=None)
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
                              line=dict(color="black", width=1),
                              name=name,
                              showlegend=showlegend)
    return edge_trace


def main(use_wandb: bool = False, save_figure: bool = True, show_figure: bool = True):
    mesh_vertices, mesh_faces, gripper_vertices, gripper_faces = get_mesh(stride=1)

    noise = np.random.normal(scale=0.002, size=mesh_vertices.shape)
    noise = np.cumsum(noise, axis=0)
    noisy_vertices = mesh_vertices + noise
    fig = get_figure(predicted_positions=noisy_vertices,
                     reference_positions=mesh_vertices,
                     mesh_faces=mesh_faces,
                     gripper_positions=gripper_vertices,
                     gripper_faces=gripper_faces)

    if save_figure:
        plotly.io.write_html(fig, file="3d_visualization.html", include_plotlyjs='cdn')

    if use_wandb:
        # we have to transform the plotly plot to html to properly render the animation in wandb because the backend
        # of wandb does not support plotly animations
        logger = wandb.init(project="playground",  # name of the whole project
                            name="3d_visualization",  # individual repetitions
                            )
        logger.log({"my_chart": wandb.Html(plotly.io.to_html(fig, include_plotlyjs='cdn'))})

    if show_figure:
        fig.show()


if __name__ == '__main__':
    main()
