import pickle
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay



class AbstractColors:
    """
    Abstract class for Color subclasses
    """

    def __init__(self):
        self._colors = None

    def __call__(self, color_id):
        raise NotImplementedError("AbstractColors can not be called")

    @property
    def colors(self):
        return self._colors


class WheelColors(AbstractColors):
    """
    Provides num_colors equidistant colors for plotting
    """

    def __init__(self, num_colors: int):
        """
        Creates the colors used for plotting. The colors used here are the traditional 10 matplotlib colors.
        This may not be enough for some applications
        """
        super().__init__()
        self.color_map = plt.cm.hsv
        self.num_colors = num_colors
        self._colors = [self.color_map((x + 0.5) / num_colors) for x in np.arange(0, num_colors)]
        # prepare colobar
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', self._colors, self.color_map.N)
        self._scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_colors))
        self._bounds = np.arange(num_colors + 1)

    def __call__(self, color_id: int) -> Tuple[float]:
        """
        get the i-th color of the color_wheel. Will raise an exception for invalid color ids
        :param color_id: the color to be returned
        :return: A pyplot color as an RGBA tuple
        """
        assert 0 <= color_id < self.num_colors, f"Color_id must be in [0,{self.num_colors}), given {color_id}"
        return self._colors[color_id]

    def as_list(self) -> List[Tuple[float]]:
        return self._colors

    def draw_colorbar(self, label: str = "Colorbar", horizontal: bool = False) -> matplotlib.pyplot.colorbar:
        if self.num_colors > 1:
            ticks = self._bounds[:-1]
            squishing_factor = int(len(ticks) // 30) + 1
            ticks = ticks[::squishing_factor]
            colorbar = plt.colorbar(self._scalar_mappable, spacing='proportional', ticks=ticks, boundaries=self._bounds,
                                    orientation="horizontal" if horizontal else "vertical")
            colorbar.set_label(label, rotation=None if horizontal else 90)


def get_mesh(stride: int):
    with open("../../datasets/lts_gns/deformable_plate/deformable_plate_test.pkl", "rb") as file:
        rollout_data = pickle.load(file)
    # Define the vertices of the tetrahedron (you might need to change these to match your specific case)
    first_trajectory = rollout_data[0]

    pointclouds = first_trajectory["pcd_points"][::stride]
    mesh_vertices = np.array(first_trajectory["nodes_grid"])[::stride]
    mesh_faces = first_trajectory["triangles_grid"]  # connectivity does not change over time
    collider_vertices = np.array(first_trajectory["nodes_collider"])[::stride]
    collider_faces = first_trajectory["triangles_collider"]  # connectivity does not change over time
    return mesh_vertices, mesh_faces, collider_vertices, collider_faces, pointclouds


def ifp(pointcloud: np.ndarray, init_convex_hull=True, target_points: int = 81) -> np.ndarray:
    points = np.empty(shape=(target_points, 2))
    ptr = 0
    if init_convex_hull:
        from scipy.spatial import ConvexHull
        convex_hull = ConvexHull(pointcloud).vertices
        length = convex_hull.shape[0]
        points[:length] = pointcloud[convex_hull]
        ptr = ptr + length
    else:
        # add the first point
        points[0] = pointcloud[0]
        ptr = ptr + 1

    while ptr < target_points:
        # find point with largest distance to all other points
        distances = np.linalg.norm(pointcloud[:, None] - points[:ptr], axis=2)
        min_distances = np.min(distances, axis=1)
        max_distance_idx = np.argmax(min_distances)
        points[ptr] = pointcloud[max_distance_idx]
        ptr = ptr + 1

    distances = np.linalg.norm(points[:, None] - points, axis=2)
    # exclude the diagonal
    distances = distances[~np.eye(distances.shape[0], dtype=bool)].reshape(distances.shape[0], -1)
    max_min_distance = np.max(np.min(distances, axis=1))

    return points, max_min_distance


def ploot(points, max_min_distance, color):
    # create a neighborhood graph of all points where the distance is smaller than the max_min_distance
    distances = np.linalg.norm(points[:, None] - points, axis=2)
    neighborhood_graph = distances < np.sqrt(2) * max_min_distance
    # plot all edges in the neighborhood graph
    # for i in range(neighborhood_graph.shape[0]):
    #     for j in range(neighborhood_graph.shape[1]):
    #         if neighborhood_graph[i, j]:
    #             plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color=color, alpha=0.2)
    # plt.scatter(pointcloud[:, 0], pointcloud[:, 1], color="r")
    plt.scatter(points[:, 0], points[:, 1], color=color, alpha=.3)



def method_1(color_wheel, num_steps, pointcloud_trajectory, num_points: int = 81):
    plt.figure(figsize=(10, 10))
    # plt.scatter(pointcloud_trajectory[0][:, 0], pointcloud_trajectory[0][:, 1], color="black", alpha=.3)
    for step, pointcloud in enumerate(pointcloud_trajectory[:num_steps]):
        pointcloud = offset_pointcloud(pointcloud, offset=step*0.01)
        sub_pointcloud, max_min_distance = ifp(pointcloud, init_convex_hull=False, target_points=num_points)

        if step % 1 == 0:
            ploot(sub_pointcloud, max_min_distance, color_wheel(step))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def method_2(color_wheel, num_steps, pointcloud_trajectory):
    plt.figure(figsize=(10, 10))
    first_pointcloud = pointcloud_trajectory[0]
    sub_pointcloud, max_min_distance = ifp(first_pointcloud, target_points=81)
    ploot(sub_pointcloud, max_min_distance, color_wheel(0))
    for step, pointcloud in enumerate(pointcloud_trajectory[1:num_steps]):
        #
        # pointcloud = noise_pointcloud(pointcloud)

        sub_pointcloud = get_matching_points(sub_pointcloud, pointcloud)

        if step % 10 == 0:
            ploot(sub_pointcloud, max_min_distance, color_wheel(step + 1))
    # equal the aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def method_3(color_wheel, num_steps, pointcloud_trajectory, num_points: int = 81):
    plt.figure(figsize=(10, 10))
    first_pointcloud = pointcloud_trajectory[0]
    sub_pointcloud, _ = ifp(first_pointcloud, target_points=num_points)

    tri = Delaunay(sub_pointcloud)

    plot_from_delaunay(sub_pointcloud, tri, color_wheel(0))

    for step in range(num_steps):
        next_pointcloud = pointcloud_trajectory[step]
        next_sub_pointcloud = get_matching_points(sub_pointcloud, next_pointcloud)
        next_sub_pointcloud = noise_pointcloud(next_sub_pointcloud, offset=0.015)
        next_sub_pointcloud = get_assignment(next_pointcloud, next_sub_pointcloud)

        plot_from_delaunay(next_sub_pointcloud, tri, color_wheel(step + 1))
        sub_pointcloud = next_sub_pointcloud
    # equal the aspect ratio
    color_wheel.draw_colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_from_delaunay(points, tri, color):
    plt.triplot(points[:, 0], points[:, 1], tri.simplices, color=color, alpha=0.2)
    plt.plot(points[:, 0], points[:, 1], 'o', color=color, alpha=0.2)


def offset_pointcloud(pointcloud, offset=0.0005):
    # noise = np.random.normal(scale=0.0005, size=pointcloud.shape)
    # noise = np.cumsum(noise, axis=0)
    pointcloud = pointcloud + offset
    return pointcloud

def noise_pointcloud(pointcloud, offset=0.0005):
    noise = np.random.normal(scale=offset, size=pointcloud.shape)
    pointcloud = pointcloud + noise
    return pointcloud

def get_matching_points(sub_pointcloud, pointcloud):
    """
    Get the closest matching points in the pointcloud to the sub_pointcloud
    Args:
        sub_pointcloud: A (small) sub-pointcloud, where each point should be matched by the closest point in the
                        pointcloud. Has shape (N, 2)
        pointcloud: A (large) pointcloud, where each point should be matched by the closest point in the
                        sub_pointcloud. Has shape (M, 2)
    Returns: An array of shape (N, 2) where each point is the closest point in the pointcloud to the corresponding
                point in the sub_pointcloud

    """
    distances = np.linalg.norm(sub_pointcloud[:, None] - pointcloud, axis=2)
    min_distance_idx = np.argmin(distances, axis=1)
    return pointcloud[min_distance_idx]


def get_assignment(pointcloud1, pointcloud2):
    """
    Perform the Hungarian algorithm on the two pointclouds to find the best matching points
    Args:
        pointcloud1:
        pointcloud2:

    Returns: An array of shape (N, 2) where each point is the assigned point of sub_pointcloud2 that is closest to
        the corresponding point in sub_pointcloud1

    """
    if len(pointcloud2) != len(pointcloud1):
        print("stop")
    from scipy.optimize import linear_sum_assignment
    cost = np.linalg.norm(pointcloud1 - pointcloud2[:, None], axis=2) ** 2

    row_ind, col_ind = linear_sum_assignment(cost)
    assert np.all(row_ind[1:] >= row_ind[:-1], axis=0), "row_ind is not sorted"
    inv_col_ind = np.empty(col_ind.size, dtype=np.int32)
    for i in np.arange(col_ind.size):
        inv_col_ind[col_ind[i]] = i

    # permutation = np.empty_like(pointcloud2)
    # permutation[row_ind%81] = pointcloud2[col_ind]
    #
    # return permutation
    return pointcloud2[inv_col_ind], inv_col_ind, col_ind

def main():
    # logger = wandb.init(project="playground",  # name of the whole project
    #                     name="3d_visualization",  # individual repetitions
    #                     )
    mesh_vertices, mesh_faces, collider_vertices, collider_faces, pointcloud_trajectory = get_mesh(stride=1)

    num_steps = 10
    color_wheel = WheelColors(num_colors=num_steps+1)

    # method_1(color_wheel, num_steps, pointcloud_trajectory, num_points=81)
    # method_2(color_wheel, num_steps, pointcloud_trajectory)
    method_3(color_wheel, num_steps, pointcloud_trajectory, num_points=150)

    # we have to transform the plotly plot to html to properly render the animation in wandb because the backend
    # of wandb does not support plotly animations
    # logger.log({"my_chart": wandb.Html(plotly.io.to_html(fig, include_plotlyjs='cdn'))})
    # fig.show()


if __name__ == '__main__':
    main()
