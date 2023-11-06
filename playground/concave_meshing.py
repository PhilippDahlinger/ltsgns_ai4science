import open3d as o3d
import trimesh
import numpy as np


def ball_pivoting_reconstruction(xy, radii=None):
    """Given a 3D point cloud, get unstructured mesh using ball pivoting algorithm

    Based on this stack overflow code snippet:
    https://stackoverflow.com/questions/56965268/how-do-i-convert-a-3d-point-cloud-ply-into-a-mesh-with-faces-and-vertices

    Parameters
    ----------
    xy: [n_points, 2]
        input point cloud, numpy array
    radii: [n_radii]
        list of radiuses to use for the ball pivoting algorithm

    Returns
    -------
    mesh: trimesh Mesh object with verts, faces and normals of the mesh

    """
    xyz = np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)
    # estimate normals first
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # Manually set the normals to [0., 0., 1] for all points
    normal_vector = np.array([0., 0., 1.])
    pcd.normals = o3d.utility.Vector3dVector(np.tile(normal_vector, (len(xyz), 1)))

    # heuristic to estimate the radius of a rolling ball
    if radii is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1.5 * avg_dist
        radii = [radius, radius * 2]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                           vertex_normals=np.asarray(mesh.vertex_normals))

    # try to fix normals with Trimesh
    mesh.fix_normals()

    # save mesh:
    # mesh.export('../logs/mesh.obj')

    return mesh

if __name__ == "__main__":
    # create random point cloud
    np.random.seed(42)
    xy = np.random.rand(100, 2)

    mesh = ball_pivoting_reconstruction(xy, radii=[.1, 0.2,])
    x = mesh.vertices
    edges = mesh.edges

    # plot
    import matplotlib.pyplot as plt
    plt.scatter(x[:, 0], x[:, 1])
    for edge in edges:
        plt.plot(x[edge, 0], x[edge, 1], c="r")
    plt.show()

    print("stop")

