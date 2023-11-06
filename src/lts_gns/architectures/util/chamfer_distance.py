import numpy as np
import torch


def padded_chamfer_distance(pointcloud1: torch.Tensor, padded_pointcloud2: torch.Tensor,
                            density_aware: bool = False,
                            forward_only: bool = False,
                            point_reduction: str = "mean") -> torch.Tensor:
    """
    Compute the Chamfer Distance between two point clouds. The second pointcloud can be padded with nan values.
    Args:
        pointcloud1 (torch.Tensor): Point cloud 1, shape ([Batch_dims], P1, D)
        padded_pointcloud2 (torch.Tensor): Point cloud 2, shape (Single_Batch_dim, P2, D) Single_Batch_dim has to be the
                                           shape of the last batch dimension of pointcloud1.
        density_aware (bool): If True, the Chamfer Distance is "density aware", i.e.
          the distance is scaled exponentially. Instead of d(x,y) = sqrt((x-y)^2), the distance is
            d(x,y) = 1-exp(-((x-y)^2)). This is useful if the point clouds have different densities.
        forward_only (bool): If True, only compute the forward distance (from pointcloud1 to pointcloud2).
        point_reduction (str): How to reduce the point dimension. Can be "mean" or "sum".
    Returns:
        chamfer_dist (torch.Tensor): Shape ([Batch_dims]) tensor representing the Chamfer Distance between the point clouds
          of each batch.
    """
    if point_reduction == "mean":
        aggregation = lambda x: torch.nanmean(x, dim=-1)
    elif point_reduction == "sum":
        aggregation = lambda x: torch.nansum(x, dim=-1)
    else:
        raise ValueError(f"Unknown point reduction {point_reduction}")
    # Replace nan values with 0, so that the gradient can be computed
    nan_mask = torch.isnan(padded_pointcloud2)
    padded_pointcloud2 = torch.where(nan_mask, torch.zeros_like(padded_pointcloud2), padded_pointcloud2)
    # Compute pairwise distance matrices
    dist_matrix = torch.cdist(pointcloud1, padded_pointcloud2) ** 2  # shape (Batch, P1, P2)
    if density_aware:
        dist_matrix = 1 - torch.exp(-dist_matrix)
    # now replace the indices where originally was nan with inf. We first have to create a mask for the output
    # remove the D dimension (since it is removed during the cdist computation)
    dist_matrix_nan_mask = torch.any(nan_mask, dim=-1)
    # get the size of the first point cloud and repeat the mask along this dimension
    pc1_size = pointcloud1.shape[-2]
    dist_matrix_nan_mask = dist_matrix_nan_mask[:, None, :]
    dist_matrix_nan_mask = dist_matrix_nan_mask.repeat(1, pc1_size, 1)
    # get the batch shape of the first point cloud and repeat the mask along this dimension
    pc1_batch_shape = pointcloud1.shape[:-2]
    # the last batch dimension of the first point cloud has to be the same as the batch dimension of the second point cloud
    assert pc1_batch_shape[-1] == padded_pointcloud2.shape[0]
    if len(pc1_batch_shape) == 1:
        # we are done
        pass
    elif len(pc1_batch_shape) == 2:
        # repeat this additional batch dimension
        dist_matrix_nan_mask = dist_matrix_nan_mask[None, :, :, :]
        dist_matrix_nan_mask = dist_matrix_nan_mask.repeat(pc1_batch_shape[0], 1, 1, 1)
    else:
        raise NotImplementedError(f"Batch shape {pc1_batch_shape} not supported, only batch with 1 or 2 dims are supported.")
    # now replace the indices where originally was nan with inf
    dist_matrix[dist_matrix_nan_mask] = float("inf")

    # # replace the nan values to infinity to take the correct min
    # dist_matrix = torch.nan_to_num(dist_matrix, nan=float("inf"))

    # Compute the minimum distance from points in pointcloud1 to pointcloud2 and vice versa
    forward_distance, _ = torch.min(dist_matrix, dim=-1)  # shape (Batch, P1,)
    # replace the inf values to nan, to take the correct aggregation
    forward_distance = torch.nan_to_num(forward_distance, posinf=float("nan"))
    forward_distance = aggregation(forward_distance)  # shape (Batch,)

    if forward_only:
        chamfer_dist = forward_distance
    else:
        # Average the minimum distances
        backward_distance, _ = torch.min(dist_matrix, dim=-2)  # shape (Batch, P2,)
        # replace the inf values to nan, to take the correct aggregation
        backward_distance = torch.nan_to_num(backward_distance, posinf=float("nan"))
        backward_distance = aggregation(backward_distance)  # shape (Batch,)
        chamfer_dist = 0.5 * (forward_distance + backward_distance)  # shape (Batch,)
    return chamfer_dist

def chamfer_distance(pointcloud1: torch.Tensor, pointcloud2: torch.Tensor,
                     density_aware: bool = False,
                     forward_only: bool = False,
                     point_reduction: str = "mean") -> torch.Tensor:
    """
    Compute the Chamfer Distance between two point clouds. No Nan values are allowed.
    Args:
        pointcloud1 (torch.Tensor): Point cloud 1, shape ([Batch_dims], P1, D)
        pointcloud2 (torch.Tensor): Point cloud 2, shape (Single_Batch_dim, P2, D) Single_Batch_dim has to be the
                                           shape of the last batch dimension of pointcloud1.
        density_aware (bool): If True, the Chamfer Distance is "density aware", i.e.
          the distance is scaled exponentially. Instead of d(x,y) = sqrt((x-y)^2), the distance is
            d(x,y) = 1-exp(-((x-y)^2)). This is useful if the point clouds have different densities.
        forward_only (bool): If True, only compute the forward distance (from pointcloud1 to pointcloud2).
        point_reduction (str): How to reduce the point dimension. Can be "mean" or "sum".
    Returns:
        chamfer_dist (torch.Tensor): Shape ([Batch_dims]) tensor representing the Chamfer Distance between the point clouds
          of each batch.
    """
    if point_reduction == "mean":
        aggregation = lambda x: torch.nanmean(x, dim=-1)
    elif point_reduction == "sum":
        aggregation = lambda x: torch.nansum(x, dim=-1)
    else:
        raise ValueError(f"Unknown point reduction {point_reduction}")
    # Compute pairwise distance matrices
    dist_matrix = torch.cdist(pointcloud1, pointcloud2) ** 2  # shape (Batch, P1, P2)
    if density_aware:
        dist_matrix = 1 - torch.exp(-dist_matrix)
    # Compute the minimum distance from points in pointcloud1 to pointcloud2 and vice versa
    forward_distance, _ = torch.min(dist_matrix, dim=-1)  # shape (Batch, P1,)
    forward_distance = aggregation(forward_distance)  # shape (Batch,)
    if forward_only:
        chamfer_dist = forward_distance
    else:
        # Average the minimum distances
        backward_distance, _ = torch.min(dist_matrix, dim=-2)  # shape (Batch, P2,)
        backward_distance = aggregation(backward_distance)  # shape (Batch,)
        chamfer_dist = 0.5 * (forward_distance + backward_distance)  # shape (Batch,)
    return chamfer_dist