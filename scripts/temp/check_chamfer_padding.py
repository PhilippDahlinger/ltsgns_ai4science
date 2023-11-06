# TODO: test the chamfer distance with padded point clouds

import torch

from lts_gns.architectures.util.chamfer_distance import padded_chamfer_distance, chamfer_distance

torch.manual_seed(42)

pc1a = torch.rand(15, 2)
pc1b = torch.rand(13, 2)
pc2 = torch.rand(2, 6, 2)
pc2b = pc2.clone()
pc2.requires_grad_()
pc2b.requires_grad_()
# print(chamfer_distance(pc2, pc1a,  density_aware=True, forward_only=False, point_reduction="mean"))
# print(chamfer_distance(pc2, pc1b, density_aware=True, forward_only=False, point_reduction="mean"))

point_cloud_positions = [pc1a, pc1b]
# point_cloud_positions = [pc1a]

padded_point_cloud_positions = torch.full((len(point_cloud_positions), 16, point_cloud_positions[0].shape[-1]), float("nan"))
for idx, point_cloud in enumerate(point_cloud_positions):
    padded_point_cloud_positions[idx, :point_cloud.shape[0]] = point_cloud
    torch.sum(chamfer_distance(pc2[idx], point_cloud, density_aware=True, forward_only=False, point_reduction="mean")).backward()
print(pc2.grad)
result = padded_chamfer_distance(pc2b, padded_point_cloud_positions, density_aware=True, forward_only=False, point_reduction="mean")
torch.sum(result).backward()
print(pc2b.grad)
