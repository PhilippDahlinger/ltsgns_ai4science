from typing import Optional, Tuple, Union, List

import torch

from lts_gns.architectures.normalizers.world_to_model_normalizer import WorldToModelNormalizer
from lts_gns.util import keys


def compute_model_velocities(data_dict, world_to_model_normalizer: WorldToModelNormalizer, dt=1.0):
    world_velocities = (data_dict[keys.NEXT_MESH_POS] - data_dict[keys.MESH]) / dt
    # have the normalized velocities (e.g. in model space) as labels
    model_velocities = world_to_model_normalizer(world_velocities)
    return model_velocities


def get_one_hot_features_and_types(input_list: Union[List[int], List[List[int]]],
                                   device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds one-hot feature tensor indicating the edge/node type from a list of numbers per type or indices per type.
    Assumes that the types are ordered in the same way as the input list.

    Args:
        input_list: List of numbers of nodes per type or list of lists of indices per type
        device: Device to compute on

    Returns: A tuple of two tensors containing the one-hot features and the type as number
        features: One-hot features Tensor, e.g., "(0,0,1,0)" for a node of type 2
        types: Tensor containing the type as number, e.g., "2" for a node of type 2
    """
    if isinstance(input_list[0], int):  # the input is a list of counts
        counts = input_list
        indices_per_type = [list(range(sum(counts[:i]), sum(counts[:i+1]))) for i in range(len(counts))]
    else:  # the input is a list of list of indices
        indices_per_type = input_list
    total_num = sum(len(indices) for indices in indices_per_type)
    num_types = len(indices_per_type)
    features = torch.zeros(total_num, num_types, device=device)
    types = torch.zeros(total_num, device=device)

    for type_idx, indices in enumerate(indices_per_type):
        features[indices, type_idx] = 1
        types[indices] = type_idx
    return features, types
