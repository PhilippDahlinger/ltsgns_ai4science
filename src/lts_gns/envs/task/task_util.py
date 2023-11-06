from __future__ import annotations

from typing import Tuple

import torch


def generate_context_target_subindices(l_task: int, l_subtask: int,
                                       sampling_type: str) -> Tuple[torch.Tensor, int | None]:
    """
    Generates the indices for the context and target subtask. The index for the target is always the complement of the
    context indices, i.e., we want to predict everything that we do not know from the context
    Args:
        l_task: The length of the trajectory to sub-sample from
        l_subtask: The length of the subtask
        sampling_type: The sampling type. Either 
            * "uniform" for random sampling of length l_subtask indices or
            * "slice" for a random slice of length l_subtask
            * "slice_from_start" for a slice of length l_subtask starting at index 0

    Returns: A tuple of (context_indices, starting_index|None)

    """
    # todo @niklas the target index is never used. Do we still need that?
    if sampling_type == "uniform":
        random_perm = torch.randperm(l_task)
        sub_index_context = random_perm[:l_subtask]
        # sub_index_target = random_perm[l_subtask:]
        start_idx = None
    elif sampling_type == "slice":
        start_idx = torch.randint(0, l_task - l_subtask + 1, (1,)).item()
        sub_index_context = torch.arange(start_idx, start_idx + l_subtask)
        # sub_index_target = torch.cat((torch.arange(start_idx), torch.arange(start_idx + l_subtask, l_task)))
    elif sampling_type == "slice_from_start":
        start_idx = 0
        sub_index_context = torch.arange(start_idx, start_idx + l_subtask)
    else:
        raise ValueError(f"Unknown sampling type {sampling_type}")
    # return sub_index_context, sub_index_target
    return sub_index_context, start_idx
