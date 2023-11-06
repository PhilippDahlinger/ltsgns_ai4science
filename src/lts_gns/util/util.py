from functools import update_wrapper, partial
from typing import List, Any, Callable, Dict, Iterable

import numpy as np
import torch
from torch_geometric.data import Data, Batch


def bisect(elt_list: List[Any], query: int, key=Callable[[Any], int]) -> int:
    """
    Given a sorted (small to high) list of elements, return the index of the element which key is smaller or equal than
    the query. Uses a bisection algorithm.
    Raises an exception if no element is found.
    :param elt_list:
    :param query:
    :param key:
    :return:
    """
    if len(elt_list) == 0:
        raise ValueError("Empty list")
    if len(elt_list) == 1:
        if key(elt_list[0]) <= query:
            return 0
        else:
            raise ValueError("No element found")
    else:
        mid_index = len(elt_list) // 2
        mid_elt = elt_list[mid_index]
        if key(mid_elt) <= query:
            return mid_index + bisect(elt_list[mid_index:], query, key)
        else:
            return bisect(elt_list[:mid_index], query, key)


def to_numpy(dict_or_tensor: torch.Tensor | Dict[str, torch.Tensor]) -> np.array:
    """
    Converts a tensor to a numpy array
    """
    if isinstance(dict_or_tensor, dict):
        return {key: to_numpy(value) for key, value in dict_or_tensor.items()}
    else:
        return dict_or_tensor.detach().cpu().numpy()


def prefix_dict(dictionary: Dict[str, Any], prefix: str, separator: str = "_") -> Dict[str, Any]:
    """
    Prefixes all keys of a dictionary with a given prefix
    Args:
        prefix:
        dictionary:
        separator: String/character to separate prefix and key

    Returns:

    """
    return {prefix + separator + key: value for key, value in dictionary.items()}


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def polyak_update(
        params: Iterable[torch.nn.Parameter],
        target_params: Iterable[torch.nn.Parameter],
        tau: float,
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93

    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    from itertools import zip_longest
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def node_type_mask(graph_or_batch: Batch | Data, key: str) -> torch.Tensor:
    if isinstance(graph_or_batch, Batch):
        mask: torch.Tensor = graph_or_batch.node_type == graph_or_batch.node_type_description[0].index(key)
    elif isinstance(graph_or_batch, Data):
        mask: torch.Tensor = graph_or_batch.node_type == graph_or_batch.node_type_description.index(key)
    else:
        raise ValueError(f"Unknown type of batch: {type(graph_or_batch)}")
    return mask


def edge_type_mask(graph_or_batch: Batch | Data, key: str) -> torch.Tensor:
    if isinstance(graph_or_batch, Batch):
        mask: torch.Tensor = graph_or_batch.edge_type == graph_or_batch.edge_type_description[0].index(key)
    elif isinstance(graph_or_batch, Data):
        mask: torch.Tensor = graph_or_batch.edge_type == graph_or_batch.edge_type_description.index(key)
    else:
        raise ValueError(f"Unknown type of batch: {type(graph_or_batch)}")
    return mask


def print_gpu_mem_snapshot(header: str = ""):
    import nvidia_smi

    nvidia_smi.nvmlInit()

    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(header)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

    nvidia_smi.nvmlShutdown()
