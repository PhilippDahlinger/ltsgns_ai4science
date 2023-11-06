from __future__ import annotations  # necessary import for the create_subtask return typehint to work

from typing import List, Tuple

import torch
from torch_geometric.data import Data

from lts_gns.envs.task.task_util import generate_context_target_subindices


class Task:
    """
    Object which captures a List of Data elements (usually one trajectory). Can be used to create subtasks and
    construct a dataloader from it.
    """

    def __init__(self, trajectory: List[Data]):
        self._trajectory: List[Data] = trajectory

    @property
    def trajectory(self) -> List[Data]:
        return self._trajectory

    def split(self, min_length: int, max_length: int, sampling_type: str) -> Tuple[Task, int | None]:
        """
        Creates auxiliary tasks (e.g. a cutout of the whole trajectory) according to the given length and sampling type.
            Is called from GNSEnvironment
        Args:
            min_length: Minimum length of the subtask
            max_length: Maximum length of the subtask
            sampling_type: Sampling type of the subtask. Either "random" or "slice"

        Returns: A tuple of (Split_task, starting_position)

        """
        task_length = torch.randint(low=min_length, high=max_length + 1, size=(1,)).item()
        sub_index_context, start_position = generate_context_target_subindices(l_task=len(self.trajectory),
                                                                               l_subtask=task_length,
                                                                               sampling_type=sampling_type)
        sub_context_trajectory = [self.trajectory[i] for i in sub_index_context]
        return Task(sub_context_trajectory), start_position

    def get_subtask(self, start_idx: int, end_idx: int, deepcopy: bool = False,
                    device: str | torch.device = None) -> Task:
        """
        Returns a subtask of the current task with specific start and stop indices.
        """
        # todo factor this out?
        if deepcopy:
            sub_context_trajectory = [data.clone() for data in self.trajectory[start_idx:end_idx]]
        else:
            sub_context_trajectory = self.trajectory[start_idx:end_idx]

        task = Task(sub_context_trajectory)
        if device is not None:
            task.to(device)
        return task

    def to(self, device: torch.device | str) -> None:
        """
        Move this task to the given device.
        Args:
            device:

        Returns:

        """
        for data in self.trajectory:
            data.to(device)

    def print_property(self, property_name: str) -> None:
        """
        Prints the given property of all datapoints in the trajectory. Useful for debugging purposes.
        Args:
            property_name: Name of the property to print

        Returns: None

        """
        print(f"Printing {property_name} of all datapoints in the trajectory")
        for data in self.trajectory:
            print(data[property_name])

    def __len__(self) -> int:
        return len(self.trajectory)

    def __getitem__(self, item):
        return self.trajectory[item]

    def __repr__(self) -> str:
        return f"Task with {len(self)} datapoints like: {self[0]}"
