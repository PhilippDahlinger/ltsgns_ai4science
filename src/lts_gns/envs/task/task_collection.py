from typing import List

import numpy as np

from lts_gns.envs.task.task import Task


class TaskCollection:
    def __init__(self, task_list: List[Task]):
        self._task_list = task_list
        self._subtask_list = []
        self._subtask_mapping = []  # mapping from subtask index to task index
        self._subtask_positions = []  # mapping from subtask index to starting position in task

    def create_subtasks(self, subtasks_per_task: int, min_subtask_length: int,
                        max_subtask_length: int, sampling_type: str) -> None:
        """
        Creates subtasks from the original tasks. Sorts the subtasks by length.
        Saves both the subtask list and a mapping back to the original tasks internally.
        Args:
            subtasks_per_task: Number of subtasks to create per task
            min_subtask_length: Minimum length of the subtask
            max_subtask_length: Maximum length of the subtask
            sampling_type: Sampling type of the subtask. Either "random" or "slice"

        Returns:

        """
        assert len(self._subtask_list) == 0, "Already created subtasks."
        for task_position, task in enumerate(self._task_list):
            for n_aux_tasks_per_task in range(subtasks_per_task):
                subtask, starting_position = task.split(min_length=min_subtask_length,
                                                        max_length=max_subtask_length,
                                                        sampling_type=sampling_type)
                self._subtask_list.append(subtask)
                self._subtask_mapping.append(task_position)
                if sampling_type == "slice":
                    self._subtask_positions.append(starting_position)

        self._subtask_mapping = np.array(self._subtask_mapping)
        if len(self._subtask_positions) > 0:
            self._subtask_positions = np.array(self._subtask_positions)

    @property
    def subtask_list(self) -> List[Task]:
        """
        A list of sub-tasks that are created from the original list of tasks
        Returns:

        """
        assert len(self._subtask_list) > 0, "No subtasks created yet. Call create_subtasks first."
        return self._subtask_list

    @property
    def subtask_mapping(self) -> np.array:
        """
        A mapping from the subtask to the original task. Has shape (len(self.subtask_list), ), where each entry is the
        index of the original task, starting at 0.
        Returns:

        """
        return self._subtask_mapping

    def map_subtasks_to_tasks(self, subtask_index: int | List[int] | np.array) -> Task | List[Task]:
        """
        Maps subtask indices to the original tasks.
        """
        if isinstance(subtask_index, (int, np.int32)):
            return self._task_list[self.subtask_mapping[subtask_index]]
        elif isinstance(subtask_index, list) or isinstance(subtask_index, np.ndarray):
            return [self._task_list[self.subtask_mapping[i]] for i in subtask_index]
        else:
            raise ValueError(f"Invalid type for subtask_index: {type(subtask_index)}")

    @property
    def subtask_positions(self) -> np.array:
        """
        Returns the *starting* positions of the subtasks in the original task.
        Returns:

        """
        assert len(self._subtask_positions) > 0, "Subtasks have no starting positions. " \
                                                 "Make sure to use 'slice' as a sampling_type to sample them"
        return self._subtask_positions

    @property
    def task_list(self) -> List[Task]:
        return self._task_list

    def __len__(self) -> int:
        return len(self._task_list)

    def __getitem__(self, item) -> Task:
        return self._task_list[item]

    def __repr__(self) -> str:
        return f"List of {len(self)} Tasks, with Task0='{self[0]}'"
