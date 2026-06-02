"""Abstract task/dataset interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from roundtable.core.types import TaskInput


class TaskDataset(ABC):
    """Abstract base class for task datasets.

    Subclass this to load your own evaluation data. A dataset yields
    TaskInput objects that are fed to the RoundtablePolicy.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of tasks in the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> TaskInput:
        """Get a task by index."""
        ...

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def get_dimensions(self) -> list[str]:
        """Return the evaluation dimensions for this dataset.

        These become the columns of the confidence-weight table.
        For ScienceEval, these are the subtask domains (physics, biology, etc.).
        For ScienceNarrative, these are rubric dimensions
        (creativity, scientific rigor, logical coherence, etc.).

        Returns:
            List of dimension names.
        """
        ...
