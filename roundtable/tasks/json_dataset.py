"""Generic JSON dataset loader for custom tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roundtable.core.types import TaskInput
from roundtable.tasks.base import TaskDataset


class JsonDataset(TaskDataset):
    """Generic dataset loaded from a JSON file.

    This is the simplest way to bring your own tasks into Roundtable Policy.

    Expected data format (JSON):
        [
            {
                "query": "Your question or prompt here",
                "task_id": "optional_id",
                "domain": "optional_domain",
                "reference_answer": "optional ground truth",
            },
            ...
        ]

    Args:
        data_path: Path to the JSON file.
        dimensions: Evaluation dimensions for the confidence table.
        query_key: JSON key for the query text (default "query").
    """

    def __init__(
        self,
        data_path: str | Path,
        dimensions: list[str] | None = None,
        query_key: str = "query",
    ) -> None:
        raw: list[dict[str, Any]] = json.loads(Path(data_path).read_text())

        self._tasks: list[TaskInput] = []
        for item in raw:
            self._tasks.append(
                TaskInput(
                    query=item[query_key],
                    task_id=item.get("task_id", ""),
                    domain=item.get("domain", ""),
                    reference_answer=item.get("reference_answer"),
                    metadata={
                        k: v for k, v in item.items()
                        if k not in {query_key, "task_id", "domain", "reference_answer"}
                    },
                )
            )

        self._dimensions = dimensions or ["default"]

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, index: int) -> TaskInput:
        return self._tasks[index]

    def get_dimensions(self) -> list[str]:
        return list(self._dimensions)
