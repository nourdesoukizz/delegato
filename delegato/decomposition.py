"""LLM-powered recursive task decomposition engine."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from delegato.llm import LLMError, complete_json
from delegato.models import Task, TaskDAG, VerificationMethod, VerificationSpec

logger = logging.getLogger(__name__)

DECOMPOSITION_SYSTEM_PROMPT = """\
You are a task decomposition engine. Given a high-level goal, break it into \
the smallest independent sub-tasks that can be:
1. Assigned to a single agent
2. Verified with a concrete acceptance criterion

For each sub-task, specify:
- goal: what needs to be done
- required_capabilities: list of skills needed (strings)
- verification_method: how to check the output (one of: llm_judge, regex, schema, function, none)
- verification_criteria: specific description of what "correct" looks like
- dependencies: list of indices (0-based) of other sub-tasks that must complete first

Return JSON: {"subtasks": [...]}\
"""


class DecompositionError(Exception):
    """Raised when task decomposition fails."""


class DecompositionEngine:
    """Decomposes high-level tasks into sub-task DAGs using an LLM."""

    def __init__(
        self,
        *,
        model: str = "openai/gpt-4o",
        max_depth: int = 3,
        max_subtasks: int = 6,
        llm_call: Callable[..., Any] | None = None,
    ):
        self.model = model
        self.max_depth = max_depth
        self.max_subtasks = max_subtasks
        self._llm_call = llm_call or self._default_llm_call

    async def _default_llm_call(self, messages: list[dict[str, str]]) -> dict:
        """Default LLM call using complete_json."""
        return await complete_json(model=self.model, messages=messages)

    async def decompose(self, task: Task) -> TaskDAG:
        """Decompose a task into a DAG of sub-tasks."""
        dag = TaskDAG(root_task_id=task.id)
        dag.add_task(task)
        await self._decompose_recursive(task, dag, depth=0)
        return dag

    async def _decompose_recursive(
        self, task: Task, dag: TaskDAG, depth: int
    ) -> None:
        """Recursively decompose a task, adding sub-tasks to the DAG."""
        if depth >= self.max_depth:
            return

        messages = [
            {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Decompose this task: {task.goal}"},
        ]

        try:
            response = await self._llm_call(messages)
        except (LLMError, Exception) as exc:
            raise DecompositionError(
                f"LLM call failed during decomposition: {exc}"
            ) from exc

        subtask_defs = self._parse_response(response)
        if not subtask_defs:
            return

        # Enforce max_subtasks limit
        subtask_defs = subtask_defs[: self.max_subtasks]

        # Create sub-tasks and collect them for dependency resolution
        created_tasks: list[Task] = []
        for subtask_def in subtask_defs:
            goal = subtask_def.get("goal", "").strip()
            if not goal:
                logger.warning("Skipping sub-task with empty goal")
                continue

            verification_method = self._resolve_verification_method(
                subtask_def.get("verification_method", "none")
            )

            subtask = Task(
                goal=goal,
                required_capabilities=subtask_def.get("required_capabilities", []),
                verification=VerificationSpec(
                    method=verification_method,
                    criteria=subtask_def.get("verification_criteria", ""),
                ),
                parent_id=task.id,
                priority=task.priority,
                reversibility=task.reversibility,
                complexity=max(1, task.complexity - 1),
            )
            created_tasks.append(subtask)

        # Add tasks to DAG with dependency resolution
        for idx, subtask in enumerate(created_tasks):
            raw_deps = subtask_defs[idx].get("dependencies", [])
            depends_on = self._resolve_dependencies(raw_deps, created_tasks)
            dag.add_task(subtask, depends_on=depends_on)

        # Recurse on sub-tasks with NONE verification
        for subtask in created_tasks:
            if subtask.verification.method == VerificationMethod.NONE:
                await self._decompose_recursive(subtask, dag, depth + 1)

    def _parse_response(self, response: Any) -> list[dict]:
        """Extract subtask definitions from LLM response."""
        if isinstance(response, dict):
            subtasks = response.get("subtasks", [])
            if isinstance(subtasks, list):
                return subtasks
        return []

    def _resolve_verification_method(self, method_str: str) -> VerificationMethod:
        """Convert string to VerificationMethod, defaulting to LLM_JUDGE on invalid."""
        try:
            return VerificationMethod(method_str)
        except ValueError:
            logger.warning(
                "Invalid verification method '%s', falling back to LLM_JUDGE",
                method_str,
            )
            return VerificationMethod.LLM_JUDGE

    def _resolve_dependencies(
        self, raw_deps: list, created_tasks: list[Task]
    ) -> list[str]:
        """Convert index-based dependencies to task IDs, skipping invalid indices."""
        resolved = []
        for dep_idx in raw_deps:
            if isinstance(dep_idx, int) and 0 <= dep_idx < len(created_tasks):
                resolved.append(created_tasks[dep_idx].id)
            else:
                logger.warning("Skipping invalid dependency index: %s", dep_idx)
        return resolved
