"""Core data models for delegato — Pydantic models, enums, and validators."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VerificationMethod(str, Enum):
    LLM_JUDGE = "llm_judge"
    REGEX = "regex"
    SCHEMA = "schema"
    FUNCTION = "function"
    NONE = "none"


class Reversibility(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DelegationEventType(str, Enum):
    TASK_DECOMPOSED = "task_decomposed"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"
    TRUST_UPDATED = "trust_updated"
    TRUST_CIRCUIT_BREAK = "trust_circuit_break"
    TASK_REASSIGNED = "task_reassigned"
    ESCALATED = "escalated"


# ── Models ───────────────────────────────────────────────────────────────────


class Permission(BaseModel):
    resource: str
    action: str
    scope: str = "*"
    expiry: datetime | None = None


class VerificationSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: VerificationMethod
    criteria: str = ""
    json_schema: dict | None = None
    custom_fn: Callable | None = None
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    judges: int = Field(default=1, ge=1)
    consensus_threshold: float = Field(default=0.66, ge=0.0, le=1.0)


class TaskResult(BaseModel):
    task_id: str
    agent_id: str
    output: Any = None
    success: bool
    verified: bool = False
    verification_details: str = ""
    cost: float = 0.0
    delegation_overhead: float = 0.0
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Task(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    required_capabilities: list[str] = []
    verification: VerificationSpec
    parent_id: str | None = None
    priority: int = Field(default=3, ge=1, le=5)
    complexity: int = Field(default=3, ge=1, le=5)
    reversibility: Reversibility = Reversibility.MEDIUM
    max_retries: int = Field(default=2, ge=0)
    timeout_seconds: float = 60.0
    metadata: dict = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING

    @field_validator("required_capabilities")
    @classmethod
    def capabilities_non_empty_strings(cls, v: list[str]) -> list[str]:
        if any(not cap.strip() for cap in v):
            raise ValueError("Capability names must be non-empty strings")
        return v


class Contract(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    task: Task
    agent_id: str
    permissions: list[Permission] = []
    verification: VerificationSpec
    monitoring_interval: float = 5.0
    max_cost: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    result: TaskResult | None = None
    attempt: int = Field(default=1, ge=1)


class Agent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    capabilities: list[str]
    handler: Callable[[Task], Awaitable[TaskResult]]
    trust_scores: dict[str, float] = Field(default_factory=dict)
    transparency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    max_concurrent: int = Field(default=1, ge=1)
    current_load: int = Field(default=0, ge=0)
    metadata: dict = Field(default_factory=dict)

    @field_validator("trust_scores")
    @classmethod
    def trust_scores_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for cap, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Trust score for '{cap}' must be between 0.0 and 1.0, got {score}"
                )
        return v


class DelegationEvent(BaseModel):
    type: DelegationEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task_id: str | None = None
    agent_id: str | None = None
    data: dict = Field(default_factory=dict)


class AuditEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str
    task_id: str | None = None
    agent_id: str | None = None
    details: dict = Field(default_factory=dict)


class TrustRecord(BaseModel):
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DelegationResult(BaseModel):
    task: Task
    success: bool
    output: Any = None
    subtask_results: list[TaskResult] = []
    audit_log: list[AuditEntry] = []
    total_cost: float = 0.0
    total_delegation_overhead: float = 0.0
    total_duration: float = 0.0
    reassignments: int = 0


class TaskDAG(BaseModel):
    """Directed acyclic graph of tasks with dependency edges."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tasks: dict[str, Task] = Field(default_factory=dict)
    dependencies: dict[str, list[str]] = Field(default_factory=dict)
    dependents: dict[str, list[str]] = Field(default_factory=dict)
    root_task_id: str | None = None

    def add_task(self, task: Task, depends_on: list[str] | None = None) -> None:
        """Add a task to the DAG with optional dependency edges."""
        self.tasks[task.id] = task
        self.dependencies[task.id] = depends_on or []
        if task.id not in self.dependents:
            self.dependents[task.id] = []

        for dep_id in self.dependencies[task.id]:
            if dep_id not in self.tasks:
                raise KeyError(f"Dependency task '{dep_id}' not found in DAG")
            if dep_id not in self.dependents:
                self.dependents[dep_id] = []
            self.dependents[dep_id].append(task.id)

        self._check_cycle(task.id)

    def _check_cycle(self, start_id: str) -> None:
        """Detect cycles using DFS from the newly added node."""
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            stack.add(node_id)
            for dep_id in self.dependents.get(node_id, []):
                if dep_id not in visited:
                    if dfs(dep_id):
                        return True
                elif dep_id in stack:
                    return True
            stack.discard(node_id)
            return False

        if dfs(start_id):
            # Roll back: remove the task that caused the cycle
            for dep_id in self.dependencies.get(start_id, []):
                if start_id in self.dependents.get(dep_id, []):
                    self.dependents[dep_id].remove(start_id)
            del self.tasks[start_id]
            del self.dependencies[start_id]
            if start_id in self.dependents:
                del self.dependents[start_id]
            raise ValueError(f"Adding task '{start_id}' would create a cycle")

    def get_task(self, task_id: str) -> Task:
        """Retrieve a task by ID."""
        if task_id not in self.tasks:
            raise KeyError(f"Task '{task_id}' not found in DAG")
        return self.tasks[task_id]

    def get_ready_tasks(self, completed: set[str] | None = None) -> list[Task]:
        """Return tasks whose dependencies are all satisfied."""
        completed = completed or set()
        ready = []
        for task_id, task in self.tasks.items():
            if task_id in completed:
                continue
            deps = self.dependencies.get(task_id, [])
            if all(dep_id in completed for dep_id in deps):
                ready.append(task)
        return ready

    def topological_sort(self) -> list[Task]:
        """Return tasks in topological order using Kahn's algorithm."""
        in_degree: dict[str, int] = {
            tid: len(deps) for tid, deps in self.dependencies.items()
        }
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result: list[Task] = []

        while queue:
            node_id = queue.pop(0)
            result.append(self.tasks[node_id])
            for dep_id in self.dependents.get(node_id, []):
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

        if len(result) != len(self.tasks):
            raise ValueError("Cycle detected in TaskDAG during topological sort")

        return result

    def get_all_tasks(self) -> list[Task]:
        """Return all tasks in the DAG."""
        return list(self.tasks.values())
