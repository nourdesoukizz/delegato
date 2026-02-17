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
