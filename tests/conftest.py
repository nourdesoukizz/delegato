"""Shared fixtures and factory functions for delegato tests."""

from __future__ import annotations

import pytest

from delegato.assignment import AssignmentScorer
from delegato.audit import AuditLog
from delegato.events import EventBus
from delegato.models import (
    Agent,
    Reversibility,
    Task,
    TaskResult,
    VerificationMethod,
    VerificationSpec,
)
from delegato.trust import TrustTracker


# ── Factory functions ────────────────────────────────────────────────────────


def make_task(
    task_id: str = "t1",
    goal: str = "Test task",
    capabilities: list[str] | None = None,
    method: VerificationMethod = VerificationMethod.NONE,
    criteria: str = "",
    complexity: int = 3,
    reversibility: Reversibility = Reversibility.MEDIUM,
    max_retries: int = 2,
    timeout: float = 60.0,
    **kwargs,
) -> Task:
    return Task(
        id=task_id,
        goal=goal,
        required_capabilities=["code"] if capabilities is None else capabilities,
        verification=VerificationSpec(method=method, criteria=criteria),
        complexity=complexity,
        reversibility=reversibility,
        max_retries=max_retries,
        timeout_seconds=timeout,
        **kwargs,
    )


def make_agent(
    agent_id: str = "a1",
    capabilities: list[str] | None = None,
    handler=None,
    max_concurrent: int = 1,
    current_load: int = 0,
    metadata: dict | None = None,
) -> Agent:
    async def default_handler(task):
        return TaskResult(
            task_id=task.id, agent_id=agent_id, output="done", success=True
        )

    return Agent(
        id=agent_id,
        name=f"Agent {agent_id}",
        capabilities=capabilities or ["code"],
        handler=handler or default_handler,
        max_concurrent=max_concurrent,
        current_load=current_load,
        metadata=metadata or {},
    )


def make_result(
    task_id: str = "t1",
    agent_id: str = "a1",
    output="done",
    success: bool = True,
    **kwargs,
) -> TaskResult:
    return TaskResult(
        task_id=task_id, agent_id=agent_id, output=output, success=success, **kwargs
    )


def mock_decompose_llm(subtasks: list[dict] | None = None):
    """Return a mock LLM callable that produces decomposition output."""
    if subtasks is None:
        subtasks = [
            {
                "goal": "Sub-task 1",
                "required_capabilities": ["code"],
                "verification_method": "none",
                "verification_criteria": "",
                "dependencies": [],
            }
        ]

    async def mock_call(messages):
        return {"subtasks": subtasks}

    return mock_call


def mock_verify_llm(score: float = 1.0, reasoning: str = "ok"):
    """Return a mock LLM callable that produces verification output."""

    async def mock_call(messages):
        return {"score": score, "reasoning": reasoning}

    return mock_call


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def trust_tracker(event_bus: EventBus) -> TrustTracker:
    return TrustTracker(event_bus=event_bus)


@pytest.fixture
def audit_log(event_bus: EventBus) -> AuditLog:
    return AuditLog(event_bus=event_bus)


@pytest.fixture
def assignment_scorer() -> AssignmentScorer:
    return AssignmentScorer()
