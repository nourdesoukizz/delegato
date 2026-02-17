"""Tests for delegato core models, enums, and validators."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from delegato.models import (
    Agent,
    AuditEntry,
    Contract,
    DelegationEvent,
    DelegationEventType,
    DelegationResult,
    Permission,
    Reversibility,
    Task,
    TaskResult,
    TaskStatus,
    TrustRecord,
    VerificationMethod,
    VerificationSpec,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _dummy_handler(task: Task) -> TaskResult:
    return TaskResult(task_id=task.id, agent_id="test", success=True, output="ok")


def _make_verification(**kwargs) -> VerificationSpec:
    defaults = {"method": VerificationMethod.NONE}
    defaults.update(kwargs)
    return VerificationSpec(**defaults)


def _make_task(**kwargs) -> Task:
    defaults = {"goal": "test task", "verification": _make_verification()}
    defaults.update(kwargs)
    return Task(**defaults)


def _make_agent(**kwargs) -> Agent:
    defaults = {
        "name": "test-agent",
        "capabilities": ["testing"],
        "handler": _dummy_handler,
    }
    defaults.update(kwargs)
    return Agent(**defaults)


# ── Enum Tests ───────────────────────────────────────────────────────────────


class TestEnums:
    def test_task_status_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.ASSIGNED == "assigned"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"

    def test_verification_method_values(self):
        assert VerificationMethod.LLM_JUDGE == "llm_judge"
        assert VerificationMethod.REGEX == "regex"
        assert VerificationMethod.SCHEMA == "schema"
        assert VerificationMethod.FUNCTION == "function"
        assert VerificationMethod.NONE == "none"

    def test_reversibility_values(self):
        assert Reversibility.HIGH == "high"
        assert Reversibility.MEDIUM == "medium"
        assert Reversibility.LOW == "low"

    def test_delegation_event_type_values(self):
        assert DelegationEventType.TASK_DECOMPOSED == "task_decomposed"
        assert DelegationEventType.TRUST_CIRCUIT_BREAK == "trust_circuit_break"
        assert DelegationEventType.ESCALATED == "escalated"


# ── Permission Tests ─────────────────────────────────────────────────────────


class TestPermission:
    def test_create_permission(self):
        p = Permission(resource="filesystem", action="read")
        assert p.resource == "filesystem"
        assert p.action == "read"
        assert p.scope == "*"
        assert p.expiry is None

    def test_permission_with_scope(self):
        p = Permission(resource="api:github", action="write", scope="repos/myorg/*")
        assert p.scope == "repos/myorg/*"

    def test_permission_json_roundtrip(self):
        p = Permission(resource="network", action="execute", scope="/tmp/*")
        data = p.model_dump_json()
        p2 = Permission.model_validate_json(data)
        assert p == p2


# ── VerificationSpec Tests ───────────────────────────────────────────────────


class TestVerificationSpec:
    def test_defaults(self):
        v = VerificationSpec(method=VerificationMethod.LLM_JUDGE)
        assert v.threshold == 0.7
        assert v.judges == 1
        assert v.consensus_threshold == 0.66

    def test_threshold_out_of_range(self):
        with pytest.raises(Exception):
            VerificationSpec(method=VerificationMethod.NONE, threshold=1.5)

    def test_threshold_negative(self):
        with pytest.raises(Exception):
            VerificationSpec(method=VerificationMethod.NONE, threshold=-0.1)

    def test_judges_minimum(self):
        with pytest.raises(Exception):
            VerificationSpec(method=VerificationMethod.NONE, judges=0)

    def test_multi_judge_config(self):
        v = VerificationSpec(
            method=VerificationMethod.LLM_JUDGE, judges=3, consensus_threshold=0.8
        )
        assert v.judges == 3
        assert v.consensus_threshold == 0.8


# ── Task Tests ───────────────────────────────────────────────────────────────


class TestTask:
    def test_create_with_defaults(self):
        t = _make_task()
        assert t.priority == 3
        assert t.complexity == 3
        assert t.reversibility == Reversibility.MEDIUM
        assert t.max_retries == 2
        assert t.timeout_seconds == 60.0
        assert t.status == TaskStatus.PENDING
        assert t.id  # auto-generated UUID

    def test_priority_out_of_range_high(self):
        with pytest.raises(Exception):
            _make_task(priority=6)

    def test_priority_out_of_range_low(self):
        with pytest.raises(Exception):
            _make_task(priority=0)

    def test_complexity_out_of_range(self):
        with pytest.raises(Exception):
            _make_task(complexity=10)

    def test_empty_capability_string_rejected(self):
        with pytest.raises(Exception):
            _make_task(required_capabilities=["web_search", ""])

    def test_whitespace_capability_rejected(self):
        with pytest.raises(Exception):
            _make_task(required_capabilities=["  "])

    def test_valid_capabilities(self):
        t = _make_task(required_capabilities=["web_search", "summarization"])
        assert t.required_capabilities == ["web_search", "summarization"]

    def test_json_roundtrip(self):
        t = _make_task(
            goal="test roundtrip",
            required_capabilities=["search"],
            priority=4,
            complexity=2,
        )
        data = t.model_dump_json()
        t2 = Task.model_validate_json(data)
        assert t2.goal == "test roundtrip"
        assert t2.priority == 4
        assert t2.complexity == 2
        assert t2.required_capabilities == ["search"]

    def test_metadata(self):
        t = _make_task(metadata={"key": "value", "count": 42})
        assert t.metadata["key"] == "value"
        assert t.metadata["count"] == 42


# ── TaskResult Tests ─────────────────────────────────────────────────────────


class TestTaskResult:
    def test_create_task_result(self):
        r = TaskResult(task_id="t1", agent_id="a1", success=True, output="done")
        assert r.verified is False
        assert r.cost == 0.0
        assert r.duration_seconds == 0.0

    def test_json_roundtrip(self):
        r = TaskResult(
            task_id="t1",
            agent_id="a1",
            success=False,
            output={"error": "timeout"},
            verified=False,
            verification_details="Agent timed out",
        )
        data = r.model_dump_json()
        r2 = TaskResult.model_validate_json(data)
        assert r2.success is False
        assert r2.output == {"error": "timeout"}


# ── Agent Tests ──────────────────────────────────────────────────────────────


class TestAgent:
    def test_create_agent(self):
        a = _make_agent()
        assert a.name == "test-agent"
        assert a.capabilities == ["testing"]
        assert a.transparency_score == 0.5
        assert a.max_concurrent == 1
        assert a.current_load == 0

    def test_trust_score_out_of_range(self):
        with pytest.raises(Exception):
            _make_agent(trust_scores={"testing": 1.5})

    def test_trust_score_negative(self):
        with pytest.raises(Exception):
            _make_agent(trust_scores={"testing": -0.1})

    def test_valid_trust_scores(self):
        a = _make_agent(trust_scores={"testing": 0.8, "coding": 0.6})
        assert a.trust_scores["testing"] == 0.8

    def test_transparency_out_of_range(self):
        with pytest.raises(Exception):
            _make_agent(transparency_score=1.1)


# ── Contract Tests ───────────────────────────────────────────────────────────


class TestContract:
    def test_create_contract(self):
        t = _make_task()
        v = _make_verification()
        c = Contract(task=t, agent_id="agent-1", verification=v)
        assert c.agent_id == "agent-1"
        assert c.attempt == 1
        assert c.result is None
        assert c.completed_at is None

    def test_contract_with_permissions(self):
        t = _make_task()
        v = _make_verification()
        perms = [Permission(resource="fs", action="read", scope="/data/*")]
        c = Contract(task=t, agent_id="a1", verification=v, permissions=perms)
        assert len(c.permissions) == 1
        assert c.permissions[0].resource == "fs"


# ── DelegationEvent Tests ───────────────────────────────────────────────────


class TestDelegationEvent:
    def test_create_event(self):
        e = DelegationEvent(
            type=DelegationEventType.TASK_COMPLETED,
            task_id="t1",
            agent_id="a1",
            data={"output": "done"},
        )
        assert e.type == DelegationEventType.TASK_COMPLETED
        assert e.task_id == "t1"
        assert e.data["output"] == "done"
        assert isinstance(e.timestamp, datetime)

    def test_event_json_roundtrip(self):
        e = DelegationEvent(type=DelegationEventType.ESCALATED, data={"reason": "all failed"})
        data = e.model_dump_json()
        e2 = DelegationEvent.model_validate_json(data)
        assert e2.type == DelegationEventType.ESCALATED


# ── AuditEntry Tests ─────────────────────────────────────────────────────────


class TestAuditEntry:
    def test_create_audit_entry(self):
        a = AuditEntry(event_type="task_completed", task_id="t1", agent_id="a1")
        assert a.event_type == "task_completed"
        assert a.id  # auto-generated

    def test_audit_entry_with_details(self):
        a = AuditEntry(
            event_type="trust_updated",
            details={"old_score": 0.5, "new_score": 0.55},
        )
        assert a.details["old_score"] == 0.5


# ── TrustRecord Tests ───────────────────────────────────────────────────────


class TestTrustRecord:
    def test_defaults(self):
        r = TrustRecord()
        assert r.score == 0.5
        assert isinstance(r.last_updated, datetime)

    def test_score_out_of_range(self):
        with pytest.raises(Exception):
            TrustRecord(score=1.5)

    def test_score_negative(self):
        with pytest.raises(Exception):
            TrustRecord(score=-0.1)


# ── DelegationResult Tests ──────────────────────────────────────────────────


class TestDelegationResult:
    def test_create_result(self):
        t = _make_task()
        r = DelegationResult(task=t, success=True, output="final output")
        assert r.success is True
        assert r.total_cost == 0.0
        assert r.reassignments == 0
        assert r.subtask_results == []
        assert r.audit_log == []
