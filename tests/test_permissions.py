"""Tests for delegato permission manager."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from delegato.events import EventBus
from delegato.models import (
    Agent,
    Contract,
    DelegationEvent,
    DelegationEventType,
    Permission,
    Reversibility,
    Task,
    TaskResult,
    VerificationMethod,
    VerificationSpec,
)
from delegato.permissions import (
    CircuitBreakResult,
    ComplexityFloorResult,
    PermissionManager,
    PrivilegeEscalationError,
)
from delegato.trust import TrustTracker


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_perm(resource="filesystem", action="read", scope="*", expiry=None) -> Permission:
    return Permission(resource=resource, action=action, scope=scope, expiry=expiry)


def _make_task(**kwargs) -> Task:
    defaults = {
        "goal": "Test task",
        "verification": VerificationSpec(method=VerificationMethod.NONE),
        "complexity": 3,
        "reversibility": Reversibility.MEDIUM,
    }
    defaults.update(kwargs)
    return Task(**defaults)


async def _dummy_handler(task):
    return TaskResult(task_id=task.id, agent_id="a1", output="done", success=True)


def _make_agent(agent_id="a1", capabilities=None, trust_scores=None) -> Agent:
    return Agent(
        id=agent_id,
        name=f"Agent {agent_id}",
        capabilities=capabilities or ["web_search"],
        handler=_dummy_handler,
        trust_scores=trust_scores or {},
    )


def _make_contract(agent_id="a1", result=None) -> Contract:
    task = _make_task()
    return Contract(
        task=task,
        agent_id=agent_id,
        verification=task.verification,
        result=result,
    )


# ── Scope Subset Tests ───────────────────────────────────────────────────────


class TestScopeSubset:
    def test_wildcard_parent(self):
        pm = PermissionManager()
        assert pm._scope_is_subset("/data/reports", "*") is True

    def test_equal_scopes(self):
        pm = PermissionManager()
        assert pm._scope_is_subset("/data/reports", "/data/reports") is True

    def test_narrower_child(self):
        pm = PermissionManager()
        assert pm._scope_is_subset("/data/reports/2024", "/data/*") is True

    def test_broader_child_fails(self):
        pm = PermissionManager()
        assert pm._scope_is_subset("/data/*", "/data/reports") is False

    def test_exact_match_no_wildcard(self):
        pm = PermissionManager()
        assert pm._scope_is_subset("/tmp/file.txt", "/tmp/file.txt") is True

    def test_child_wildcard_parent_no_wildcard(self):
        pm = PermissionManager()
        assert pm._scope_is_subset("/data/*", "/data/reports") is False

    def test_nested_wildcards(self):
        pm = PermissionManager()
        assert pm._scope_is_subset("/data/reports/*", "/data/*") is True

    def test_no_match(self):
        pm = PermissionManager()
        assert pm._scope_is_subset("/other/path", "/data/*") is False


# ── Attenuate Tests ──────────────────────────────────────────────────────────


class TestAttenuate:
    def test_valid_subset_passes(self):
        pm = PermissionManager()
        parent = [_make_perm(scope="/data/*")]
        child = [_make_perm(scope="/data/reports")]
        result = pm.attenuate(parent, child)
        assert len(result) == 1

    def test_broader_child_raises(self):
        pm = PermissionManager()
        parent = [_make_perm(scope="/data/reports")]
        child = [_make_perm(scope="/data/*")]
        with pytest.raises(PrivilegeEscalationError):
            pm.attenuate(parent, child)

    def test_different_resource_raises(self):
        pm = PermissionManager()
        parent = [_make_perm(resource="filesystem")]
        child = [_make_perm(resource="network")]
        with pytest.raises(PrivilegeEscalationError):
            pm.attenuate(parent, child)

    def test_different_action_raises(self):
        pm = PermissionManager()
        parent = [_make_perm(action="read")]
        child = [_make_perm(action="write")]
        with pytest.raises(PrivilegeEscalationError):
            pm.attenuate(parent, child)

    def test_expired_child_filtered(self):
        pm = PermissionManager()
        parent = [_make_perm()]
        past = datetime.now(UTC) - timedelta(hours=1)
        child = [_make_perm(expiry=past)]
        result = pm.attenuate(parent, child)
        assert len(result) == 0


# ── Check Permission Tests ───────────────────────────────────────────────────


class TestCheckPermission:
    def test_exact_match(self):
        pm = PermissionManager()
        perm = _make_perm(resource="api", action="read", scope="/users")
        granted = [_make_perm(resource="api", action="read", scope="/users")]
        assert pm.check_permission(perm, granted) is True

    def test_no_match(self):
        pm = PermissionManager()
        perm = _make_perm(resource="api", action="write", scope="/users")
        granted = [_make_perm(resource="api", action="read", scope="/users")]
        assert pm.check_permission(perm, granted) is False


# ── Check Expiry Tests ───────────────────────────────────────────────────────


class TestCheckExpiry:
    def test_filters_expired(self):
        pm = PermissionManager()
        past = datetime.now(UTC) - timedelta(hours=1)
        perms = [_make_perm(expiry=past)]
        assert len(pm.check_expiry(perms)) == 0

    def test_keeps_valid(self):
        pm = PermissionManager()
        future = datetime.now(UTC) + timedelta(hours=1)
        perms = [_make_perm(expiry=future)]
        assert len(pm.check_expiry(perms)) == 1

    def test_none_expiry_always_valid(self):
        pm = PermissionManager()
        perms = [_make_perm(expiry=None)]
        assert len(pm.check_expiry(perms)) == 1


# ── Complexity Floor Tests ───────────────────────────────────────────────────


class TestComplexityFloor:
    def test_eligible_task(self):
        tracker = TrustTracker()
        tracker.register_agent("a1", ["web_search"])
        # Bump trust above 0.7
        tracker._trust_records["a1"]["web_search"].score = 0.8

        pm = PermissionManager(trust_tracker=tracker)
        task = _make_task(
            complexity=1,
            reversibility=Reversibility.HIGH,
            required_capabilities=["web_search"],
        )
        agent = _make_agent("a1", capabilities=["web_search"])
        result = pm.check_complexity_floor(task, [agent])
        assert result.eligible is True
        assert result.agent_id == "a1"

    def test_complexity_too_high(self):
        pm = PermissionManager()
        task = _make_task(complexity=3, reversibility=Reversibility.HIGH)
        result = pm.check_complexity_floor(task, [_make_agent()])
        assert result.eligible is False
        assert "complexity" in result.reason.lower()

    def test_reversibility_not_high(self):
        pm = PermissionManager()
        task = _make_task(complexity=1, reversibility=Reversibility.LOW)
        result = pm.check_complexity_floor(task, [_make_agent()])
        assert result.eligible is False
        assert "reversibility" in result.reason.lower()

    def test_no_trusted_agent(self):
        tracker = TrustTracker()
        tracker.register_agent("a1", ["web_search"])
        # Trust stays at default 0.5, below 0.7 threshold

        pm = PermissionManager(trust_tracker=tracker)
        task = _make_task(
            complexity=1,
            reversibility=Reversibility.HIGH,
            required_capabilities=["web_search"],
        )
        agent = _make_agent("a1", capabilities=["web_search"])
        result = pm.check_complexity_floor(task, [agent])
        assert result.eligible is False

    def test_no_matching_capabilities(self):
        tracker = TrustTracker()
        tracker.register_agent("a1", ["coding"])
        tracker._trust_records["a1"]["coding"].score = 0.9

        pm = PermissionManager(trust_tracker=tracker)
        task = _make_task(
            complexity=1,
            reversibility=Reversibility.HIGH,
            required_capabilities=["web_search"],
        )
        agent = _make_agent("a1", capabilities=["coding"])
        result = pm.check_complexity_floor(task, [agent])
        assert result.eligible is False


# ── Circuit Breaker Tests ────────────────────────────────────────────────────


class TestCircuitBreaker:
    async def test_large_drop_triggers(self):
        pm = PermissionManager()
        contract = _make_contract("a1")
        result = await pm.check_circuit_breaker("a1", "web_search", 0.8, 0.4, [contract])
        assert result is not None
        assert result.trust_drop == pytest.approx(0.4)
        assert contract.id in result.paused_contract_ids

    async def test_small_drop_no_op(self):
        pm = PermissionManager()
        result = await pm.check_circuit_breaker("a1", "web_search", 0.8, 0.7, [])
        assert result is None

    async def test_event_emitted(self):
        events: list[DelegationEvent] = []

        async def listener(event):
            events.append(event)

        bus = EventBus()
        bus.on(DelegationEventType.TRUST_CIRCUIT_BREAK, listener)
        pm = PermissionManager(event_bus=bus)
        await pm.check_circuit_breaker("a1", "web_search", 0.8, 0.3, [])
        assert len(events) == 1
        assert events[0].type == DelegationEventType.TRUST_CIRCUIT_BREAK
        assert events[0].data["trust_drop"] == pytest.approx(0.5)

    async def test_paused_contracts_identified(self):
        pm = PermissionManager()
        active = _make_contract("a1", result=None)
        completed_result = TaskResult(task_id="t1", agent_id="a1", output="done", success=True)
        completed = _make_contract("a1", result=completed_result)
        other_agent = _make_contract("a2", result=None)

        result = await pm.check_circuit_breaker(
            "a1", "web_search", 0.9, 0.2, [active, completed, other_agent]
        )
        assert result is not None
        assert active.id in result.paused_contract_ids
        assert completed.id not in result.paused_contract_ids
        assert other_agent.id not in result.paused_contract_ids
