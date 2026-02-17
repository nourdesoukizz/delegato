"""Tests for delegato Delegator — main orchestrator, all LLM calls mocked."""

from __future__ import annotations

import pytest

from delegato.assignment import AssignmentScorer
from delegato.audit import AuditLog
from delegato.coordination import EscalationError
from delegato.decomposition import DecompositionEngine
from delegato.delegator import Delegator
from delegato.events import EventBus
from delegato.models import (
    Agent,
    DelegationEvent,
    DelegationEventType,
    Reversibility,
    Task,
    TaskResult,
    VerificationMethod,
    VerificationSpec,
)
from delegato.permissions import PermissionManager
from delegato.trust import TrustTracker
from delegato.verification import VerificationEngine, VerificationResult


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_task(
    task_id="t1",
    goal="Test task",
    capabilities=None,
    method=VerificationMethod.NONE,
    complexity=3,
    reversibility=Reversibility.MEDIUM,
    **kwargs,
) -> Task:
    return Task(
        id=task_id,
        goal=goal,
        required_capabilities=capabilities or ["code"],
        verification=VerificationSpec(method=method),
        complexity=complexity,
        reversibility=reversibility,
        **kwargs,
    )


async def _good_handler(task):
    return TaskResult(task_id=task.id, agent_id="a1", output="done", success=True)


async def _good_handler_a2(task):
    return TaskResult(task_id=task.id, agent_id="a2", output="done", success=True)


def _make_agent(agent_id="a1", capabilities=None, handler=None) -> Agent:
    return Agent(
        id=agent_id,
        name=f"Agent {agent_id}",
        capabilities=capabilities or ["code"],
        handler=handler or _good_handler,
    )


def _mock_decompose_llm(subtasks=None):
    """Mock LLM that returns a decomposition with given subtasks."""
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


def _mock_verify_llm(score=1.0):
    async def mock_call(messages):
        return {"score": score, "reasoning": "ok"}
    return mock_call


# ── Delegator Init ──────────────────────────────────────────────────────────


class TestDelegatorInit:
    def test_default_construction(self):
        d = Delegator()
        assert d._agents == []
        assert d._event_bus is not None
        assert d._trust_tracker is not None
        assert d._assignment_scorer is not None
        assert d._decomposition_engine is not None
        assert d._verification_engine is not None
        assert d._permission_manager is not None
        assert d._audit_log is not None

    def test_custom_components_injected(self):
        bus = EventBus()
        tracker = TrustTracker()
        scorer = AssignmentScorer()
        d = Delegator(
            event_bus=bus,
            trust_tracker=tracker,
            assignment_scorer=scorer,
        )
        assert d._event_bus is bus
        assert d._trust_tracker is tracker
        assert d._assignment_scorer is scorer

    def test_agents_registered_in_trust_tracker(self):
        agent = _make_agent()
        d = Delegator(agents=[agent])
        scores = d.get_trust_scores()
        assert agent.id in scores

    def test_llm_call_passed_through(self):
        call_log = []

        async def custom_llm(messages):
            call_log.append(messages)
            return {"subtasks": []}

        d = Delegator(llm_call=custom_llm)
        # The llm_call should be wired to the decomposition engine
        assert d._decomposition_engine._llm_call is custom_llm


# ── Register / Remove Agent ─────────────────────────────────────────────────


class TestRegisterRemoveAgent:
    def test_register(self):
        d = Delegator()
        agent = _make_agent()
        d.register_agent(agent)
        assert len(d._agents) == 1
        assert d._agents[0].id == agent.id

    def test_remove(self):
        agent = _make_agent()
        d = Delegator(agents=[agent])
        d.remove_agent(agent.id)
        assert len(d._agents) == 0

    def test_remove_nonexistent_is_noop(self):
        d = Delegator()
        d.remove_agent("nonexistent")
        assert len(d._agents) == 0

    def test_register_multiple(self):
        d = Delegator()
        a1 = _make_agent(agent_id="a1")
        a2 = _make_agent(agent_id="a2", handler=_good_handler_a2)
        d.register_agent(a1)
        d.register_agent(a2)
        assert len(d._agents) == 2


# ── Complexity Floor ────────────────────────────────────────────────────────


class TestComplexityFloor:
    async def test_fast_path_eligible(self):
        """Low-complexity, high-reversibility tasks with trusted agent use fast path."""
        tracker = TrustTracker()
        agent = _make_agent()
        tracker.register_agent(agent.id, agent.capabilities)
        # Set trust high enough for floor
        tracker._trust_records[agent.id]["code"].score = 0.9

        d = Delegator(
            agents=[agent],
            trust_tracker=tracker,
            llm_call=_mock_decompose_llm(),
        )
        task = _make_task(complexity=1, reversibility=Reversibility.HIGH)
        result = await d.run(task)
        assert result.success is True

    async def test_fast_path_skips_decomposition(self):
        """Fast path should not call the decomposition engine."""
        decompose_called = False

        async def tracking_llm(messages):
            nonlocal decompose_called
            decompose_called = True
            return {"subtasks": []}

        tracker = TrustTracker()
        agent = _make_agent()
        tracker.register_agent(agent.id, agent.capabilities)
        tracker._trust_records[agent.id]["code"].score = 0.9

        d = Delegator(
            agents=[agent],
            trust_tracker=tracker,
            llm_call=tracking_llm,
        )
        task = _make_task(complexity=1, reversibility=Reversibility.HIGH)
        await d.run(task)
        assert decompose_called is False

    async def test_non_eligible_uses_full_path(self):
        """High-complexity tasks go through decomposition."""
        decompose_called = False
        original_llm = _mock_decompose_llm()

        async def tracking_llm(messages):
            nonlocal decompose_called
            decompose_called = True
            return await original_llm(messages)

        agent = _make_agent()
        d = Delegator(agents=[agent], llm_call=tracking_llm)
        task = _make_task(complexity=4, reversibility=Reversibility.MEDIUM)
        await d.run(task)
        assert decompose_called is True

    async def test_fast_path_still_verifies(self):
        """Fast path should still run verification."""
        verify_called = False

        async def verify_llm(messages):
            nonlocal verify_called
            verify_called = True
            return {"score": 1.0, "reasoning": "ok"}

        tracker = TrustTracker()
        agent = _make_agent()
        tracker.register_agent(agent.id, agent.capabilities)
        tracker._trust_records[agent.id]["code"].score = 0.9

        d = Delegator(
            agents=[agent],
            trust_tracker=tracker,
            verification_engine=VerificationEngine(llm_call=verify_llm),
            llm_call=_mock_decompose_llm(),
        )
        task = _make_task(
            complexity=1,
            reversibility=Reversibility.HIGH,
            method=VerificationMethod.LLM_JUDGE,
        )
        result = await d.run(task)
        assert verify_called is True
        assert result.success is True


# ── Full Pipeline ───────────────────────────────────────────────────────────


class TestFullPipeline:
    async def test_decompose_and_execute(self):
        agent = _make_agent()
        d = Delegator(agents=[agent], llm_call=_mock_decompose_llm())
        task = _make_task()
        result = await d.run(task)
        assert result.success is True
        assert len(result.subtask_results) >= 1

    async def test_dependencies_ordering(self):
        """Tasks with dependencies execute in correct order."""
        execution_order = []

        async def tracking_handler(task):
            execution_order.append(task.goal)
            return TaskResult(task_id=task.id, agent_id="a1", output="done", success=True)

        agent = _make_agent(handler=tracking_handler)

        subtasks = [
            {
                "goal": "First",
                "required_capabilities": ["code"],
                "verification_method": "regex",
                "verification_criteria": "done",
                "dependencies": [],
            },
            {
                "goal": "Second",
                "required_capabilities": ["code"],
                "verification_method": "regex",
                "verification_criteria": "done",
                "dependencies": [0],
            },
        ]

        d = Delegator(agents=[agent], llm_call=_mock_decompose_llm(subtasks))
        task = _make_task()
        result = await d.run(task)
        assert result.success is True
        assert execution_order == ["First", "Second"]

    async def test_failure_partial_results(self):
        """Failed tasks still appear in subtask_results."""
        async def bad_handler(task):
            return TaskResult(task_id=task.id, agent_id="a1", output="bad", success=True)

        agent = _make_agent(handler=bad_handler)

        subtasks = [
            {
                "goal": "Will fail",
                "required_capabilities": ["code"],
                "verification_method": "llm_judge",
                "verification_criteria": "Must be good",
                "dependencies": [],
            },
        ]

        d = Delegator(
            agents=[agent],
            llm_call=_mock_decompose_llm(subtasks),
            verification_engine=VerificationEngine(llm_call=_mock_verify_llm(score=0.0)),
            max_reassignments=0,
        )
        task = _make_task(max_retries=0)
        result = await d.run(task)
        assert result.success is False
        assert len(result.subtask_results) >= 1

    async def test_reassignment_counted(self):
        """Reassignments are tracked in the result."""
        async def a1_handler(task):
            return TaskResult(task_id=task.id, agent_id="a1", output="bad", success=True)

        async def a2_handler(task):
            return TaskResult(task_id=task.id, agent_id="a2", output="good", success=True)

        async def selective_llm(messages):
            content = messages[1]["content"]
            if "bad" in content:
                return {"score": 0.0, "reasoning": "bad"}
            return {"score": 1.0, "reasoning": "good"}

        a1 = _make_agent(agent_id="a1", handler=a1_handler)
        a2 = _make_agent(agent_id="a2", handler=a2_handler)

        subtasks = [
            {
                "goal": "Do work",
                "required_capabilities": ["code"],
                "verification_method": "llm_judge",
                "verification_criteria": "Must be good",
                "dependencies": [],
            },
        ]

        d = Delegator(
            agents=[a1, a2],
            llm_call=_mock_decompose_llm(subtasks),
            verification_engine=VerificationEngine(llm_call=selective_llm),
        )
        task = _make_task(max_retries=0)
        result = await d.run(task)
        assert result.reassignments >= 1

    async def test_total_cost_accumulated(self):
        """Total cost is sum of all subtask costs."""
        async def costed_handler(task):
            return TaskResult(task_id=task.id, agent_id="a1", output="done", success=True, cost=0.5)

        agent = _make_agent(handler=costed_handler)

        subtasks = [
            {"goal": "Task 1", "required_capabilities": ["code"], "verification_method": "none", "dependencies": []},
            {"goal": "Task 2", "required_capabilities": ["code"], "verification_method": "none", "dependencies": []},
        ]

        d = Delegator(agents=[agent], llm_call=_mock_decompose_llm(subtasks))
        task = _make_task()
        result = await d.run(task)
        assert result.total_cost >= 1.0  # 2 tasks × 0.5 each


# ── Event Forwarding ────────────────────────────────────────────────────────


class TestEventForwarding:
    async def test_on_forwards(self):
        events: list[DelegationEvent] = []

        async def listener(event):
            events.append(event)

        d = Delegator(llm_call=_mock_decompose_llm())
        agent = _make_agent()
        d.register_agent(agent)
        d.on(DelegationEventType.TASK_COMPLETED, listener)

        task = _make_task()
        await d.run(task)
        completed = [e for e in events if e.type == DelegationEventType.TASK_COMPLETED]
        assert len(completed) >= 1

    async def test_on_all_forwards(self):
        events: list[DelegationEvent] = []

        async def listener(event):
            events.append(event)

        d = Delegator(llm_call=_mock_decompose_llm())
        agent = _make_agent()
        d.register_agent(agent)
        d.on_all(listener)

        task = _make_task()
        await d.run(task)
        assert len(events) > 0

    async def test_task_decomposed_emitted(self):
        events: list[DelegationEvent] = []

        async def listener(event):
            events.append(event)

        d = Delegator(llm_call=_mock_decompose_llm())
        agent = _make_agent()
        d.register_agent(agent)
        d.on(DelegationEventType.TASK_DECOMPOSED, listener)

        task = _make_task()
        await d.run(task)
        decomposed = [e for e in events if e.type == DelegationEventType.TASK_DECOMPOSED]
        assert len(decomposed) == 1


# ── Public API ──────────────────────────────────────────────────────────────


class TestPublicAPI:
    def test_get_trust_scores(self):
        agent = _make_agent()
        d = Delegator(agents=[agent])
        scores = d.get_trust_scores()
        assert isinstance(scores, dict)
        assert agent.id in scores

    def test_get_audit_log_empty(self):
        d = Delegator()
        log = d.get_audit_log()
        assert isinstance(log, list)
        assert len(log) == 0

    def test_register_verifier(self):
        d = Delegator()

        def my_verifier(task, result):
            return True

        d.register_verifier("my_custom", my_verifier)
        assert "my_custom" in d._verification_engine._custom_verifiers

    async def test_audit_entries_after_run(self):
        agent = _make_agent()
        d = Delegator(agents=[agent], llm_call=_mock_decompose_llm())
        task = _make_task()
        await d.run(task)
        log = d.get_audit_log()
        assert len(log) > 0


# ── Edge Cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    async def test_no_agents_failure(self):
        d = Delegator(llm_call=_mock_decompose_llm())
        task = _make_task()
        result = await d.run(task)
        # No agents → all subtasks fail → overall failure
        assert result.success is False

    async def test_escalation_returns_success_false(self):
        """When all recovery options are exhausted, result.success is False."""
        async def bad_handler(task):
            return TaskResult(task_id=task.id, agent_id="a1", output="bad", success=True)

        agent = _make_agent(handler=bad_handler)

        subtasks = [
            {
                "goal": "Will fail",
                "required_capabilities": ["code"],
                "verification_method": "llm_judge",
                "verification_criteria": "Must be good",
                "dependencies": [],
            },
        ]

        d = Delegator(
            agents=[agent],
            llm_call=_mock_decompose_llm(subtasks),
            verification_engine=VerificationEngine(llm_call=_mock_verify_llm(score=0.0)),
            max_reassignments=0,
        )
        task = _make_task(max_retries=0)
        result = await d.run(task)
        assert result.success is False
