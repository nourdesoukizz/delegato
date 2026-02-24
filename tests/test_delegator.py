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


# Capabilities spread across agents — no single agent covers ≥60%, allowing
# the decomposition gate to pass through to Tier 2.
_DECOMPOSE_CAPS = ["code", "analysis", "web_search"]


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


def _mock_force_decomposition_llm(subtasks=None):
    """Mock LLM that fails smart route verification, forcing decomposition path."""
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
    verify_count = 0

    async def mock_call(messages):
        nonlocal verify_count
        system = messages[0]["content"].lower()
        if "task decomposition" in system:
            return {"subtasks": subtasks}
        verify_count += 1
        if verify_count == 1:  # First = smart route → fail
            return {"score": 0.3, "reasoning": "needs decomposition"}
        return {"score": 1.0, "reasoning": "good"}  # Subsequent = subtask verify → pass

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
        """High-complexity tasks go through decomposition (after smart route fails)."""
        decompose_called = False

        async def tracking_llm(messages):
            nonlocal decompose_called
            system = messages[0]["content"].lower()
            if "task decomposition" in system:
                decompose_called = True
                return {"subtasks": [
                    {
                        "goal": "Sub-task 1",
                        "required_capabilities": ["code"],
                        "verification_method": "none",
                        "verification_criteria": "",
                        "dependencies": [],
                    }
                ]}
            # Verification — fail smart route
            return {"score": 0.3, "reasoning": "needs decomposition"}

        agent = _make_agent()
        d = Delegator(agents=[agent], llm_call=tracking_llm)
        task = _make_task(
            capabilities=_DECOMPOSE_CAPS,
            complexity=4, reversibility=Reversibility.MEDIUM,
            method=VerificationMethod.LLM_JUDGE,
        )
        await d.run(task)
        assert decompose_called is True

    async def test_fast_path_skips_verification(self):
        """Fast path should skip verification — trust established by complexity floor."""
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
        assert verify_called is False
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

        d = Delegator(agents=[agent], llm_call=_mock_force_decomposition_llm(subtasks))
        task = _make_task(capabilities=_DECOMPOSE_CAPS, method=VerificationMethod.LLM_JUDGE)
        result = await d.run(task)
        assert result.success is True
        assert "First" in execution_order
        assert "Second" in execution_order
        assert execution_order.index("First") < execution_order.index("Second")

    async def test_failure_partial_results(self):
        """Failed tasks still appear in subtask_results."""
        async def bad_handler(task):
            return TaskResult(task_id=task.id, agent_id="a1", output="bad", success=False)

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
        # Fallback also fails because handler returns success=False
        assert result.success is False

    async def test_reassignment_counted(self):
        """Reassignments are tracked in the result."""
        a1_calls = 0

        async def a1_handler(task):
            nonlocal a1_calls
            a1_calls += 1
            # First call is smart route — return failure to bypass without trust update
            return TaskResult(
                task_id=task.id, agent_id="a1", output="bad",
                success=a1_calls > 1,
            )

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
            smart_route_max_attempts=1,
        )
        task = _make_task(capabilities=_DECOMPOSE_CAPS, max_retries=0)
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

        d = Delegator(agents=[agent], llm_call=_mock_force_decomposition_llm(subtasks))
        task = _make_task(capabilities=_DECOMPOSE_CAPS, method=VerificationMethod.LLM_JUDGE)
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

        d = Delegator(llm_call=_mock_force_decomposition_llm())
        agent = _make_agent()
        d.register_agent(agent)
        d.on(DelegationEventType.TASK_DECOMPOSED, listener)

        task = _make_task(capabilities=_DECOMPOSE_CAPS, method=VerificationMethod.LLM_JUDGE)
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
            return TaskResult(task_id=task.id, agent_id="a1", output="bad", success=False)

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


# ── Fallback Single Agent ──────────────────────────────────────────────────


class TestFallbackSingleAgent:
    async def test_fallback_on_decomposition_exception(self):
        """When decomposition raises, fallback picks best agent and produces output."""
        async def exploding_llm(messages):
            raise RuntimeError("LLM crashed")

        agent = _make_agent()
        d = Delegator(agents=[agent], llm_call=exploding_llm)
        task = _make_task()
        result = await d.run(task)
        assert result.success is True
        assert result.output == "done"

    async def test_fallback_on_empty_dag(self):
        """When decomposition returns 0 subtasks, fallback produces output."""
        agent = _make_agent()
        d = Delegator(agents=[agent], llm_call=_mock_decompose_llm(subtasks=[]))
        task = _make_task()
        result = await d.run(task)
        assert result.success is True
        assert result.output == "done"

    async def test_fallback_on_all_subtasks_failed(self):
        """When all subtasks fail, fallback picks best agent and succeeds."""
        call_count = 0

        async def handler_fails_then_succeeds(task):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First call = smart route, second = subtask — both fail verification
                return TaskResult(task_id=task.id, agent_id="a1", output="bad", success=True)
            # Third call is the fallback — succeed directly
            return TaskResult(task_id=task.id, agent_id="a1", output="good", success=True)

        agent = _make_agent(handler=handler_fails_then_succeeds)

        subtasks = [
            {
                "goal": "Will fail verification",
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
        task = _make_task(capabilities=_DECOMPOSE_CAPS, max_retries=0, method=VerificationMethod.LLM_JUDGE)
        result = await d.run(task)
        assert result.success is True
        assert result.output == "good"

    async def test_fallback_no_agents_still_fails(self):
        """When no agents are available, fallback returns failure gracefully."""
        async def exploding_llm(messages):
            raise RuntimeError("LLM crashed")

        d = Delegator(llm_call=exploding_llm)
        task = _make_task()
        result = await d.run(task)
        assert result.success is False
        assert result.output is None


# ── Smart Route ────────────────────────────────────────────────────────


class TestSmartRoute:
    async def test_smart_route_passes_skips_decomposition(self):
        """When smart route verification passes, decomposition is skipped entirely."""
        decompose_called = False

        async def tracking_llm(messages):
            nonlocal decompose_called
            system = messages[0]["content"].lower()
            if "task decomposition" in system:
                decompose_called = True
                return {"subtasks": []}
            return {"score": 1.0, "reasoning": "good"}

        agent = _make_agent()
        d = Delegator(agents=[agent], llm_call=tracking_llm)
        task = _make_task(method=VerificationMethod.LLM_JUDGE)
        result = await d.run(task)
        assert result.success is True
        assert result.output == "done"
        assert decompose_called is False

    async def test_smart_route_fails_falls_through_to_decompose(self):
        """When smart route verification fails, task falls through to decomposition."""
        decompose_called = False

        async def routing_llm(messages):
            nonlocal decompose_called
            system = messages[0]["content"].lower()
            if "task decomposition" in system:
                decompose_called = True
                return {"subtasks": [
                    {
                        "goal": "Sub-task 1",
                        "required_capabilities": ["code"],
                        "verification_method": "none",
                        "verification_criteria": "",
                        "dependencies": [],
                    }
                ]}
            return {"score": 0.3, "reasoning": "insufficient"}

        agent = _make_agent()
        d = Delegator(agents=[agent], llm_call=routing_llm)
        task = _make_task(capabilities=_DECOMPOSE_CAPS, method=VerificationMethod.LLM_JUDGE)
        result = await d.run(task)
        assert decompose_called is True
        assert result.success is True

    async def test_smart_route_agent_exception_falls_through(self):
        """When agent handler raises, smart route falls through to decomposition."""
        call_count = 0

        async def exploding_then_good(task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Agent crashed")
            return TaskResult(task_id=task.id, agent_id="a1", output="done", success=True)

        # Use regex verification on subtask to prevent recursive decomposition
        subtasks = [
            {
                "goal": "Sub-task 1",
                "required_capabilities": ["code"],
                "verification_method": "regex",
                "verification_criteria": ".",
                "dependencies": [],
            }
        ]
        agent = _make_agent(handler=exploding_then_good)
        d = Delegator(agents=[agent], llm_call=_mock_decompose_llm(subtasks))
        task = _make_task()
        result = await d.run(task)
        assert result.success is True
        assert call_count == 2  # 1 smart route crash + 1 subtask success

    async def test_smart_route_no_agents_falls_through(self):
        """When no suitable agent exists, smart route falls through."""
        d = Delegator(llm_call=_mock_decompose_llm())
        task = _make_task()
        result = await d.run(task)
        # No agents → smart route None → decomposition → no agents → fallback → no agents → fail
        assert result.success is False

    async def test_smart_route_updates_trust_on_success(self):
        """Smart route updates trust score when verification passes."""
        agent = _make_agent()
        d = Delegator(
            agents=[agent],
            llm_call=_mock_verify_llm(score=1.0),
        )
        scores_before = d.get_trust_scores()
        trust_before = scores_before[agent.id]["trust"]["code"]

        task = _make_task(method=VerificationMethod.LLM_JUDGE)
        await d.run(task)

        scores_after = d.get_trust_scores()
        trust_after = scores_after[agent.id]["trust"]["code"]
        assert trust_after > trust_before

    async def test_smart_route_updates_trust_on_failure(self):
        """Smart route updates trust score when verification fails."""
        agent = _make_agent()

        async def fail_verify_then_crash_decompose(messages):
            system = messages[0]["content"].lower()
            if "task decomposition" in system:
                raise RuntimeError("decomposition failed")
            return {"score": 0.3, "reasoning": "bad"}

        d = Delegator(
            agents=[agent],
            llm_call=fail_verify_then_crash_decompose,
        )
        scores_before = d.get_trust_scores()
        trust_before = scores_before[agent.id]["trust"]["code"]

        task = _make_task(method=VerificationMethod.LLM_JUDGE)
        await d.run(task)

        scores_after = d.get_trust_scores()
        trust_after = scores_after[agent.id]["trust"]["code"]
        assert trust_after < trust_before

    async def test_smart_route_tries_second_agent_on_first_failure(self):
        """First agent fails verification, second succeeds — no decomposition."""
        decompose_called = False
        verify_count = 0

        async def routing_llm(messages):
            nonlocal decompose_called, verify_count
            system = messages[0]["content"].lower()
            if "task decomposition" in system:
                decompose_called = True
                return {"subtasks": []}
            verify_count += 1
            if verify_count == 1:
                return {"score": 0.3, "reasoning": "bad"}
            return {"score": 1.0, "reasoning": "good"}

        a1 = _make_agent(agent_id="a1")
        a2 = _make_agent(agent_id="a2", handler=_good_handler_a2)
        d = Delegator(agents=[a1, a2], llm_call=routing_llm)
        task = _make_task(method=VerificationMethod.LLM_JUDGE)
        result = await d.run(task)
        assert result.success is True
        assert decompose_called is False
        assert verify_count == 2  # 2 smart route attempts

    async def test_smart_route_both_agents_fail_falls_through(self):
        """Both agents fail verification — falls through to decomposition."""
        decompose_called = False

        async def always_fail_verify(messages):
            nonlocal decompose_called
            system = messages[0]["content"].lower()
            if "task decomposition" in system:
                decompose_called = True
                return {"subtasks": [
                    {
                        "goal": "Sub-task 1",
                        "required_capabilities": ["code"],
                        "verification_method": "none",
                        "verification_criteria": "",
                        "dependencies": [],
                    }
                ]}
            return {"score": 0.3, "reasoning": "bad"}

        a1 = _make_agent(agent_id="a1")
        a2 = _make_agent(agent_id="a2", handler=_good_handler_a2)
        d = Delegator(agents=[a1, a2], llm_call=always_fail_verify)
        task = _make_task(capabilities=_DECOMPOSE_CAPS, method=VerificationMethod.LLM_JUDGE)
        result = await d.run(task)
        assert decompose_called is True

    async def test_smart_route_single_agent_skips_second(self):
        """Only 1 agent registered — tries 1, falls through if it fails."""
        verify_count = 0

        async def counting_verify(messages):
            nonlocal verify_count
            system = messages[0]["content"].lower()
            if "task decomposition" in system:
                return {"subtasks": [
                    {
                        "goal": "Sub-task 1",
                        "required_capabilities": ["code"],
                        "verification_method": "none",
                        "verification_criteria": "",
                        "dependencies": [],
                    }
                ]}
            verify_count += 1
            return {"score": 0.3, "reasoning": "bad"}

        agent = _make_agent()
        d = Delegator(agents=[agent], llm_call=counting_verify)
        task = _make_task(method=VerificationMethod.LLM_JUDGE)
        await d.run(task)
        assert verify_count == 1  # Only 1 attempt, not 2

    async def test_decomposition_gate_blocks_when_coverage_sufficient(self):
        """Task with capabilities covered by one agent — decomposition skipped."""
        decompose_called = False

        async def tracking_llm(messages):
            nonlocal decompose_called
            system = messages[0]["content"].lower()
            if "task decomposition" in system:
                decompose_called = True
                return {"subtasks": []}
            return {"score": 0.3, "reasoning": "bad"}

        # Agent covers 2/2 = 100% ≥ 60% → gate blocks
        agent = _make_agent(capabilities=["code", "analysis"])
        d = Delegator(agents=[agent], llm_call=tracking_llm)
        task = _make_task(capabilities=["code", "analysis"], method=VerificationMethod.LLM_JUDGE)
        result = await d.run(task)
        assert decompose_called is False
        # Fallback fires instead of decomposition
        assert result.success is True

    async def test_decomposition_gate_allows_when_no_coverage(self):
        """Task with 3 capabilities, agent covers 1 — decomposition proceeds."""
        decompose_called = False

        async def tracking_llm(messages):
            nonlocal decompose_called
            system = messages[0]["content"].lower()
            if "task decomposition" in system:
                decompose_called = True
                return {"subtasks": [
                    {
                        "goal": "Sub-task 1",
                        "required_capabilities": ["code"],
                        "verification_method": "none",
                        "verification_criteria": "",
                        "dependencies": [],
                    }
                ]}
            return {"score": 0.3, "reasoning": "bad"}

        # Agent covers 1/3 = 33% < 60% → gate allows decomposition
        agent = _make_agent(capabilities=["code"])
        d = Delegator(agents=[agent], llm_call=tracking_llm)
        task = _make_task(
            capabilities=["code", "analysis", "web_search"],
            method=VerificationMethod.LLM_JUDGE,
        )
        result = await d.run(task)
        assert decompose_called is True

    async def test_smart_route_max_attempts_configurable(self):
        """Set max_attempts=3, verify 3 agents tried."""
        verify_count = 0

        async def counting_verify(messages):
            nonlocal verify_count
            system = messages[0]["content"].lower()
            if "task decomposition" in system:
                return {"subtasks": [
                    {
                        "goal": "Sub-task 1",
                        "required_capabilities": ["code"],
                        "verification_method": "none",
                        "verification_criteria": "",
                        "dependencies": [],
                    }
                ]}
            verify_count += 1
            return {"score": 0.3, "reasoning": "bad"}

        a1 = _make_agent(agent_id="a1")
        a2 = _make_agent(agent_id="a2", handler=_good_handler_a2)
        a3 = _make_agent(agent_id="a3", handler=_good_handler)
        d = Delegator(
            agents=[a1, a2, a3],
            llm_call=counting_verify,
            smart_route_max_attempts=3,
        )
        task = _make_task(method=VerificationMethod.LLM_JUDGE)
        await d.run(task)
        assert verify_count == 3  # All 3 agents tried
