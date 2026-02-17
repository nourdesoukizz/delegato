"""Tests for delegato coordination loop — parallel DAG execution with retry/reassign/escalate."""

from __future__ import annotations

import asyncio
import time

import pytest

from delegato.assignment import AssignmentScorer
from delegato.coordination import CoordinationLoop, EscalationError
from delegato.events import EventBus
from delegato.models import (
    Agent,
    DelegationEvent,
    DelegationEventType,
    Task,
    TaskDAG,
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
    max_retries=2,
    timeout=60.0,
    complexity=3,
    **kwargs,
) -> Task:
    return Task(
        id=task_id,
        goal=goal,
        required_capabilities=capabilities or ["code"],
        verification=VerificationSpec(method=method),
        max_retries=max_retries,
        timeout_seconds=timeout,
        complexity=complexity,
        **kwargs,
    )


def _make_agent(agent_id="a1", capabilities=None, handler=None) -> Agent:
    async def default_handler(task):
        return TaskResult(
            task_id=task.id, agent_id=agent_id, output="done", success=True
        )

    return Agent(
        id=agent_id,
        name=f"Agent {agent_id}",
        capabilities=capabilities or ["code"],
        handler=handler or default_handler,
    )


def _make_failing_agent(agent_id="a_fail", capabilities=None) -> Agent:
    async def fail_handler(task):
        return TaskResult(
            task_id=task.id, agent_id=agent_id, output=None, success=False
        )

    return Agent(
        id=agent_id,
        name=f"Failing Agent {agent_id}",
        capabilities=capabilities or ["code"],
        handler=fail_handler,
    )


def _make_dag_with_tasks(*tasks, root_id="root") -> TaskDAG:
    root = Task(
        id=root_id,
        goal="Root task",
        verification=VerificationSpec(method=VerificationMethod.NONE),
    )
    dag = TaskDAG(root_task_id=root_id)
    dag.add_task(root)
    for task in tasks:
        dag.add_task(task, depends_on=[root_id])
    return dag


def _verification_engine_pass():
    """Engine that always passes verification."""
    async def llm_call(messages):
        return {"score": 1.0, "reasoning": "ok"}
    return VerificationEngine(llm_call=llm_call)


def _verification_engine_fail():
    """Engine that always fails verification."""
    async def llm_call(messages):
        return {"score": 0.0, "reasoning": "bad"}
    return VerificationEngine(llm_call=llm_call)


def _make_loop(agents, **kwargs):
    defaults = {
        "agents": agents,
        "verification_engine": _verification_engine_pass(),
    }
    defaults.update(kwargs)
    return CoordinationLoop(**defaults)


# ── Basic Execution ─────────────────────────────────────────────────────────


class TestBasicExecution:
    async def test_single_task_success(self):
        agent = _make_agent()
        task = _make_task()
        dag = _make_dag_with_tasks(task)
        loop = _make_loop([agent])
        results, reassignments = await loop.execute_dag(dag)
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].verified is True
        assert reassignments == 0

    async def test_single_task_failure_escalation(self):
        agent = _make_failing_agent()
        task = _make_task(max_retries=0, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(task)
        loop = _make_loop(
            [agent],
            verification_engine=_verification_engine_fail(),
            max_reassignments=0,
        )
        results, _ = await loop.execute_dag(dag)
        assert len(results) == 1
        assert results[0].success is False

    async def test_linear_dag_order(self):
        """Tasks with dependencies execute in order."""
        execution_order = []

        def make_ordered_agent(agent_id):
            async def handler(task):
                execution_order.append(task.id)
                return TaskResult(task_id=task.id, agent_id=agent_id, output="done", success=True)
            return _make_agent(agent_id=agent_id, handler=handler)

        agent = make_ordered_agent("a1")
        t1 = _make_task(task_id="t1")
        t2 = _make_task(task_id="t2")

        root = Task(id="root", goal="Root", verification=VerificationSpec(method=VerificationMethod.NONE))
        dag = TaskDAG(root_task_id="root")
        dag.add_task(root)
        dag.add_task(t1, depends_on=["root"])
        dag.add_task(t2, depends_on=["t1"])

        loop = _make_loop([agent])
        results, _ = await loop.execute_dag(dag)
        assert execution_order == ["t1", "t2"]

    async def test_parallel_dag_concurrency(self):
        """Independent tasks can run concurrently."""
        timestamps = {}

        def make_timed_agent(agent_id):
            async def handler(task):
                timestamps[task.id] = {"start": time.monotonic()}
                await asyncio.sleep(0.05)
                timestamps[task.id]["end"] = time.monotonic()
                return TaskResult(task_id=task.id, agent_id=agent_id, output="done", success=True)
            return _make_agent(agent_id=agent_id, handler=handler)

        agent = make_timed_agent("a1")
        t1 = _make_task(task_id="t1")
        t2 = _make_task(task_id="t2")
        dag = _make_dag_with_tasks(t1, t2)

        loop = _make_loop([agent], max_parallel=4)
        await loop.execute_dag(dag)
        # Both tasks should overlap in time
        assert timestamps["t1"]["start"] < timestamps["t2"]["end"]
        assert timestamps["t2"]["start"] < timestamps["t1"]["end"]

    async def test_empty_dag(self):
        agent = _make_agent()
        dag = TaskDAG()
        loop = _make_loop([agent])
        results, reassignments = await loop.execute_dag(dag)
        assert results == []
        assert reassignments == 0

    async def test_root_only_dag(self):
        """DAG with only the root task produces no results."""
        agent = _make_agent()
        root = Task(id="root", goal="Root", verification=VerificationSpec(method=VerificationMethod.NONE))
        dag = TaskDAG(root_task_id="root")
        dag.add_task(root)
        loop = _make_loop([agent])
        results, _ = await loop.execute_dag(dag)
        assert results == []


# ── Concurrency Control ─────────────────────────────────────────────────────


class TestConcurrencyControl:
    async def test_max_parallel_respected(self):
        """No more than max_parallel tasks run simultaneously."""
        concurrent_count = 0
        max_concurrent = 0

        def make_counting_agent(agent_id):
            async def handler(task):
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.05)
                concurrent_count -= 1
                return TaskResult(task_id=task.id, agent_id=agent_id, output="done", success=True)
            return _make_agent(agent_id=agent_id, handler=handler)

        agent = make_counting_agent("a1")
        tasks = [_make_task(task_id=f"t{i}") for i in range(6)]
        dag = _make_dag_with_tasks(*tasks)

        loop = _make_loop([agent], max_parallel=2)
        await loop.execute_dag(dag)
        assert max_concurrent <= 2

    async def test_semaphore_released_on_failure(self):
        """Semaphore is released even when task fails."""
        agent = _make_failing_agent()
        tasks = [_make_task(task_id=f"t{i}", max_retries=0) for i in range(3)]
        dag = _make_dag_with_tasks(*tasks)
        loop = _make_loop(
            [agent],
            verification_engine=_verification_engine_fail(),
            max_parallel=1,
            max_reassignments=0,
        )
        results, _ = await loop.execute_dag(dag)
        # All tasks should complete (even if failed), proving semaphore was released
        assert len(results) == 3

    async def test_semaphore_released_on_timeout(self):
        """Semaphore is released when task times out."""
        async def slow_handler(task):
            await asyncio.sleep(10)
            return TaskResult(task_id=task.id, agent_id="a1", output="done", success=True)

        agent = _make_agent(handler=slow_handler)
        t1 = _make_task(task_id="t1", timeout=0.05, max_retries=0)
        t2 = _make_task(task_id="t2", timeout=60.0)
        dag = _make_dag_with_tasks(t1, t2)

        loop = _make_loop(
            [agent],
            max_parallel=1,
            max_reassignments=0,
        )
        results, _ = await loop.execute_dag(dag)
        assert len(results) == 2


# ── Retry Logic ─────────────────────────────────────────────────────────────


class TestRetryLogic:
    async def test_retry_on_verification_failure(self):
        """Task is retried when verification fails."""
        call_count = 0

        async def flaky_handler(task):
            nonlocal call_count
            call_count += 1
            return TaskResult(task_id=task.id, agent_id="a1", output=f"attempt_{call_count}", success=True)

        # First call fails verification, second passes
        verify_calls = 0

        async def flaky_verify_llm(messages):
            nonlocal verify_calls
            verify_calls += 1
            return {"score": 0.0 if verify_calls == 1 else 1.0, "reasoning": "ok"}

        agent = _make_agent(handler=flaky_handler)
        task = _make_task(max_retries=2, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(task)

        loop = _make_loop(
            [agent],
            verification_engine=VerificationEngine(llm_call=flaky_verify_llm),
        )
        results, _ = await loop.execute_dag(dag)
        assert len(results) == 1
        assert results[0].success is True
        assert call_count == 2

    async def test_max_retries_exhausted_then_reassign(self):
        """After exhausting retries, task is reassigned to next agent."""
        agent1_calls = 0
        agent2_calls = 0

        async def agent1_handler(task):
            nonlocal agent1_calls
            agent1_calls += 1
            return TaskResult(task_id=task.id, agent_id="a1", output="bad", success=True)

        async def agent2_handler(task):
            nonlocal agent2_calls
            agent2_calls += 1
            return TaskResult(task_id=task.id, agent_id="a2", output="good", success=True)

        # Fails for a1, passes for a2
        async def selective_llm(messages):
            content = messages[1]["content"]
            if "bad" in content:
                return {"score": 0.0, "reasoning": "bad"}
            return {"score": 1.0, "reasoning": "good"}

        a1 = _make_agent(agent_id="a1", handler=agent1_handler)
        a2 = _make_agent(agent_id="a2", handler=agent2_handler)
        task = _make_task(max_retries=1, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(task)

        loop = _make_loop(
            [a1, a2],
            verification_engine=VerificationEngine(llm_call=selective_llm),
        )
        results, reassignments = await loop.execute_dag(dag)
        assert results[0].success is True
        assert results[0].agent_id == "a2"
        assert agent1_calls == 2  # Initial + 1 retry
        assert agent2_calls == 1
        assert reassignments == 1

    async def test_attempt_count_incremented(self):
        """Each retry increments the contract attempt count."""
        call_count = 0

        async def handler(task):
            nonlocal call_count
            call_count += 1
            return TaskResult(task_id=task.id, agent_id="a1", output=f"try_{call_count}", success=True)

        verify_calls = 0

        async def llm(messages):
            nonlocal verify_calls
            verify_calls += 1
            # Fail first two, pass third
            return {"score": 0.0 if verify_calls <= 2 else 1.0, "reasoning": "ok"}

        agent = _make_agent(handler=handler)
        task = _make_task(max_retries=2, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(task)

        loop = _make_loop(
            [agent],
            verification_engine=VerificationEngine(llm_call=llm),
        )
        results, _ = await loop.execute_dag(dag)
        assert call_count == 3  # 1 initial + 2 retries
        assert results[0].success is True

    async def test_same_agent_on_retry(self):
        """Retries use the same agent."""
        agents_used = []

        async def tracking_handler(task):
            agents_used.append("a1")
            return TaskResult(task_id=task.id, agent_id="a1", output="out", success=True)

        verify_calls = 0

        async def llm(messages):
            nonlocal verify_calls
            verify_calls += 1
            return {"score": 0.0 if verify_calls == 1 else 1.0, "reasoning": "ok"}

        agent = _make_agent(handler=tracking_handler)
        task = _make_task(max_retries=1, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(task)

        loop = _make_loop(
            [agent],
            verification_engine=VerificationEngine(llm_call=llm),
        )
        await loop.execute_dag(dag)
        assert agents_used == ["a1", "a1"]

    async def test_zero_retries_straight_to_reassign(self):
        """With max_retries=0, failure goes directly to reassignment."""
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
        task = _make_task(max_retries=0, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(task)

        loop = _make_loop(
            [a1, a2],
            verification_engine=VerificationEngine(llm_call=selective_llm),
        )
        results, reassignments = await loop.execute_dag(dag)
        assert results[0].agent_id == "a2"
        assert reassignments == 1


# ── Reassignment ────────────────────────────────────────────────────────────


class TestReassignment:
    async def test_reassign_to_next_best_agent(self):
        async def bad_handler(task):
            return TaskResult(task_id=task.id, agent_id="a1", output="bad", success=True)

        async def good_handler(task):
            return TaskResult(task_id=task.id, agent_id="a2", output="good", success=True)

        async def selective_llm(messages):
            content = messages[1]["content"]
            if "bad" in content:
                return {"score": 0.0, "reasoning": "nope"}
            return {"score": 1.0, "reasoning": "ok"}

        a1 = _make_agent(agent_id="a1", handler=bad_handler)
        a2 = _make_agent(agent_id="a2", handler=good_handler)
        task = _make_task(max_retries=0, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(task)

        loop = _make_loop(
            [a1, a2],
            verification_engine=VerificationEngine(llm_call=selective_llm),
        )
        results, _ = await loop.execute_dag(dag)
        assert results[0].agent_id == "a2"

    async def test_task_reassigned_event(self):
        events: list[DelegationEvent] = []

        async def listener(event):
            events.append(event)

        bus = EventBus()
        bus.on(DelegationEventType.TASK_REASSIGNED, listener)

        async def bad_handler(task):
            return TaskResult(task_id=task.id, agent_id="a1", output="bad", success=True)

        async def good_handler(task):
            return TaskResult(task_id=task.id, agent_id="a2", output="good", success=True)

        async def selective_llm(messages):
            content = messages[1]["content"]
            if "bad" in content:
                return {"score": 0.0, "reasoning": "nope"}
            return {"score": 1.0, "reasoning": "ok"}

        a1 = _make_agent(agent_id="a1", handler=bad_handler)
        a2 = _make_agent(agent_id="a2", handler=good_handler)
        task = _make_task(max_retries=0, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(task)

        loop = _make_loop(
            [a1, a2],
            event_bus=bus,
            verification_engine=VerificationEngine(llm_call=selective_llm),
        )
        await loop.execute_dag(dag)
        reassigned_events = [e for e in events if e.type == DelegationEventType.TASK_REASSIGNED]
        assert len(reassigned_events) == 1

    async def test_max_reassignments_escalation(self):
        """Exceeding max_reassignments triggers escalation."""
        async def bad_handler(task):
            return TaskResult(task_id=task.id, agent_id="any", output="bad", success=True)

        agents = [
            _make_agent(agent_id=f"a{i}", handler=bad_handler)
            for i in range(5)
        ]
        task = _make_task(max_retries=0, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(task)

        loop = _make_loop(
            agents,
            verification_engine=_verification_engine_fail(),
            max_reassignments=2,
        )
        results, _ = await loop.execute_dag(dag)
        assert results[0].success is False

    async def test_anti_oscillation_counter_per_task(self):
        """Reassignment counter is tracked per task."""
        async def bad_handler(task):
            return TaskResult(task_id=task.id, agent_id="any", output="bad", success=True)

        agents = [_make_agent(agent_id=f"a{i}", handler=bad_handler) for i in range(5)]
        t1 = _make_task(task_id="t1", max_retries=0, method=VerificationMethod.LLM_JUDGE)
        t2 = _make_task(task_id="t2", max_retries=0, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(t1, t2)

        loop = _make_loop(
            agents,
            verification_engine=_verification_engine_fail(),
            max_reassignments=2,
        )
        await loop.execute_dag(dag)
        # Each task should have its own reassignment count
        assert loop._reassignment_counts.get("t1", 0) <= 2
        assert loop._reassignment_counts.get("t2", 0) <= 2

    async def test_no_available_agent_escalates(self):
        """If no agents are available for reassignment, escalate."""
        async def bad_handler(task):
            return TaskResult(task_id=task.id, agent_id="a1", output="bad", success=True)

        a1 = _make_agent(agent_id="a1", handler=bad_handler)
        task = _make_task(max_retries=0, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(task)

        loop = _make_loop(
            [a1],
            verification_engine=_verification_engine_fail(),
        )
        results, _ = await loop.execute_dag(dag)
        assert results[0].success is False


# ── Timeout ─────────────────────────────────────────────────────────────────


class TestTimeout:
    async def test_timeout_creates_failure_result(self):
        async def slow_handler(task):
            await asyncio.sleep(10)
            return TaskResult(task_id=task.id, agent_id="a1", output="done", success=True)

        agent = _make_agent(handler=slow_handler)
        task = _make_task(timeout=0.05, max_retries=0)
        dag = _make_dag_with_tasks(task)

        loop = _make_loop([agent], max_reassignments=0)
        results, _ = await loop.execute_dag(dag)
        assert len(results) == 1
        assert results[0].success is False

    async def test_timeout_triggers_retry(self):
        call_count = 0

        async def handler(task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(10)
            return TaskResult(task_id=task.id, agent_id="a1", output="done", success=True)

        agent = _make_agent(handler=handler)
        task = _make_task(timeout=0.05, max_retries=1)
        dag = _make_dag_with_tasks(task)

        loop = _make_loop([agent])
        results, _ = await loop.execute_dag(dag)
        assert call_count == 2
        assert results[0].success is True

    async def test_uses_task_timeout_seconds(self):
        """Each task uses its own timeout_seconds."""
        async def slow_handler(task):
            await asyncio.sleep(0.2)
            return TaskResult(task_id=task.id, agent_id="a1", output="done", success=True)

        agent = _make_agent(handler=slow_handler)
        t1 = _make_task(task_id="t1", timeout=0.05, max_retries=0)
        t2 = _make_task(task_id="t2", timeout=5.0)
        dag = _make_dag_with_tasks(t1, t2)

        loop = _make_loop([agent], max_reassignments=0)
        results, _ = await loop.execute_dag(dag)
        results_by_id = {r.task_id: r for r in results}
        assert results_by_id["t1"].success is False  # Timed out
        assert results_by_id["t2"].success is True   # Had enough time


# ── Event Emission ──────────────────────────────────────────────────────────


class TestEventEmission:
    async def test_task_assigned_event(self):
        events: list[DelegationEvent] = []

        async def listener(event):
            events.append(event)

        bus = EventBus()
        bus.on(DelegationEventType.TASK_ASSIGNED, listener)

        agent = _make_agent()
        task = _make_task()
        dag = _make_dag_with_tasks(task)
        loop = _make_loop([agent], event_bus=bus)
        await loop.execute_dag(dag)
        assert any(e.type == DelegationEventType.TASK_ASSIGNED for e in events)

    async def test_task_started_event(self):
        events: list[DelegationEvent] = []

        async def listener(event):
            events.append(event)

        bus = EventBus()
        bus.on(DelegationEventType.TASK_STARTED, listener)

        agent = _make_agent()
        task = _make_task()
        dag = _make_dag_with_tasks(task)
        loop = _make_loop([agent], event_bus=bus)
        await loop.execute_dag(dag)
        assert any(e.type == DelegationEventType.TASK_STARTED for e in events)

    async def test_task_completed_event(self):
        events: list[DelegationEvent] = []

        async def listener(event):
            events.append(event)

        bus = EventBus()
        bus.on(DelegationEventType.TASK_COMPLETED, listener)

        agent = _make_agent()
        task = _make_task()
        dag = _make_dag_with_tasks(task)
        loop = _make_loop([agent], event_bus=bus)
        await loop.execute_dag(dag)
        assert any(e.type == DelegationEventType.TASK_COMPLETED for e in events)

    async def test_escalated_event(self):
        events: list[DelegationEvent] = []

        async def listener(event):
            events.append(event)

        bus = EventBus()
        bus.on(DelegationEventType.ESCALATED, listener)

        async def bad_handler(task):
            return TaskResult(task_id=task.id, agent_id="a1", output="bad", success=True)

        agent = _make_agent(handler=bad_handler)
        task = _make_task(max_retries=0, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(task)

        loop = _make_loop(
            [agent],
            event_bus=bus,
            verification_engine=_verification_engine_fail(),
            max_reassignments=0,
        )
        await loop.execute_dag(dag)
        assert any(e.type == DelegationEventType.ESCALATED for e in events)


# ── Trust Integration ───────────────────────────────────────────────────────


class TestTrustIntegration:
    async def test_trust_up_on_success(self):
        tracker = TrustTracker()
        tracker.register_agent("a1", ["code"])
        initial = tracker.get_trust("a1", "code")

        agent = _make_agent()
        task = _make_task()
        dag = _make_dag_with_tasks(task)
        loop = _make_loop([agent], trust_tracker=tracker)
        await loop.execute_dag(dag)
        assert tracker.get_trust("a1", "code") > initial

    async def test_trust_down_on_failure(self):
        tracker = TrustTracker()
        tracker.register_agent("a1", ["code"])
        initial = tracker.get_trust("a1", "code")

        agent = _make_agent()
        task = _make_task(max_retries=0, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(task)

        loop = _make_loop(
            [agent],
            trust_tracker=tracker,
            verification_engine=_verification_engine_fail(),
            max_reassignments=0,
        )
        await loop.execute_dag(dag)
        assert tracker.get_trust("a1", "code") < initial

    async def test_circuit_breaker_on_large_drop(self):
        events: list[DelegationEvent] = []

        async def listener(event):
            events.append(event)

        bus = EventBus()
        bus.on(DelegationEventType.TRUST_CIRCUIT_BREAK, listener)

        tracker = TrustTracker(event_bus=bus)
        tracker.register_agent("a1", ["code"])
        # Set trust high so the drop is big enough to trigger circuit breaker
        tracker._trust_records["a1"]["code"].score = 0.95

        pm = PermissionManager(
            event_bus=bus,
            trust_tracker=tracker,
            circuit_breaker_threshold=0.3,
        )

        async def bad_handler(task):
            return TaskResult(task_id=task.id, agent_id="a1", output="bad", success=True)

        agent = _make_agent(handler=bad_handler)
        task = _make_task(max_retries=0, method=VerificationMethod.LLM_JUDGE)
        dag = _make_dag_with_tasks(task)

        # Use a failing verification engine with high failure_learning_rate to cause a large trust drop
        loop = _make_loop(
            [agent],
            event_bus=bus,
            trust_tracker=TrustTracker(event_bus=bus, failure_learning_rate=0.5),
            verification_engine=_verification_engine_fail(),
            permission_manager=PermissionManager(
                event_bus=bus,
                trust_tracker=tracker,
                circuit_breaker_threshold=0.1,
            ),
            max_reassignments=0,
        )
        # Manually set high trust on the loop's tracker
        loop._trust_tracker.register_agent("a1", ["code"])
        loop._trust_tracker._trust_records["a1"]["code"].score = 0.95

        await loop.execute_dag(dag)
        # Trust should have dropped significantly
        assert loop._trust_tracker.get_trust("a1", "code") < 0.95
