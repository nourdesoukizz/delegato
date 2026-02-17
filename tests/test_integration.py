"""End-to-end integration tests through the Delegator public API.

All LLM calls are mocked via the ``llm_call`` parameter — no API keys needed.

NOTE: Subtasks use ``verification_method: "regex"`` (not ``"none"``) to avoid
triggering recursive decomposition in the DecompositionEngine.
"""

from __future__ import annotations

import pytest

from delegato import (
    Agent,
    DelegationEvent,
    DelegationEventType,
    DelegationResult,
    Delegator,
    Reversibility,
    Task,
    TaskResult,
    VerificationMethod,
    VerificationSpec,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_task(
    task_id: str = "t1",
    goal: str = "integration test task",
    capabilities: list[str] | None = None,
    method: VerificationMethod = VerificationMethod.NONE,
    criteria: str = "",
    complexity: int = 3,
    reversibility: Reversibility = Reversibility.MEDIUM,
    max_retries: int = 2,
    **kwargs,
) -> Task:
    return Task(
        id=task_id,
        goal=goal,
        required_capabilities=capabilities or ["code"],
        verification=VerificationSpec(method=method, criteria=criteria),
        complexity=complexity,
        reversibility=reversibility,
        max_retries=max_retries,
        **kwargs,
    )


def _make_agent(
    agent_id: str = "a1",
    capabilities: list[str] | None = None,
    handler=None,
) -> Agent:
    async def default_handler(task):
        return TaskResult(
            task_id=task.id, agent_id=agent_id, output="done", success=True, cost=0.01
        )

    return Agent(
        id=agent_id,
        name=f"Agent {agent_id}",
        capabilities=capabilities or ["code"],
        handler=handler or default_handler,
    )


def _sub(goal, deps=None, method="regex", criteria="."):
    """Shorthand for a subtask dict. Defaults to regex verification to avoid recursion."""
    return {
        "goal": goal,
        "required_capabilities": ["code"],
        "verification_method": method,
        "verification_criteria": criteria,
        "dependencies": deps or [],
    }


def _mock_decompose(subtasks: list[dict]):
    """Return a mock LLM callable that routes decomposition vs verification."""

    async def llm_call(messages):
        system = messages[0]["content"].lower()
        if "task decomposition" in system:
            return {"subtasks": subtasks}
        # verification judge
        return {"score": 1.0, "reasoning": "ok"}

    return llm_call


def _mock_routing_llm(subtasks: list[dict], verify_fn=None):
    """LLM that routes decomposition calls and optionally custom verification."""

    async def llm_call(messages):
        system = messages[0]["content"].lower()
        if "task decomposition" in system:
            return {"subtasks": subtasks}
        if verify_fn is not None:
            return await verify_fn(messages)
        return {"score": 1.0, "reasoning": "ok"}

    return llm_call


# ── TestFullPipelineIntegration ──────────────────────────────────────────────


class TestFullPipelineIntegration:
    async def test_single_subtask_pipeline(self):
        """One subtask flows through decompose → assign → execute → verify."""
        agent = _make_agent()
        subtasks = [_sub("Do work")]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))
        result = await d.run(_make_task())
        assert result.success is True
        assert len(result.subtask_results) == 1

    async def test_three_subtask_sequential_pipeline(self):
        """Three sequential subtasks execute in dependency order."""
        order: list[str] = []

        async def tracking_handler(task):
            order.append(task.goal)
            return TaskResult(task_id=task.id, agent_id="a1", output="done", success=True)

        agent = _make_agent(handler=tracking_handler)
        subtasks = [
            _sub("Step 1"),
            _sub("Step 2", deps=[0]),
            _sub("Step 3", deps=[1]),
        ]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))
        result = await d.run(_make_task())
        assert result.success is True
        assert order == ["Step 1", "Step 2", "Step 3"]

    async def test_parallel_independent_subtasks(self):
        """Independent subtasks all complete."""
        agent = _make_agent()
        subtasks = [_sub("A"), _sub("B"), _sub("C")]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))
        result = await d.run(_make_task())
        assert result.success is True
        assert len(result.subtask_results) == 3

    async def test_output_assembly(self):
        """Results from all subtasks are assembled in the final output."""
        async def numbered_handler(task):
            return TaskResult(
                task_id=task.id, agent_id="a1",
                output=f"result-{task.goal}", success=True,
            )

        agent = _make_agent(handler=numbered_handler)
        subtasks = [_sub("one"), _sub("two")]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))
        result = await d.run(_make_task())
        assert result.success is True
        outputs = [r.output for r in result.subtask_results]
        assert "result-one" in outputs
        assert "result-two" in outputs

    async def test_pipeline_with_llm_judge_verification(self):
        """LLM judge verification passes when score exceeds threshold."""
        agent = _make_agent()
        subtasks = [
            _sub("Produce output", method="llm_judge", criteria="Must be correct"),
        ]

        async def always_pass(messages):
            return {"score": 0.95, "reasoning": "excellent"}

        llm = _mock_routing_llm(subtasks, verify_fn=always_pass)
        d = Delegator(agents=[agent], llm_call=llm)
        result = await d.run(_make_task())
        assert result.success is True


# ── TestMultiTaskDAG ─────────────────────────────────────────────────────────


class TestMultiTaskDAG:
    async def test_diamond_dag_dependencies(self):
        """Diamond DAG: A → (B, C) → D executes correctly."""
        order: list[str] = []

        async def tracking(task):
            order.append(task.goal)
            return TaskResult(task_id=task.id, agent_id="a1", output="ok", success=True)

        agent = _make_agent(handler=tracking)
        subtasks = [
            _sub("A"),
            _sub("B", deps=[0]),
            _sub("C", deps=[0]),
            _sub("D", deps=[1, 2]),
        ]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))
        result = await d.run(_make_task())
        assert result.success is True
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    async def test_fan_out_fan_in(self):
        """Fan-out from A to B,C,D then fan-in to E."""
        order: list[str] = []

        async def tracking(task):
            order.append(task.goal)
            return TaskResult(task_id=task.id, agent_id="a1", output="ok", success=True)

        agent = _make_agent(handler=tracking)
        subtasks = [
            _sub("A"),
            _sub("B", deps=[0]),
            _sub("C", deps=[0]),
            _sub("D", deps=[0]),
            _sub("E", deps=[1, 2, 3]),
        ]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))
        result = await d.run(_make_task())
        assert result.success is True
        assert order.index("A") == 0
        assert order.index("E") == len(order) - 1

    async def test_deterministic_sequential_order(self):
        """Strictly sequential tasks always execute in order."""
        order: list[str] = []

        async def tracking(task):
            order.append(task.goal)
            return TaskResult(task_id=task.id, agent_id="a1", output="ok", success=True)

        agent = _make_agent(handler=tracking)
        subtasks = [
            _sub("1"),
            _sub("2", deps=[0]),
            _sub("3", deps=[1]),
            _sub("4", deps=[2]),
        ]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))
        result = await d.run(_make_task())
        assert result.success is True
        assert order == ["1", "2", "3", "4"]


# ── TestRetryAndReassignment ────────────────────────────────────────────────


class TestRetryAndReassignment:
    async def test_retry_then_succeed(self):
        """Task fails once, retries with same agent, succeeds."""
        call_count = 0

        async def flaky_handler(task):
            nonlocal call_count
            call_count += 1
            output = "bad" if call_count == 1 else "good"
            return TaskResult(
                task_id=task.id, agent_id="a1", output=output, success=True
            )

        async def verify_fn(messages):
            content = messages[1]["content"]
            if "bad" in content:
                return {"score": 0.1, "reasoning": "bad output"}
            return {"score": 0.9, "reasoning": "good output"}

        agent = _make_agent(handler=flaky_handler)
        subtasks = [
            _sub("Flaky task", method="llm_judge", criteria="Must be good"),
        ]
        llm = _mock_routing_llm(subtasks, verify_fn=verify_fn)
        d = Delegator(agents=[agent], llm_call=llm)
        result = await d.run(_make_task(max_retries=2))
        assert result.success is True
        assert call_count == 2

    async def test_reassignment_after_retries_exhausted(self):
        """When agent a1 exhausts retries, task is reassigned to a2."""
        async def bad_handler(task):
            return TaskResult(
                task_id=task.id, agent_id="a1", output="always bad", success=True
            )

        async def good_handler(task):
            return TaskResult(
                task_id=task.id, agent_id="a2", output="good", success=True
            )

        async def verify_fn(messages):
            content = messages[1]["content"]
            if "always bad" in content:
                return {"score": 0.1, "reasoning": "bad"}
            return {"score": 0.9, "reasoning": "good"}

        a1 = _make_agent(agent_id="a1", handler=bad_handler)
        a2 = _make_agent(agent_id="a2", handler=good_handler)
        subtasks = [
            _sub("Needs reassignment", method="llm_judge", criteria="Must be good"),
        ]
        llm = _mock_routing_llm(subtasks, verify_fn=verify_fn)
        d = Delegator(agents=[a1, a2], llm_call=llm)
        result = await d.run(_make_task(max_retries=0))
        assert result.success is True
        assert result.reassignments >= 1

    async def test_escalation_when_all_agents_fail(self):
        """When all agents fail, the result is failure."""
        async def bad_handler(task):
            return TaskResult(
                task_id=task.id, agent_id="a1", output="bad", success=True
            )

        async def verify_fn(messages):
            return {"score": 0.0, "reasoning": "always fail"}

        agent = _make_agent(handler=bad_handler)
        subtasks = [
            _sub("Impossible", method="llm_judge", criteria="Cannot pass"),
        ]
        llm = _mock_routing_llm(subtasks, verify_fn=verify_fn)
        d = Delegator(agents=[agent], llm_call=llm, max_reassignments=0)
        result = await d.run(_make_task(max_retries=0))
        assert result.success is False

    async def test_partial_results_preserved_on_failure(self):
        """Successful subtasks are preserved even when another fails."""
        async def mixed_handler(task):
            if "fail" in task.goal.lower():
                return TaskResult(
                    task_id=task.id, agent_id="a1", output="bad", success=True
                )
            return TaskResult(
                task_id=task.id, agent_id="a1", output="good", success=True
            )

        async def verify_fn(messages):
            content = messages[1]["content"]
            if "bad" in content:
                return {"score": 0.0, "reasoning": "bad"}
            return {"score": 1.0, "reasoning": "good"}

        agent = _make_agent(handler=mixed_handler)
        subtasks = [
            _sub("Good task", method="llm_judge", criteria="ok"),
            _sub("Fail task", method="llm_judge", criteria="ok"),
        ]
        llm = _mock_routing_llm(subtasks, verify_fn=verify_fn)
        d = Delegator(agents=[agent], llm_call=llm, max_reassignments=0)
        result = await d.run(_make_task(max_retries=0))
        assert len(result.subtask_results) == 2
        successes = [r for r in result.subtask_results if r.success]
        failures = [r for r in result.subtask_results if not r.success]
        assert len(successes) >= 1
        assert len(failures) >= 1


# ── TestTrustScoreLifecycle ──────────────────────────────────────────────────


class TestTrustScoreLifecycle:
    async def test_trust_increases_after_success(self):
        agent = _make_agent()
        subtasks = [_sub("Do work")]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))
        scores_before = d.get_trust_scores()
        trust_before = scores_before[agent.id]["trust"]["code"]

        await d.run(_make_task())

        scores_after = d.get_trust_scores()
        trust_after = scores_after[agent.id]["trust"]["code"]
        assert trust_after > trust_before

    async def test_trust_decreases_after_failure(self):
        async def bad_handler(task):
            return TaskResult(
                task_id=task.id, agent_id="a1", output="bad", success=True
            )

        async def verify_fn(messages):
            return {"score": 0.0, "reasoning": "fail"}

        agent = _make_agent(handler=bad_handler)
        subtasks = [
            _sub("Fail", method="llm_judge", criteria="pass"),
        ]
        llm = _mock_routing_llm(subtasks, verify_fn=verify_fn)
        d = Delegator(agents=[agent], llm_call=llm, max_reassignments=0)
        scores_before = d.get_trust_scores()
        trust_before = scores_before[agent.id]["trust"]["code"]

        await d.run(_make_task(max_retries=0))

        scores_after = d.get_trust_scores()
        trust_after = scores_after[agent.id]["trust"]["code"]
        assert trust_after < trust_before

    async def test_trust_reflects_multiple_tasks(self):
        agent = _make_agent()
        subtasks = [_sub("Do work")]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))

        await d.run(_make_task(task_id="t1"))
        scores_mid = d.get_trust_scores()
        trust_mid = scores_mid[agent.id]["trust"]["code"]

        await d.run(_make_task(task_id="t2"))
        scores_after = d.get_trust_scores()
        trust_after = scores_after[agent.id]["trust"]["code"]

        # Each success should push trust higher
        assert trust_after > trust_mid


# ── TestEventPropagation ─────────────────────────────────────────────────────


class TestEventPropagation:
    async def test_complete_event_sequence(self):
        """Successful pipeline emits decomposed, assigned, started, completed events."""
        events: list[DelegationEvent] = []

        async def collector(event):
            events.append(event)

        agent = _make_agent()
        subtasks = [_sub("Work")]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))
        d.on_all(collector)
        await d.run(_make_task())

        types = [e.type for e in events]
        assert DelegationEventType.TASK_DECOMPOSED in types
        assert DelegationEventType.TASK_ASSIGNED in types
        assert DelegationEventType.TASK_STARTED in types
        assert DelegationEventType.TASK_COMPLETED in types

    async def test_failure_event_sequence(self):
        """Failed pipeline emits verification_failed and escalated events."""
        events: list[DelegationEvent] = []

        async def collector(event):
            events.append(event)

        async def bad_handler(task):
            return TaskResult(
                task_id=task.id, agent_id="a1", output="bad", success=True
            )

        async def verify_fn(messages):
            return {"score": 0.0, "reasoning": "fail"}

        agent = _make_agent(handler=bad_handler)
        subtasks = [
            _sub("Fail", method="llm_judge", criteria="pass"),
        ]
        llm = _mock_routing_llm(subtasks, verify_fn=verify_fn)
        # Let Delegator create its own VerificationEngine so it shares the event bus
        d = Delegator(agents=[agent], llm_call=llm, max_reassignments=0)
        d.on_all(collector)
        await d.run(_make_task(max_retries=0))

        types = [e.type for e in events]
        assert DelegationEventType.VERIFICATION_FAILED in types
        assert DelegationEventType.ESCALATED in types

    async def test_event_data_populated(self):
        """Events carry relevant data payloads."""
        events: list[DelegationEvent] = []

        async def collector(event):
            events.append(event)

        agent = _make_agent()
        subtasks = [_sub("Work")]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))
        d.on_all(collector)
        await d.run(_make_task())

        decomposed = [e for e in events if e.type == DelegationEventType.TASK_DECOMPOSED]
        assert len(decomposed) == 1
        assert "subtask_count" in decomposed[0].data

        assigned = [e for e in events if e.type == DelegationEventType.TASK_ASSIGNED]
        assert len(assigned) >= 1
        assert assigned[0].agent_id is not None


# ── TestAuditLog ─────────────────────────────────────────────────────────────


class TestAuditLog:
    async def test_audit_log_records_all_events(self):
        agent = _make_agent()
        subtasks = [_sub("Work")]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))
        await d.run(_make_task())
        log = d.get_audit_log()
        assert len(log) > 0
        event_types = {entry.event_type for entry in log}
        assert "task_decomposed" in event_types
        assert "task_completed" in event_types

    async def test_audit_entries_have_timestamps(self):
        agent = _make_agent()
        subtasks = [_sub("Work")]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))
        await d.run(_make_task())
        log = d.get_audit_log()
        for entry in log:
            assert entry.timestamp is not None


# ── TestComplexityFloorIntegration ───────────────────────────────────────────


class TestComplexityFloorIntegration:
    async def test_floor_bypasses_decomposition(self):
        """Low-complexity high-reversibility task skips decomposition."""
        decompose_called = False

        async def tracking_llm(messages):
            nonlocal decompose_called
            system = messages[0]["content"].lower()
            if "task decomposition" in system:
                decompose_called = True
            return {"subtasks": [], "score": 1.0, "reasoning": "ok"}

        from delegato.trust import TrustTracker

        tracker = TrustTracker()
        agent = _make_agent()
        tracker.register_agent(agent.id, agent.capabilities)
        tracker._trust_records[agent.id]["code"].score = 0.9

        d = Delegator(
            agents=[agent],
            trust_tracker=tracker,
            llm_call=tracking_llm,
        )
        task = _make_task(
            complexity=1,
            reversibility=Reversibility.HIGH,
        )
        result = await d.run(task)
        assert result.success is True
        assert decompose_called is False

    async def test_ineligible_uses_full_pipeline(self):
        """High-complexity task uses decomposition."""
        decompose_called = False

        async def tracking_llm(messages):
            nonlocal decompose_called
            system = messages[0]["content"].lower()
            if "task decomposition" in system:
                decompose_called = True
                return {"subtasks": [
                    _sub("Sub"),
                ]}
            return {"score": 1.0, "reasoning": "ok"}

        agent = _make_agent()
        d = Delegator(agents=[agent], llm_call=tracking_llm)
        task = _make_task(complexity=4, reversibility=Reversibility.MEDIUM)
        result = await d.run(task)
        assert decompose_called is True


# ── TestCostAccumulation ─────────────────────────────────────────────────────


class TestCostAccumulation:
    async def test_cost_summed_across_subtasks(self):
        async def costed_handler(task):
            return TaskResult(
                task_id=task.id, agent_id="a1", output="ok", success=True, cost=0.25
            )

        agent = _make_agent(handler=costed_handler)
        subtasks = [_sub("A"), _sub("B"), _sub("C")]
        d = Delegator(agents=[agent], llm_call=_mock_decompose(subtasks))
        result = await d.run(_make_task())
        assert result.total_cost >= 0.75  # 3 × 0.25
