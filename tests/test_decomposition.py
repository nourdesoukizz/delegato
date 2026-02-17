"""Tests for delegato decomposition engine — all LLM calls are mocked."""

from __future__ import annotations

import pytest

from delegato.decomposition import DecompositionEngine, DecompositionError
from delegato.models import (
    Reversibility,
    Task,
    VerificationMethod,
    VerificationSpec,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_task(**kwargs) -> Task:
    defaults = {
        "goal": "Research AI in healthcare",
        "verification": VerificationSpec(method=VerificationMethod.LLM_JUDGE),
    }
    defaults.update(kwargs)
    return Task(**defaults)


def _mock_llm_response(subtasks: list[dict]):
    """Return an async callable that returns a fixed decomposition response."""
    async def mock_call(messages):
        return {"subtasks": subtasks}
    return mock_call


def _basic_subtasks():
    return [
        {
            "goal": "Search for AI healthcare papers",
            "required_capabilities": ["web_search"],
            "verification_method": "schema",
            "verification_criteria": "Returns at least 3 results",
            "dependencies": [],
        },
        {
            "goal": "Analyze search results",
            "required_capabilities": ["data_analysis"],
            "verification_method": "llm_judge",
            "verification_criteria": "Extracts key claims",
            "dependencies": [0],
        },
        {
            "goal": "Write summary report",
            "required_capabilities": ["summarization"],
            "verification_method": "llm_judge",
            "verification_criteria": "500 words, 3+ examples",
            "dependencies": [1],
        },
    ]


# ── Basic Decomposition Tests ────────────────────────────────────────────────


class TestDecomposition:
    async def test_basic_decomposition(self):
        engine = DecompositionEngine(llm_call=_mock_llm_response(_basic_subtasks()))
        task = _make_task()
        dag = await engine.decompose(task)
        # Root task + 3 subtasks
        assert len(dag.get_all_tasks()) == 4
        assert dag.root_task_id == task.id

    async def test_subtask_goals(self):
        engine = DecompositionEngine(llm_call=_mock_llm_response(_basic_subtasks()))
        dag = await engine.decompose(_make_task())
        goals = [t.goal for t in dag.get_all_tasks()]
        assert "Search for AI healthcare papers" in goals
        assert "Analyze search results" in goals
        assert "Write summary report" in goals

    async def test_dependency_resolution(self):
        engine = DecompositionEngine(llm_call=_mock_llm_response(_basic_subtasks()))
        dag = await engine.decompose(_make_task())
        # Get subtasks (exclude root)
        subtasks = [t for t in dag.get_all_tasks() if t.goal != "Research AI in healthcare"]
        search = next(t for t in subtasks if "Search" in t.goal)
        analyze = next(t for t in subtasks if "Analyze" in t.goal)
        write = next(t for t in subtasks if "Write" in t.goal)
        # search has no deps
        assert dag.dependencies[search.id] == []
        # analyze depends on search
        assert search.id in dag.dependencies[analyze.id]
        # write depends on analyze
        assert analyze.id in dag.dependencies[write.id]

    async def test_topological_order(self):
        engine = DecompositionEngine(llm_call=_mock_llm_response(_basic_subtasks()))
        dag = await engine.decompose(_make_task())
        order = dag.topological_sort()
        ids = [t.id for t in order]
        subtasks = [t for t in dag.get_all_tasks() if t.goal != "Research AI in healthcare"]
        search = next(t for t in subtasks if "Search" in t.goal)
        analyze = next(t for t in subtasks if "Analyze" in t.goal)
        write = next(t for t in subtasks if "Write" in t.goal)
        assert ids.index(search.id) < ids.index(analyze.id)
        assert ids.index(analyze.id) < ids.index(write.id)


# ── Max Subtasks Enforcement ─────────────────────────────────────────────────


class TestMaxSubtasks:
    async def test_max_subtasks_enforced(self):
        many_subtasks = [
            {
                "goal": f"subtask {i}",
                "required_capabilities": [],
                "verification_method": "llm_judge",
                "verification_criteria": "test",
                "dependencies": [],
            }
            for i in range(10)
        ]
        engine = DecompositionEngine(
            llm_call=_mock_llm_response(many_subtasks),
            max_subtasks=4,
        )
        dag = await engine.decompose(_make_task())
        # Root + at most 4 subtasks
        assert len(dag.get_all_tasks()) <= 5


# ── Max Depth Tests ──────────────────────────────────────────────────────────


class TestMaxDepth:
    async def test_max_depth_stops_recursion(self):
        """Tasks with NONE verification should NOT recurse beyond max_depth."""
        call_count = 0

        async def counting_llm(messages):
            nonlocal call_count
            call_count += 1
            return {
                "subtasks": [
                    {
                        "goal": f"sub-{call_count}",
                        "required_capabilities": [],
                        "verification_method": "none",
                        "verification_criteria": "",
                        "dependencies": [],
                    }
                ]
            }

        engine = DecompositionEngine(llm_call=counting_llm, max_depth=2)
        dag = await engine.decompose(_make_task())
        # depth 0: decompose root → sub-1 (NONE)
        # depth 1: decompose sub-1 → sub-2 (NONE)
        # depth 2: stop (max_depth reached)
        assert call_count == 2


# ── Recursive Decomposition Tests ────────────────────────────────────────────


class TestRecursiveDecomposition:
    async def test_recurse_on_none_verification(self):
        """Sub-tasks with NONE verification should be recursively decomposed."""
        call_count = 0

        async def recursive_llm(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "subtasks": [
                        {
                            "goal": "vague task",
                            "required_capabilities": [],
                            "verification_method": "none",
                            "verification_criteria": "",
                            "dependencies": [],
                        }
                    ]
                }
            else:
                return {
                    "subtasks": [
                        {
                            "goal": "concrete task",
                            "required_capabilities": ["coding"],
                            "verification_method": "regex",
                            "verification_criteria": "matches pattern",
                            "dependencies": [],
                        }
                    ]
                }

        engine = DecompositionEngine(llm_call=recursive_llm, max_depth=3)
        dag = await engine.decompose(_make_task())
        # root + vague task + concrete task
        assert len(dag.get_all_tasks()) == 3
        goals = [t.goal for t in dag.get_all_tasks()]
        assert "concrete task" in goals

    async def test_no_recurse_on_non_none_verification(self):
        """Sub-tasks with non-NONE verification should NOT be recursively decomposed."""
        call_count = 0

        async def mock_llm(messages):
            nonlocal call_count
            call_count += 1
            return {
                "subtasks": [
                    {
                        "goal": "verified task",
                        "required_capabilities": [],
                        "verification_method": "llm_judge",
                        "verification_criteria": "check quality",
                        "dependencies": [],
                    }
                ]
            }

        engine = DecompositionEngine(llm_call=mock_llm, max_depth=3)
        dag = await engine.decompose(_make_task())
        # Should only call LLM once (for root decomposition, no recursion)
        assert call_count == 1


# ── Error Handling Tests ─────────────────────────────────────────────────────


class TestErrorHandling:
    async def test_llm_error_raises_decomposition_error(self):
        async def failing_llm(messages):
            raise Exception("API down")

        engine = DecompositionEngine(llm_call=failing_llm)
        with pytest.raises(DecompositionError, match="LLM call failed"):
            await engine.decompose(_make_task())

    async def test_empty_response(self):
        engine = DecompositionEngine(llm_call=_mock_llm_response([]))
        dag = await engine.decompose(_make_task())
        # Only root task
        assert len(dag.get_all_tasks()) == 1

    async def test_invalid_response_format(self):
        async def bad_llm(messages):
            return "not a dict"

        engine = DecompositionEngine(llm_call=bad_llm)
        dag = await engine.decompose(_make_task())
        # Should handle gracefully, only root
        assert len(dag.get_all_tasks()) == 1

    async def test_empty_goal_skipped(self):
        subtasks = [
            {
                "goal": "",
                "required_capabilities": [],
                "verification_method": "llm_judge",
                "verification_criteria": "",
                "dependencies": [],
            },
            {
                "goal": "valid task",
                "required_capabilities": [],
                "verification_method": "llm_judge",
                "verification_criteria": "check",
                "dependencies": [],
            },
        ]
        engine = DecompositionEngine(llm_call=_mock_llm_response(subtasks))
        dag = await engine.decompose(_make_task())
        goals = [t.goal for t in dag.get_all_tasks()]
        assert "" not in goals
        assert "valid task" in goals


# ── Invalid Input Handling ───────────────────────────────────────────────────


class TestInvalidInputs:
    async def test_invalid_verification_method_falls_back(self):
        subtasks = [
            {
                "goal": "task with bad method",
                "required_capabilities": [],
                "verification_method": "invalid_method",
                "verification_criteria": "test",
                "dependencies": [],
            }
        ]
        engine = DecompositionEngine(llm_call=_mock_llm_response(subtasks))
        dag = await engine.decompose(_make_task())
        subtask = [t for t in dag.get_all_tasks() if t.goal == "task with bad method"][0]
        assert subtask.verification.method == VerificationMethod.LLM_JUDGE

    async def test_bad_dependency_indices_skipped(self):
        subtasks = [
            {
                "goal": "task a",
                "required_capabilities": [],
                "verification_method": "llm_judge",
                "verification_criteria": "test",
                "dependencies": [99, -1, "invalid"],
            }
        ]
        engine = DecompositionEngine(llm_call=_mock_llm_response(subtasks))
        dag = await engine.decompose(_make_task())
        subtask = [t for t in dag.get_all_tasks() if t.goal == "task a"][0]
        assert dag.dependencies[subtask.id] == []


# ── Parent Property Inheritance ──────────────────────────────────────────────


class TestPropertyInheritance:
    async def test_inherits_priority(self):
        engine = DecompositionEngine(llm_call=_mock_llm_response(_basic_subtasks()))
        task = _make_task(priority=5)
        dag = await engine.decompose(task)
        for t in dag.get_all_tasks():
            if t.id != task.id:
                assert t.priority == 5

    async def test_inherits_reversibility(self):
        engine = DecompositionEngine(llm_call=_mock_llm_response(_basic_subtasks()))
        task = _make_task(reversibility=Reversibility.LOW)
        dag = await engine.decompose(task)
        for t in dag.get_all_tasks():
            if t.id != task.id:
                assert t.reversibility == Reversibility.LOW

    async def test_complexity_decremented(self):
        engine = DecompositionEngine(llm_call=_mock_llm_response(_basic_subtasks()))
        task = _make_task(complexity=4)
        dag = await engine.decompose(task)
        for t in dag.get_all_tasks():
            if t.id != task.id:
                assert t.complexity == 3

    async def test_complexity_floor_at_one(self):
        engine = DecompositionEngine(llm_call=_mock_llm_response(_basic_subtasks()))
        task = _make_task(complexity=1)
        dag = await engine.decompose(task)
        for t in dag.get_all_tasks():
            if t.id != task.id:
                assert t.complexity == 1

    async def test_parent_id_set(self):
        engine = DecompositionEngine(llm_call=_mock_llm_response(_basic_subtasks()))
        task = _make_task()
        dag = await engine.decompose(task)
        for t in dag.get_all_tasks():
            if t.id != task.id:
                assert t.parent_id == task.id
