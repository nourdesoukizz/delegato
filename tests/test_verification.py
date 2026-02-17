"""Tests for delegato verification engine — all LLM calls are mocked."""

from __future__ import annotations

import pytest

from delegato.events import EventBus
from delegato.models import (
    DelegationEvent,
    DelegationEventType,
    Task,
    TaskResult,
    VerificationMethod,
    VerificationSpec,
)
from delegato.verification import VerificationEngine, VerificationError, VerificationResult


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_task(method=VerificationMethod.NONE, **kwargs) -> Task:
    spec_kwargs = {"method": method}
    spec_kwargs.update(kwargs)
    return Task(goal="Test task", verification=VerificationSpec(**spec_kwargs))


def _make_result(output="hello world", success=True) -> TaskResult:
    return TaskResult(task_id="t1", agent_id="a1", output=output, success=success)


def _mock_llm(score=0.9, reasoning="Good"):
    async def mock_call(messages):
        return {"score": score, "reasoning": reasoning}
    return mock_call


# ── NONE Verification ────────────────────────────────────────────────────────


class TestVerifyNone:
    async def test_always_passes(self):
        engine = VerificationEngine()
        task = _make_task(method=VerificationMethod.NONE)
        result = _make_result()
        vr = await engine.verify(task, result)
        assert vr.passed is True
        assert vr.score == 1.0
        assert vr.method == VerificationMethod.NONE


# ── REGEX Verification ───────────────────────────────────────────────────────


class TestVerifyRegex:
    async def test_match_passes(self):
        engine = VerificationEngine()
        task = _make_task(method=VerificationMethod.REGEX, criteria=r"hello\s+world")
        vr = await engine.verify(task, _make_result(output="hello world"))
        assert vr.passed is True
        assert vr.score == 1.0

    async def test_no_match_fails(self):
        engine = VerificationEngine()
        task = _make_task(method=VerificationMethod.REGEX, criteria=r"^goodbye$")
        vr = await engine.verify(task, _make_result(output="hello world"))
        assert vr.passed is False
        assert vr.score == 0.0

    async def test_bad_regex_raises(self):
        engine = VerificationEngine()
        task = _make_task(method=VerificationMethod.REGEX, criteria=r"[invalid")
        with pytest.raises(VerificationError, match="Invalid regex"):
            await engine.verify(task, _make_result())

    async def test_non_string_output_converted(self):
        engine = VerificationEngine()
        task = _make_task(method=VerificationMethod.REGEX, criteria=r"42")
        vr = await engine.verify(task, _make_result(output=42))
        assert vr.passed is True


# ── SCHEMA Verification ──────────────────────────────────────────────────────


class TestVerifySchema:
    async def test_valid_passes(self):
        engine = VerificationEngine()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        task = _make_task(method=VerificationMethod.SCHEMA, json_schema=schema)
        vr = await engine.verify(task, _make_result(output={"name": "Alice"}))
        assert vr.passed is True
        assert vr.score == 1.0

    async def test_invalid_fails(self):
        engine = VerificationEngine()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        task = _make_task(method=VerificationMethod.SCHEMA, json_schema=schema)
        vr = await engine.verify(task, _make_result(output={"age": 30}))
        assert vr.passed is False
        assert vr.score == 0.0

    async def test_string_json_parsed(self):
        engine = VerificationEngine()
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        task = _make_task(method=VerificationMethod.SCHEMA, json_schema=schema)
        vr = await engine.verify(task, _make_result(output='{"x": 1}'))
        assert vr.passed is True

    async def test_missing_schema_raises(self):
        engine = VerificationEngine()
        task = _make_task(method=VerificationMethod.SCHEMA)
        with pytest.raises(VerificationError, match="requires json_schema"):
            await engine.verify(task, _make_result())

    async def test_invalid_json_string_fails(self):
        engine = VerificationEngine()
        schema = {"type": "object"}
        task = _make_task(method=VerificationMethod.SCHEMA, json_schema=schema)
        vr = await engine.verify(task, _make_result(output="not json"))
        assert vr.passed is False


# ── FUNCTION Verification ────────────────────────────────────────────────────


class TestVerifyFunction:
    async def test_bool_true(self):
        engine = VerificationEngine()
        task = _make_task(
            method=VerificationMethod.FUNCTION,
            custom_fn=lambda t, r: True,
        )
        vr = await engine.verify(task, _make_result())
        assert vr.passed is True

    async def test_bool_false(self):
        engine = VerificationEngine()
        task = _make_task(
            method=VerificationMethod.FUNCTION,
            custom_fn=lambda t, r: False,
        )
        vr = await engine.verify(task, _make_result())
        assert vr.passed is False

    async def test_returns_verification_result(self):
        def custom(task, result):
            return VerificationResult(
                passed=True, score=0.95, details="Custom OK", method=VerificationMethod.FUNCTION
            )

        engine = VerificationEngine()
        task = _make_task(method=VerificationMethod.FUNCTION, custom_fn=custom)
        vr = await engine.verify(task, _make_result())
        assert vr.score == 0.95
        assert vr.details == "Custom OK"

    async def test_async_fn(self):
        async def async_custom(task, result):
            return True

        engine = VerificationEngine()
        task = _make_task(method=VerificationMethod.FUNCTION, custom_fn=async_custom)
        vr = await engine.verify(task, _make_result())
        assert vr.passed is True

    async def test_missing_fn_raises(self):
        engine = VerificationEngine()
        task = _make_task(method=VerificationMethod.FUNCTION)
        with pytest.raises(VerificationError, match="requires custom_fn"):
            await engine.verify(task, _make_result())

    async def test_exception_raises(self):
        def bad_fn(task, result):
            raise ValueError("boom")

        engine = VerificationEngine()
        task = _make_task(method=VerificationMethod.FUNCTION, custom_fn=bad_fn)
        with pytest.raises(VerificationError, match="Custom verifier raised"):
            await engine.verify(task, _make_result())


# ── LLM_JUDGE Verification ───────────────────────────────────────────────────


class TestVerifyLLMJudge:
    async def test_single_judge_pass(self):
        engine = VerificationEngine(llm_call=_mock_llm(score=0.9))
        task = _make_task(method=VerificationMethod.LLM_JUDGE, criteria="Is correct", threshold=0.7)
        vr = await engine.verify(task, _make_result())
        assert vr.passed is True
        assert vr.score == 0.9

    async def test_single_judge_fail(self):
        engine = VerificationEngine(llm_call=_mock_llm(score=0.3))
        task = _make_task(method=VerificationMethod.LLM_JUDGE, criteria="Is correct", threshold=0.7)
        vr = await engine.verify(task, _make_result())
        assert vr.passed is False
        assert vr.score == 0.3

    async def test_llm_error_raises(self):
        async def failing_llm(messages):
            raise Exception("API down")

        engine = VerificationEngine(llm_call=failing_llm)
        task = _make_task(method=VerificationMethod.LLM_JUDGE, criteria="test")
        with pytest.raises(VerificationError, match="LLM judge call failed"):
            await engine.verify(task, _make_result())


# ── Multi-Judge Consensus ────────────────────────────────────────────────────


class TestMultiJudgeConsensus:
    async def test_majority_passes(self):
        call_count = 0

        async def multi_llm(messages):
            nonlocal call_count
            call_count += 1
            # 2 out of 3 judges pass
            return {"score": 0.9 if call_count != 2 else 0.3, "reasoning": "ok"}

        engine = VerificationEngine(llm_call=multi_llm)
        task = _make_task(
            method=VerificationMethod.LLM_JUDGE,
            criteria="test",
            threshold=0.7,
            judges=3,
            consensus_threshold=0.66,
        )
        vr = await engine.verify(task, _make_result())
        assert vr.passed is True
        assert len(vr.judge_scores) == 3

    async def test_minority_fails(self):
        call_count = 0

        async def multi_llm(messages):
            nonlocal call_count
            call_count += 1
            # Only 1 out of 3 passes
            return {"score": 0.9 if call_count == 1 else 0.3, "reasoning": "ok"}

        engine = VerificationEngine(llm_call=multi_llm)
        task = _make_task(
            method=VerificationMethod.LLM_JUDGE,
            criteria="test",
            threshold=0.7,
            judges=3,
            consensus_threshold=0.66,
        )
        vr = await engine.verify(task, _make_result())
        assert vr.passed is False

    async def test_different_perspectives_used(self):
        perspectives_seen = []

        async def tracking_llm(messages):
            system_msg = messages[0]["content"]
            perspectives_seen.append(system_msg)
            return {"score": 0.9, "reasoning": "ok"}

        engine = VerificationEngine(llm_call=tracking_llm)
        task = _make_task(
            method=VerificationMethod.LLM_JUDGE,
            criteria="test",
            threshold=0.7,
            judges=3,
        )
        await engine.verify(task, _make_result())
        # Each judge should get a different perspective
        assert len(perspectives_seen) == 3
        assert perspectives_seen[0] != perspectives_seen[1]
        assert perspectives_seen[1] != perspectives_seen[2]


# ── Custom Verifiers ─────────────────────────────────────────────────────────


class TestCustomVerifiers:
    async def test_register_and_use(self):
        def custom(task, result):
            return VerificationResult(
                passed=True, score=0.99, details="Custom pass", method=VerificationMethod.FUNCTION
            )

        engine = VerificationEngine()
        engine.register_verifier("none", custom)
        task = _make_task(method=VerificationMethod.NONE)
        vr = await engine.verify(task, _make_result())
        assert vr.score == 0.99
        assert vr.details == "Custom pass"

    async def test_override_builtin(self):
        def custom_regex(task, result):
            return VerificationResult(
                passed=True, score=0.5, details="Custom regex", method=VerificationMethod.REGEX
            )

        engine = VerificationEngine()
        engine.register_verifier("regex", custom_regex)
        task = _make_task(method=VerificationMethod.REGEX, criteria=r"^no_match$")
        vr = await engine.verify(task, _make_result(output="hello"))
        # Custom should override, so it passes even though regex wouldn't match
        assert vr.passed is True
        assert vr.details == "Custom regex"


# ── Event Emission ───────────────────────────────────────────────────────────


class TestEventEmission:
    async def test_pass_emits_verification_passed(self):
        events: list[DelegationEvent] = []

        async def listener(event):
            events.append(event)

        bus = EventBus()
        bus.on(DelegationEventType.VERIFICATION_PASSED, listener)
        engine = VerificationEngine(event_bus=bus)
        task = _make_task(method=VerificationMethod.NONE)
        await engine.verify(task, _make_result())
        assert len(events) == 1
        assert events[0].type == DelegationEventType.VERIFICATION_PASSED

    async def test_fail_emits_verification_failed(self):
        events: list[DelegationEvent] = []

        async def listener(event):
            events.append(event)

        bus = EventBus()
        bus.on(DelegationEventType.VERIFICATION_FAILED, listener)
        engine = VerificationEngine(event_bus=bus)
        task = _make_task(method=VerificationMethod.REGEX, criteria=r"^no_match$")
        await engine.verify(task, _make_result(output="hello"))
        assert len(events) == 1
        assert events[0].type == DelegationEventType.VERIFICATION_FAILED
