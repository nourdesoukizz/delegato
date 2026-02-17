"""Verification engine with multi-judge LLM consensus."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from delegato.events import EventBus
from delegato.models import (
    DelegationEvent,
    DelegationEventType,
    Task,
    TaskResult,
    VerificationMethod,
    VerificationSpec,
)


class VerificationError(Exception):
    """Raised when verification cannot be performed."""


class VerificationResult(BaseModel):
    passed: bool
    score: float
    details: str
    judge_scores: list[float] = []
    method: VerificationMethod


# Perspective prompts cycled across multiple judges
_JUDGE_PERSPECTIVES = [
    "Evaluate strictly — only pass if the output fully meets every criterion.",
    "Evaluate charitably — pass if the output reasonably addresses the criteria.",
    "Evaluate for completeness — does the output cover all aspects of the criteria?",
    "Evaluate for accuracy — is the output factually correct relative to the criteria?",
    "Evaluate for clarity — is the output well-structured and easy to understand?",
]


class VerificationEngine:
    """Verifies task outputs using configurable verification methods."""

    def __init__(
        self,
        *,
        model: str = "openai/gpt-4o",
        event_bus: EventBus | None = None,
        llm_call: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self._event_bus = event_bus
        self._llm_call = llm_call or self._default_llm_call
        self._custom_verifiers: dict[str, Callable] = {}

    async def _default_llm_call(self, messages: list[dict[str, str]]) -> dict:
        from delegato.llm import complete_json

        return await complete_json(model=self.model, messages=messages)

    def register_verifier(self, name: str, fn: Callable) -> None:
        """Register a custom verifier that takes (task, result) and returns a VerificationResult."""
        self._custom_verifiers[name] = fn

    async def verify(self, task: Task, result: TaskResult) -> VerificationResult:
        """Verify a task result against its verification spec."""
        spec = task.verification

        # Check custom verifiers first
        method_name = spec.method.value
        if method_name in self._custom_verifiers:
            vr = await self._call_custom(self._custom_verifiers[method_name], task, result)
        elif spec.method == VerificationMethod.NONE:
            vr = self._verify_none()
        elif spec.method == VerificationMethod.REGEX:
            vr = self._verify_regex(spec, result)
        elif spec.method == VerificationMethod.SCHEMA:
            vr = self._verify_schema(spec, result)
        elif spec.method == VerificationMethod.FUNCTION:
            vr = await self._verify_function(spec, task, result)
        elif spec.method == VerificationMethod.LLM_JUDGE:
            vr = await self._verify_llm_judge(spec, task, result)
        else:
            raise VerificationError(f"Unknown verification method: {spec.method}")

        # Emit event
        if self._event_bus:
            event_type = (
                DelegationEventType.VERIFICATION_PASSED
                if vr.passed
                else DelegationEventType.VERIFICATION_FAILED
            )
            await self._event_bus.emit(
                DelegationEvent(
                    type=event_type,
                    task_id=task.id,
                    data={
                        "score": vr.score,
                        "method": vr.method.value,
                        "details": vr.details,
                    },
                )
            )

        return vr

    # ── Built-in verifiers ───────────────────────────────────────────────

    @staticmethod
    def _verify_none() -> VerificationResult:
        return VerificationResult(
            passed=True,
            score=1.0,
            details="No verification required",
            method=VerificationMethod.NONE,
        )

    @staticmethod
    def _verify_regex(spec: VerificationSpec, result: TaskResult) -> VerificationResult:
        output_str = str(result.output)
        try:
            match = re.search(spec.criteria, output_str)
        except re.error as exc:
            raise VerificationError(f"Invalid regex pattern: {exc}") from exc

        passed = match is not None
        return VerificationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            details=f"Regex {'matched' if passed else 'did not match'}: {spec.criteria}",
            method=VerificationMethod.REGEX,
        )

    @staticmethod
    def _verify_schema(spec: VerificationSpec, result: TaskResult) -> VerificationResult:
        if spec.json_schema is None:
            raise VerificationError("Schema verification requires json_schema in VerificationSpec")

        try:
            import jsonschema
        except ImportError as exc:
            raise VerificationError(
                "jsonschema package required for schema verification"
            ) from exc

        output = result.output
        # Parse string output as JSON
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError as exc:
                return VerificationResult(
                    passed=False,
                    score=0.0,
                    details=f"Output is not valid JSON: {exc}",
                    method=VerificationMethod.SCHEMA,
                )

        try:
            jsonschema.validate(output, spec.json_schema)
            return VerificationResult(
                passed=True,
                score=1.0,
                details="Output matches JSON schema",
                method=VerificationMethod.SCHEMA,
            )
        except jsonschema.ValidationError as exc:
            return VerificationResult(
                passed=False,
                score=0.0,
                details=f"Schema validation failed: {exc.message}",
                method=VerificationMethod.SCHEMA,
            )

    @staticmethod
    async def _call_custom(
        fn: Callable, task: Task, result: TaskResult
    ) -> VerificationResult:
        """Call a custom verifier, handling sync/async and various return types."""
        try:
            if inspect.iscoroutinefunction(fn):
                rv = await fn(task, result)
            else:
                rv = fn(task, result)
        except Exception as exc:
            raise VerificationError(f"Custom verifier raised: {exc}") from exc

        if isinstance(rv, VerificationResult):
            return rv
        if isinstance(rv, bool):
            return VerificationResult(
                passed=rv,
                score=1.0 if rv else 0.0,
                details="Custom verifier returned bool",
                method=VerificationMethod.FUNCTION,
            )
        raise VerificationError(f"Custom verifier returned unexpected type: {type(rv)}")

    async def _verify_function(
        self, spec: VerificationSpec, task: Task, result: TaskResult
    ) -> VerificationResult:
        if spec.custom_fn is None:
            raise VerificationError("Function verification requires custom_fn in VerificationSpec")

        return await self._call_custom(spec.custom_fn, task, result)

    async def _verify_llm_judge(
        self, spec: VerificationSpec, task: Task, result: TaskResult
    ) -> VerificationResult:
        if spec.judges == 1:
            return await self._single_judge(spec, task, result, perspective_idx=0)

        # Multi-judge consensus
        judge_tasks = [
            self._single_judge(spec, task, result, perspective_idx=i)
            for i in range(spec.judges)
        ]
        try:
            judge_results = await asyncio.gather(*judge_tasks)
        except VerificationError:
            raise
        except Exception as exc:
            raise VerificationError(f"Multi-judge verification failed: {exc}") from exc

        scores = [jr.score for jr in judge_results]
        passed_count = sum(1 for jr in judge_results if jr.passed)
        consensus_met = (passed_count / len(judge_results)) >= spec.consensus_threshold

        return VerificationResult(
            passed=consensus_met,
            score=sum(scores) / len(scores),
            details=f"Multi-judge: {passed_count}/{len(judge_results)} passed (threshold: {spec.consensus_threshold})",
            judge_scores=scores,
            method=VerificationMethod.LLM_JUDGE,
        )

    async def _single_judge(
        self,
        spec: VerificationSpec,
        task: Task,
        result: TaskResult,
        perspective_idx: int,
    ) -> VerificationResult:
        perspective = _JUDGE_PERSPECTIVES[perspective_idx % len(_JUDGE_PERSPECTIVES)]

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a verification judge. {perspective}\n"
                    "Score the output 0.0 to 1.0 based on how well it meets the criteria.\n"
                    'Return JSON: {"score": <float>, "reasoning": "<string>"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task goal: {task.goal}\n"
                    f"Acceptance criteria: {spec.criteria}\n"
                    f"Output to verify: {result.output}"
                ),
            },
        ]

        try:
            response = await self._llm_call(messages)
        except Exception as exc:
            raise VerificationError(f"LLM judge call failed: {exc}") from exc

        score = self._extract_score(response)
        passed = score >= spec.threshold

        return VerificationResult(
            passed=passed,
            score=score,
            details=response.get("reasoning", "") if isinstance(response, dict) else str(response),
            judge_scores=[score],
            method=VerificationMethod.LLM_JUDGE,
        )

    @staticmethod
    def _extract_score(response: Any) -> float:
        if isinstance(response, dict):
            score = response.get("score", 0.0)
            try:
                return max(0.0, min(1.0, float(score)))
            except (TypeError, ValueError):
                return 0.0
        return 0.0
