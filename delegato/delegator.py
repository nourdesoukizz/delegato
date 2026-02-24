"""Delegator — main orchestrator and public API for delegato."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from delegato.assignment import AssignmentScorer
from delegato.audit import AuditLog
from delegato.coordination import CoordinationLoop, EscalationError
from delegato.decomposition import DecompositionEngine
from delegato.events import EventBus
from delegato.models import (
    Agent,
    AuditEntry,
    Contract,
    DelegationEvent,
    DelegationEventType,
    DelegationResult,
    Task,
    TaskResult,
)
from delegato.permissions import PermissionManager
from delegato.trust import TrustTracker
from delegato.verification import VerificationEngine

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """\
You are an output synthesis engine. You receive the original task goal and \
multiple sub-task outputs. Your job is to merge them into ONE coherent response \
that looks like it was written by a single author.

Rules:
1. Match the original task's format, length, and structure requirements exactly.
2. Remove duplication — if two sub-tasks cover the same point, keep the best version.
3. Fill gaps — if the original task asks for something no sub-task fully covered, \
   infer it from context or note it briefly.
4. Create narrative flow — add transitions, unify tone, fix references.
5. The final output must be a COMPLETE, STANDALONE answer to the original task.
6. Do NOT mention sub-tasks, agents, or the synthesis process in your output.

Return JSON: {"output": "<the merged response>"}\
"""


class Delegator:
    """Main orchestrator — decomposes, assigns, executes, verifies, and adapts."""

    def __init__(
        self,
        *,
        agents: list[Agent] | None = None,
        model: str = "openai/gpt-4o",
        event_bus: EventBus | None = None,
        trust_tracker: TrustTracker | None = None,
        assignment_scorer: AssignmentScorer | None = None,
        decomposition_engine: DecompositionEngine | None = None,
        verification_engine: VerificationEngine | None = None,
        permission_manager: PermissionManager | None = None,
        audit_log: AuditLog | None = None,
        llm_call: Callable[..., Any] | None = None,
        max_parallel: int = 4,
        max_reassignments: int = 3,
        smart_route_max_attempts: int = 2,
    ) -> None:
        self._model = model
        self._max_parallel = max_parallel
        self._max_reassignments = max_reassignments
        self._smart_route_max_attempts = smart_route_max_attempts
        self._llm_call = llm_call

        # Wire up components — create defaults if not provided
        self._event_bus = event_bus or EventBus()
        self._trust_tracker = trust_tracker or TrustTracker(event_bus=self._event_bus)
        self._assignment_scorer = assignment_scorer or AssignmentScorer()
        self._permission_manager = permission_manager or PermissionManager(
            event_bus=self._event_bus,
            trust_tracker=self._trust_tracker,
        )
        self._decomposition_engine = decomposition_engine or DecompositionEngine(
            model=model,
            llm_call=llm_call,
        )
        self._verification_engine = verification_engine or VerificationEngine(
            model=model,
            event_bus=self._event_bus,
            llm_call=llm_call,
        )
        self._audit_log = audit_log or AuditLog(event_bus=self._event_bus)

        # Agent registry
        self._agents: list[Agent] = []
        if agents:
            for agent in agents:
                self.register_agent(agent)

    def register_agent(self, agent: Agent) -> None:
        """Add an agent to the registry and register in trust tracker."""
        self._agents.append(agent)
        self._trust_tracker.register_agent(agent.id, agent.capabilities)

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry. No-op if not found."""
        self._agents = [a for a in self._agents if a.id != agent_id]

    def register_verifier(self, name: str, fn: Callable) -> None:
        """Register a custom verification function."""
        self._verification_engine.register_verifier(name, fn)

    def on(
        self,
        event_type: DelegationEventType,
        callback: Callable[[DelegationEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to a specific delegation event type."""
        self._event_bus.on(event_type, callback)

    def on_all(
        self,
        callback: Callable[[DelegationEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to all delegation events."""
        self._event_bus.on_all(callback)

    def get_trust_scores(self) -> dict:
        """Return current trust scores for all agents."""
        return self._trust_tracker.get_all_scores()

    def get_audit_log(self) -> list[AuditEntry]:
        """Return full audit trail."""
        return self._audit_log.get_all()

    async def run(self, task: Task) -> DelegationResult:
        """Full delegation pipeline: complexity floor → smart route → decompose → coordinate → assemble."""
        start = time.monotonic()

        # ── Tier 0: Complexity floor ────────────────────────────────────
        floor_result = self._permission_manager.check_complexity_floor(
            task, self._agents
        )
        if floor_result.eligible and floor_result.agent_id:
            return await self._fast_path(task, floor_result.agent_id, start)

        # ── Tier 1: Smart route ─────────────────────────────────────────
        smart_result = await self._smart_route(task, start)
        if smart_result is not None:
            return smart_result

        # ── Decomposition gate ──────────────────────────────────────────
        if not self._should_decompose(task):
            return await self._fallback_single_agent(
                task, start, "Single agent has sufficient capability coverage"
            )

        # ── Tier 2: Decompose ───────────────────────────────────────────
        try:
            dag = await self._decomposition_engine.decompose(task)
        except Exception as exc:
            return await self._fallback_single_agent(
                task, start, f"Decomposition failed: {exc}"
            )

        # Check for empty DAG (decomposition returned 0 subtasks)
        executable_tasks = [tid for tid in dag.tasks if tid != dag.root_task_id]
        if not executable_tasks:
            return await self._fallback_single_agent(
                task, start, "Decomposition produced empty DAG (0 subtasks)"
            )

        await self._event_bus.emit(
            DelegationEvent(
                type=DelegationEventType.TASK_DECOMPOSED,
                task_id=task.id,
                data={"subtask_count": len(dag.tasks) - 1},  # Exclude root
            )
        )

        # ── Coordinate ──────────────────────────────────────────────────
        loop = CoordinationLoop(
            agents=self._agents,
            event_bus=self._event_bus,
            trust_tracker=self._trust_tracker,
            assignment_scorer=self._assignment_scorer,
            verification_engine=self._verification_engine,
            permission_manager=self._permission_manager,
            max_parallel=self._max_parallel,
            max_reassignments=self._max_reassignments,
        )

        results, reassignments = await loop.execute_dag(dag)

        # If all subtasks failed, fall back to single-agent execution
        if not any(r.success for r in results):
            return await self._fallback_single_agent(
                task, start, "All subtask results failed"
            )

        # ── Assemble ────────────────────────────────────────────────────
        elapsed = time.monotonic() - start
        all_success = all(r.success for r in results) and len(results) > 0
        total_cost = sum(r.cost for r in results)

        # Collect outputs
        successful_outputs = [r.output for r in results if r.success]
        if len(successful_outputs) > 1 and self._llm_call is not None:
            output = await self._synthesize_output(task, successful_outputs)
        elif len(successful_outputs) == 1:
            output = successful_outputs[0]
        else:
            output = successful_outputs or None

        return DelegationResult(
            task=task,
            success=all_success,
            output=output,
            subtask_results=results,
            total_cost=total_cost,
            total_duration=elapsed,
            reassignments=reassignments,
        )

    async def _synthesize_output(
        self, task: Task, outputs: list[Any]
    ) -> Any:
        """Merge multiple subtask outputs into a single coherent answer."""
        numbered = "\n\n".join(
            f"--- Sub-task {i+1} output ---\n{o}" for i, o in enumerate(outputs)
        )
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original task goal:\n{task.goal}\n\n"
                    f"Sub-task outputs to merge:\n{numbered}"
                ),
            },
        ]
        try:
            response = await self._llm_call(messages)
            if isinstance(response, dict) and "output" in response:
                return response["output"]
            # If the LLM returned something unexpected, fall back
            logger.warning("Synthesis returned unexpected format, falling back to concatenation")
            return "\n\n".join(str(o) for o in outputs)
        except Exception:
            logger.warning("Synthesis failed, falling back to concatenation", exc_info=True)
            return "\n\n".join(str(o) for o in outputs)

    async def _fast_path(
        self, task: Task, agent_id: str, start: float
    ) -> DelegationResult:
        """Execute a simple task directly — bypass decomposition."""
        agent = self._find_agent(agent_id)
        if agent is None:
            elapsed = time.monotonic() - start
            return DelegationResult(
                task=task,
                success=False,
                output=None,
                total_duration=elapsed,
            )

        await self._event_bus.emit(
            DelegationEvent(
                type=DelegationEventType.TASK_ASSIGNED,
                task_id=task.id,
                agent_id=agent.id,
            )
        )

        try:
            result = await agent.handler(task)
        except Exception:
            elapsed = time.monotonic() - start
            return DelegationResult(
                task=task,
                success=False,
                output=None,
                total_duration=elapsed,
            )

        # Fast path skips verification — trust was already established
        # by the complexity floor check
        elapsed = time.monotonic() - start

        return DelegationResult(
            task=task,
            success=result.success,
            output=result.output,
            subtask_results=[result],
            total_cost=result.cost,
            total_duration=elapsed,
        )

    async def _smart_route(
        self, task: Task, start: float
    ) -> DelegationResult | None:
        """Tier 1: try top-N ranked agents with verification.

        Iterates up to ``smart_route_max_attempts`` agents from ``rank_agents()``.
        Returns DelegationResult on first verification pass, or None to fall
        through to decomposition gate / Tier 2.
        """
        ranked = self._assignment_scorer.rank_agents(
            task, self._agents, self._trust_tracker
        )
        if not ranked:
            return None

        max_attempts = min(self._smart_route_max_attempts, len(ranked))

        for i in range(max_attempts):
            agent, score = ranked[i]
            if score < self._assignment_scorer.min_threshold:
                break  # remaining agents are below threshold

            await self._event_bus.emit(
                DelegationEvent(
                    type=DelegationEventType.TASK_ASSIGNED,
                    task_id=task.id,
                    agent_id=agent.id,
                )
            )

            try:
                result = await agent.handler(task)
            except Exception:
                logger.debug("Smart route agent %s raised, trying next", agent.id)
                continue

            if not result.success:
                continue

            # Verify output quality
            task_result = TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                output=result.output,
                success=result.success,
                cost=result.cost,
                duration_seconds=result.duration_seconds,
            )
            vr = await self._verification_engine.verify(task, task_result)

            # Update trust based on verification
            if task.required_capabilities:
                cap = task.required_capabilities[0]
                await self._trust_tracker.update_trust(
                    agent.id, cap, verified=vr.passed
                )

            if not vr.passed:
                logger.info(
                    "Smart route attempt %d/%d failed (agent=%s, score=%.2f) "
                    "for task %s",
                    i + 1,
                    max_attempts,
                    agent.id,
                    vr.score,
                    task.id,
                )
                continue

            # Success — return immediately, skip decomposition
            elapsed = time.monotonic() - start
            await self._event_bus.emit(
                DelegationEvent(
                    type=DelegationEventType.TASK_COMPLETED,
                    task_id=task.id,
                    agent_id=agent.id,
                    data={"tier": "smart_route", "attempt": i + 1},
                )
            )
            return DelegationResult(
                task=task,
                success=True,
                output=result.output,
                subtask_results=[task_result],
                total_cost=result.cost,
                total_duration=elapsed,
            )

        # All attempts exhausted
        logger.info(
            "Smart route exhausted %d attempts for task %s, falling through",
            max_attempts,
            task.id,
        )
        return None

    async def _fallback_single_agent(
        self, task: Task, start: float, reason: str
    ) -> DelegationResult:
        """Last resort: run original task with single best agent, no decomposition."""
        logger.warning("Fallback triggered: %s (task=%s)", reason, task.id)
        agent = self._assignment_scorer.select_best(
            task, self._agents, self._trust_tracker
        )
        if agent is None:
            elapsed = time.monotonic() - start
            return DelegationResult(
                task=task, success=False, output=None, total_duration=elapsed,
            )

        try:
            result = await agent.handler(task)
        except Exception:
            elapsed = time.monotonic() - start
            return DelegationResult(
                task=task, success=False, output=None, total_duration=elapsed,
            )

        elapsed = time.monotonic() - start
        return DelegationResult(
            task=task,
            success=result.success,
            output=result.output,
            subtask_results=[result],
            total_cost=result.cost,
            total_duration=elapsed,
        )

    def _should_decompose(self, task: Task) -> bool:
        """Only decompose if no single agent covers majority of capabilities."""
        if not task.required_capabilities:
            return False
        for agent in self._agents:
            matched = len(set(task.required_capabilities) & set(agent.capabilities))
            if matched >= len(task.required_capabilities) * 0.6:
                return False  # Single agent covers enough
        return True

    def _find_agent(self, agent_id: str) -> Agent | None:
        """Lookup helper."""
        for agent in self._agents:
            if agent.id == agent_id:
                return agent
        return None
