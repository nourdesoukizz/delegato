"""Delegator — main orchestrator and public API for delegato."""

from __future__ import annotations

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
    ) -> None:
        self._model = model
        self._max_parallel = max_parallel
        self._max_reassignments = max_reassignments

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
        """Full delegation pipeline: complexity floor → decompose → coordinate → assemble."""
        start = time.monotonic()

        # ── Complexity floor check ──────────────────────────────────────
        floor_result = self._permission_manager.check_complexity_floor(
            task, self._agents
        )
        if floor_result.eligible and floor_result.agent_id:
            return await self._fast_path(task, floor_result.agent_id, start)

        # ── Decompose ───────────────────────────────────────────────────
        try:
            dag = await self._decomposition_engine.decompose(task)
        except Exception as exc:
            elapsed = time.monotonic() - start
            return DelegationResult(
                task=task,
                success=False,
                output=None,
                total_duration=elapsed,
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

        # ── Assemble ────────────────────────────────────────────────────
        elapsed = time.monotonic() - start
        all_success = all(r.success for r in results) and len(results) > 0
        total_cost = sum(r.cost for r in results)

        # Collect outputs
        output = [r.output for r in results if r.success] or None
        if output and len(output) == 1:
            output = output[0]

        return DelegationResult(
            task=task,
            success=all_success,
            output=output,
            subtask_results=results,
            total_cost=total_cost,
            total_duration=elapsed,
            reassignments=reassignments,
        )

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

        # Verify even on fast path
        vr = await self._verification_engine.verify(task, result)

        # Update trust
        if task.required_capabilities:
            cap = task.required_capabilities[0]
            await self._trust_tracker.update_trust(agent.id, cap, verified=vr.passed)

        elapsed = time.monotonic() - start

        return DelegationResult(
            task=task,
            success=vr.passed,
            output=result.output,
            subtask_results=[result],
            total_cost=result.cost,
            total_duration=elapsed,
        )

    def _find_agent(self, agent_id: str) -> Agent | None:
        """Lookup helper."""
        for agent in self._agents:
            if agent.id == agent_id:
                return agent
        return None
