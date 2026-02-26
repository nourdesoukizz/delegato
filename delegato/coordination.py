"""Coordination loop — parallel DAG execution with retry, reassignment, and escalation."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

from delegato.events import EventBus
from delegato.models import (
    Agent,
    Contract,
    DelegationEvent,
    DelegationEventType,
    Task,
    TaskDAG,
    TaskResult,
    VerificationMethod,
    VerificationSpec,
)
from delegato.assignment import AssignmentScorer
from delegato.permissions import PermissionManager
from delegato.trust import TrustTracker
from delegato.verification import VerificationEngine, VerificationResult


class EscalationError(Exception):
    """Raised when a task exhausts all recovery options."""


class CoordinationLoop:
    """Executes a TaskDAG in parallel batches with retry, reassignment, and escalation."""

    def __init__(
        self,
        *,
        agents: list[Agent],
        event_bus: EventBus | None = None,
        trust_tracker: TrustTracker | None = None,
        assignment_scorer: AssignmentScorer | None = None,
        verification_engine: VerificationEngine | None = None,
        permission_manager: PermissionManager | None = None,
        max_parallel: int = 4,
        max_reassignments: int = 3,
    ) -> None:
        self._agents = list(agents)
        self._event_bus = event_bus
        self._trust_tracker = trust_tracker or TrustTracker()
        self._assignment_scorer = assignment_scorer or AssignmentScorer()
        self._verification_engine = verification_engine or VerificationEngine()
        self._permission_manager = permission_manager
        self._max_parallel = max_parallel
        self._max_reassignments = max_reassignments

        # Per-run state — reset each execute_dag call
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._results: dict[str, TaskResult] = {}
        self._contracts: list[Contract] = []
        self._reassignment_counts: dict[str, int] = {}
        self._total_reassignments: int = 0

    async def execute_dag(self, dag: TaskDAG) -> tuple[list[TaskResult], int]:
        """Execute all tasks in the DAG in topological-batch order.

        Returns (list of TaskResults, total reassignment count).
        Skips the root task (container task, not executable).
        """
        # Reset per-run state
        self._completed = set()
        self._failed = set()
        self._results = {}
        self._contracts = []
        self._reassignment_counts = {}
        self._total_reassignments = 0

        # Mark the root task as "completed" so its dependents can run
        if dag.root_task_id:
            self._completed.add(dag.root_task_id)

        sem = asyncio.Semaphore(self._max_parallel)

        while True:
            ready = dag.get_ready_tasks(self._completed | self._failed)
            # Filter out root task and already-processed tasks
            executable = [
                t
                for t in ready
                if t.id != dag.root_task_id
                and t.id not in self._completed
                and t.id not in self._failed
            ]

            if not executable:
                break

            coros = [self._execute_single_task(task, sem) for task in executable]
            await asyncio.gather(*coros)

        return list(self._results.values()), self._total_reassignments

    async def _execute_single_task(self, task: Task, sem: asyncio.Semaphore) -> None:
        """Assess, contract, execute+verify a single task. Handle failure if needed."""
        async with sem:
            # Assess — select best agent
            agent = self._assignment_scorer.select_best(
                task, self._agents, self._trust_tracker
            )
            if agent is None:
                # No suitable agent — escalate immediately
                await self._escalate(task, "No suitable agent found")
                return

            await self._emit(
                DelegationEvent(
                    type=DelegationEventType.TASK_ASSIGNED,
                    task_id=task.id,
                    agent_id=agent.id,
                )
            )

            # Contract
            contract = Contract(
                task=task,
                agent_id=agent.id,
                verification=task.verification,
            )
            self._contracts.append(contract)

            # Execute + verify
            await self._emit(
                DelegationEvent(
                    type=DelegationEventType.TASK_STARTED,
                    task_id=task.id,
                    agent_id=agent.id,
                )
            )

            result, vr = await self._run_and_verify(task, agent, contract)

            if vr.passed:
                result = TaskResult(
                    task_id=result.task_id,
                    agent_id=result.agent_id,
                    output=result.output,
                    success=True,
                    verified=True,
                    verification_details=vr.details,
                    cost=result.cost,
                    duration_seconds=result.duration_seconds,
                )
                self._results[task.id] = result
                self._completed.add(task.id)
                contract.result = result
                contract.completed_at = datetime.now(UTC)
                await self._emit(
                    DelegationEvent(
                        type=DelegationEventType.TASK_COMPLETED,
                        task_id=task.id,
                        agent_id=agent.id,
                        data={"verified": True},
                    )
                )
            else:
                # Handle failure — retry / reassign / escalate
                await self._handle_failure(task, agent, contract, result, vr)

    async def _run_and_verify(
        self, task: Task, agent: Agent, contract: Contract,
        *, verify_task: Task | None = None,
    ) -> tuple[TaskResult, VerificationResult]:
        """Execute agent handler with timeout, verify output, update trust, check circuit breaker.

        Args:
            task: The task to execute (may include retry feedback in goal).
            verify_task: If provided, use this task for verification instead of ``task``.
                         This separates agent feedback from verification context.
        """
        vtask = verify_task or task
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                agent.handler(task), timeout=task.timeout_seconds
            )
            elapsed = time.monotonic() - start
            result = TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                output=result.output,
                success=result.success,
                cost=result.cost,
                duration_seconds=elapsed,
            )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            result = TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                output=None,
                success=False,
                duration_seconds=elapsed,
                verification_details="Task timed out",
            )
            # Timeout — create a failing verification result
            vr = VerificationResult(
                passed=False,
                score=0.0,
                details="Task timed out",
                method=task.verification.method,
            )
            # Update trust for timeout
            if task.required_capabilities:
                cap = task.required_capabilities[0]
                old_trust = self._trust_tracker.get_trust(agent.id, cap)
                await self._trust_tracker.update_trust(agent.id, cap, verified=False)
                new_trust = self._trust_tracker.get_trust(agent.id, cap)
                # Check circuit breaker
                if self._permission_manager:
                    await self._permission_manager.check_circuit_breaker(
                        agent.id, cap, old_trust, new_trust, self._contracts
                    )
            return result, vr
        except Exception:
            elapsed = time.monotonic() - start
            result = TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                output=None,
                success=False,
                duration_seconds=elapsed,
                verification_details="Agent handler raised an exception",
            )
            vr = VerificationResult(
                passed=False,
                score=0.0,
                details="Agent handler raised an exception",
                method=task.verification.method,
            )
            return result, vr

        # Verify against original task (vtask) to avoid feedback text contaminating judgement
        vr = await self._verification_engine.verify(vtask, result)

        # Update trust
        if vtask.required_capabilities:
            cap = task.required_capabilities[0]
            old_trust = self._trust_tracker.get_trust(agent.id, cap)
            await self._trust_tracker.update_trust(agent.id, cap, verified=vr.passed)
            new_trust = self._trust_tracker.get_trust(agent.id, cap)
            # Check circuit breaker
            if self._permission_manager:
                await self._permission_manager.check_circuit_breaker(
                    agent.id, cap, old_trust, new_trust, self._contracts
                )

        return result, vr

    async def _handle_failure(
        self,
        task: Task,
        agent: Agent,
        contract: Contract,
        result: TaskResult,
        vr: VerificationResult,
    ) -> None:
        """Retry same agent → reassign to next-best → escalate. Uses a while loop."""
        current_agent = agent
        attempt = 1
        excluded_agents: set[str] = set()

        while True:
            # Try retrying with the same agent
            if attempt <= task.max_retries:
                attempt += 1
                # Append format correction hints for SCHEMA/REGEX failures
                retry_task = task
                if vr.method in (VerificationMethod.SCHEMA, VerificationMethod.REGEX):
                    retry_task = task.model_copy(update={
                        "goal": (
                            f"{task.goal}\n\n"
                            f"IMPORTANT FORMAT CORRECTION: Your previous output failed "
                            f"format validation. Error: {vr.details}\n"
                            f"Please ensure your output matches the required format exactly."
                        )
                    })
                elif vr.method == VerificationMethod.LLM_JUDGE and vr.details:
                    retry_task = task.model_copy(update={
                        "goal": (
                            f"{task.goal}\n\n"
                            f"IMPORTANT: Your previous attempt did not meet quality criteria. "
                            f"Feedback: {vr.details}\n"
                            f"Please address this feedback in your response."
                        )
                    })
                contract_retry = Contract(
                    task=task,
                    agent_id=current_agent.id,
                    verification=task.verification,
                    attempt=attempt,
                )
                self._contracts.append(contract_retry)
                result, vr = await self._run_and_verify(
                    retry_task, current_agent, contract_retry, verify_task=task,
                )
                if vr.passed:
                    result = TaskResult(
                        task_id=result.task_id,
                        agent_id=result.agent_id,
                        output=result.output,
                        success=True,
                        verified=True,
                        verification_details=vr.details,
                        cost=result.cost,
                        duration_seconds=result.duration_seconds,
                    )
                    self._results[task.id] = result
                    self._completed.add(task.id)
                    contract_retry.result = result
                    contract_retry.completed_at = datetime.now(UTC)
                    await self._emit(
                        DelegationEvent(
                            type=DelegationEventType.TASK_COMPLETED,
                            task_id=task.id,
                            agent_id=current_agent.id,
                            data={"verified": True},
                        )
                    )
                    return
                continue

            # Retries exhausted for current agent — try reassignment
            excluded_agents.add(current_agent.id)
            task_reassignments = self._reassignment_counts.get(task.id, 0)

            if task_reassignments >= self._max_reassignments:
                await self._escalate(task, "Max reassignments reached")
                return

            available = self._get_available_agents(excluded_agents)
            next_agent = self._assignment_scorer.select_best(
                task, available, self._trust_tracker
            )

            if next_agent is None:
                await self._escalate(task, "No available agents for reassignment")
                return

            self._reassignment_counts[task.id] = task_reassignments + 1
            self._total_reassignments += 1
            current_agent = next_agent
            attempt = 1  # Reset attempt counter for new agent

            await self._emit(
                DelegationEvent(
                    type=DelegationEventType.TASK_REASSIGNED,
                    task_id=task.id,
                    agent_id=next_agent.id,
                    data={"reassignment_count": self._reassignment_counts[task.id]},
                )
            )

            # Execute with new agent — include format/quality hints if last failure was related
            retry_task = task
            if vr.method in (VerificationMethod.SCHEMA, VerificationMethod.REGEX):
                retry_task = task.model_copy(update={
                    "goal": (
                        f"{task.goal}\n\n"
                        f"IMPORTANT FORMAT CORRECTION: Your previous output failed "
                        f"format validation. Error: {vr.details}\n"
                        f"Please ensure your output matches the required format exactly."
                    )
                })
            elif vr.method == VerificationMethod.LLM_JUDGE and vr.details:
                retry_task = task.model_copy(update={
                    "goal": (
                        f"{task.goal}\n\n"
                        f"IMPORTANT: Your previous attempt did not meet quality criteria. "
                        f"Feedback: {vr.details}\n"
                        f"Please address this feedback in your response."
                    )
                })
            contract_new = Contract(
                task=task,
                agent_id=next_agent.id,
                verification=task.verification,
            )
            self._contracts.append(contract_new)
            result, vr = await self._run_and_verify(
                retry_task, next_agent, contract_new, verify_task=task,
            )
            if vr.passed:
                result = TaskResult(
                    task_id=result.task_id,
                    agent_id=result.agent_id,
                    output=result.output,
                    success=True,
                    verified=True,
                    verification_details=vr.details,
                    cost=result.cost,
                    duration_seconds=result.duration_seconds,
                )
                self._results[task.id] = result
                self._completed.add(task.id)
                contract_new.result = result
                contract_new.completed_at = datetime.now(UTC)
                await self._emit(
                    DelegationEvent(
                        type=DelegationEventType.TASK_COMPLETED,
                        task_id=task.id,
                        agent_id=next_agent.id,
                        data={"verified": True},
                    )
                )
                return
            # Loop continues to retry / reassign again

    async def _escalate(self, task: Task, reason: str) -> None:
        """Mark task as failed and emit ESCALATED event."""
        fail_result = TaskResult(
            task_id=task.id,
            agent_id="",
            output=None,
            success=False,
            verification_details=reason,
        )
        self._results[task.id] = fail_result
        self._failed.add(task.id)
        await self._emit(
            DelegationEvent(
                type=DelegationEventType.ESCALATED,
                task_id=task.id,
                data={"reason": reason},
            )
        )
        await self._emit(
            DelegationEvent(
                type=DelegationEventType.TASK_FAILED,
                task_id=task.id,
                data={"reason": reason},
            )
        )

    def _find_agent_by_id(self, agent_id: str) -> Agent | None:
        """Lookup helper."""
        for agent in self._agents:
            if agent.id == agent_id:
                return agent
        return None

    def _get_available_agents(self, exclude: set[str]) -> list[Agent]:
        """Filter agents for reassignment, excluding already-tried agents."""
        return [a for a in self._agents if a.id not in exclude]

    async def _emit(self, event: DelegationEvent) -> None:
        """Guard: only emit if event bus is present."""
        if self._event_bus:
            await self._event_bus.emit(event)
