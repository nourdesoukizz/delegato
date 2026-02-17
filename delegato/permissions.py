"""Permission manager with privilege attenuation and circuit breakers."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from delegato.events import EventBus
from delegato.models import (
    Agent,
    Contract,
    DelegationEvent,
    DelegationEventType,
    Permission,
    Reversibility,
    Task,
)
from delegato.trust import TrustTracker


class PrivilegeEscalationError(Exception):
    """Raised when a child attempts to acquire broader permissions than its parent."""


class CircuitBreakResult(BaseModel):
    agent_id: str
    old_trust: float
    new_trust: float
    trust_drop: float
    paused_contract_ids: list[str] = []


class ComplexityFloorResult(BaseModel):
    eligible: bool
    reason: str
    agent_id: str | None = None


class PermissionManager:
    """Manages scoped permissions, privilege attenuation, and circuit breakers."""

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        trust_tracker: TrustTracker | None = None,
        circuit_breaker_threshold: float = 0.3,
        complexity_floor_trust: float = 0.7,
    ) -> None:
        self._event_bus = event_bus
        self._trust_tracker = trust_tracker
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._complexity_floor_trust = complexity_floor_trust

    def attenuate(
        self, parent_perms: list[Permission], child_perms: list[Permission]
    ) -> list[Permission]:
        """Validate that child permissions are a subset of parent permissions.

        Returns the valid child permissions after filtering expired ones.
        Raises PrivilegeEscalationError if any child permission is broader than parent.
        """
        valid_child = self.check_expiry(child_perms)

        for child_perm in valid_child:
            if not self.check_permission(child_perm, parent_perms):
                raise PrivilegeEscalationError(
                    f"Child permission ({child_perm.resource}:{child_perm.action}:{child_perm.scope}) "
                    f"is not covered by parent permissions"
                )

        return valid_child

    def check_permission(
        self, perm: Permission, granted: list[Permission]
    ) -> bool:
        """Check if a single permission is covered by any of the granted permissions."""
        valid_granted = self.check_expiry(granted)
        return any(self.is_subset(perm, g) for g in valid_granted)

    def is_subset(self, child: Permission, parent: Permission) -> bool:
        """Check if child permission is a subset of parent permission."""
        if child.resource != parent.resource:
            return False
        if child.action != parent.action:
            return False
        return self._scope_is_subset(child.scope, parent.scope)

    def check_expiry(self, perms: list[Permission]) -> list[Permission]:
        """Filter out expired permissions."""
        now = datetime.now(UTC)
        return [p for p in perms if p.expiry is None or p.expiry > now]

    def check_complexity_floor(
        self, task: Task, agents: list[Agent]
    ) -> ComplexityFloorResult:
        """Check if a task is eligible for the complexity floor bypass.

        Eligible if complexity <= 2, reversibility HIGH, and a trusted agent exists.
        """
        if task.complexity > 2:
            return ComplexityFloorResult(
                eligible=False,
                reason=f"Task complexity {task.complexity} exceeds floor threshold of 2",
            )

        if task.reversibility != Reversibility.HIGH:
            return ComplexityFloorResult(
                eligible=False,
                reason=f"Task reversibility {task.reversibility.value} is not HIGH",
            )

        # Find a trusted agent with matching capabilities
        for agent in agents:
            if not task.required_capabilities or set(task.required_capabilities) <= set(
                agent.capabilities
            ):
                # Check trust score
                if self._trust_tracker:
                    if task.required_capabilities:
                        cap = task.required_capabilities[0]
                        trust = self._trust_tracker.get_trust(agent.id, cap)
                    else:
                        trust = self._complexity_floor_trust  # default pass if no cap
                else:
                    # No trust tracker — use agent's own trust scores
                    if task.required_capabilities:
                        cap = task.required_capabilities[0]
                        trust = agent.trust_scores.get(cap, 0.0)
                    else:
                        trust = self._complexity_floor_trust

                if trust >= self._complexity_floor_trust:
                    return ComplexityFloorResult(
                        eligible=True,
                        reason="Task qualifies for complexity floor bypass",
                        agent_id=agent.id,
                    )

        return ComplexityFloorResult(
            eligible=False,
            reason="No trusted agent with matching capabilities found",
        )

    async def check_circuit_breaker(
        self,
        agent_id: str,
        cap: str,
        old_trust: float,
        new_trust: float,
        contracts: list[Contract],
    ) -> CircuitBreakResult | None:
        """Check if a trust drop triggers the circuit breaker.

        Returns CircuitBreakResult if triggered, None otherwise.
        """
        trust_drop = old_trust - new_trust

        if trust_drop < self._circuit_breaker_threshold:
            return None

        # Find active contracts for this agent
        paused_ids = [c.id for c in contracts if c.agent_id == agent_id and c.result is None]

        result = CircuitBreakResult(
            agent_id=agent_id,
            old_trust=old_trust,
            new_trust=new_trust,
            trust_drop=trust_drop,
            paused_contract_ids=paused_ids,
        )

        if self._event_bus:
            await self._event_bus.emit(
                DelegationEvent(
                    type=DelegationEventType.TRUST_CIRCUIT_BREAK,
                    agent_id=agent_id,
                    data={
                        "capability": cap,
                        "old_trust": old_trust,
                        "new_trust": new_trust,
                        "trust_drop": trust_drop,
                        "paused_contract_ids": paused_ids,
                    },
                )
            )

        return result

    @staticmethod
    def _scope_is_subset(child_scope: str, parent_scope: str) -> bool:
        """Check if child scope is within parent scope.

        Rules:
        - Parent "*" covers any child
        - Equal scopes are allowed
        - Parent ending with "*" covers children with matching prefix
        - Child ending with "*" but parent doesn't is broader (rejected)
        """
        if parent_scope == "*":
            return True

        if child_scope == parent_scope:
            return True

        # Child is a wildcard but parent isn't — child is broader
        if child_scope.endswith("*") and not parent_scope.endswith("*"):
            return False

        # Parent ends with wildcard — check prefix match
        if parent_scope.endswith("*"):
            prefix = parent_scope[:-1]
            return child_scope.startswith(prefix)

        return False
