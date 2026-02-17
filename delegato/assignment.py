"""Assignment scorer — ranks agents for a given task using multi-objective scoring."""

from __future__ import annotations

from delegato.models import Agent, Task
from delegato.trust import TrustTracker


class AssignmentScorer:
    """Scores and ranks agents for task assignment."""

    def __init__(
        self,
        *,
        w1: float = 0.35,
        w2: float = 0.30,
        w3: float = 0.20,
        w4: float = 0.15,
        min_threshold: float = 0.3,
    ) -> None:
        self.w1 = w1  # capability_match weight
        self.w2 = w2  # trust_score weight
        self.w3 = w3  # availability weight
        self.w4 = w4  # cost_efficiency weight
        self.min_threshold = min_threshold

    def score_agent(self, task: Task, agent: Agent, trust_tracker: TrustTracker) -> float:
        """Compute weighted score for a single agent on a task."""
        capability_match = self._capability_match(task, agent)
        trust_score = self._trust_score(task, agent, trust_tracker)
        availability = self._availability(agent)
        cost_efficiency = self._cost_efficiency(agent)

        return (
            self.w1 * capability_match
            + self.w2 * trust_score
            + self.w3 * availability
            + self.w4 * cost_efficiency
        )

    def rank_agents(
        self, task: Task, agents: list[Agent], trust_tracker: TrustTracker
    ) -> list[tuple[Agent, float]]:
        """Score all agents and return sorted list (best first), filtered to agents with at least one matching capability."""
        scored: list[tuple[Agent, float]] = []
        for agent in agents:
            if not self._has_any_capability(task, agent):
                continue
            score = self.score_agent(task, agent, trust_tracker)
            scored.append((agent, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def select_best(
        self, task: Task, agents: list[Agent], trust_tracker: TrustTracker
    ) -> Agent | None:
        """Return the top-scoring agent above threshold, or None."""
        ranked = self.rank_agents(task, agents, trust_tracker)
        if ranked and ranked[0][1] >= self.min_threshold:
            return ranked[0][0]
        return None

    # ── Private scoring factors ──────────────────────────────────────────

    @staticmethod
    def _has_any_capability(task: Task, agent: Agent) -> bool:
        """Check if agent has at least one matching capability."""
        if not task.required_capabilities:
            return True
        return bool(set(task.required_capabilities) & set(agent.capabilities))

    @staticmethod
    def _capability_match(task: Task, agent: Agent) -> float:
        """1.0 if agent has all required capabilities, partial match scored proportionally."""
        if not task.required_capabilities:
            return 1.0
        matched = len(set(task.required_capabilities) & set(agent.capabilities))
        return matched / len(task.required_capabilities)

    @staticmethod
    def _trust_score(task: Task, agent: Agent, trust_tracker: TrustTracker) -> float:
        """Agent's trust score for the primary required capability (first in list)."""
        if not task.required_capabilities:
            # No specific capability required — average across all agent capabilities
            if not agent.capabilities:
                return 0.5
            scores = [trust_tracker.get_trust(agent.id, cap) for cap in agent.capabilities]
            return sum(scores) / len(scores)
        # Use the first required capability as the primary one
        return trust_tracker.get_trust(agent.id, task.required_capabilities[0])

    @staticmethod
    def _availability(agent: Agent) -> float:
        """(max_concurrent - current_load) / max_concurrent."""
        return (agent.max_concurrent - agent.current_load) / agent.max_concurrent

    @staticmethod
    def _cost_efficiency(agent: Agent) -> float:
        """Normalized inverse of agent cost. Default 1.0 if no cost info."""
        cost_per_token = agent.metadata.get("cost_per_token", 0.0)
        if cost_per_token <= 0:
            return 1.0
        # Simple inverse normalization — lower cost = higher score
        # Cap at 1.0 for very cheap agents
        return min(1.0, 1.0 / (1.0 + cost_per_token * 1000))
