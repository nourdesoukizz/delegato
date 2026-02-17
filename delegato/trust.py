"""Trust tracker with time-based decay and transparency scores."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from delegato.events import EventBus
from delegato.models import DelegationEvent, DelegationEventType, TrustRecord


class TrustTracker:
    """Maintains per-agent, per-capability trust scores with time-based decay."""

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        success_learning_rate: float = 0.1,
        failure_learning_rate: float = 0.2,
        default_trust: float = 0.5,
        decay_window: timedelta = timedelta(hours=72),
        decay_rate: float = 0.01,
    ) -> None:
        self._event_bus = event_bus
        self._success_learning_rate = success_learning_rate
        self._failure_learning_rate = failure_learning_rate
        self._default_trust = default_trust
        self._decay_window = decay_window
        self._decay_rate = decay_rate

        # agent_id → capability → TrustRecord
        self._trust_records: dict[str, dict[str, TrustRecord]] = {}
        # agent_id → transparency score
        self._transparency_scores: dict[str, float] = {}

    def register_agent(self, agent_id: str, capabilities: list[str]) -> None:
        """Register a new agent with cold-start trust scores."""
        if agent_id not in self._trust_records:
            self._trust_records[agent_id] = {}
        for cap in capabilities:
            if cap not in self._trust_records[agent_id]:
                self._trust_records[agent_id][cap] = TrustRecord(
                    score=self._default_trust
                )
        if agent_id not in self._transparency_scores:
            self._transparency_scores[agent_id] = self._default_trust

    def get_trust(self, agent_id: str, capability: str) -> float:
        """Return the decayed trust score for an agent's capability."""
        records = self._trust_records.get(agent_id, {})
        record = records.get(capability)
        if record is None:
            return self._default_trust

        return self._apply_decay(record)

    def get_transparency(self, agent_id: str) -> float:
        """Return the transparency score for an agent."""
        return self._transparency_scores.get(agent_id, self._default_trust)

    async def update_trust(self, agent_id: str, capability: str, verified: bool) -> None:
        """Update trust score based on task verification outcome."""
        if agent_id not in self._trust_records:
            self._trust_records[agent_id] = {}

        record = self._trust_records[agent_id].get(capability)
        old_score = record.score if record else self._default_trust

        if verified:
            new_score = old_score + self._success_learning_rate * (1.0 - old_score)
        else:
            new_score = old_score - self._failure_learning_rate * old_score

        # Clamp to [0.0, 1.0]
        new_score = max(0.0, min(1.0, new_score))

        self._trust_records[agent_id][capability] = TrustRecord(
            score=new_score,
            last_updated=datetime.now(UTC),
        )

        if self._event_bus:
            await self._event_bus.emit(
                DelegationEvent(
                    type=DelegationEventType.TRUST_UPDATED,
                    agent_id=agent_id,
                    data={
                        "capability": capability,
                        "old_score": old_score,
                        "new_score": new_score,
                        "verified": verified,
                    },
                )
            )

    def update_transparency(self, agent_id: str, delta: float) -> None:
        """Adjust an agent's transparency score by delta."""
        current = self._transparency_scores.get(agent_id, self._default_trust)
        self._transparency_scores[agent_id] = max(0.0, min(1.0, current + delta))

    def get_all_scores(self) -> dict:
        """Return all trust and transparency scores for the public API."""
        result: dict[str, dict] = {}
        for agent_id, caps in self._trust_records.items():
            result[agent_id] = {
                "trust": {
                    cap: self._apply_decay(record) for cap, record in caps.items()
                },
                "transparency": self._transparency_scores.get(
                    agent_id, self._default_trust
                ),
            }
        return result

    def _apply_decay(self, record: TrustRecord) -> float:
        """Apply time-based decay toward 0.5 (neutral)."""
        now = datetime.now(UTC)
        elapsed = now - record.last_updated
        if elapsed <= self._decay_window:
            return record.score

        hours_stale = (elapsed - self._decay_window).total_seconds() / 3600
        decay_factor = min(1.0, self._decay_rate * hours_stale)
        return record.score + (0.5 - record.score) * decay_factor
