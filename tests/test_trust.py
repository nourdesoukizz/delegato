"""Standalone tests for the TrustTracker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from delegato.events import EventBus
from delegato.models import DelegationEventType, TrustRecord
from delegato.trust import TrustTracker


# ── TestRegisterAgent ────────────────────────────────────────────────────────


class TestRegisterAgent:
    """Tests for TrustTracker.register_agent()."""

    def test_cold_start_trust_half(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        assert trust_tracker.get_trust("a1", "code") == 0.5

    def test_multiple_capabilities(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code", "search", "summarize"])
        assert trust_tracker.get_trust("a1", "code") == 0.5
        assert trust_tracker.get_trust("a1", "search") == 0.5
        assert trust_tracker.get_trust("a1", "summarize") == 0.5

    def test_idempotent_registration(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        # Manually set trust to something different
        trust_tracker._trust_records["a1"]["code"] = TrustRecord(score=0.9)

        # Re-registering should NOT reset existing capability
        trust_tracker.register_agent("a1", ["code", "new_cap"])
        assert trust_tracker.get_trust("a1", "code") == pytest.approx(0.9, abs=0.01)
        assert trust_tracker.get_trust("a1", "new_cap") == 0.5

    def test_default_transparency(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        assert trust_tracker.get_transparency("a1") == 0.5


# ── TestGetTrust ─────────────────────────────────────────────────────────────


class TestGetTrust:
    """Tests for TrustTracker.get_trust()."""

    def test_unknown_agent_returns_default(self, trust_tracker: TrustTracker):
        assert trust_tracker.get_trust("unknown", "code") == 0.5

    def test_unknown_capability_returns_default(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        assert trust_tracker.get_trust("a1", "unknown_cap") == 0.5

    def test_returns_registered_score(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker._trust_records["a1"]["code"] = TrustRecord(score=0.75)
        assert trust_tracker.get_trust("a1", "code") == pytest.approx(0.75, abs=0.01)


# ── TestUpdateTrust ──────────────────────────────────────────────────────────


class TestUpdateTrust:
    """Tests for TrustTracker.update_trust()."""

    async def test_success_increases(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        await trust_tracker.update_trust("a1", "code", verified=True)
        # 0.5 + 0.1 * (1.0 - 0.5) = 0.55
        assert trust_tracker.get_trust("a1", "code") == pytest.approx(0.55, abs=0.01)

    async def test_failure_decreases(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        await trust_tracker.update_trust("a1", "code", verified=False)
        # 0.5 - 0.2 * 0.5 = 0.40
        assert trust_tracker.get_trust("a1", "code") == pytest.approx(0.40, abs=0.01)

    async def test_success_at_high_trust(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker._trust_records["a1"]["code"] = TrustRecord(score=0.9)
        await trust_tracker.update_trust("a1", "code", verified=True)
        # 0.9 + 0.1 * (1.0 - 0.9) = 0.91
        assert trust_tracker.get_trust("a1", "code") == pytest.approx(0.91, abs=0.01)

    async def test_failure_at_low_trust(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker._trust_records["a1"]["code"] = TrustRecord(score=0.1)
        await trust_tracker.update_trust("a1", "code", verified=False)
        # 0.1 - 0.2 * 0.1 = 0.08
        assert trust_tracker.get_trust("a1", "code") == pytest.approx(0.08, abs=0.01)

    async def test_clamped_to_one(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker._trust_records["a1"]["code"] = TrustRecord(score=0.99)
        await trust_tracker.update_trust("a1", "code", verified=True)
        assert trust_tracker.get_trust("a1", "code") <= 1.0

    async def test_clamped_to_zero(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker._trust_records["a1"]["code"] = TrustRecord(score=0.01)
        await trust_tracker.update_trust("a1", "code", verified=False)
        assert trust_tracker.get_trust("a1", "code") >= 0.0

    async def test_unregistered_agent_creates_record(self, trust_tracker: TrustTracker):
        # Agent not registered beforehand — update_trust should create record
        await trust_tracker.update_trust("new_agent", "code", verified=True)
        # 0.5 + 0.1 * (1.0 - 0.5) = 0.55
        assert trust_tracker.get_trust("new_agent", "code") == pytest.approx(
            0.55, abs=0.01
        )

    async def test_custom_learning_rates(self, event_bus: EventBus):
        tracker = TrustTracker(
            event_bus=event_bus,
            success_learning_rate=0.5,
            failure_learning_rate=0.5,
        )
        tracker.register_agent("a1", ["code"])

        await tracker.update_trust("a1", "code", verified=True)
        # 0.5 + 0.5 * (1.0 - 0.5) = 0.75
        assert tracker.get_trust("a1", "code") == pytest.approx(0.75, abs=0.01)

    async def test_emits_trust_updated_event(self, event_bus: EventBus, trust_tracker: TrustTracker):
        received: list = []

        async def listener(event):
            received.append(event)

        event_bus.on(DelegationEventType.TRUST_UPDATED, listener)
        trust_tracker.register_agent("a1", ["code"])
        await trust_tracker.update_trust("a1", "code", verified=True)

        assert len(received) == 1
        assert received[0].data["capability"] == "code"
        assert received[0].data["verified"] is True

    async def test_no_event_bus_no_error(self):
        tracker = TrustTracker()  # no event_bus
        tracker.register_agent("a1", ["code"])
        await tracker.update_trust("a1", "code", verified=True)
        assert tracker.get_trust("a1", "code") == pytest.approx(0.55, abs=0.01)


# ── TestUpdateTransparency ───────────────────────────────────────────────────


class TestUpdateTransparency:
    """Tests for TrustTracker.update_transparency()."""

    def test_positive_delta(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker.update_transparency("a1", 0.1)
        assert trust_tracker.get_transparency("a1") == pytest.approx(0.6, abs=0.01)

    def test_negative_delta(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker.update_transparency("a1", -0.2)
        assert trust_tracker.get_transparency("a1") == pytest.approx(0.3, abs=0.01)

    def test_clamped_to_one(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker.update_transparency("a1", 0.9)
        assert trust_tracker.get_transparency("a1") <= 1.0

    def test_clamped_to_zero(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker.update_transparency("a1", -0.9)
        assert trust_tracker.get_transparency("a1") >= 0.0

    def test_unregistered_agent_uses_default(self, trust_tracker: TrustTracker):
        trust_tracker.update_transparency("unknown", 0.1)
        # default 0.5 + 0.1 = 0.6
        assert trust_tracker.get_transparency("unknown") == pytest.approx(0.6, abs=0.01)


# ── TestTimeBasedDecay ───────────────────────────────────────────────────────


class TestTimeBasedDecay:
    """Tests for _apply_decay() via get_trust()."""

    def test_no_decay_within_window(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker._trust_records["a1"]["code"] = TrustRecord(
            score=0.9, last_updated=datetime.now(UTC) - timedelta(hours=10)
        )
        # Within 72h window → no decay
        assert trust_tracker.get_trust("a1", "code") == pytest.approx(0.9, abs=0.01)

    def test_decay_after_window_toward_neutral(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker._trust_records["a1"]["code"] = TrustRecord(
            score=0.9, last_updated=datetime.now(UTC) - timedelta(hours=172)
        )
        # 100h past the 72h window, decay_factor = min(1.0, 0.01 * 100) = 1.0
        # decayed = 0.9 + (0.5 - 0.9) * 1.0 = 0.5
        decayed = trust_tracker.get_trust("a1", "code")
        assert decayed < 0.9
        assert decayed >= 0.5  # moves toward 0.5

    def test_decay_from_low_score_toward_neutral(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker._trust_records["a1"]["code"] = TrustRecord(
            score=0.1, last_updated=datetime.now(UTC) - timedelta(hours=172)
        )
        # 100h past window → decay_factor = 1.0
        # decayed = 0.1 + (0.5 - 0.1) * 1.0 = 0.5
        decayed = trust_tracker.get_trust("a1", "code")
        assert decayed > 0.1
        assert decayed <= 0.5

    def test_max_decay_factor_capped(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker._trust_records["a1"]["code"] = TrustRecord(
            score=0.9, last_updated=datetime.now(UTC) - timedelta(hours=1000)
        )
        # Very stale → decay_factor capped at 1.0
        decayed = trust_tracker.get_trust("a1", "code")
        assert decayed == pytest.approx(0.5, abs=0.01)

    def test_custom_decay_window_and_rate(self, event_bus: EventBus):
        tracker = TrustTracker(
            event_bus=event_bus,
            decay_window=timedelta(hours=24),
            decay_rate=0.1,
        )
        tracker.register_agent("a1", ["code"])
        tracker._trust_records["a1"]["code"] = TrustRecord(
            score=0.9, last_updated=datetime.now(UTC) - timedelta(hours=34)
        )
        # 10h past 24h window, decay_factor = min(1.0, 0.1 * 10) = 1.0
        # decayed = 0.9 + (0.5 - 0.9) * 1.0 = 0.5
        assert tracker.get_trust("a1", "code") == pytest.approx(0.5, abs=0.01)


# ── TestGetAllScores ─────────────────────────────────────────────────────────


class TestGetAllScores:
    """Tests for TrustTracker.get_all_scores()."""

    def test_returns_all_agents(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker.register_agent("a2", ["search"])
        scores = trust_tracker.get_all_scores()
        assert "a1" in scores
        assert "a2" in scores

    def test_includes_trust_and_transparency(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        scores = trust_tracker.get_all_scores()
        assert "trust" in scores["a1"]
        assert "transparency" in scores["a1"]

    def test_applies_decay_in_output(self, trust_tracker: TrustTracker):
        trust_tracker.register_agent("a1", ["code"])
        trust_tracker._trust_records["a1"]["code"] = TrustRecord(
            score=0.9, last_updated=datetime.now(UTC) - timedelta(hours=200)
        )
        scores = trust_tracker.get_all_scores()
        # Decay should have been applied
        assert scores["a1"]["trust"]["code"] < 0.9

    def test_empty_tracker_returns_empty(self, trust_tracker: TrustTracker):
        assert trust_tracker.get_all_scores() == {}
