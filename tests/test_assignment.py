"""Standalone tests for the AssignmentScorer."""

from __future__ import annotations

import pytest

from delegato.assignment import AssignmentScorer
from delegato.trust import TrustTracker

from tests.conftest import make_agent, make_task


# ── TestCapabilityMatch ──────────────────────────────────────────────────────


class TestCapabilityMatch:
    """Tests for AssignmentScorer._capability_match()."""

    def test_full_match(self, assignment_scorer: AssignmentScorer):
        task = make_task(capabilities=["code", "search"])
        agent = make_agent(capabilities=["code", "search", "summarize"])
        assert AssignmentScorer._capability_match(task, agent) == 1.0

    def test_partial_match(self, assignment_scorer: AssignmentScorer):
        task = make_task(capabilities=["code", "search"])
        agent = make_agent(capabilities=["code"])
        assert AssignmentScorer._capability_match(task, agent) == 0.5

    def test_no_required_caps(self, assignment_scorer: AssignmentScorer):
        task = make_task(capabilities=[])
        agent = make_agent(capabilities=["code"])
        assert AssignmentScorer._capability_match(task, agent) == 1.0

    def test_no_matching_caps(self, assignment_scorer: AssignmentScorer):
        task = make_task(capabilities=["code", "search"])
        agent = make_agent(capabilities=["summarize"])
        assert AssignmentScorer._capability_match(task, agent) == 0.0


# ── TestTrustScore ───────────────────────────────────────────────────────────


class TestTrustScore:
    """Tests for AssignmentScorer._trust_score()."""

    def test_uses_first_required_capability(self):
        tracker = TrustTracker()
        tracker.register_agent("a1", ["code", "search"])
        tracker._trust_records["a1"]["code"].score = 0.8
        tracker._trust_records["a1"]["search"].score = 0.3

        task = make_task(capabilities=["code", "search"])
        agent = make_agent(agent_id="a1", capabilities=["code", "search"])

        score = AssignmentScorer._trust_score(task, agent, tracker)
        assert score == pytest.approx(0.8, abs=0.01)

    def test_no_required_caps_averages_agent_caps(self):
        tracker = TrustTracker()
        tracker.register_agent("a1", ["code", "search"])
        tracker._trust_records["a1"]["code"].score = 0.8
        tracker._trust_records["a1"]["search"].score = 0.6

        task = make_task(capabilities=[])
        agent = make_agent(agent_id="a1", capabilities=["code", "search"])

        score = AssignmentScorer._trust_score(task, agent, tracker)
        assert score == pytest.approx(0.7, abs=0.01)

    def test_no_caps_at_all_returns_half(self):
        tracker = TrustTracker()
        task = make_task(capabilities=[])
        agent = make_agent(capabilities=[])

        score = AssignmentScorer._trust_score(task, agent, tracker)
        assert score == 0.5

    def test_unknown_agent_returns_default(self):
        tracker = TrustTracker()
        task = make_task(capabilities=["code"])
        agent = make_agent(agent_id="unknown", capabilities=["code"])

        score = AssignmentScorer._trust_score(task, agent, tracker)
        assert score == 0.5  # default trust


# ── TestAvailability ─────────────────────────────────────────────────────────


class TestAvailability:
    """Tests for AssignmentScorer._availability()."""

    def test_fully_available(self):
        agent = make_agent(max_concurrent=5, current_load=0)
        assert AssignmentScorer._availability(agent) == 1.0

    def test_partially_loaded(self):
        agent = make_agent(max_concurrent=4, current_load=2)
        assert AssignmentScorer._availability(agent) == 0.5

    def test_fully_loaded(self):
        agent = make_agent(max_concurrent=3, current_load=3)
        assert AssignmentScorer._availability(agent) == 0.0


# ── TestCostEfficiency ───────────────────────────────────────────────────────


class TestCostEfficiency:
    """Tests for AssignmentScorer._cost_efficiency()."""

    def test_no_cost_info(self):
        agent = make_agent(metadata={})
        assert AssignmentScorer._cost_efficiency(agent) == 1.0

    def test_zero_cost(self):
        agent = make_agent(metadata={"cost_per_token": 0.0})
        assert AssignmentScorer._cost_efficiency(agent) == 1.0

    def test_low_cost_formula(self):
        agent = make_agent(metadata={"cost_per_token": 0.001})
        # 1.0 / (1.0 + 0.001 * 1000) = 1.0 / 2.0 = 0.5
        assert AssignmentScorer._cost_efficiency(agent) == pytest.approx(0.5, abs=0.01)

    def test_high_cost_formula(self):
        agent = make_agent(metadata={"cost_per_token": 0.01})
        # 1.0 / (1.0 + 0.01 * 1000) = 1.0 / 11.0 ≈ 0.091
        result = AssignmentScorer._cost_efficiency(agent)
        assert result == pytest.approx(1.0 / 11.0, abs=0.01)


# ── TestScoreAgent ───────────────────────────────────────────────────────────


class TestScoreAgent:
    """Tests for AssignmentScorer.score_agent()."""

    def test_perfect_agent_composite_score(self, assignment_scorer: AssignmentScorer):
        tracker = TrustTracker()
        tracker.register_agent("a1", ["code"])
        tracker._trust_records["a1"]["code"].score = 1.0

        task = make_task(capabilities=["code"])
        agent = make_agent(
            agent_id="a1",
            capabilities=["code"],
            max_concurrent=5,
            current_load=0,
        )

        score = assignment_scorer.score_agent(task, agent, tracker)
        # cap=1.0, trust=1.0, avail=1.0, cost=1.0
        # 0.35*1 + 0.30*1 + 0.20*1 + 0.15*1 = 1.0
        assert score == pytest.approx(1.0, abs=0.01)

    def test_specific_weighted_sum(self, assignment_scorer: AssignmentScorer):
        tracker = TrustTracker()
        tracker.register_agent("a1", ["code"])
        tracker._trust_records["a1"]["code"].score = 0.8

        task = make_task(capabilities=["code"])
        agent = make_agent(
            agent_id="a1",
            capabilities=["code"],
            max_concurrent=4,
            current_load=2,
            metadata={"cost_per_token": 0.001},
        )

        score = assignment_scorer.score_agent(task, agent, tracker)
        # cap=1.0, trust=0.8, avail=0.5, cost=0.5
        # 0.35*1.0 + 0.30*0.8 + 0.20*0.5 + 0.15*0.5
        expected = 0.35 * 1.0 + 0.30 * 0.8 + 0.20 * 0.5 + 0.15 * 0.5
        assert score == pytest.approx(expected, abs=0.01)

    def test_custom_weights_apply(self):
        scorer = AssignmentScorer(w1=1.0, w2=0.0, w3=0.0, w4=0.0)
        tracker = TrustTracker()
        tracker.register_agent("a1", ["code"])

        task = make_task(capabilities=["code", "search"])
        agent = make_agent(agent_id="a1", capabilities=["code"])

        score = scorer.score_agent(task, agent, tracker)
        # Only capability_match matters: 1/2 = 0.5
        assert score == pytest.approx(0.5, abs=0.01)


# ── TestRankAgents ───────────────────────────────────────────────────────────


class TestRankAgents:
    """Tests for AssignmentScorer.rank_agents()."""

    def test_descending_order(self, assignment_scorer: AssignmentScorer):
        tracker = TrustTracker()
        tracker.register_agent("a1", ["code"])
        tracker.register_agent("a2", ["code"])
        tracker._trust_records["a1"]["code"].score = 0.9
        tracker._trust_records["a2"]["code"].score = 0.3

        task = make_task(capabilities=["code"])
        agents = [
            make_agent(agent_id="a1", capabilities=["code"]),
            make_agent(agent_id="a2", capabilities=["code"]),
        ]

        ranked = assignment_scorer.rank_agents(task, agents, tracker)
        assert ranked[0][0].id == "a1"
        assert ranked[1][0].id == "a2"
        assert ranked[0][1] > ranked[1][1]

    def test_filters_non_matching_agents(self, assignment_scorer: AssignmentScorer):
        tracker = TrustTracker()
        task = make_task(capabilities=["code"])
        agents = [
            make_agent(agent_id="a1", capabilities=["code"]),
            make_agent(agent_id="a2", capabilities=["search"]),  # no overlap
        ]

        ranked = assignment_scorer.rank_agents(task, agents, tracker)
        assert len(ranked) == 1
        assert ranked[0][0].id == "a1"

    def test_empty_list(self, assignment_scorer: AssignmentScorer):
        tracker = TrustTracker()
        task = make_task(capabilities=["code"])
        assert assignment_scorer.rank_agents(task, [], tracker) == []

    def test_all_filtered_out(self, assignment_scorer: AssignmentScorer):
        tracker = TrustTracker()
        task = make_task(capabilities=["code"])
        agents = [make_agent(agent_id="a1", capabilities=["search"])]
        assert assignment_scorer.rank_agents(task, agents, tracker) == []

    def test_no_required_caps_includes_all(self, assignment_scorer: AssignmentScorer):
        tracker = TrustTracker()
        task = make_task(capabilities=[])
        agents = [
            make_agent(agent_id="a1", capabilities=["code"]),
            make_agent(agent_id="a2", capabilities=["search"]),
        ]

        ranked = assignment_scorer.rank_agents(task, agents, tracker)
        assert len(ranked) == 2


# ── TestSelectBest ───────────────────────────────────────────────────────────


class TestSelectBest:
    """Tests for AssignmentScorer.select_best()."""

    def test_returns_top_agent(self, assignment_scorer: AssignmentScorer):
        tracker = TrustTracker()
        tracker.register_agent("a1", ["code"])
        tracker._trust_records["a1"]["code"].score = 0.9

        task = make_task(capabilities=["code"])
        agents = [make_agent(agent_id="a1", capabilities=["code"])]

        best = assignment_scorer.select_best(task, agents, tracker)
        assert best is not None
        assert best.id == "a1"

    def test_none_below_threshold(self):
        # Use a very high threshold so no agent qualifies
        scorer = AssignmentScorer(min_threshold=0.99)
        tracker = TrustTracker()
        tracker.register_agent("a1", ["code"])
        tracker._trust_records["a1"]["code"].score = 0.3

        task = make_task(capabilities=["code"])
        agents = [make_agent(agent_id="a1", capabilities=["code"])]

        assert scorer.select_best(task, agents, tracker) is None

    def test_custom_threshold(self):
        scorer = AssignmentScorer(min_threshold=0.5)
        tracker = TrustTracker()
        tracker.register_agent("a1", ["code"])
        tracker._trust_records["a1"]["code"].score = 0.8

        task = make_task(capabilities=["code"])
        agents = [make_agent(agent_id="a1", capabilities=["code"])]

        best = scorer.select_best(task, agents, tracker)
        assert best is not None
        assert best.id == "a1"

    def test_no_agents(self, assignment_scorer: AssignmentScorer):
        tracker = TrustTracker()
        task = make_task(capabilities=["code"])
        assert assignment_scorer.select_best(task, [], tracker) is None

    def test_all_filtered(self, assignment_scorer: AssignmentScorer):
        tracker = TrustTracker()
        task = make_task(capabilities=["code"])
        agents = [make_agent(agent_id="a1", capabilities=["unrelated"])]
        assert assignment_scorer.select_best(task, agents, tracker) is None
