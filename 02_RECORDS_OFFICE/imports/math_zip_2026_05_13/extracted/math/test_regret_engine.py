"""Tests for pypeline.math.regret_engine — RFC-0018 §3.12 CFR."""
import pytest
from pypeline.math.regret_engine import compute_regret_weighted_strategy, rank_actions_by_ev
from pypeline.math.types import MathAction, MathCounterfactual


def _action(id: str, ev: float, risk: float = 0.0) -> MathAction:
    return MathAction(id=id, type="test", expected_value=ev, risk=risk)


def _cf(action: str, prob: float = 0.5, loss: float = 0.0) -> MathCounterfactual:
    return MathCounterfactual(
        id=f"cf_{action}",
        action=action,
        predicted_actor_response="neutral",
        probability=prob,
        loss_bound=loss,
    )


class TestComputeRegretWeightedStrategy:
    def test_empty_actions_returns_empty(self):
        assert compute_regret_weighted_strategy([], []) == {}

    def test_single_action_gets_full_probability(self):
        actions = [_action("a0", ev=1.0)]
        result = compute_regret_weighted_strategy(actions, [])
        assert result == pytest.approx({"a0": 1.0})

    def test_probabilities_sum_to_one(self):
        actions = [_action(f"a{i}", ev=float(i)) for i in range(5)]
        result = compute_regret_weighted_strategy(actions, [])
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(v >= 0 for v in result.values())

    def test_dominant_action_gets_highest_weight(self):
        actions = [_action("low", ev=0.1), _action("high", ev=10.0)]
        result = compute_regret_weighted_strategy(actions, [], iterations=2000)
        assert result["high"] > result["low"]

    def test_all_ids_present_in_result(self):
        actions = [_action(f"a{i}", ev=float(i)) for i in range(4)]
        result = compute_regret_weighted_strategy(actions, [])
        assert set(result.keys()) == {"a0", "a1", "a2", "a3"}

    def test_counterfactuals_influence_strategy(self):
        # a0 has high EV but high loss counterfactual; a1 is safer
        actions = [_action("a0", ev=5.0, risk=0.0), _action("a1", ev=2.0, risk=0.0)]
        cfs = [_cf("a0", prob=0.9, loss=0.9)]  # a0's counterfactual nearly cancels its EV
        result = compute_regret_weighted_strategy(actions, cfs, iterations=2000)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_equal_actions_get_equal_weight(self):
        actions = [_action(f"a{i}", ev=1.0) for i in range(3)]
        result = compute_regret_weighted_strategy(actions, [], iterations=500)
        for v in result.values():
            assert v == pytest.approx(1.0 / 3, abs=0.05)

    def test_risk_penalizes_action(self):
        actions = [_action("risky", ev=5.0, risk=4.5), _action("safe", ev=1.0, risk=0.0)]
        result = compute_regret_weighted_strategy(actions, [], iterations=2000)
        assert result["safe"] > result["risky"]


class TestRankActionsByEv:
    def test_empty_returns_empty(self):
        assert rank_actions_by_ev([], []) == []

    def test_ranks_descending_by_ev_minus_risk(self):
        a0 = _action("a0", ev=1.0, risk=0.0)   # net = 1.0
        a1 = _action("a1", ev=3.0, risk=2.5)   # net = 0.5
        a2 = _action("a2", ev=5.0, risk=1.0)   # net = 4.0
        result = rank_actions_by_ev([a0, a1, a2], [])
        assert [r.id for r in result] == ["a2", "a0", "a1"]

    def test_single_action(self):
        a = _action("only", ev=3.0)
        assert rank_actions_by_ev([a], []) == [a]
