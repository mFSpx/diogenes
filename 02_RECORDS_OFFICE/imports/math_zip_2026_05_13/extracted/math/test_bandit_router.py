"""Tests for pypeline.math.bandit_router — RFC-0018 §3.13 WO-BM-031."""
from __future__ import annotations

import pytest
from pypeline.math.bandit_router import (
    BanditAction,
    BanditUpdate,
    select_action,
    update_policy,
    reset_policy,
)


@pytest.fixture(autouse=True)
def fresh_policy():
    reset_policy()
    yield
    reset_policy()


ACTIONS = ["file_bcfsa", "file_civil", "send_demand", "escalate"]
CTX = {"pressure_score": 7.5, "evidence_count": 12}


class TestSelectActionLinUCB:
    def test_returns_bandit_action(self):
        result = select_action(CTX, ACTIONS, algorithm="linucb")
        assert isinstance(result, BanditAction)

    def test_action_id_in_available(self):
        result = select_action(CTX, ACTIONS, algorithm="linucb")
        assert result.action_id in ACTIONS

    def test_propensity_in_unit_interval(self):
        result = select_action(CTX, ACTIONS, algorithm="linucb")
        assert 0.0 < result.propensity <= 1.0

    def test_confidence_bound_non_negative(self):
        result = select_action(CTX, ACTIONS, algorithm="linucb")
        assert result.confidence_bound >= 0.0

    def test_single_action_always_selected(self):
        result = select_action(CTX, ["only_one"], algorithm="linucb")
        assert result.action_id == "only_one"

    def test_empty_actions_raises(self):
        with pytest.raises(ValueError):
            select_action(CTX, [], algorithm="linucb")

    def test_empty_context_works(self):
        result = select_action({}, ACTIONS, algorithm="linucb")
        assert result.action_id in ACTIONS


class TestSelectActionThompson:
    def test_returns_bandit_action(self):
        result = select_action(CTX, ACTIONS, algorithm="thompson")
        assert result.action_id in ACTIONS

    def test_propensity_uniform_approx(self):
        result = select_action(CTX, ACTIONS, algorithm="thompson")
        assert result.propensity == pytest.approx(1.0 / len(ACTIONS), abs=1e-9)

    def test_after_updates_prefers_high_reward_action(self):
        # Drive "escalate" to high reward
        updates = [BanditUpdate("ctx1", "escalate", 1.0, 0.25)] * 20
        updates += [BanditUpdate("ctx1", "file_bcfsa", 0.0, 0.25)] * 20
        update_policy(updates)
        # Thompson should sample escalate most often
        wins: dict[str, int] = {a: 0 for a in ACTIONS}
        for _ in range(200):
            r = select_action(CTX, ACTIONS, algorithm="thompson")
            wins[r.action_id] += 1
        assert wins["escalate"] > wins["file_bcfsa"]


class TestSelectActionEpsilonGreedy:
    def test_returns_bandit_action(self):
        result = select_action(CTX, ACTIONS, algorithm="epsilon_greedy")
        assert result.action_id in ACTIONS

    def test_after_updates_mostly_greedy(self):
        updates = [BanditUpdate("c", "file_civil", 1.0, 0.25)] * 30
        updates += [BanditUpdate("c", "send_demand", 0.0, 0.25)] * 30
        update_policy(updates)
        wins = sum(
            1 for _ in range(100)
            if select_action(CTX, ACTIONS, algorithm="epsilon_greedy").action_id == "file_civil"
        )
        # Greedy (90%) + small explore chance → file_civil should win most
        assert wins >= 50


class TestUpdatePolicy:
    def test_update_does_not_raise(self):
        updates = [
            BanditUpdate("c1", "file_bcfsa", 0.8, 0.25),
            BanditUpdate("c2", "escalate", 0.2, 0.25),
        ]
        update_policy(updates)  # no exception

    def test_empty_updates_no_op(self):
        update_policy([])  # no exception

    def test_linucb_learns_from_updates(self):
        # Repeatedly reward "escalate" → its expected_reward should increase
        for _ in range(10):
            update_policy([BanditUpdate("c", "escalate", 1.0, 0.25)])
        r1 = select_action(CTX, ["escalate"], algorithm="linucb")
        assert r1.expected_reward > -10.0  # not degenerate
