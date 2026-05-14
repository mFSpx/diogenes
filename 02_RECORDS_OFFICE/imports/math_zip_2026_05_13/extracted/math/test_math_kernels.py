"""Tests for RFC-0018 math kernels — bayes_claim_kernel + regret_engine."""
from __future__ import annotations

import pytest

from pypeline.math.types import (
    MathAction,
    MathCounterfactual,
    MathEvidence,
    MathHypothesis,
)
from pypeline.math.bayes_claim_kernel import (
    update_hypothesis,
    compute_log_likelihood_ratio,
)
from pypeline.math.regret_engine import (
    compute_regret_weighted_strategy,
    rank_actions_by_ev,
)


# ===========================================================================
# bayes_claim_kernel — update_hypothesis (RFC-0018 §3.10)
# ===========================================================================

class TestUpdateHypothesis:
    _EV = MathEvidence(
        id="ev_001",
        source="bc_registry",
        hash="a" * 64,
        admissibility=1.0,
        reliability=0.9,
    )

    def _hyp(self, posterior: float = 0.5) -> MathHypothesis:
        return MathHypothesis(id="h001", prior=0.5, posterior=posterior)

    def test_strong_supporting_evidence_raises_posterior(self) -> None:
        h = self._hyp(0.5)
        updated = update_hypothesis(h, self._EV, likelihood_ratio=10.0)
        assert updated.posterior > h.posterior
        assert updated.posterior == pytest.approx(10.0 / (10.0 + 1.0), abs=1e-6)

    def test_weak_evidence_lr_one_is_noop(self) -> None:
        h = self._hyp(0.7)
        updated = update_hypothesis(h, self._EV, likelihood_ratio=1.0)
        assert updated.posterior == pytest.approx(0.7, abs=1e-6)

    def test_strong_disconfirming_evidence_lowers_posterior(self) -> None:
        h = self._hyp(0.8)
        updated = update_hypothesis(h, self._EV, likelihood_ratio=0.1)
        assert updated.posterior < h.posterior

    def test_lr_zero_makes_posterior_zero(self) -> None:
        h = self._hyp(0.6)
        updated = update_hypothesis(h, self._EV, likelihood_ratio=0.0)
        assert updated.posterior == pytest.approx(0.0)

    def test_updated_prior_equals_old_posterior(self) -> None:
        h = self._hyp(0.6)
        updated = update_hypothesis(h, self._EV, likelihood_ratio=5.0)
        assert updated.prior == pytest.approx(0.6)

    def test_negative_lr_raises_value_error(self) -> None:
        h = self._hyp(0.5)
        with pytest.raises(ValueError, match="non-negative"):
            update_hypothesis(h, self._EV, likelihood_ratio=-1.0)

    def test_posterior_clamped_to_one(self) -> None:
        h = self._hyp(0.9999)
        updated = update_hypothesis(h, self._EV, likelihood_ratio=1e12)
        assert updated.posterior <= 1.0

    def test_prior_zero_is_absorbing_state(self) -> None:
        h = MathHypothesis(id="h_abs", prior=0.0, posterior=0.0)
        updated = update_hypothesis(h, self._EV, likelihood_ratio=1000.0)
        assert updated.posterior == pytest.approx(0.0)

    def test_compute_log_lr_raises_not_implemented(self) -> None:
        from pypeline.math.types import MathClaim
        claim = MathClaim(
            id="c1", text="t", subject="s", predicate="p",
            object="o", source="src", timestamp="2026-01-01T00:00:00Z"
        )
        with pytest.raises(NotImplementedError):
            compute_log_likelihood_ratio(claim, "h001", [])


# ===========================================================================
# regret_engine — rank_actions_by_ev (RFC-0018 §3.12)
# ===========================================================================

class TestRankActionsByEV:
    def _action(self, action_id: str, ev: float, risk: float = 0.0) -> MathAction:
        return MathAction(id=action_id, type="filing", expected_value=ev, risk=risk)

    def test_empty_list_returns_empty(self) -> None:
        assert rank_actions_by_ev([], []) == []

    def test_rank_descending_by_ev_minus_risk(self) -> None:
        a1 = self._action("a1", ev=8.0, risk=1.0)  # net 7.0
        a2 = self._action("a2", ev=5.0, risk=0.0)  # net 5.0
        a3 = self._action("a3", ev=10.0, risk=6.0)  # net 4.0
        ranked = rank_actions_by_ev([a1, a2, a3], [])
        assert ranked[0].id == "a1"
        assert ranked[1].id == "a2"
        assert ranked[2].id == "a3"

    def test_single_action_returned(self) -> None:
        a = self._action("solo", ev=5.0)
        assert rank_actions_by_ev([a], []) == [a]

    def test_high_risk_action_demoted(self) -> None:
        safe = self._action("safe", ev=3.0, risk=0.0)    # net 3.0
        risky = self._action("risky", ev=9.0, risk=7.0)  # net 2.0
        ranked = rank_actions_by_ev([risky, safe], [])
        assert ranked[0].id == "safe"

    def test_cfr_strategy_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            compute_regret_weighted_strategy([], [], iterations=10)
