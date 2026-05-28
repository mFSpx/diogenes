"""Tests for pypeline.math.counterfactual_effects — RFC-0018 §3.11."""
import pytest
import numpy as np
from pypeline.math.counterfactual_effects import (
    CausalEffect,
    estimate_causal_effect,
    estimate_heterogeneous_effects,
    run_refutation_suite,
)


def _synthetic_data(n: int = 100, true_effect: float = 2.0, seed: int = 42) -> dict:
    """Linear DGP: Y = true_effect * T + 0.5 * X + noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    T = (X + rng.standard_normal(n) > 0).astype(float)
    Y = true_effect * T + 0.5 * X + rng.standard_normal(n) * 0.5
    return {"T": T.tolist(), "Y": Y.tolist(), "X": X.tolist()}


class TestEstimateCausalEffect:
    def test_returns_causal_effect(self):
        data = _synthetic_data()
        result = estimate_causal_effect("T", "Y", ["X"], data)
        assert isinstance(result, CausalEffect)

    def test_ate_close_to_true_effect(self):
        data = _synthetic_data(n=300, true_effect=2.0)
        result = estimate_causal_effect("T", "Y", ["X"], data)
        assert result.ate_estimate is not None
        assert abs(result.ate_estimate - 2.0) < 1.0  # Within 1 unit

    def test_returns_confidence_interval(self):
        data = _synthetic_data()
        result = estimate_causal_effect("T", "Y", ["X"], data)
        assert result.ate_confidence_interval is not None
        lo, hi = result.ate_confidence_interval
        assert lo <= hi

    def test_ci_contains_true_effect(self):
        data = _synthetic_data(n=500, true_effect=2.0)
        result = estimate_causal_effect("T", "Y", ["X"], data)
        if result.ate_confidence_interval:
            lo, hi = result.ate_confidence_interval
            assert lo <= 2.5 and hi >= 1.5  # Reasonable CI around 2.0

    def test_no_confounders(self):
        rng = np.random.default_rng(7)
        n = 100
        T = rng.integers(0, 2, n).astype(float)
        Y = 3.0 * T + rng.standard_normal(n)
        data = {"T": T.tolist(), "Y": Y.tolist()}
        result = estimate_causal_effect("T", "Y", [], data)
        assert result.ate_estimate is not None
        assert abs(result.ate_estimate - 3.0) < 1.5

    def test_effect_id_unique(self):
        data = _synthetic_data()
        r1 = estimate_causal_effect("T", "Y", ["X"], data)
        r2 = estimate_causal_effect("T", "Y", ["X"], data)
        assert r1.effect_id != r2.effect_id

    def test_refutation_methods_set(self):
        data = _synthetic_data()
        result = estimate_causal_effect("T", "Y", ["X"], data)
        assert len(result.refutation_methods) > 0


class TestEstimateHeterogeneousEffects:
    def test_returns_dict(self):
        data = _synthetic_data(n=200)
        result = estimate_heterogeneous_effects("T", "Y", ["X"], data)
        assert isinstance(result, dict)

    def test_subgroup_keys_present(self):
        data = _synthetic_data(n=200)
        result = estimate_heterogeneous_effects("T", "Y", ["X"], data)
        assert len(result) > 0

    def test_no_confounders_returns_overall(self):
        rng = np.random.default_rng(5)
        n = 80
        T = rng.integers(0, 2, n).astype(float)
        Y = 2.0 * T + rng.standard_normal(n)
        data = {"T": T.tolist(), "Y": Y.tolist()}
        result = estimate_heterogeneous_effects("T", "Y", [], data)
        assert "overall" in result

    def test_effect_values_are_floats(self):
        data = _synthetic_data(n=200)
        result = estimate_heterogeneous_effects("T", "Y", ["X"], data)
        for v in result.values():
            assert isinstance(v, float)


class TestRunRefutationSuite:
    def _make_effect(self, ate: float, ci: tuple | None = None, passed: bool = True) -> CausalEffect:
        return CausalEffect(
            effect_id="test_effect",
            treatment="T",
            outcome="Y",
            confounders=("X",),
            ate_estimate=ate,
            ate_confidence_interval=ci or (ate - 0.5, ate + 0.5),
            refutation_passed=passed,
            refutation_methods=("placebo_treatment",),
            heterogeneous_effects={},
        )

    def test_returns_dict(self):
        effect = self._make_effect(2.0)
        result = run_refutation_suite(effect)
        assert isinstance(result, dict)

    def test_default_methods_present(self):
        effect = self._make_effect(2.0)
        result = run_refutation_suite(effect)
        assert "placebo_treatment" in result
        assert "data_subset" in result
        assert "random_common_cause" in result

    def test_custom_methods(self):
        effect = self._make_effect(2.0)
        result = run_refutation_suite(effect, methods=["placebo_treatment"])
        assert set(result.keys()) == {"placebo_treatment"}

    def test_none_ate_returns_all_false(self):
        effect = CausalEffect(
            effect_id="none_ate",
            treatment="T",
            outcome="Y",
            confounders=(),
            ate_estimate=None,
            ate_confidence_interval=None,
            refutation_passed=False,
            refutation_methods=(),
            heterogeneous_effects={},
        )
        result = run_refutation_suite(effect)
        assert all(v is False for v in result.values())

    def test_bool_values(self):
        effect = self._make_effect(3.0)
        result = run_refutation_suite(effect)
        for v in result.values():
            assert isinstance(v, bool)
