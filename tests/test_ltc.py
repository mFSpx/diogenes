from __future__ import annotations

import numpy as np
import pytest

from ALGOS import ltc


def test_ltc_run_returns_final_state_by_default():
    cell = ltc.LTCCell(input_size=4, hidden_size=3, seed="ltc-test")
    seq = ltc.LTCSequence(cell)
    observations = [
        (0.0, ltc.feature_hash("alpha", dim=4)),
        (1800.0, ltc.feature_hash("bravo", dim=4)),
    ]

    out = seq.run(observations)

    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)


def test_ltc_run_can_return_states_when_requested():
    cell = ltc.LTCCell(input_size=4, hidden_size=2, seed="ltc-test-states")
    seq = ltc.LTCSequence(cell)
    observations = [
        (0.0, ltc.feature_hash("alpha", dim=4)),
        (3600.0, ltc.feature_hash("bravo", dim=4)),
        (7200.0, ltc.feature_hash("charlie", dim=4)),
    ]

    states = seq.run(observations, return_states=True)

    assert isinstance(states, list)
    assert len(states) == 3
    assert all(isinstance(s, np.ndarray) for s in states)
    assert all(s.shape == (2,) for s in states)


def test_ltc_feature_and_hidden_size_caps_are_enforced():
    with pytest.raises(ValueError, match="input_size"):
        ltc.LTCCell(input_size=ltc.MAX_FEATURE_DIM + 1, hidden_size=2, seed="ltc-cap")

    with pytest.raises(ValueError, match="hidden_size"):
        ltc.LTCCell(input_size=2, hidden_size=ltc.MAX_HIDDEN_SIZE + 1, seed="ltc-cap")


def test_ltc_feature_hash_enforces_dimension_cap():
    with pytest.raises(ValueError, match="dimension"):
        ltc.feature_hash("too-wide-feature", dim=ltc.MAX_FEATURE_DIM + 1)


def test_ltc_step_caps_substeps_and_delta_t(monkeypatch):
    cell = ltc.LTCCell(input_size=4, hidden_size=3, seed="ltc-substep")
    x0 = np.zeros(3)
    x1 = np.ones(4)

    baseline = ltc.LTCCell(input_size=4, hidden_size=3, seed="ltc-substep")
    x_base = baseline.step(x0.copy(), x1, dt=ltc.MAX_DELTA_T, max_sub_dt=ltc.MAX_SUB_DT)

    x_cap = cell.step(x0.copy(), x1, dt=ltc.MAX_DELTA_T * 10.0, max_sub_dt=ltc.MAX_SUB_DT)
    assert np.allclose(x_cap, x_base)

    calls = {"n": 0}
    original_sigmoid = ltc._sigmoid

    def counting_sigmoid(values):
        calls["n"] += 1
        return original_sigmoid(values)

    monkeypatch.setattr(ltc, "_sigmoid", counting_sigmoid)
    # Make dt large enough that, without a substep cap this would explode.
    _ = cell.step(np.zeros(3), np.ones(4), dt=ltc.MAX_DELTA_T * 5, max_sub_dt=ltc.MAX_SUB_DT)
    assert calls["n"] <= 2 * ltc.MAX_SUB_STEPS
