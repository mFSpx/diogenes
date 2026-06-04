#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pytest

from ALGOS import diffusion_forcing as df


def test_diffusion_forcing_loss_validates_shape_mismatch():
    x0 = np.zeros((2, 3), dtype=np.float32)
    eps_pred = np.zeros((2, 2), dtype=np.float32)
    t_seq = np.array([0, 1], dtype=np.int64)
    alpha_bars = df.noise_schedule(10, schedule="cosine")
    with pytest.raises(ValueError, match="shape"):
        df.diffusion_forcing_loss(x0, eps_pred, t_seq, alpha_bars, np.random.default_rng(0))


def test_sample_causal_t_seq_validates_inputs():
    with pytest.raises(ValueError, match="clean_prefix"):
        df.sample_causal_t_seq(5, 10, clean_prefix=99)
    with pytest.raises(ValueError, match="T"):
        df.sample_causal_t_seq(5, 0)


def test_diffusion_forcing_loss_rejects_out_of_range_timesteps():
    x0 = np.zeros((2, 2), dtype=np.float32)
    eps_pred = np.zeros((2, 2), dtype=np.float32)
    alpha_bars = df.noise_schedule(3, schedule="cosine")
    with pytest.raises(ValueError, match="timesteps"):
        df.diffusion_forcing_loss(x0, eps_pred, np.array([0, 99], dtype=np.int64), alpha_bars, np.random.default_rng(0))


def test_diffusion_forcing_loss_streams_without_allocating_full_noisy_tensor(monkeypatch):
    alpha_bars = df.noise_schedule(4, schedule="cosine")
    x0 = np.zeros((3, 2), dtype=np.float64)
    eps_pred = np.zeros((3, 2), dtype=np.float64)
    t_seq = np.array([0, 1, 2], dtype=np.int64)

    def _fail(*args, **kwargs):
        raise AssertionError("add_noise_sequence should not be called")

    monkeypatch.setattr(df, "add_noise_sequence", _fail)
    loss = df.diffusion_forcing_loss(x0, eps_pred, t_seq, alpha_bars, np.random.default_rng(0))
    assert isinstance(loss, float)
    assert np.isfinite(loss)
