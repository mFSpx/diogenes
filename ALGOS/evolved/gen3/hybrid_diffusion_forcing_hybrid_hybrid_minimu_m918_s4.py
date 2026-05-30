# DARWIN HAMMER — match 918, survivor 4
# gen: 3
# parent_a: diffusion_forcing.py (gen0)
# parent_b: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s2.py (gen2)
# born: 2026-05-29T23:31:39Z

"""Hybrid Diffusion‑Epistemic Bayesian Planner.

This module fuses two parent algorithms:

* **diffusion_forcing.py** – token‑wise diffusion where each position i receives an
  independent timestep t_i and a corresponding noise level ᾱ_{t_i}.  The loss
  weights each token by λ(t_i).

* **hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s2.py** – a Bayesian
  minimum‑cost tree where every node carries a `CertaintyFlag`.  The flag is
  converted to a probability p ∈ [0,1] and used as a prior/likelihood in a
  Bayesian update; the posterior then scales edge costs.

**Mathematical bridge**

For token i we have two independent sources of belief:

1. **Diffusion prior** p_i^{diff} = ᾱ_{t_i} (the expected signal‑to‑noise ratio after
   corrupting the token at timestep t_i).

2. **Epistemic prior** p_i^{epi} = confidence_bps/10 000 (from a `CertaintyFlag`).

Treating both as independent Bernoulli beliefs, we perform a Bayesian update
with a false‑positive rate ϵ_fp:

\[
p_i^{post} = \frac{p_i^{diff}\,p_i^{epi}}
                  {p_i^{diff}\,p_i^{epi} + (1-p_i^{diff})(1-p_i^{epi})\,
                   \epsilon_{fp}} .
\]

The posterior p_i^{post} is used in two places:

* as a **per‑token loss weight** λ_i = λ(t_i)·p_i^{post} in the diffusion‑forcing
  objective,
* as an **edge weight** in a chain‑structured minimum‑cost tree (each edge
  connects token i → i+1).  The total tree cost is the sum of these posterior‑
  weighted edges.

The three public functions below illustrate the hybrid operation:

* `combined_token_posterior` – computes p_i^{post} from a timestep and a flag.
* `hybrid_diffusion_forcing_loss` – diffusion loss weighted by the posterior.
* `hybrid_chain_tree_cost` – minimum‑cost chain tree cost using the same posterior.

All other utilities (noise schedule, noise addition, weighting) are retained
from the original diffusion forcing implementation.  The module is self‑contained
and runs a smoke test when executed as a script."""

from __future__ import annotations

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Diffusion Forcing utilities (adapted)
# ----------------------------------------------------------------------


def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """Cumulative noise schedule ᾱ_t for t = 0…T.

    Returns an array of shape (T+1,) with ᾱ_0 = 1.0 (clean) and ᾱ_T ≈ 0.0 (pure noise).
    """
    if schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        alpha_bars = np.clip(alpha_bars, a_min=1e-12, a_max=1.0)
        return alpha_bars
    elif schedule == "linear":
        # Linear decay from 1 to 0.
        return np.linspace(1.0, 0.0, T + 1)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")


def weighting_lambda(alpha_bar: np.ndarray, t: int) -> float:
    """Signal‑to‑noise based weighting λ(t) = 1 / (1 - ᾱ_t)."""
    # Prevent division by zero when ᾱ_t is 1 (t=0).
    eps = 1e-12
    return 1.0 / max(1.0 - alpha_bar[t], eps)


def add_noise_token(x0: np.ndarray, t: int, alpha_bar: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Corrupt a single token x0 with timestep t using the schedule ᾱ."""
    sqrt_alpha = math.sqrt(alpha_bar[t])
    sqrt_one_minus = math.sqrt(1.0 - alpha_bar[t])
    epsilon = rng.normal(size=x0.shape)
    return sqrt_alpha * x0 + sqrt_one_minus * epsilon


def add_noise_sequence(
    x0_seq: np.ndarray,
    t_seq: np.ndarray,
    alpha_bar: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply per‑token diffusion noise to an entire sequence.

    Parameters
    ----------
    x0_seq : np.ndarray shape (N, D)
        Clean token embeddings.
    t_seq : np.ndarray shape (N,)
        Integer timesteps in [0, T].
    alpha_bar : np.ndarray shape (T+1,)
        Cumulative schedule.
    rng : np.random.Generator, optional
        Random generator; if None a new default RNG is created.

    Returns
    -------
    np.ndarray shape (N, D) – noisy sequence.
    """
    if rng is None:
        rng = np.random.default_rng()
    noisy = np.empty_like(x0_seq)
    for i, (x0, t) in enumerate(zip(x0_seq, t_seq)):
        noisy[i] = add_noise_token(x0, int(t), alpha_bar, rng)
    return noisy


def sample_causal_t_seq(N: int, T: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a causal timestep sequence for planning.

    The first k tokens (unknown length) are set to 0 (clean prefix);
    the remaining tokens are drawn uniformly from {1,…,T} (noisy suffix).
    """
    if rng is None:
        rng = np.random.default_rng()
    # Randomly decide a split point (including the possibility of no clean prefix).
    split = rng.integers(0, N + 1)
    t_seq = np.empty(N, dtype=int)
    t_seq[:split] = 0
    if split < N:
        t_seq[split:] = rng.integers(1, T + 1, size=N - split)
    return t_seq


# ----------------------------------------------------------------------
# Parent B – Epistemic certainty helpers (adapted)
# ----------------------------------------------------------------------


EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    """Metadata‑rich epistemic confidence descriptor."""
    label: str
    confidence_bps: int  # basis points, 0 … 10 000
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10_000")


def confidence_to_probability(flag: CertaintyFlag) -> float:
    """Map a CertaintyFlag to a probability p ∈ [0,1]."""
    return flag.confidence_bps / 10_000.0


# ----------------------------------------------------------------------
# Hybrid bridge – Bayesian fusion of diffusion prior and epistemic prior
# ----------------------------------------------------------------------


def combined_token_posterior(
    t: int,
    T: int,
    flag: CertaintyFlag,
    schedule: str = "cosine",
    false_positive: float = 1e-3,
) -> float:
    """Compute the posterior belief p_post for a single token.

    Parameters
    ----------
    t : int
        Diffusion timestep for the token (0 ≤ t ≤ T).
    T : int
        Total diffusion steps.
    flag : CertaintyFlag
        Epistemic confidence attached to the token.
    schedule : str
        Noise schedule name passed to :func:`noise_schedule`.
    false_positive : float
        Small constant ϵ_fp representing the chance that both priors are wrong.

    Returns
    -------
    float
        Posterior probability in (0,1).
    """
    alpha_bar = noise_schedule(T, schedule)[t]  # diffusion prior p_diff = ᾱ_t
    p_diff = float(alpha_bar)
    p_epi = confidence_to_probability(flag)

    numerator = p_diff * p_epi
    denominator = numerator + (1.0 - p_diff) * (1.0 - p_epi) * false_positive
    # Guard against pathological zero denominator.
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def hybrid_diffusion_forcing_loss(
    x0_seq: np.ndarray,
    epsilon_seq: np.ndarray,
    t_seq: np.ndarray,
    flags: List[CertaintyFlag],
    T: int,
    schedule: str = "cosine",
    false_positive: float = 1e-3,
) -> float:
    """Diffusion‑forcing loss weighted by epistemic‑Bayesian posteriors.

    The model prediction function ``epsilon_theta`` is abstracted as a callable
    that returns a noisy estimate for each token.  For the smoke test we use a
    simple identity function.

    Parameters
    ----------
    x0_seq : np.ndarray shape (N, D)
        Clean token embeddings.
    epsilon_seq : np.ndarray shape (N, D)
        Ground‑truth noise ε used to corrupt the sequence.
    t_seq : np.ndarray shape (N,)
        Per‑token diffusion timesteps.
    flags : list of CertaintyFlag, length N
        Epistemic confidence for each token.
    T : int
        Total diffusion steps.
    schedule : str
        Noise schedule identifier.
    false_positive : float
        Bayesian false‑positive rate.

    Returns
    -------
    float
        Scalar loss value.
    """
    if len(x0_seq) != len(t_seq) or len(x0_seq) != len(flags):
        raise ValueError("Lengths of sequence, timesteps and flags must match")

    alpha_bar = noise_schedule(T, schedule)

    # Simple placeholder model: predicts the true epsilon (perfect denoiser).
    def epsilon_theta(noisy_seq: np.ndarray, t_seq_local: np.ndarray) -> np.ndarray:
        # In practice this would be a neural network; here we just return the ground truth.
        return epsilon_seq

    # Compute the model output once (identical for all tokens in this placeholder).
    pred_eps = epsilon_theta(add_noise_sequence(x0_seq, t_seq, alpha_bar), t_seq)

    total_loss = 0.0
    for i, (t, flag) in enumerate(zip(t_seq, flags)):
        lam = weighting_lambda(alpha_bar, int(t))
        posterior = combined_token_posterior(int(t), T, flag, schedule, false_positive)
        weight = lam * posterior
        diff = epsilon_seq[i] - pred_eps[i]
        total_loss += weight * np.mean(diff ** 2)  # MSE per token
    return total_loss


def hybrid_chain_tree_cost(
    t_seq: np.ndarray,
    flags: List[CertaintyFlag],
    T: int,
    schedule: str = "cosine",
    false_positive: float = 1e-3,
) -> float:
    """Minimum‑cost chain tree where each edge weight = posterior of its source token.

    The chain connects token i → i+1 for i = 0 … N‑2.  The total cost is the sum of
    posterior‑scaled edge weights.

    Parameters
    ----------
    t_seq : np.ndarray shape (N,)
        Per‑token diffusion timesteps.
    flags : list of CertaintyFlag, length N
        Epistemic confidence per token.
    T : int
        Total diffusion steps.
    schedule : str
        Noise schedule name.
    false_positive : float
        Bayesian false‑positive rate.

    Returns
    -------
    float
        Total tree cost.
    """
    if len(t_seq) != len(flags):
        raise ValueError("Lengths of timesteps and flags must match")
    N = len(t_seq)
    if N < 2:
        return 0.0

    total_cost = 0.0
    for i in range(N - 1):
        post_i = combined_token_posterior(int(t_seq[i]), T, flags[i], schedule, false_positive)
        # Edge cost could also incorporate a physical distance; we use unit distance.
        total_cost += post_i
    return total_cost


def aggregate_hybrid_certainty(
    flags: List[CertaintyFlag],
    t_seq: np.ndarray,
    T: int,
    schedule: str = "cosine",
    false_positive: float = 1e-3,
) -> float:
    """Aggregate overall certainty as the product of per‑token posteriors."""
    if len(flags) != len(t_seq):
        raise ValueError("Lengths must match")
    prod = 1.0
    for t, flag in zip(t_seq, flags):
        prod *= combined_token_posterior(int(t), T, flag, schedule, false_positive)
    return prod


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Simple deterministic test with a tiny sequence.
    rng = np.random.default_rng(42)

    N = 5          # sequence length
    D = 3          # embedding dimension
    T = 10         # diffusion steps

    # Random clean embeddings.
    x0 = rng.normal(size=(N, D))

    # Ground‑truth noise (same shape as x0).
    epsilon = rng.normal(size=(N, D))

    # Sample a causal timestep sequence.
    t_seq = sample_causal_t_seq(N, T, rng)

    # Create a matching list of CertaintyFlag objects with varying confidence.
    labels = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
    flags = [
        CertaintyFlag(label=lbl, confidence_bps=bp, authority_class="test", rationale="smoke")
        for lbl, bp in zip(labels, [9000, 7000, 5000, 2000, 1000])
    ]

    # Compute hybrid loss.
    loss = hybrid_diffusion_forcing_loss(x0, epsilon, t_seq, flags, T)
    print(f"Hybrid diffusion‑forcing loss: {loss:.6f}")

    # Compute hybrid chain tree cost.
    tree_cost = hybrid_chain_tree_cost(t_seq, flags, T)
    print(f"Hybrid chain tree cost: {tree_cost:.6f}")

    # Aggregate overall certainty.
    agg_cert = aggregate_hybrid_certainty(flags, t_seq, T)
    print(f"Aggregate hybrid certainty (product of posteriors): {agg_cert:.6e}")

    # Verify that loss is non‑negative and tree cost is non‑negative.
    assert loss >= 0.0
    assert tree_cost >= 0.0
    assert 0.0 <= agg_cert <= 1.0

    sys.exit(0)