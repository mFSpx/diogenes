# DARWIN HAMMER — match 488, survivor 1
# gen: 4
# parent_a: hybrid_cockpit_metrics_hybrid_hybrid_ternar_m229_s3.py (gen3)
# parent_b: hybrid_hybrid_liquid_time_c_diffusion_forcing_m16_s0.py (gen2)
# born: 2026-05-29T23:29:08Z

"""
Hybrid Cockpit‑Metrics & Liquid‑Time‑Constant Diffusion (Hybrid A+B)

Parent A: cockpit metrics with adaptive pruning (anti_slop_ratio, cockpit_honesty,
social_interaction, evasion_delta).
Parent B: signature‑based similarity and diffusion‑forcing noise schedule
(signature, similarity, noise_schedule).

Mathematical bridge:
Both parents expose scalar quality measures (honesty, slop ratio, similarity) that
can be used as weighting factors for each other.  The hybrid constructs a
*pruning probability* from the honesty‑based metrics of A and feeds it into the
noise schedule of B, thereby modulating diffusion intensity according to how
trustworthy the current claim set is.  Conversely, the similarity between two
token sets modulates the strength of the social interaction update, allowing
the vector dynamics to be damped when the underlying evidence is weak.
"""

import math
import random
import sys
from pathlib import Path
from typing import Sequence, List

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Parent A core components
# ----------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Fraction of claims that are backed by evidence, clipped to [0,1]."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Honesty = proportion of displayed claims that are known to be OK."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))


def social_interaction(
    x: Vector,
    g_best: Vector,
    k: int = 1,
    r1: float | None = None,
    seed: int | str | None = None,
) -> List[float]:
    """Particle‑swarm‑style attraction toward the global best."""
    if len(x) != len(g_best):
        raise ValueError("x and g_best must share dimension")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    rng = random.Random(seed)
    r = rng.random() if r1 is None else r1
    if not (0.0 <= r <= 1.0):
        raise ValueError("r1 must be in [0, 1]")
    return [xi + r * (gj - k * xi) for xi, gj in zip(x, g_best)]


def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 0.1) -> float:
    """Decaying factor that encourages evasion early and stabilises later."""
    if t_max <= 0:
        raise ValueError("t_max must be positive")
    return delta_max * (1.0 - (t / t_max)) ** alpha


def hybrid_pruning_schedule(
    claims_with_evidence: int,
    total_claims_emitted: int,
    displayed_ok: int,
    unknown_displayed_as_ok: int,
    t: int,
    t_max: int,
) -> float:
    """
    Combined pruning probability:
        p = (1 - honesty) * anti_slop_ratio * evasion_delta
    Clipped to [0,1].
    """
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    slop = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    delta = evasion_delta(t, t_max)
    prob = (1.0 - honesty) * slop * delta
    return max(0.0, min(1.0, prob))


# ----------------------------------------------------------------------
# Parent B core components
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash used for MinHash‑style signatures."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    # Simple rolling hash to stay within stdlib (blake2b is allowed)
    import hashlib

    return int.from_bytes(
        hashlib.blake2b(data, digest_size=8).digest(), "big"
    )


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """MinHash‑style signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k  # maximal value when empty
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity based on equal hash entries."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """Diffusion‑forcing noise schedule (alpha‑bars)."""
    if T <= 0:
        raise ValueError("T must be positive")
    if schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        return np.clip(alpha_bars, 1e-9, 1.0)
    elif schedule == "linear":
        beta_start = 1e-4
        beta_end = 0.02
        betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
        alphas = 1.0 - betas
        return np.clip(np.cumprod(alphas), 1e-9, 1.0)
    else:
        raise ValueError(f"unknown schedule '{schedule}'")


# ----------------------------------------------------------------------
# Hybrid operations (integrating A and B)
# ----------------------------------------------------------------------
def hybrid_noise_weighted_by_honesty(
    displayed_ok: int,
    unknown_displayed_as_ok: int,
    T: int,
    schedule: str = "cosine",
) -> np.ndarray:
    """
    Scale the diffusion noise schedule by the cockpit honesty metric.
    High honesty → less aggressive noise (more trust in the data).
    """
    base = noise_schedule(T, schedule)
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    # Map honesty ∈ [0,1] to a damping factor ∈ [0.5,1.0]
    damping = 0.5 + 0.5 * honesty
    return base * damping


def hybrid_social_update(
    x: Vector,
    g_best: Vector,
    claims_with_evidence: int,
    total_claims_emitted: int,
    displayed_ok: int,
    unknown_displayed_as_ok: int,
    t: int,
    t_max: int,
    k: int = 1,
    seed: int | str | None = None,
) -> List[float]:
    """
    Perform a social interaction step whose step size is attenuated by the
    pruning probability derived from the cockpit metrics.
    """
    prune_prob = hybrid_pruning_schedule(
        claims_with_evidence,
        total_claims_emitted,
        displayed_ok,
        unknown_displayed_as_ok,
        t,
        t_max,
    )
    # The interaction strength is reduced proportionally to the pruning probability.
    raw = social_interaction(x, g_best, k=k, seed=seed)
    return [xi * (1.0 - prune_prob) for xi in raw]


def hybrid_similarity_pruned(
    tokens_a: List[str],
    tokens_b: List[str],
    claims_with_evidence: int,
    total_claims_emitted: int,
    displayed_ok: int,
    unknown_displayed_as_ok: int,
    t: int,
    t_max: int,
    k: int = 128,
) -> float:
    """
    Compute token similarity, then diminish it according to the current pruning
    probability (i.e., low‑trust contexts reduce the effective similarity).
    """
    sig_a = signature(tokens_a, k)
    sig_b = signature(tokens_b, k)
    base_sim = similarity(sig_a, sig_b)
    prune_prob = hybrid_pruning_schedule(
        claims_with_evidence,
        total_claims_emitted,
        displayed_ok,
        unknown_displayed_as_ok,
        t,
        t_max,
    )
    # Final similarity is linearly blended with a baseline of 0.
    return base_sim * (1.0 - prune_prob)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy data for a quick sanity run
    x_vec = [0.2, -0.5, 1.0]
    g_best_vec = [0.0, 0.0, 0.0]

    # Metrics for pruning / honesty
    claims_with_evidence = 70
    total_claims_emitted = 100
    displayed_ok = 40
    unknown_displayed_as_ok = 10
    t, t_max = 3, 10

    # Tokens for similarity
    tokens1 = ["alpha", "beta", "gamma", "delta"]
    tokens2 = ["alpha", "epsilon", "gamma", "zeta"]

    # Hybrid noise schedule
    noise = hybrid_noise_weighted_by_honesty(
        displayed_ok, unknown_displayed_as_ok, T=20, schedule="cosine"
    )
    print("Hybrid noise schedule (first 5 values):", noise[:5])

    # Hybrid social update
    updated = hybrid_social_update(
        x_vec,
        g_best_vec,
        claims_with_evidence,
        total_claims_emitted,
        displayed_ok,
        unknown_displayed_as_ok,
        t,
        t_max,
        seed=42,
    )
    print("Hybrid social update result:", updated)

    # Hybrid similarity
    sim = hybrid_similarity_pruned(
        tokens1,
        tokens2,
        claims_with_evidence,
        total_claims_emitted,
        displayed_ok,
        unknown_displayed_as_ok,
        t,
        t_max,
    )
    print("Hybrid similarity (pruned):", sim)

    sys.exit(0)