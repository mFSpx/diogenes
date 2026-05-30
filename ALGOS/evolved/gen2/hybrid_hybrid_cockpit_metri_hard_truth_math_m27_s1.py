# DARWIN HAMMER — match 27, survivor 1
# gen: 2
# parent_a: hybrid_cockpit_metrics_rectified_flow_m10_s2.py (gen1)
# parent_b: hard_truth_math.py (gen0)
# born: 2026-05-29T23:26:23Z

# hybrid_cockpit_rectified_flow_hard_truth_math.py
"""
Hybrid module unifying the cockpit honesty/evidence metrics (Parent A: hybrid_cockpit_metrics_rectified_flow_m10_s2) with the hard-truth LSM vector features/math (Parent B: hard_truth_math.py).

Mathematical bridge
-------------------
The core of the rectified-flow family is the constant-velocity vector field v*(x0, x1) = x1 - x0, driving a straight-line interpolation Z_t = t·x1 + (1-t)·x0. The cockpit metrics provide a scalar *trust* value in the interval [0,1]. We treat this scalar as a multiplicative factor on the ideal velocity to obtain a *trust-weighted* velocity field v_hybrid(x0, x1; h) = h · (x1 - x0), where h ∈ [0,1] is any cockpit metric.

The hard-truth LSM vector features/math from Parent B can be used to compute the similarity score between two text strings. By combining the trust-weighted velocity from the cockpit metrics with the LSM vector similarity score, we can create a hybrid system that adapts the trust-weighted velocity based on the similarity between text strings.

Functions
---------
* `hybrid_flow_target`: Metric-scaled target velocity.
* `hybrid_flow_loss`: MSE between a model prediction and the scaled target.
* `hybrid_euler_solve`: Euler integration that adapts the step size using the audit-debt count as a regulariser and the LSM vector similarity score.
"""

import math
import random
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – cockpit metrics (re-implemented for internal use)
# ---------------------------------------------------------------------------

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known-good, clamped to [0, 1]."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))


def audit_debt(exports_missing_audit_step: int, total_exports: int) -> float:
    """Ratio of exports missing audit steps, clamped to [0, 1]."""
    return 1.0 if total_exports <= 0 else max(0.0, min(1.0, exports_missing_audit_step / total_exports))


# ---------------------------------------------------------------------------
# Parent B – hard-truth LSM vector features/math (re-implemented for internal use)
# ---------------------------------------------------------------------------

def words(text: str) -> list[str]:
    """Split a text into words."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> dict[str, float]:
    """Compute the LSM vector for a text string."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {cat: sum(cnt[w] for w in vocab) / total for cat, vocab in FUNCTION_CATS.items()}


def lsm_score(a: dict[str, float], b: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Compute the similarity score between two LSM vectors."""
    detail: dict[str, float] = {}
    vals: list[float] = []
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
        vals.append(score)
    return sum(vals) / len(vals), detail


# ---------------------------------------------------------------------------
# Hybrid functions
# ---------------------------------------------------------------------------

def hybrid_flow_target(x0: np.ndarray, x1: np.ndarray, h: float) -> np.ndarray:
    """Metric-scaled target velocity."""
    return h * (x1 - x0)


def hybrid_flow_loss(model_prediction: np.ndarray, target_velocity: np.ndarray) -> float:
    """MSE between a model prediction and the scaled target."""
    return np.mean((model_prediction - target_velocity) ** 2)


def hybrid_euler_solve(x0: np.ndarray, v0: np.ndarray, dt: float, h: float, lsm_score: float) -> np.ndarray:
    """Euler integration that adapts the step size using the audit-debt count as a regulariser and the LSM vector similarity score."""
    v_hybrid = hybrid_flow_target(x0, v0, h)
    x1 = x0 + dt * v_hybrid
    v1 = v0 + dt * (1 - lsm_score) * v_hybrid
    return x1, v1


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate some random data
    np.random.seed(0)
    x0 = np.random.rand(2)
    v0 = np.random.rand(2)
    dt = 0.1
    h = cockpit_honesty(10, 20)
    lsm_score, _ = lsm_score(lsm_vector("Hello world"), lsm_vector("Hello universe"))

    # Run the hybrid Euler solver
    x1, v1 = hybrid_euler_solve(x0, v0, dt, h, lsm_score)

    # Print the results
    print("x0:", x0)
    print("v0:", v0)
    print("x1:", x1)
    print("v1:", v1)