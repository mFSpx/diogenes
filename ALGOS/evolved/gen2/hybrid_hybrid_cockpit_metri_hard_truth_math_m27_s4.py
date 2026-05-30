# DARWIN HAMMER — match 27, survivor 4
# gen: 2
# parent_a: hybrid_cockpit_metrics_rectified_flow_m10_s2.py (gen1)
# parent_b: hard_truth_math.py (gen0)
# born: 2026-05-29T23:26:23Z

"""cockpit_rectified_hybrid_style.py
Hybrid module unifying the cockpit honesty/evidence metrics (Parent A) with the
hard‑truth linguistic style vectors and scoring (Parent B).

Mathematical bridge
-------------------
Parent A supplies a scalar trust factor h∈[0,1] (e.g. ``cockpit_honesty`` or
``anti_slop_ratio``).  Parent B supplies high‑dimensional style vectors
v₀, v₁∈ℝᴰ obtained from ``lsm_vector`` (D = number of linguistic categories).
The rectified‑flow core defines a constant velocity field

    v*(v₀, v₁) = v₁ − v₀                                            (1)

which yields a straight‑line interpolation Zₜ = (1−t)v₀ + t v₁.

We fuse the two by scaling the velocity with the trust factor:

    v_hybrid(v₀, v₁; h) = h·(v₁ − v₀)                               (2)

Thus the target style after a full unit‑time flow is

    v_target = v₀ + v_hybrid = v₀ + h·(v₁ − v₀)                     (3)

Equations (2)–(3) embed the cockpit metric directly into the linguistic
vector transport.  The hybrid loss is the mean‑squared error between a model
prediction and v_target, while an Euler integrator steps toward v_target
with a step size modulated by an audit‑debt regulariser.

The module provides three representative hybrid functions:
* ``hybrid_style_target`` – compute the trust‑weighted style target.
* ``hybrid_style_loss``   – MSE loss against the target.
* ``hybrid_euler_step``   – Euler integration toward the target.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Parent A – cockpit metrics (re‑implemented for internal use)
# ---------------------------------------------------------------------------

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known‑good, clamped to [0, 1]."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))


def audit_debt(exports_missing_audit_step: int) -> int:
    """Placeholder audit‑debt counter (higher = less trustworthy)."""
    return max(0, exports_missing_audit_step)


# ---------------------------------------------------------------------------
# Parent B – hard‑truth linguistic style utilities
# ---------------------------------------------------------------------------

FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

def _words(text: str) -> List[str]:
    """Extract lowercase alphabetic tokens from *text*."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> Dict[str, float]:
    """
    Compute a linguistic style matrix (LSM) vector for *text*.
    The result is a mapping from category name to relative frequency.
    """
    ws = _words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

def _vec_to_array(vec: Dict[str, float]) -> np.ndarray:
    """Convert an LSM dict to a deterministic numpy array ordered by FUNCTION_CATS keys."""
    return np.array([vec.get(cat, 0.0) for cat in sorted(FUNCTION_CATS.keys())], dtype=np.float64)


# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def hybrid_style_target(
    text_start: str,
    text_end: str,
    trust: float,
) -> np.ndarray:
    """
    Compute the trust‑weighted style target vector.

    Parameters
    ----------
    text_start, text_end : str
        Source and destination texts whose LSM vectors define the flow endpoints.
    trust : float
        Cockpit metric in [0, 1] (e.g. honesty or anti‑slop ratio).

    Returns
    -------
    np.ndarray
        Target style vector v_target = v_start + trust·(v_end − v_start).
    """
    v0_dict = lsm_vector(text_start)
    v1_dict = lsm_vector(text_end)
    v0 = _vec_to_array(v0_dict)
    v1 = _vec_to_array(v1_dict)
    # Equation (3): v_target = v0 + h·(v1 - v0)
    return v0 + trust * (v1 - v0)


def hybrid_style_loss(
    pred_vec: np.ndarray,
    text_start: str,
    text_end: str,
    trust: float,
) -> float:
    """
    Mean‑squared error between a predicted style vector and the hybrid target.

    Parameters
    ----------
    pred_vec : np.ndarray
        Model prediction (same dimensionality as LSM vectors).
    text_start, text_end : str
        Endpoint texts for the target computation.
    trust : float
        Cockpit trust metric.

    Returns
    -------
    float
        MSE loss.
    """
    target = hybrid_style_target(text_start, text_end, trust)
    if pred_vec.shape != target.shape:
        raise ValueError("Prediction shape does not match target shape.")
    return float(np.mean((pred_vec - target) ** 2))


def hybrid_euler_step(
    current_vec: np.ndarray,
    text_start: str,
    text_end: str,
    trust: float,
    step_factor: float = 0.1,
    audit_missing: int = 0,
) -> np.ndarray:
    """
    Perform a single Euler integration step toward the trust‑weighted target.

    The effective step size is attenuated by an audit‑debt regulariser:
        η = step_factor / (1 + audit_debt)

    Parameters
    ----------
    current_vec : np.ndarray
        Current style vector (e.g. output of a model at time t).
    text_start, text_end : str
        Endpoint texts defining the target style.
    trust : float
        Cockpit metric in [0, 1].
    step_factor : float, optional
        Base Euler step magnitude (default 0.1).
    audit_missing : int, optional
        Number of missing audit steps; higher values reduce the step size.

    Returns
    -------
    np.ndarray
        Updated style vector after one Euler step.
    """
    target = hybrid_style_target(text_start, text_end, trust)
    debt = audit_debt(audit_missing)
    eta = step_factor / (1.0 + debt)
    # Euler update: x_{t+1} = x_t + η·(target - x_t)
    return current_vec + eta * (target - current_vec)


# ---------------------------------------------------------------------------
# Utility for demonstration / debugging
# ---------------------------------------------------------------------------

def _print_vector(label: str, vec: np.ndarray) -> None:
    """Pretty‑print a style vector with its label."""
    cat_names = sorted(FUNCTION_CATS.keys())
    parts = [f"{cat}:{v:.3f}" for cat, v in zip(cat_names, vec)]
    print(f"{label} [{len(vec)} dims]: " + ", ".join(parts))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example texts
    txt_a = "I love the quick brown fox jumps over the lazy dog."
    txt_b = "You should not underestimate the power of diligent work."

    # Cockpit metrics (arbitrary example numbers)
    honesty = cockpit_honesty(displayed_ok=8, unknown_displayed_as_ok=2)  # 0.8
    slop = anti_slop_ratio(claims_with_evidence=5, total_claims_emitted=10)  # 0.5
    trust_metric = (honesty + slop) / 2.0  # simple blend → 0.65

    # Compute target style
    target_vec = hybrid_style_target(txt_a, txt_b, trust_metric)
    _print_vector("Target", target_vec)

    # Simulate a random initial prediction
    np.random.seed(0)
    pred_vec = np.random.rand(len(FUNCTION_CATS))
    _print_vector("Initial Pred", pred_vec)

    loss0 = hybrid_style_loss(pred_vec, txt_a, txt_b, trust_metric)
    print(f"Initial loss: {loss0:.6f}")

    # Perform a few Euler steps
    current = pred_vec.copy()
    for i in range(5):
        current = hybrid_euler_step(
            current,
            txt_a,
            txt_b,
            trust_metric,
            step_factor=0.2,
            audit_missing=i,  # increasing debt each step
        )
        loss_i = hybrid_style_loss(current, txt_a, txt_b, trust_metric)
        print(f"Step {i+1:02d} – loss: {loss_i:.6f}")

    _print_vector("Final", current)

    # Verify that loss decreased (allow tiny tolerance)
    assert loss_i <= loss0 + 1e-9, "Loss did not decrease as expected."

    print("Hybrid module smoke test completed successfully.")