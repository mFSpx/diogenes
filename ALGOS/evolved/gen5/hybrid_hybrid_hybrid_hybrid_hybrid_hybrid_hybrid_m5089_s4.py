# DARWIN HAMMER — match 5089, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1035_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m2639_s0.py (gen4)
# born: 2026-05-29T23:59:53Z

import sys
import math
import random
import re
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex feature extraction
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def _extract_counts(text: str) -> dict:
    """Return raw counts of evidence‑related and planning‑related tokens."""
    evidence = len(EVIDENCE_RE.findall(text))
    planning = len(PLANNING_RE.findall(text))
    return {"evidence": evidence, "planning": planning}


def feature_vector(text: str) -> np.ndarray:
    """2‑element column vector [evidence; planning] as float64."""
    cnts = _extract_counts(text)
    return np.array([cnts["evidence"], cnts["planning"]], dtype=np.float64).reshape(2, 1)


def confidence_from_vector(v: np.ndarray, eps: float = 1e-8) -> float:
    """
    Map a 2‑element column vector to a scalar confidence ρ ∈ (0,1).

    Uses a smooth sigmoid on the L2 norm to avoid a dead‑zone when v = 0.
    """
    norm = np.linalg.norm(v)
    # sigmoid-like mapping that respects (0,1) range even for norm=0
    return 1.0 / (1.0 + math.exp(-(norm - 1.0))) * (1.0 - eps) + eps


# ----------------------------------------------------------------------
# Parent B – TTT‑Linear core with gradient descent
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize weight matrix W ~ N(0,scale²)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in), dtype=np.float64) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    """Quadratic loss L = ||W·x – target||² (target defaults to x)."""
    if target is None:
        target = x
    diff = W @ x - target
    return float(np.sum(diff ** 2))


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Gradient ∂L/∂W = 2·(W·x – target)·xᵀ."""
    if target is None:
        target = x
    diff = W @ x - target
    return 2.0 * diff @ x.T


# ----------------------------------------------------------------------
# Fractional (Caputo) memory utilities – normalized kernel
# ----------------------------------------------------------------------
def caputo_kernel(alpha: float, length: int) -> np.ndarray:
    """
    Return a normalized Caputo power‑law kernel κ_k = (k+1)^{-α} / Γ(1‑α)
    for k = 0 … length‑1 such that Σ κ_k = 1.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0,1)")
    coeff = 1.0 / math.gamma(1.0 - alpha)
    ks = np.arange(1, length + 1, dtype=np.float64)  # k+1
    raw = coeff * ks ** (-alpha)
    return raw / raw.sum()  # normalization


def fractional_weighted_sum(
    grad_conf_history: List[Tuple[np.ndarray, float]],
    alpha: float,
) -> np.ndarray:
    """
    Compute Σ κ_k·c_k·g_k where:
        g_k – gradient (newest first)
        c_k – confidence at the time g_k was recorded
        κ_k – normalized Caputo kernel (newest ↔ k=0)
    """
    if not grad_conf_history:
        raise ValueError("gradient‑confidence history is empty")
    grads, confs = zip(*grad_conf_history)  # unzip
    kernel = caputo_kernel(alpha, len(grads))
    weighted = sum(k * c * g for k, c, g in zip(kernel, confs, grads))
    return weighted


# ----------------------------------------------------------------------
# Hybrid operations – deeper integration of confidence
# ----------------------------------------------------------------------
def hybrid_step(
    W: np.ndarray,
    x: np.ndarray,
    eta: float,
    alpha: float,
    grad_conf_history: List[Tuple[np.ndarray, float]],
    confidence: float,
    max_history: int = 50,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, float]]]:
    """
    Perform one hybrid update with confidence‑aware fractional memory.

    Steps
    -----
    1. Compute current gradient g.
    2. Store (g, confidence) at the front of the history.
    3. Truncate history to `max_history`.
    4. Compute fractional weighted sum G = Σ κ_k·c_k·g_k.
    5. Apply scaled update ΔW = - η · G   (confidence already baked into history).

    Returns
    -------
    (W_new, updated_history)
    """
    # 1. current gradient
    g = ttt_grad(W, x)

    # 2. push (gradient, confidence) – newest first
    grad_conf_history.insert(0, (g, confidence))

    # 3. truncate
    if len(grad_conf_history) > max_history:
        grad_conf_history = grad_conf_history[:max_history]

    # 4. fractional aggregation
    G = fractional_weighted_sum(grad_conf_history, alpha)

    # 5. update (confidence already influences G)
    delta = -eta * G
    W_new = W + delta
    return W_new, grad_conf_history


def hybrid_process_text(
    text: str,
    W: np.ndarray,
    eta: float,
    alpha: float,
    grad_conf_history: List[Tuple[np.ndarray, float]],
    max_history: int = 50,
) -> Tuple[np.ndarray, float, List[Tuple[np.ndarray, float]]]:
    """
    End‑to‑end processing of a single document.

    Returns
    -------
    (W_new, confidence_used, updated_history)
    """
    v = feature_vector(text)               # (2,1)
    rho = confidence_from_vector(v)        # scalar ∈ (0,1)
    W_new, new_history = hybrid_step(
        W, v, eta, alpha, grad_conf_history, rho, max_history
    )
    return W_new, rho, new_history


def evaluate_hybrid(
    W: np.ndarray,
    texts: List[str],
    eta: float,
    alpha: float,
    max_history: int = 50,
) -> dict:
    """
    Run the hybrid system over a list of texts, accumulating loss and
    returning a summary dictionary.
    """
    grad_conf_history: List[Tuple[np.ndarray, float]] = []
    total_loss = 0.0

    for txt in texts:
        v = feature_vector(txt)
        rho = confidence_from_vector(v)
        total_loss += ttt_loss(W, v)
        W, grad_conf_history = hybrid_step(
            W, v, eta, alpha, grad_conf_history, rho, max_history
        )

    return {
        "final_weights": W,
        "average_loss": total_loss / len(texts) if texts else float("nan"),
        "history_length": len(grad_conf_history),
    }


# ----------------------------------------------------------------------
# Simple sanity‑check unit test (run when module is executed directly)
# ----------------------------------------------------------------------
def _self_test() -> None:
    sample_texts = [
        "The audit confirmed the evidence and the plan was executed.",
        "No source or checklist was provided.",
        "Verified record, but missing roadmap and schedule.",
        "Proof of concept and test steps are documented.",
    ]

    # Initialise a 2×2 weight matrix (matches 2‑dimensional feature space)
    W0 = init_ttt(d_in=2, scale=0.05, seed=42)

    # Hyper‑parameters
    learning_rate = 0.01
    fractional_alpha = 0.6

    result = evaluate_hybrid(
        W=W0,
        texts=sample_texts,
        eta=learning_rate,
        alpha=fractional_alpha,
        max_history=30,
    )

    print("Final weight matrix:\n", result["final_weights"])
    print("Average loss:", result["average_loss"])
    print("History length:", result["history_length"])


if __name__ == "__main__":
    _self_test()
    sys.exit(0)