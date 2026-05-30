# DARWIN HAMMER — match 2751, survivor 6
# gen: 6
# parent_a: hybrid_honeybee_store_hybrid_hybrid_hybrid_m836_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m675_s0.py (gen5)
# born: 2026-05-29T23:45:37Z

"""HybridStoreGeometric Algorithm
Integrates:
- Parent A: honeybee_store dynamics with SSIM‑based weighting.
- Parent B: geometric‑product guided Test‑Time Training (TTT) with stylometry features.

Mathematical bridge:
1. Text is transformed into a feature vector (Parent A).
2. SSIM between this vector and a reference vector yields a similarity gain g∈[0,1].
3. The store delta Δs from the honeybee dynamics (Parent A) scales the regularization of the weight matrix W.
4. The TTT gradient update of W (Parent B) is multiplied by the SSIM gain g, fusing the two topologies into a single feedback loop.
"""

import math
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# -------------------- Constants (Parent A) --------------------
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
K1 = 0.01            # regularization weight for store delta
K2 = 0.03            # SSIM constants
L = 255.0            # dynamic range for SSIM

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)

# -------------------- Parent A Core Functions --------------------
def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = ALPHA,
    beta: float = BETA,
    dt: float = DT,
) -> Tuple[float, float]:
    """
    Honeybee store dynamics.
    Returns (new_store, delta) where delta = α·Σinflow − β·Σoutflow.
    """
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta


def _ssim_index(x: np.ndarray, y: np.ndarray) -> float:
    """
    Simplified 1‑D SSIM between vectors x and y.
    """
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator) if denominator != 0 else 0.0


def extract_feature_vector(text: str) -> np.ndarray:
    """
    Produce a 9‑dim feature vector from the input text.
    Positive categories add weight, negative categories subtract weight.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = Counter(tokens)

    # simple heuristic: presence of evidence tokens boosts the first dimension
    evidence_score = 1 if EVIDENCE_RE.search(text) else 0

    pos_counts = np.array([counts.get(feat, 0) for feat in _FEATURE_ORDER], dtype=np.float64)
    vec = pos_counts * _POSITIVE_WEIGHTS + evidence_score * _POSITIVE_WEIGHTS[0]

    neg_counts = np.array([counts.get(feat, 0) for feat in _FEATURE_ORDER], dtype=np.float64)
    vec -= neg_counts * _NEGATIVE_WEIGHTS

    return vec


def ssim_gain(feature_vec: np.ndarray, reference_vec: np.ndarray) -> float:
    """
    Normalised SSIM gain in [0, 1] used to weight gradient updates.
    """
    return max(0.0, min(1.0, _ssim_index(feature_vec, reference_vec)))


# -------------------- Parent B Core Functions --------------------
def init_weight_matrix(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """
    Initialise the TTT weight matrix W ∈ ℝ^{d_out×d_in}.
    """
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    """
    Quadratic loss ‖W·x − t‖² where t = target if supplied else x.
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """
    Gradient of the quadratic loss w.r.t. W: 2·(W·x − t)·xᵀ.
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)


def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                if j < n - 1:
                    lst.pop(j)
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def geometric_product(a: dict, b: dict) -> dict:
    """
    Geometric product of two multivectors a and b.
    Each multivector is a dict mapping frozenset of basis indices → coefficient.
    """
    result = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            result[blade_out] = result.get(blade_out, 0.0) + sign * coef_a * coef_b
    return result


# -------------------- Hybrid Operations --------------------
def hybrid_store_update(
    store: float,
    inflow: List[float],
    outflow: List[float],
) -> Tuple[float, float]:
    """
    Wrapper around Parent A's store update.
    """
    return update_store(store, inflow, outflow)


def hybrid_gradient_step(
    W: np.ndarray,
    x: np.ndarray,
    store_delta: float,
    ssim_gain_val: float,
    lr: float = 0.01,
) -> np.ndarray:
    """
    Perform a TTT gradient descent step on W.
    The gradient is scaled by the SSIM gain and regularised by the store delta.
    """
    grad = ttt_grad(W, x)                     # raw gradient
    grad_scaled = ssim_gain_val * grad        # SSIM weighting

    # regularisation term proportional to Δstore (acts like weight decay towards identity)
    reg = K1 * store_delta * np.eye(W.shape[0], W.shape[1])
    W_new = W - lr * (grad_scaled + reg)
    return W_new


def hybrid_step(
    store: float,
    inflow: List[float],
    outflow: List[float],
    W: np.ndarray,
    text: str,
    reference_vec: np.ndarray,
    lr: float = 0.01,
) -> Tuple[float, np.ndarray, float]:
    """
    Complete hybrid iteration:
    1. Update store → (store', Δs).
    2. Extract feature vector from text.
    3. Compute SSIM gain g with respect to reference_vec.
    4. Update weight matrix W using g and Δs.
    Returns (new_store, new_W, g).
    """
    new_store, delta_s = hybrid_store_update(store, inflow, outflow)

    feat_vec = extract_feature_vector(text)

    g = ssim_gain(feat_vec, reference_vec)

    # Use the feature vector as the TTT input x (projected to appropriate size)
    # If dimensions mismatch, truncate or pad with zeros.
    d_in = W.shape[1]
    if feat_vec.size < d_in:
        x = np.pad(feat_vec, (0, d_in - feat_vec.size))
    else:
        x = feat_vec[:d_in]

    new_W = hybrid_gradient_step(W, x, delta_s, g, lr=lr)
    return new_store, new_W, g


# -------------------- Smoke Test --------------------
if __name__ == "__main__":
    # Initialise store and weight matrix
    store_val = 0.0
    inflow_vals = [random.uniform(0, 2) for _ in range(3)]
    outflow_vals = [random.uniform(0, 1) for _ in range(2)]

    dim = 9  # same as feature vector length
    W = init_weight_matrix(d_in=dim, d_out=dim, seed=42)

    # Example texts
    txt = "The evidence was verified and the planning was solid, but there was a delay."
    ref = np.zeros(dim)  # neutral reference

    # Run a single hybrid iteration
    new_store, new_W, gain = hybrid_step(
        store=store_val,
        inflow=inflow_vals,
        outflow=outflow_vals,
        W=W,
        text=txt,
        reference_vec=ref,
        lr=0.005,
    )

    print(f"Store updated: {store_val:.4f} → {new_store:.4f}")
    print(f"SSIM gain: {gain:.4f}")
    print(f"Weight matrix norm before: {np.linalg.norm(W):.4f}")
    print(f"Weight matrix norm after : {np.linalg.norm(new_W):.4f}")