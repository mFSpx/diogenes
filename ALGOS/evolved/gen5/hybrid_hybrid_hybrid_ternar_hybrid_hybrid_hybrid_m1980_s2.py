# DARWIN HAMMER — match 1980, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m170_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s5.py (gen3)
# born: 2026-05-29T23:40:20Z

"""Hybrid Ternary‑Morphology Analyzer

Parents:
- hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m170_s2.py (Algorithm A)
- hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s5.py (Algorithm B)

Mathematical bridge:
Algorithm A supplies a deterministic ternary vector **t**∈{-1,0,1}^D (D=12) derived
from a hashed command payload. Algorithm B supplies a set of continuous
morphology descriptors **m** = [sphericity, flatness, righting‑time] computed
from a physical object's dimensions.

The hybrid fuses them by interpreting the first *K* components of the ternary
vector (K=3) as a signed selector that sparsely modulates the morphology
features:

    **x** = **t**₁:ₖ ⊙ **m**          (⊙ denotes element‑wise product)

A ternary‑linear regression model with weight vector **w**∈ℝᴷ and bias *b*
maps the modulated features to a scalar decision score:

    ŷ = **w**·**x** + b

During training the weight update is driven by a *split‑gain* metric – the
reduction in mean‑squared error that would be obtained by applying the
current ternary mask. This couples the sparse selection logic of A with the
state‑space morphology evaluation of B into a single unified system.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants from Parent A
# ----------------------------------------------------------------------
TERNARY_DIMS = 12          # dimensionality of the full ternary vector
SELECT_DIM = 3            # number of ternary components used to mask morphology

# ----------------------------------------------------------------------
# Helper utilities (Parent A)
# ----------------------------------------------------------------------
def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def payload_hash(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> str:
    """Deterministic SHA‑256 of the command envelope."""
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ternary_vector(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> np.ndarray:
    """Generate a ternary vector based on the payload hash."""
    h = payload_hash(raw_command, normalized_intent, context)
    hi = int(h, 16)
    vec = np.empty(TERNARY_DIMS, dtype=int)
    for i in range(TERNARY_DIMS):
        vec[i] = (hi % 3) - 1          # maps {0,1,2} → {-1,0,1}
        hi //= 3
    return vec


# ----------------------------------------------------------------------
# Morphology utilities (Parent B)
# ----------------------------------------------------------------------
class Morphology:
    """Container for physical dimensions."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    """Sphericity = (volume)^(1/3) / longest edge."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    volume = length * width * height
    return volume ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length + width) / (2 * height)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """
    Simplified righting‑time model.
    τ = b * (mass * g * lever) / (k * width * height)
    """
    g = 9.80665  # gravity m/s²
    numerator = b * (m.mass * g * neck_lever)
    denominator = k * m.width * m.height
    if denominator == 0:
        raise ZeroDivisionError("Invalid geometry for righting time")
    return numerator / denominator


def morphology_features(m: Morphology) -> np.ndarray:
    """Return a length‑3 vector of the three descriptors."""
    s = sphericity_index(m.length, m.width, m.height)
    f = flatness_index(m.length, m.width, m.height)
    r = righting_time_index(m)
    return np.array([s, f, r], dtype=float)


# ----------------------------------------------------------------------
# Hybrid core (fusion of A & B)
# ----------------------------------------------------------------------
def masked_features(tern_vec: np.ndarray, morph_feat: np.ndarray) -> np.ndarray:
    """
    Apply the first SELECT_DIM components of the ternary vector as a signed
    mask to the morphology features.
    """
    mask = tern_vec[:SELECT_DIM].astype(float)   # values -1,0,1 → float
    return mask * morph_feat


def split_gain(y_true: np.ndarray, y_pred_before: np.ndarray,
               y_pred_after: np.ndarray) -> float:
    """
    Compute reduction in mean‑squared error caused by a candidate split.
    Positive gain indicates improvement.
    """
    mse_before = np.mean((y_true - y_pred_before) ** 2)
    mse_after = np.mean((y_true - y_pred_after) ** 2)
    return mse_before - mse_after


def train_step(tern_vec: np.ndarray,
               morph_feat: np.ndarray,
               target: float,
               w: np.ndarray,
               b: float,
               lr: float = 0.01) -> Tuple[np.ndarray, float]:
    """
    One gradient‑descent update using the split‑gain as a learning signal.
    Returns updated (w, b).
    """
    x = masked_features(tern_vec, morph_feat)          # shape (SELECT_DIM,)
    y_hat = np.dot(w, x) + b

    # Gradient of MSE loss L = (y_hat - target)^2
    grad_y = 2.0 * (y_hat - target)
    grad_w = grad_y * x
    grad_b = grad_y

    # Compute tentative update and its gain
    w_new = w - lr * grad_w
    b_new = b - lr * grad_b
    y_hat_new = np.dot(w_new, x) + b_new

    gain = split_gain(np.array([target]), np.array([y_hat]), np.array([y_hat_new]))
    # Accept update only if it yields positive gain
    if gain > 0:
        return w_new, b_new
    else:
        return w, b


def hybrid_predict(raw_command: str,
                   normalized_intent: str,
                   context: dict[str, Any],
                   morphology: Morphology,
                   w: np.ndarray,
                   b: float) -> float:
    """
    End‑to‑end prediction: hash → ternary mask → masked morphology → linear output.
    """
    t_vec = ternary_vector(raw_command, normalized_intent, context)
    m_feat = morphology_features(morphology)
    x = masked_features(t_vec, m_feat)
    return float(np.dot(w, x) + b)


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def demo_generation() -> None:
    """Show ternary vector creation and morphology extraction."""
    cmd = "rotate 90"
    intent = "rotate"
    ctx = {"user":"alice"}
    t = ternary_vector(cmd, intent, ctx)
    m = Morphology(0.12, 0.08, 0.04, 0.35)
    mf = morphology_features(m)
    print("Ternary vector (first 6):", t[:6])
    print("Morphology features:", mf)


def demo_training_loop(num_epochs: int = 20) -> None:
    """Simple synthetic training loop that fits the hybrid model to a target."""
    # Initialise random weights
    w = np.random.randn(SELECT_DIM)
    b = 0.0

    # Synthetic dataset: a handful of commands + morphologies + targets
    data = [
        ("push", "push", {}, Morphology(0.10, 0.07, 0.03, 0.30), 0.5),
        ("lift", "lift", {"mode":"slow"}, Morphology(0.15, 0.09, 0.05, 0.45), 0.8),
        ("rotate", "rotate", {"angle":90}, Morphology(0.12, 0.08, 0.04, 0.35), 0.6),
    ]

    for epoch in range(num_epochs):
        total_loss = 0.0
        for cmd, intent, ctx, morph, target in data:
            t = ternary_vector(cmd, intent, ctx)
            mf = morphology_features(morph)
            pred_before = np.dot(w, masked_features(t, mf)) + b
            w, b = train_step(t, mf, target, w, b, lr=0.05)
            pred_after = np.dot(w, masked_features(t, mf)) + b
            total_loss += (pred_after - target) ** 2
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} – MSE: {total_loss/len(data):.4f}")

    # Store trained parameters for later use
    np.savez("hybrid_params.npz", w=w, b=b)


def demo_inference() -> None:
    """Load trained parameters and run a prediction."""
    params = np.load("hybrid_params.npz")
    w = params["w"]
    b = float(params["b"])

    cmd = "rotate"
    intent = "rotate"
    ctx = {"angle": 90}
    morph = Morphology(0.12, 0.08, 0.04, 0.35)

    score = hybrid_predict(cmd, intent, ctx, morph, w, b)
    print(f"Hybrid prediction score for '{cmd}': {score:.4f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: Generation ===")
    demo_generation()
    print("\n=== Demo: Training ===")
    demo_training_loop(num_epochs=15)
    print("\n=== Demo: Inference ===")
    demo_inference()