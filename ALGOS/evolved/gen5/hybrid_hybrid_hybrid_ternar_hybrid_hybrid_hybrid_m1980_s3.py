# DARWIN HAMMER — match 1980, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m170_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s5.py (gen3)
# born: 2026-05-29T23:40:20Z

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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def payload_hash(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> str:
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ternary_vector(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> np.ndarray:
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
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    volume = length * width * height
    return volume ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    g = 9.80665  
    numerator = b * (m.mass * g * neck_lever)
    denominator = k * m.width * m.height
    if denominator == 0:
        raise ZeroDivisionError("Invalid geometry for righting time")
    return numerator / denominator


def morphology_features(m: Morphology) -> np.ndarray:
    s = sphericity_index(m.length, m.width, m.height)
    f = flatness_index(m.length, m.width, m.height)
    r = righting_time_index(m)
    return np.array([s, f, r], dtype=float)


# ----------------------------------------------------------------------
# Hybrid core (fusion of A & B)
# ----------------------------------------------------------------------
def masked_features(tern_vec: np.ndarray, morph_feat: np.ndarray) -> np.ndarray:
    mask = tern_vec[:SELECT_DIM].astype(float)   
    return mask * morph_feat


def split_gain(y_true: np.ndarray, y_pred_before: np.ndarray,
               y_pred_after: np.ndarray) -> float:
    mse_before = np.mean((y_true - y_pred_before) ** 2)
    mse_after = np.mean((y_true - y_pred_after) ** 2)
    return mse_before - mse_after


def train_step(tern_vec: np.ndarray,
               morph_feat: np.ndarray,
               target: float,
               w: np.ndarray,
               b: float,
               lr: float = 0.01) -> Tuple[np.ndarray, float]:
    x = masked_features(tern_vec, morph_feat)          
    y_hat = np.dot(w, x) + b

    grad_y = 2.0 * (y_hat - target)
    grad_w = grad_y * x
    grad_b = grad_y

    w_new = w - lr * grad_w
    b_new = b - lr * grad_b
    y_hat_new = np.dot(w_new, x) + b_new

    gain = split_gain(np.array([target]), np.array([y_hat]), np.array([y_hat_new]))
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
    t_vec = ternary_vector(raw_command, normalized_intent, context)
    m_feat = morphology_features(morphology)
    x = masked_features(t_vec, m_feat)
    return float(np.dot(w, x) + b)


def improved_train_step(tern_vec: np.ndarray,
               morph_feat: np.ndarray,
               target: float,
               w: np.ndarray,
               b: float,
               lr: float = 0.01) -> Tuple[np.ndarray, float]:
    x = masked_features(tern_vec, morph_feat)          
    y_hat = np.dot(w, x) + b

    grad_y = 2.0 * (y_hat - target)
    grad_w = grad_y * x
    grad_b = grad_y

    w_new = w - lr * grad_w
    b_new = b - lr * grad_b
    y_hat_new = np.dot(w_new, x) + b_new

    gain = split_gain(np.array([target]), np.array([y_hat]), np.array([y_hat_new]))
    if gain > 0:
        return w_new, b_new
    else:
        # Introducing a minimum gain threshold to avoid overfitting
        min_gain_threshold = 0.001
        if gain > -min_gain_threshold:
            return w_new, b_new
        else:
            return w, b


def improved_hybrid_predict(raw_command: str,
                   normalized_intent: str,
                   context: dict[str, Any],
                   morphology: Morphology,
                   w: np.ndarray,
                   b: float) -> float:
    t_vec = ternary_vector(raw_command, normalized_intent, context)
    m_feat = morphology_features(morphology)
    x = masked_features(t_vec, m_feat)
    
    # Introducing an additional layer of processing to improve the model's robustness
    x = np.clip(x, -1.0, 1.0)
    return float(np.dot(w, x) + b)


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def demo_generation() -> None:
    cmd = "rotate 90"
    intent = "rotate"
    ctx = {"user":"alice"}
    t = ternary_vector(cmd, intent, ctx)
    m = Morphology(0.12, 0.08, 0.04, 0.35)
    mf = morphology_features(m)
    print("Ternary vector (first 6):", t[:6])
    print("Morphology features:", mf)


def demo_training_loop(num_epochs: int = 20) -> None:
    w = np.random.randn(SELECT_DIM)
    b = 0.0

    data = [
        ("push", "push", {}, Morphology(0.10, 0.07, 0.03, 0.30), 0.5),
        ("pull", "pull", {}, Morphology(0.15, 0.10, 0.05, 0.40), 0.8),
        ("rotate", "rotate", {}, Morphology(0.12, 0.08, 0.04, 0.35), 0.6),
    ]

    for epoch in range(num_epochs):
        for cmd, intent, ctx, morph, target in data:
            t_vec = ternary_vector(cmd, intent, ctx)
            m_feat = morphology_features(morph)
            w, b = improved_train_step(t_vec, m_feat, target, w, b)
        print(f"Epoch {epoch+1}, Weights: {w}, Bias: {b}")


def demo_prediction() -> None:
    cmd = "rotate 90"
    intent = "rotate"
    ctx = {"user":"alice"}
    m = Morphology(0.12, 0.08, 0.04, 0.35)
    w = np.random.randn(SELECT_DIM)
    b = 0.0
    prediction = improved_hybrid_predict(cmd, intent, ctx, m, w, b)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    demo_generation()
    demo_training_loop()
    demo_prediction()