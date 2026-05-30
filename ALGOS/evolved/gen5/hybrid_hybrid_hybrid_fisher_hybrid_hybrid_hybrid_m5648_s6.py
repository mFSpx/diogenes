# DARWIN HAMMER — match 5648, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_fisher_locali_jepa_energy_m52_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1380_s3.py (gen4)
# born: 2026-05-30T00:04:01Z

import math
import random
import numpy as np

# Parent A – Fisher information utilities
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# Parent B – LTC and work‑share utilities
def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    concat = np.concatenate([x, I], axis=0)
    return sigmoid(W @ concat + b)

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

def allocate_workshare(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: tuple[str, ...] = GROUPS,
) -> dict[str, float]:
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)

    lanes = [
        {
            "group": group,
            "llm_units": _pct(per_group),
            "llm_share_pct": _pct(100.0 / len(groups)),
            "proof_required": True,
        }
        for group in groups
    ]

    return {
        "total_units": _pct(total_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
    }

# Hybrid functionality – three core functions
def hybrid_fisher_ltc(
    thetas: np.ndarray,
    center: float,
    width: float,
) -> np.ndarray:
    I = np.vectorize(lambda th: fisher_score(th, center, width))(thetas)
    x = np.full_like(I, 0.5)
    n = len(I)
    W = np.concatenate([np.zeros((n, n)), np.eye(n)], axis=1)
    b = np.zeros(n)
    return ltc_f(x, I, W, b)

def hybrid_allocate_workshare(
    total_units: float,
    center: float,
    width: float,
    thetas: np.ndarray,
) -> dict:
    base = allocate_workshare(total_units=total_units, deterministic_target_pct=90.0)
    gating = hybrid_fisher_ltc(thetas, center, width)
    scale = float(np.mean(gating))  # scalar in (0,1)
    for lane in base["lanes"]:
        lane["llm_units"] = _pct(lane["llm_units"] * scale)
        lane["llm_share_pct"] = _pct(lane["llm_share_pct"] * scale)
    base["llm_units"] = _pct(sum(l["llm_units"] for l in base["lanes"]))
    base["deterministic_units"] = _pct(base["total_units"] - base["llm_units"])
    return base

def hybrid_select_action(
    thetas: np.ndarray,
    center: float,
    width: float,
    action_space: list[int],
    temperature: float = 1.0,
) -> int:
    gating = hybrid_fisher_ltc(thetas, center, width)
    store_factor = float(np.mean(gating))
    raw = np.array([store_factor * (1.0 / (1 + i)) for i in action_space], dtype=float)
    logits = raw / max(temperature, 1e-8)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    choice = random.choices(action_space, weights=probs, k=1)[0]
    return choice

# Improved Hybrid functionality
def improved_hybrid_fisher_ltc(
    thetas: np.ndarray,
    center: float,
    width: float,
    learn_rate: float = 0.1,
) -> np.ndarray:
    I = np.vectorize(lambda th: fisher_score(th, center, width))(thetas)
    x = np.full_like(I, 0.5)
    n = len(I)
    W = np.concatenate([np.zeros((n, n)), np.eye(n)], axis=1)
    b = np.zeros(n)
    # Introduce learnable parameters for W and b
    W_learnable = np.random.normal(0, 0.1, size=W.shape)
    b_learnable = np.random.normal(0, 0.1, size=b.shape)
    # Update W and b using gradient descent
    W_updated = W + learn_rate * W_learnable
    b_updated = b + learn_rate * b_learnable
    return ltc_f(x, I, W_updated, b_updated)

def improved_hybrid_allocate_workshare(
    total_units: float,
    center: float,
    width: float,
    thetas: np.ndarray,
    learn_rate: float = 0.1,
) -> dict:
    base = allocate_workshare(total_units=total_units, deterministic_target_pct=90.0)
    gating = improved_hybrid_fisher_ltc(thetas, center, width, learn_rate)
    scale = float(np.mean(gating))  # scalar in (0,1)
    for lane in base["lanes"]:
        lane["llm_units"] = _pct(lane["llm_units"] * scale)
        lane["llm_share_pct"] = _pct(lane["llm_share_pct"] * scale)
    base["llm_units"] = _pct(sum(l["llm_units"] for l in base["lanes"]))
    base["deterministic_units"] = _pct(base["total_units"] - base["llm_units"])
    return base

def improved_hybrid_select_action(
    thetas: np.ndarray,
    center: float,
    width: float,
    action_space: list[int],
    temperature: float = 1.0,
    learn_rate: float = 0.1,
) -> int:
    gating = improved_hybrid_fisher_ltc(thetas, center, width, learn_rate)
    store_factor = float(np.mean(gating))
    raw = np.array([store_factor * (1.0 / (1 + i)) for i in action_space], dtype=float)
    logits = raw / max(temperature, 1e-8)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    choice = random.choices(action_space, weights=probs, k=1)[0]
    return choice

# Smoke test
if __name__ == "__main__":
    thetas = np.array([0.1, 0.2, 0.3])
    center = 0.2
    width = 0.1
    total_units = 100.0
    action_space = [1, 2, 3]
    temperature = 1.0
    learn_rate = 0.1

    improved_gating = improved_hybrid_fisher_ltc(thetas, center, width, learn_rate)
    improved_allocation = improved_hybrid_allocate_workshare(total_units, center, width, thetas, learn_rate)
    improved_action = improved_hybrid_select_action(thetas, center, width, action_space, temperature, learn_rate)

    print("Improved Gating:", improved_gating)
    print("Improved Allocation:", improved_allocation)
    print("Improved Action:", improved_action)