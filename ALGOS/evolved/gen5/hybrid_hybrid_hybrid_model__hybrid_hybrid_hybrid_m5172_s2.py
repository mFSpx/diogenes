# DARWIN HAMMER — match 5172, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_geometric_pro_m4_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1086_s1.py (gen4)
# born: 2026-05-30T00:00:15Z

"""Hybrid VRAM‑Regret Rotor Model

Parents:
- hybrid_hybrid_model_vram_sc_hybrid_geometric_pro_m4_s3.py (VRAM‑aware quaternion rotor updates)
- hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1086_s1.py (regret‑weighted bandit / MinHash utilities)

Mathematical bridge:
The rotor `R` (unit quaternion) encodes a rotation `y = R * x * ~R`.  
Its rotation angle `θ = 2·acos(R[0])` is a scalar that can be interpreted as a
“regret magnitude”.  Conversely, the regret‑weighted hidden state computed from
`MathAction` objects yields a scalar `r` that modulates the rotor update size.
The hybrid step therefore couples the two systems by scaling the infinitesimal
rotor increment generated from the bivector `x ∧ (y−x)` with the regret scalar
`r`, while the VRAM scheduler decides whether the full learning rates
`(η_w, η_r)` or reduced ones are applied.

The module provides:
- Minimal VRAM budgeting utilities.
- Quaternion / rotor arithmetic.
- Regret computation from a list of actions.
- A hybrid update that fuses rotor learning with regret‑scaled step size.
- Sequence‑level processing that respects VRAM constraints.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# VRAM‑related helpers (simplified version of Parent A)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
DEFAULT_BUDGET_MB = int(os.environ.get("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.environ.get("LUCIDOTA_VRAM_RESERVE_MB", "768"))


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 with trailing “Z”."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def gpu_memory() -> int:
    """
    Dummy GPU memory query.
    Returns free memory in megabytes.
    In a real environment this would query `nvidia-smi` or similar.
    """
    # For testing we assume a generous amount of free memory.
    return 8192


def budgeted_lr(base_lr: float, free_mem: int) -> float:
    """
    Scale learning rate according to VRAM budget.
    If free memory is below the budget, halve the learning rate.
    """
    if free_mem < DEFAULT_BUDGET_MB:
        return base_lr * 0.5
    return base_lr


# ----------------------------------------------------------------------
# Quaternion / rotor utilities (Parent A)
# ----------------------------------------------------------------------
def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of a quaternion (inverse for unit quaternions)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def apply_rotor(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate 3‑D vector `v` by unit rotor `R` using the sandwich product.
    `v` is a length‑3 array; it is promoted to a pure quaternion [0, *v].
    """
    v_q = np.concatenate(([0.0], v))
    return quat_mul(R, quat_mul(v_q, quat_conj(R)))[1:]


def rotor_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Build a unit rotor (quaternion) from a rotation axis and angle.
    Axis must be a non‑zero 3‑D vector; angle is in radians.
    """
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / norm
    half = angle * 0.5
    return np.concatenate(([math.cos(half)], math.sin(half) * axis))


# ----------------------------------------------------------------------
# Regret / MinHash utilities (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """Simple MinHash‑like signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return []
    hashes = [_hash(i, t) for i, t in enumerate(toks)]
    return sorted(hashes)[:k]


def compute_regret(actions: List[MathAction]) -> float:
    """
    Regret is defined as the gap between the maximal expected value
    and the expectation under a probability distribution proportional
    to `exp(-cost)`.  This yields a scalar ≥ 0.
    """
    if not actions:
        return 0.0
    max_ev = max(a.expected_value for a in actions)
    # softmax over negative costs to favour low‑cost actions
    costs = np.array([a.cost for a in actions])
    probs = np.exp(-costs)
    probs /= probs.sum()
    expected = sum(p * a.expected_value for p, a in zip(probs, actions))
    return max_ev - expected


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_rotor_update(
    x: np.ndarray,
    target: np.ndarray,
    R: np.ndarray,
    actions: List[MathAction],
    base_lr_w: float = 0.01,
    base_lr_r: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one hybrid learning step.

    1. Rotate input `x` with current rotor `R` → `y`.
    2. Compute error `e = target - y`.
    3. Bivector `b = x × e` (generator of the infinitesimal rotation).
    4. Regret scalar `r` from the supplied actions.
    5. Scale rotor learning rate by regret: η_r' = η_r * (1 + r).
    6. Apply VRAM‑aware budgeting to both learning rates.
    7. Update rotor: R_new = ΔR * R   (ΔR built from `b` and η_r').

    Returns the updated rotor and the raw error vector.
    """
    # 1. Forward rotation
    y = apply_rotor(R, x)

    # 2. Error
    e = target - y

    # 3. Generator bivector (cross product)
    b = np.cross(x, e)

    # 4. Regret scalar
    r = compute_regret(actions)

    # 5. Regret‑scaled rotor learning rate
    eta_r = base_lr_r * (1.0 + r)

    # 6. VRAM‑aware budgeting
    free_mem = gpu_memory()
    eta_w = budgeted_lr(base_lr_w, free_mem)
    eta_r = budgeted_lr(eta_r, free_mem)

    # 7. Rotor increment (axis = b, angle = η_r * |b|)
    angle = eta_r * np.linalg.norm(b)
    delta_R = rotor_from_axis_angle(b, angle)

    # Quaternion multiplication updates the rotor
    R_new = quat_mul(delta_R, R)
    R_new /= np.linalg.norm(R_new)  # renormalise to stay on the unit sphere

    return R_new, e


def hybrid_sequence_process(
    data: List[Tuple[np.ndarray, np.ndarray, List[MathAction]]],
    init_rotor: np.ndarray | None = None,
) -> List[np.ndarray]:
    """
    Process a sequence of (input, target, actions) tuples.
    Returns the list of rotors after each step.
    """
    R = init_rotor if init_rotor is not None else np.array([1.0, 0.0, 0.0, 0.0])
    rotors = []
    for x, target, actions in data:
        R, _ = hybrid_rotor_update(x, target, R, actions)
        rotors.append(R.copy())
    return rotors


def hybrid_predict(
    x: np.ndarray,
    rotor: np.ndarray,
) -> np.ndarray:
    """
    Simple prediction interface: rotate `x` with the supplied rotor.
    """
    return apply_rotor(rotor, x)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset
    rng = np.random.default_rng(42)

    def random_vec() -> np.ndarray:
        return rng.normal(size=3)

    # Generate dummy actions
    actions_pool = [
        MathAction(id=f"a{i}", expected_value=rng.random(), cost=rng.random())
        for i in range(5)
    ]

    sequence = []
    for _ in range(10):
        x = random_vec()
        # Target is a rotated version of x with a known rotor
        true_axis = rng.normal(size=3)
        true_angle = rng.uniform(0, math.pi / 4)
        true_R = rotor_from_axis_angle(true_axis, true_angle)
        target = apply_rotor(true_R, x)

        # Randomly shuffle actions to vary regret
        rng.shuffle(actions_pool)
        seq_actions = actions_pool[:3]

        sequence.append((x, target, seq_actions))

    # Run the hybrid processor
    final_rotors = hybrid_sequence_process(sequence)

    # Print final rotor and a sample prediction
    print("Final rotor (quaternion):", final_rotors[-1])
    sample_x = random_vec()
    pred = hybrid_predict(sample_x, final_rotors[-1])
    print("Sample input :", sample_x)
    print("Sample output:", pred)
    print("Timestamp:", now_z())