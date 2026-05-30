# DARWIN HAMMER — match 1594, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s1.py (gen3)
# parent_b: hybrid_fractional_hdc_counterfactual_effec_m38_s1.py (gen1)
# born: 2026-05-29T23:37:47Z

"""Hybrid Allocation & Causal Hyperdimensional Engine

This module fuses two distinct parents:

* **Parent A** – a weekday‑based sinusoidal weight vector allocator combined with a
  VRAM‑aware GPU selector.
* **Parent B** – a Hyperdimensional Computing (HDC) core that provides random
  hypervectors, binding, fractional‑power binding and causal‑effect estimation.

**Mathematical bridge** – Both parents manipulate *vectors* that are later
combined with *matrix‑like* operations.  The bridge is the **element‑wise
multiplication (binding)** of complex hypervectors with a **scalar weight vector**
derived from the weekday schedule.  The scalar weights scale the group hypervectors,
producing a weighted group hypervector.  Each selected GPU is encoded as a hypervector
whose amplitude is raised to a fractional power proportional to its free memory.
Binding the GPU hypervectors with the weighted group hypervector and finally bundling
(all‑add) yields a single composite hypervector that simultaneously represents
resource allocation (from Parent A) and causal strength (from Parent B).

The functions below demonstrate this hybrid operation:
1. ``weekday_weight_vector`` – sinusoidal, row‑stochastic weight vector.
2. ``hybrid_allocation_plan`` – builds the composite hypervector from groups,
   date, and VRAM‑aware GPUs.
3. ``estimate_plan_effect`` – treats the composite hypervector as a causal
   representation and returns a similarity‑based effect estimate.

All code is pure Python 3, requiring only the standard library and ``numpy``.
"""

import datetime as dt
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (Parent A)
# ---------------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768

# ---------------------------------------------------------------------------
# Helper utilities (Parent A)
# ---------------------------------------------------------------------------


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where 0 = Sunday … 6 = Saturday."""
    # Python's weekday(): Monday=0 … Sunday=6 → shift to Sunday=0
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def vram_aware_gpu_selection(
    gpus: List[Dict[str, Any]], budget_mb: int, reserve_mb: int
) -> List[Dict[str, Any]]:
    """
    Select GPUs that have sufficient VRAM to meet the budget and reserve requirements.
    """
    selected = []
    for gpu in gpus:
        if gpu.get("memory.free", 0) >= budget_mb + reserve_mb:
            selected.append(gpu)
    return selected


# ---------------------------------------------------------------------------
# Hyperdimensional primitives (Parent B)
# ---------------------------------------------------------------------------


def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """
    Generate a random hypervector of dimension ``d``.

    Supported kinds:
        * "complex" – unit‑magnitude complex vector (phase uniformly sampled).
        * "bipolar" – real vector with components in {+1, -1}.
        * "real"    – Gaussian vector normalised to unit L2 norm.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice([-1.0, 1.0], size=d).astype(np.float64)
    # real Gaussian
    vec = rng.normal(size=d)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Binding operator – element‑wise multiplication (complex or real)."""
    return a * b


def unbind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Inverse binding – element‑wise division (avoids division by zero)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, 0.0)
    return result


def fractional_power(hv: np.ndarray, exponent: float) -> np.ndarray:
    """
    Raise each component of a complex hypervector to ``exponent``.
    For real vectors the operation reduces to ``hv ** exponent``.
    """
    if np.iscomplexobj(hv):
        # Use polar representation: (r e^{iθ})^{p} = r^{p} e^{i p θ}
        magnitude = np.abs(hv) ** exponent
        angle = np.angle(hv) * exponent
        return magnitude * np.exp(1j * angle)
    return np.power(hv, exponent)


def bundle(hvs: List[np.ndarray]) -> np.ndarray:
    """
    Superposition (addition) of hypervectors followed by L2 normalisation.
    """
    if not hvs:
        raise ValueError("bundle requires at least one hypervector")
    summed = np.sum(hvs, axis=0)
    norm = np.linalg.norm(summed)
    return summed / norm if norm != 0 else summed


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity (absolute value) between two hypervectors.
    """
    dot = np.vdot(a, b)  # conjugate dot product for complex vectors
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(dot) / norm) if norm != 0 else 0.0


def cleanup(hv: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Zero out components with magnitude below ``threshold``."""
    mask = np.abs(hv) >= threshold
    return hv * mask


def encode_sequence(seq: Sequence[int], d: int = 10000, seed: int | None = None) -> np.ndarray:
    """
    Encode a sequence of integers into a hypervector by binding atomic symbols.
    Each integer is mapped to a random hypervector (seeded for reproducibility).
    """
    symbols = [random_hv(d, seed=seed + i if seed is not None else None) for i in seq]
    return bundle(symbols)


def fractional_blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend two hypervectors with a fractional exponent.
    Result ≈ a^{1‑α} ⊙ b^{α}
    """
    a_fp = fractional_power(a, 1.0 - alpha)
    b_fp = fractional_power(b, alpha)
    return bind(a_fp, b_fp)


# ---------------------------------------------------------------------------
# Hybrid structures
# ---------------------------------------------------------------------------


def generate_group_hvs(groups: Sequence[str], d: int = 10000, seed: int = 0) -> Dict[str, np.ndarray]:
    """
    Allocate a random hypervector for each group name.
    The seed is offset by the group's index to keep vectors deterministic.
    """
    hvs: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(groups):
        hvs[name] = random_hv(d, kind="complex", seed=seed + idx)
    return hvs


def encode_gpu_hv(gpu: Dict[str, Any], d: int = 10000, base_exponent: float = 1.0) -> np.ndarray:
    """
    Encode a GPU's free memory into a hypervector.
    The free memory (in MB) determines the fractional power exponent.
    """
    free_mb = float(gpu.get("memory.free", 0))
    # Normalise exponent to [0, 1] using a simple logistic scaling.
    exponent = 1.0 / (1.0 + math.exp(- (free_mb - 2048) / 512.0))
    exponent = max(0.0, min(1.0, exponent))  # clamp
    base_hv = random_hv(d, kind="complex", seed=hash(gpu.get("id", "gpu")) & 0xFFFFFFFF)
    return fractional_power(base_hv, exponent * base_exponent)


def hybrid_allocation_plan(
    *,
    groups: Tuple[str, ...] = GROUPS,
    date: dt.date,
    gpus: List[Dict[str, Any]],
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
    dim: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Build a composite hypervector representing the allocation plan.

    Steps
    -----
    1. Compute the weekday weight vector (Parent A).
    2. Generate a deterministic hypervector for each group (Parent B).
    3. Form a *weighted group hypervector* by scaling each group hv with its weight
       and bundling the result.
    4. Select VRAM‑compatible GPUs (Parent A).
    5. Encode each selected GPU as a hypervector whose fractional power reflects
       its free memory (Parent B).
    6. Bind each GPU hv with the weighted group hv and bundle all bindings into the
       final plan hypervector.

    Returns a dictionary containing:
        * ``plan_hv`` – the composite hypervector (complex, unit‑norm).
        * ``selected_gpus`` – list of GPU dicts that satisfied the VRAM budget.
        * ``weight_vector`` – the weekday weight vector (for inspection).
    """
    # 1. Weekday weight vector
    dow = (date.weekday() + 1) % 7  # align with doomsday convention
    weight_vec = weekday_weight_vector(groups, dow)

    # 2. Group hypervectors
    group_hvs = generate_group_hvs(groups, d=dim, seed=seed)

    # 3. Weighted group hypervector
    weighted_hvs = [
        weight_vec[i] * group_hvs[name] for i, name in enumerate(groups)
    ]
    weighted_group_hv = bundle(weighted_hvs)

    # 4. VRAM‑aware GPU selection
    selected = vram_aware_gpu_selection(gpus, budget_mb, reserve_mb)

    # 5. Encode GPUs
    gpu_hvs = [encode_gpu_hv(gpu, d=dim) for gpu in selected]

    # 6. Bind + bundle
    bound_hvs = [bind(gpu_hv, weighted_group_hv) for gpu_hv in gpu_hvs]
    if bound_hvs:
        plan_hv = bundle(bound_hvs)
    else:
        # No GPU satisfies the budget → fall back to the weighted group alone
        plan_hv = weighted_group_hv

    # Clean up tiny components for numerical stability
    plan_hv = cleanup(plan_hv)

    return {
        "plan_hv": plan_hv,
        "selected_gpus": selected,
        "weight_vector": weight_vec,
    }


@dataclass
class CausalEffect:
    """Simple container for a similarity‑based causal effect estimate."""
    similarity: float
    description: str = ""


def estimate_plan_effect(plan_hv: np.ndarray, target_hv: np.ndarray) -> CausalEffect:
    """
    Treat the composite plan hypervector as a causal representation and compare it
    against a *target* hypervector (e.g., an ideal allocation).  The absolute cosine
    similarity is interpreted as the magnitude of the causal effect.
    """
    sim = similarity(plan_hv, target_hv)
    desc = f"Effect similarity = {sim:.4f}"
    return CausalEffect(similarity=sim, description=desc)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Dummy group list (uses the default GROUPS)
    groups = GROUPS

    # Example date
    today = dt.date.today()

    # Mock GPU inventory
    mock_gpus = [
        {"id": "gpu0", "memory.free": 8192},
        {"id": "gpu1", "memory.free": 4096},
        {"id": "gpu2", "memory.free": 2048},
        {"id": "gpu3", "memory.free": 1024},
    ]

    # Build the hybrid allocation plan
    result = hybrid_allocation_plan(
        groups=groups,
        date=today,
        gpus=mock_gpus,
        budget_mb=DEFAULT_BUDGET_MB,
        reserve_mb=DEFAULT_RESERVE_MB,
        dim=8000,  # smaller dimension for faster test
        seed=123,
    )

    plan_hv = result["plan_hv"]
    selected = result["selected_gpus"]
    weight_vec = result["weight_vector"]

    print("Selected GPUs:", [g["id"] for g in selected])
    print("Weekday weight vector:", weight_vec)

    # Create a naive target hypervector (e.g., all‑ones bundle)
    target_hv = bundle([random_hv(8000, kind="complex", seed=999) for _ in range(3)])

    effect = estimate_plan_effect(plan_hv, target_hv)
    print(effect.description)

    # Verify that the plan hypervector is unit‑norm (within tolerance)
    norm = np.linalg.norm(plan_hv)
    print(f"Plan hypervector L2 norm = {norm:.6f}")

    sys.exit(0)