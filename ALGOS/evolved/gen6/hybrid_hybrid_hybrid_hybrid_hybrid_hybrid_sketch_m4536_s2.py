# DARWIN HAMMER — match 4536, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m503_s3.py (gen5)
# parent_b: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hdc_hy_m561_s1.py (gen5)
# born: 2026-05-29T23:56:24Z

"""Hybrid Algorithm Fusion: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m503_s3 +
hybrid_hybrid_sketches_hybr_hybrid_hybrid_hdc_hy_m561_s1

This module mathematically bridges the two parent algorithms by using the **Count‑Min Sketch**
from the sketch‑based bandit (Parent B) as a probabilistic frequency estimator that
weights the **Fisher information** derived from the Gaussian beam model of the
temperature‑dependent developmental rate (Parent A). The sketch frequencies are
flattened into a high‑dimensional vector, bound (element‑wise product) with a
symbolic random‑binary vector, and finally injected as additional dimensions into
an **RBF surrogate** model. In this way the stochastic count information modulates
the continuous Fisher‑score surface, yielding a unified hybrid estimator.

The core mathematical bridge can be expressed as:


w_i   = sketch_freq(i)                # Count‑Min sketch frequency for action i
I_i   = FisherScore(θ_i, μ, σ)        # Fisher information of Gaussian beam
z_i   = w_i * I_i                      # weight‑modulated Fisher score
v    = bind(symbol_vector(a), flatten(sketch))   # high‑dimensional binding
ŷ    = Σ_k exp(-||v - c_k||² / (2·γ²))          # RBF surrogate evaluation


The functions below implement this pipeline and expose three distinct hybrid
operations.

"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from datetime import date

# ----------------------------------------------------------------------
# Shared data structures (identical to parents, kept for compatibility)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# ----------------------------------------------------------------------
# Parent A – temperature / Fisher / weekday utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield model of temperature dependent developmental rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) *
        ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp(
        (params.delta_h_low / params.r_cal) *
        ((1.0 / params.t_low) - (1.0 / temp_k))
    )
    high = math.exp(
        (params.delta_h_high / params.r_cal) *
        ((1.0 / params.t_high) - (1.0 / temp_k))
    )
    return numerator / (1.0 + low + high)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float,
                 center: float,
                 width: float,
                 eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def weekday_weight_vector(dow: int, groups: int) -> np.ndarray:
    """Generate a normalized weight vector that varies with day‑of‑week."""
    base_angles = np.arange(groups) * (2.0 * math.pi) / groups
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec

def doomsday(year: int, month: int, day: int) -> int:
    """Return the day of week (0=Monday … 6=Sunday) for a given date."""
    return (date(year, month, day).weekday() + 1) % 7

# ----------------------------------------------------------------------
# Parent B – sketch, symbolic binding, RBF surrogate
# ----------------------------------------------------------------------
_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: list[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def count_min_sketch(items: list[str],
                     width: int = 64,
                     depth: int = 4) -> list[list[int]]:
    """Probabilistic frequency table using pairwise‑independent hashing."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f'{d}:{item}'.encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table

def symbol_vector(symbol: str, dim: int = 1024) -> list[int]:
    """Deterministic random binary vector (+1 / -1) derived from a symbol."""
    seed = int.from_bytes(hashlib.sha256(symbol.encode('utf-8')).digest()[:8], 'big')
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def bind(a: list[int], b: list[int]) -> list[int]:
    """Element‑wise binding (Hadamard product) of two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError('vectors must have equal length')
    return [x * y for x, y in zip(a, b)]

def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

@dataclass
class RBFSurrogate:
    centers: list[np.ndarray]          # list of centre vectors (high‑dim)
    gamma: float = 1.0                  # bandwidth parameter

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the surrogate output as a sum of RBFs centered at stored points."""
        if not self.centers:
            return 0.0
        diffs = [np.linalg.norm(x - c) for c in self.centers]
        return sum(gaussian_rbf(d, self.gamma) for d in diffs)

    def add_center(self, x: np.ndarray) -> None:
        self.centers.append(x)

# ----------------------------------------------------------------------
# Hybrid Functions (mathematical bridge)
# ----------------------------------------------------------------------
def hybrid_fisher_sketch(actions: list[BanditAction],
                         temperature_c: float,
                         sketch_width: int = 64,
                         sketch_depth: int = 4) -> dict[str, float]:
    """
    Compute a weight‑modulated Fisher score for each action.

    1. Build a Count‑Min sketch of the action identifiers.
    2. Flatten the sketch to obtain a frequency vector `w`.
    3. For each action i:
       - Obtain its sketch frequency estimate `w_i = min_d table[d][hash_i]`.
       - Compute Fisher information `I_i` using the Gaussian beam centred at the
         action's propensity with a width derived from the developmental rate
         at the supplied temperature.
       - Return the product `z_i = w_i * I_i`.
    """
    # 1. sketch
    ids = [a.action_id for a in actions]
    sketch = count_min_sketch(ids, width=sketch_width, depth=sketch_depth)

    # helper to get min count across rows (standard Count‑Min estimate)
    def sketch_estimate(item: str) -> int:
        estimates = []
        for d in range(sketch_depth):
            h = hashlib.sha256(f'{d}:{item}'.encode()).hexdigest()
            idx = int(h, 16) % sketch_width
            estimates.append(sketch[d][idx])
        return min(estimates)

    # 2. temperature‑dependent width for Gaussian beam
    temp_k = c_to_k(temperature_c)
    rate = developmental_rate(temp_k)
    # map rate (roughly 0‑1) to a sensible width; avoid zero
    width = max(0.01, 0.1 + rate)

    result: dict[str, float] = {}
    for a in actions:
        w_i = sketch_estimate(a.action_id)
        I_i = fisher_score(theta=a.propensity,
                           center=0.5,          # arbitrary centre for illustration
                           width=width)
        result[a.action_id] = w_i * I_i
    return result

def hybrid_rbf_binding(action: BanditAction,
                       sketch: list[list[int]],
                       surrogate: RBFSurrogate) -> float:
    """
    Bind a symbolic vector of the action with the flattened sketch,
    feed the resulting high‑dim vector to the RBF surrogate, and return the surrogate value.
    """
    # flatten sketch to a 1‑D numpy array
    flat = np.array(sketch, dtype=np.int32).flatten()
    # generate deterministic binary vector for the action
    sym = np.array(symbol_vector(action.action_id, dim=flat.shape[0]), dtype=np.int32)
    # binding via element‑wise product
    bound = np.multiply(sym, flat).astype(np.float64)
    # evaluate surrogate
    return surrogate.evaluate(bound)

def weekday_weighted_fisher(actions: list[BanditAction],
                            date_tuple: tuple[int, int, int],
                            groups: int = 7) -> dict[str, float]:
    """
    Combine the weekday weight vector (Parent A) with Fisher scores.
    The weight for each action is the dot product between the weekday vector
    and a one‑hot encoding of the action index modulo `groups`.
    """
    year, month, day = date_tuple
    dow = doomsday(year, month, day)               # 0‑6
    w_vec = weekday_weight_vector(dow, groups)    # length = groups

    result: dict[str, float] = {}
    for idx, a in enumerate(actions):
        group = idx % groups
        weight = w_vec[group]
        # Fisher score with a fixed centre/width for simplicity
        I = fisher_score(theta=a.propensity, center=0.5, width=0.2)
        result[a.action_id] = weight * I
    return result

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # create a tiny set of dummy actions
    actions = [
        BanditAction(action_id="A1", propensity=0.3, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="hybrid"),
        BanditAction(action_id="B2", propensity=0.6, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="hybrid"),
        BanditAction(action_id="C3", propensity=0.9, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="hybrid")
    ]

    # 1. Hybrid Fisher‑Sketch
    hybrid_scores = hybrid_fisher_sketch(actions, temperature_c=25.0)
    print("Hybrid Fisher‑Sketch scores:", hybrid_scores)

    # 2. Build a sketch and an RBF surrogate, then bind+evaluate
    sketch = count_min_sketch([a.action_id for a in actions])
    surrogate = RBFSurrogate(centers=[])
    # add a random centre for demonstration
    rng_center = np.random.default_rng(42).normal(size=len(sketch) * len(sketch[0]))
    surrogate.add_center(rng_center)

    for a in actions:
        val = hybrid_rbf_binding(a, sketch, surrogate)
        print(f"Surrogate value for {a.action_id}:", val)

    # 3. Weekday‑weighted Fisher
    wd_scores = weekday_weighted_fisher(actions, (2026, 5, 29))
    print("Weekday‑weighted Fisher scores:", wd_scores)

    print("Smoke test completed successfully.")