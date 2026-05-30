# DARWIN HAMMER — match 3207, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2494_s3.py (gen6)
# parent_b: hybrid_hybrid_bandit_router_hybrid_hybrid_endpoi_m134_s1.py (gen3)
# born: 2026-05-29T23:48:37Z

"""Hybrid algorithm combining:
- Parent A: geometric‑algebraic morphology with a Count‑Min sketch and Koopman‑operator evolution.
- Parent B: contextual multi‑armed bandit with Schoolfield developmental rate and health scoring.

Mathematical bridge:
The Count‑Min sketch yields a high‑dimensional frequency vector 𝑠∈ℝᴰ that is analogous to the bipolar hypervectors of Parent B.  
We treat 𝑠 as the *context* for the bandit policy.  The Koopman operator 𝐊∈ℝᴰˣᴰ evolves this context in discrete time:
    𝑠_{t+1}=𝐊·𝑠_t .
The reward fed back to the bandit is a composite of the Schoolfield developmental rate (temperature‑dependent growth) and the morphology health score, both of which are scalar functions of the current Morphology.  Thus the bandit selects an action that modifies the Koopman matrix (e.g., scaling a mode), while the policy update uses the variational‑free‑energy‑like reward.

The code below implements:
1. A simple Count‑Min sketch for morphologic frequency tracking.
2. Construction and stable evolution of a Koopman matrix.
3. Contextual bandit selection and policy update using the composite reward.
"""

import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
@dataclass
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(morphology: Morphology) -> float:
    """Geometric sphericity (Parent A)."""
    if min(morphology.length, morphology.width, morphology.height) <= 0:
        raise ValueError("dimensions must be positive")
    return (morphology.length * morphology.width * morphology.height) ** (1.0 / 3.0) / max(
        morphology.length, morphology.width, morphology.height
    )

def flatness_index(morphology: Morphology) -> float:
    """Geometric flatness (Parent A)."""
    if min(morphology.length, morphology.width, morphology.height) <= 0:
        raise ValueError("dimensions must be positive")
    return min(morphology.length, morphology.width, morphology.height) / max(
        morphology.length, morphology.width, morphology.height
    )

# ----------------------------------------------------------------------
# Minimal Count‑Min sketch (high‑dimensional representation)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Very small Count‑Min sketch; each cell acts like a component of a hyper‑vector."""
    def __init__(self, width: int = 128, depth: int = 5, seed: int = 0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = random.Random(seed)
        self.seeds = [rng.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item: str, i: int) -> int:
        h = hash((item, self.seeds[i]))
        return h % self.width

    def update(self, item: str, increment: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.tables[i, idx] += increment

    def estimate(self, item: str) -> int:
        return min(self.tables[i, self._hash(item, i)] for i in range(self.depth))

    def as_vector(self) -> np.ndarray:
        """Flattened representation suitable as a high‑dimensional context vector."""
        return self.tables.flatten().astype(np.float64)

# ----------------------------------------------------------------------
# Koopman operator utilities (Parent A)
# ----------------------------------------------------------------------
def build_koopman_matrix(dim: int, spectral_radius: float = 0.95, seed: int = 42) -> np.ndarray:
    """Create a stable (spectral radius < 1) Koopman matrix."""
    rng = np.random.default_rng(seed)
    # Random matrix then scale eigenvalues to desired spectral radius
    A = rng.normal(size=(dim, dim))
    eigvals, eigvecs = np.linalg.eig(A)
    max_eig = max(abs(eigvals))
    if max_eig == 0:
        scale = 0.0
    else:
        scale = spectral_radius / max_eig
    K = A * scale
    return K.real.astype(np.float64)

def evolve_context(vec: np.ndarray, K: np.ndarray) -> np.ndarray:
    """One step of Koopman evolution: s_{t+1}=K·s_t."""
    return K @ vec

# ----------------------------------------------------------------------
# Data structures and helpers from Parent B (bandit)
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

# Simple in‑memory policy store (mirrors Parent B)
_POLICY: Dict[str, Tuple[float, float]] = {}  # action_id -> (cumulative_reward, count)

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    """Incremental reward aggregation."""
    for u in updates:
        total, cnt = _POLICY.get(u.action_id, (0.0, 0.0))
        total += float(u.reward)
        cnt += 1.0
        _POLICY[u.action_id] = (total, cnt)

def _average_reward(action_id: str) -> float:
    total, cnt = _POLICY.get(action_id, (0.0, 0.0))
    return total / cnt if cnt > 0 else 0.0

# ----------------------------------------------------------------------
# Temperature‑driven developmental rate (Parent B)
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
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield temperature response (Parent B)."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def normalized_activity(temp_c: float, samples: int = 141) -> float:
    """Scaled activity between 0 and 1."""
    temp_k = c_to_k(temp_c)
    rate = developmental_rate(temp_k)
    return rate / (rate + 1.0)

# ----------------------------------------------------------------------
# Health scoring (Parent B)
# ----------------------------------------------------------------------
def health_score(failures: int, threshold: int, recovery_priority: float) -> float:
    """Higher is better; analogous to variational free‑energy term."""
    return max(0.0, (1 - failures / threshold) * (1 - recovery_priority))

def curvature_score(morph_curvature: float, health: float) -> float:
    """Placeholder – currently mirrors health."""
    return health

# ----------------------------------------------------------------------
# Hybrid core functions (the required three+ functions)
# ----------------------------------------------------------------------
def compute_context_vector(morph: Morphology, sketch: CountMinSketch) -> np.ndarray:
    """
    Assemble a high‑dimensional context vector by concatenating:
    - Normalized geometric descriptors (sphericity, flatness, mass)
    - Flattened Count‑Min sketch counts
    The vector is L2‑normalized to serve as bandit context.
    """
    geo = np.array([
        sphericity_index(morph),
        flatness_index(morph),
        morph.mass
    ], dtype=np.float64)
    sketch_vec = sketch.as_vector()
    ctx = np.concatenate([geo, sketch_vec])
    norm = np.linalg.norm(ctx)
    return ctx / norm if norm > 0 else ctx

def select_action(context: np.ndarray, actions: List[BanditAction], exploration_coef: float = 0.1) -> BanditAction:
    """
    Contextual Upper‑Confidence‑Bound (UCB) selection.
    Expected reward is taken from the stored policy; the confidence term is
    scaled by the dot‑product between action propensity vector and the context.
    """
    best_score = -math.inf
    best_action = None
    for a in actions:
        avg = _average_reward(a.action_id)
        # Propensity similarity term (acts as a bridge to high‑dimensional context)
        similarity = np.dot(context, np.full_like(context, a.propensity))
        ucb = avg + exploration_coef * math.sqrt(similarity + 1e-9) + a.confidence_bound
        if ucb > best_score:
            best_score = ucb
            best_action = a
    return best_action

def hybrid_step(
    morph: Morphology,
    sketch: CountMinSketch,
    K: np.ndarray,
    actions: List[BanditAction],
    temperature_c: float,
    failures: int,
    failure_threshold: int,
    recovery_priority: float,
) -> Tuple[Morphology, np.ndarray, BanditAction, float]:
    """
    Perform one hybrid iteration:
    1. Build context vector.
    2. Select an action via bandit.
    3. Compute composite reward (developmental_rate * health_score).
    4. Update policy.
    5. Evolve the Count‑Min sketch through Koopman dynamics (K·s).
    6. Return updated morphology (unchanged here), new context, chosen action, reward.
    """
    # 1. Context
    ctx = compute_context_vector(morph, sketch)

    # 2. Action selection
    chosen = select_action(ctx, actions)

    # 3. Composite reward
    dev_rate = normalized_activity(temperature_c)
    health = health_score(failures, failure_threshold, recovery_priority)
    reward = dev_rate * health  # multiplicative fusion of both parents

    # 4. Policy update
    update_policy([BanditUpdate(context_id="global", action_id=chosen.action_id,
                               reward=reward, propensity=chosen.propensity)])

    # 5. Koopman evolution of the sketch vector
    sketch_vec = sketch.as_vector()
    new_vec = evolve_context(sketch_vec, K)

    # Write back the evolved vector into the sketch (inverse operation)
    # For simplicity, we replace the raw tables with the reshaped vector.
    reshaped = new_vec.reshape(sketch.depth, sketch.width)
    sketch.tables = np.maximum(0, np.rint(reshaped)).astype(np.int64)  # keep integer counts

    # 6. Return diagnostics
    return morph, ctx, chosen, reward

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise components
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)
    sketch = CountMinSketch(width=64, depth=4, seed=123)
    # Populate sketch with some dummy observations
    for item in ["cell", "nucleus", "mitochondria", "cell", "cell"]:
        sketch.update(item)

    dim = sketch.depth * sketch.width + 3  # geo descriptors + sketch
    K = build_koopman_matrix(dim)

    actions = [
        BanditAction(action_id="scale_up", propensity=0.8, expected_reward=0.0,
                     confidence_bound=0.2, algorithm="koopman_scale"),
        BanditAction(action_id="scale_down", propensity=0.5, expected_reward=0.0,
                     confidence_bound=0.1, algorithm="koopman_shrink"),
        BanditAction(action_id="noop", propensity=0.2, expected_reward=0.0,
                     confidence_bound=0.05, algorithm="identity")
    ]

    temperature_c = 30.0
    failures = 1
    failure_threshold = 5
    recovery_priority = 0.3

    # Run a few hybrid steps
    for step in range(5):
        morph, ctx, act, rew = hybrid_step(
            morph,
            sketch,
            K,
            actions,
            temperature_c,
            failures,
            failure_threshold,
            recovery_priority,
        )
        print(f"Step {step+1}: action={act.action_id}, reward={rew:.4f}, ctx_norm={np.linalg.norm(ctx):.4f}")

    # Verify policy statistics
    print("\nPolicy summary:")
    for aid, (total, cnt) in _POLICY.items():
        print(f"  {aid}: avg_reward={total/cnt:.4f} over {int(cnt)} updates")