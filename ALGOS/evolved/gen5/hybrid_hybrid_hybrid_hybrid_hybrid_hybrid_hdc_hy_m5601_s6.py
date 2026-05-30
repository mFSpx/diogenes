# DARWIN HAMMER — match 5601, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1383_s0.py (gen4)
# parent_b: hybrid_hybrid_hdc_hybrid_hy_hybrid_hybrid_rbf_su_m182_s3.py (gen4)
# born: 2026-05-30T00:03:31Z

from __future__ import annotations

import math
import random
import hashlib
from dataclasses import dataclass
from typing import List, Sequence, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (unchanged semantics)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Vector utilities – richer than binary (+ deeper fusion)
# ----------------------------------------------------------------------
Vector = List[float]          # real‑valued high‑dimensional vectors
FloatVector = Sequence[float]


def _rng_from_seed(seed: int | str | None) -> random.Random:
    """Create a deterministic RNG from an int, str or None."""
    if isinstance(seed, str):
        seed = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    return random.Random(seed)


def random_float_vector(dim: int = 1024, seed: int | str | None = None) -> Vector:
    """Generate a dense vector with components drawn from N(0,1)."""
    rng = _rng_from_seed(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 1024) -> Vector:
    """
    Deterministic symbolic vector for a symbol.
    Uses a hash‑seeded normal distribution – preserves magnitude information.
    """
    return random_float_vector(dim, seed=symbol)


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (Hadamard product)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: List[Vector]) -> Vector:
    """
    Weighted (mean) bundling of real‑valued vectors.
    Returns the component‑wise average, preserving magnitude.
    """
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("vectors must have equal length")
    sums = np.zeros(dim, dtype=float)
    for v in vectors:
        sums += np.array(v, dtype=float)
    return sums.tolist()


def similarity(a: Vector, b: Vector) -> float:
    """
    Cosine similarity in ℝ^d (range [-1,1]).
    """
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: FloatVector, b: FloatVector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ----------------------------------------------------------------------
# RBF surrogate – now supports vector‑wise modulation
# ----------------------------------------------------------------------
@dataclass
class RBFSurrogate:
    centers: List[FloatVector]   # each centre lives in feature space
    weights: List[float]         # scalar weight per centre
    epsilon: float = 1.0         # bandwidth

    def predict(self, x: FloatVector) -> float:
        """Standard RBF network output (scalar)."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

    def copy(self) -> "RBFSurrogate":
        """Shallow copy preserving immutability of inner lists."""
        return RBFSurrogate(
            centers=[list(c) for c in self.centers],
            weights=self.weights.copy(),
            epsilon=self.epsilon,
        )


def modulate_surrogate(
    surrogate: RBFSurrogate,
    modulation_vector: Vector,
    alpha_center: float = 0.3,
    alpha_weight: float = 0.5,
) -> RBFSurrogate:
    """
    Deep fusion of the symbolic modulation into the surrogate:
      * Centres are shifted towards the modulation direction (Hadamard binding).
      * Weights are scaled by cosine similarity to the modulation.
      * Bandwidth ε is adaptively tightened/loosened based on modulation norm.
    """
    # 1️⃣ centre modulation – treat each centre as a real vector, bind with modulation
    mod_centers = [
        [
            (1 - alpha_center) * coord + alpha_center * (coord * mod_val)
            for coord, mod_val in zip(c, modulation_vector)
        ]
        for c in surrogate.centers
    ]

    # 2️⃣ weight scaling – similarity to a reference all‑ones vector captures “global intensity”
    ref = [1.0] * len(modulation_vector)
    intensity = (1 + similarity(modulation_vector, ref)) / 2  # map [-1,1] → [0,1]
    mod_weights = [w * (1 - alpha_weight + alpha_weight * intensity) for w in surrogate.weights]

    # 3️⃣ adaptive ε – larger modulation norm → narrower kernel (more confident)
    mod_norm = np.linalg.norm(modulation_vector)
    new_epsilon = surrogate.epsilon * (1 + 0.5 * math.tanh(mod_norm / len(modulation_vector)))

    return RBFSurrogate(mod_centers, mod_weights, new_epsilon)


# ----------------------------------------------------------------------
# Core hybrid functions – now action‑specific likelihoods
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """Softmax over regret‑adjusted utilities (numerically stable)."""
    if not actions:
        return {}
    cf: Dict[str, float] = {
        c.action_id: c.outcome_value * c.probability for c in counterfactuals
    }
    utilities = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions
    }
    max_u = max(utilities.values())
    exp_vals = {k: math.exp(v - max_u) for k, v in utilities.items()}
    total = sum(exp_vals.values())
    return {k: v / total for k, v in exp_vals.items()}


def build_modulation_vector(
    strategy: Dict[str, float],
    dim: int = 1024,
) -> Vector:
    """
    Probability‑weighted superposition of symbolic vectors.
    Preserves magnitude proportional to action probability.
    """
    if not strategy:
        raise ValueError("strategy must contain at least one action")
    weighted_vectors: List[Vector] = []
    for aid, prob in strategy.items():
        base = symbol_vector(aid, dim)
        weighted = [prob * x for x in base]  # scale each component
        weighted_vectors.append(weighted)
    return bundle(weighted_vectors)


def hybrid_bayesian_update(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    surrogate: RBFSurrogate,
    base_feature_vec: FloatVector,
    dim: int = 1024,
) -> Dict[str, float]:
    """
    End‑to‑end hybrid inference with action‑specific likelihoods:
      1. Regret‑weighted strategy → prior.
      2. Build a single modulation vector from the prior distribution.
      3. Modulate the surrogate (deep fusion).
      4. For each action, augment the base feature vector with its symbolic vector,
         obtain a surrogate prediction, exponentiate safely, and use as likelihood.
      5. Multiply prior by likelihood and renormalize (log‑space for stability).
    Returns the posterior distribution over action ids.
    """
    # ---- Step 1: prior ----
    prior = compute_regret_weighted_strategy(actions, counterfactuals)

    # ---- Step 2 & 3: modulation ----
    mod_vec = build_modulation_vector(prior, dim=dim)
    mod_surrogate = modulate_surrogate(surrogate, mod_vec)

    # ---- Step 4: action‑specific likelihoods ----
    # Pre‑compute surrogate prediction for the raw feature vector (shared term)
    base_pred = mod_surrogate.predict(base_feature_vec)

    # For each action we blend its symbolic vector into the feature space.
    # This yields a distinct input to the surrogate, preserving action identity.
    log_likelihoods: Dict[str, float] = {}
    for aid, prob in prior.items():
        action_vec = symbol_vector(aid, len(base_feature_vec))
        # Simple linear combination: feature + β * action_vec
        beta = 0.2  # controls influence of symbolic identity
        blended = [
            fv + beta * av for fv, av in zip(base_feature_vec, action_vec)
        ]
        pred = mod_surrogate.predict(blended)
        # Use log‑likelihood to avoid overflow: log L = pred (since L = exp(pred))
        log_likelihoods[aid] = pred

    # ---- Step 5: posterior via log‑sum‑exp ----
    max_log = max(log_likelihoods.values())
    unnorm_log = {aid: math.log(prior[aid]) + (log_likelihoods[aid] - max_log) for aid in prior}
    max_unnorm = max(unnorm_log.values())
    exp_vals = {aid: math.exp(val - max_unnorm) for aid, val in unnorm_log.items()}
    total = sum(exp_vals.values())
    if total == 0.0:
        # Degenerate case – fall back to uniform posterior
        n = len(prior)
        return {aid: 1.0 / n for aid in prior}
    return {aid: val / total for aid, val in exp_vals.items()}


def elect_leaders_via_bayesian_graph(
    posterior: Dict[str, float],
    top_n: int = 3,
) -> List[Tuple[str, float]]:
    """Return the top‑N actions sorted by posterior probability."""
    return sorted(posterior.items(), key=lambda kv: kv[1], reverse=True)[:top_n]