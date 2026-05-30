# DARWIN HAMMER — match 5204, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_tropical_maxp_m1632_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1961_s0.py (gen6)
# born: 2026-05-30T00:00:37Z

"""Hybrid Tropical‑Geometric‑Bandit Algorithm
==========================================

This module fuses the two parent algorithms:

* **Parent A** – tropical max‑plus algebra together with a variational
  free‑energy / entropy update (see ``StoreState`` and ``sigmoid``).
* **Parent B** – a contextual bandit that evaluates actions using a
  geometric‑algebra scalar product and a MinHash similarity score.

**Mathematical bridge**

Both parents operate on *feature representations* of actions and
contexts.  We therefore introduce three compatible operations:

1. **Tropical dot product** (max‑plus semiring)  

   \[
   \langle x, y\rangle_{\oplus}= \max_i (x_i + y_i)
   \]

2. **Geometric‑algebra scalar part** of the geometric product  

   \[
   \langle \mathcal{M}(a)\,\odot\,\mathcal{M}(c) \rangle_{0}
   \]

   where ``\mathcal{M}`` maps a feature vector to a multivector.
3. **MinHash Jaccard estimator** between the token sets of an action
   and a context.

The hybrid similarity used by the bandit is

\[
S(a,c)=\rho\;\langle \mathcal{M}(a)\,\odot\,\mathcal{M}(c) \rangle_{0}
      +\sigma\;\text{minhash\_sim}(a,c)
      +\tau\;\langle x_a, x_c\rangle_{\oplus},
\]

with tunable scalars ``ρ``, ``σ`` and ``τ``.  
The bandit’s exploitation term becomes the empirical reward plus ``S``,
while the exploration term is the usual UCB bonus.  The free‑energy
update from Parent A is applied to a ``StoreState`` that tracks a
running “energy level’’ of the system.

The functions below implement the three core operations and a unified
action‑selection routine that respects both probabilistic bandit learning
and the tropical‑geometric similarity measure.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, FrozenSet, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared with parents)
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

@dataclass
class StoreState:
    """Variational free‑energy store (Parent A)."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply the free‑energy balance."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """A bounded transformation of the last delta (used as a gain)."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))

# ----------------------------------------------------------------------
# Tropical max‑plus algebra
# ----------------------------------------------------------------------
def tropical_dot(x: np.ndarray, y: np.ndarray) -> float:
    """Max‑plus (tropical) dot product  ⟨x,y⟩⊕ = max_i (x_i + y_i)."""
    if x.shape != y.shape:
        raise ValueError("vectors must have the same shape")
    return float(np.max(x + y))

# ----------------------------------------------------------------------
# Simple geometric algebra (grade‑0 part only)
# ----------------------------------------------------------------------
class Multivector:
    """Very small GA engine: stores scalar and bivector components."""
    def __init__(self, components: Dict[FrozenSet[int], float] | None = None):
        # components are keyed by a frozenset of basis indices
        self.comp: Dict[FrozenSet[int], float] = components or {}

    def __getitem__(self, blade: FrozenSet[int]) -> float:
        return self.comp.get(blade, 0.0)

    def __setitem__(self, blade: FrozenSet[int], value: float) -> None:
        self.comp[blade] = value

    def __add__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self.comp.copy())
        for b, v in other.comp.items():
            result[b] = result[b] + v
        return result

    def geometric_product(self, other: "Multivector") -> "Multivector":
        """Naïve geometric product: only scalar‑scalar and scalar‑blade terms."""
        result = Multivector()
        for b1, v1 in self.comp.items():
            for b2, v2 in other.comp.items():
                # grade addition (XOR of basis sets) – sign ignored for simplicity
                new_blade = frozenset(b1.symmetric_difference(b2))
                result[new_blade] = result[new_blade] + v1 * v2
        return result

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component."""
        return self.comp.get(frozenset(), 0.0)

def vector_to_multivector(vec: np.ndarray) -> Multivector:
    """Map a real vector to a multivector with each entry as a basis blade."""
    mv = Multivector()
    for idx, val in enumerate(vec):
        mv[frozenset({idx})] = float(val)
    # also store the scalar part as the mean (arbitrary but deterministic)
    mv[frozenset()] = float(np.mean(vec))
    return mv

# ----------------------------------------------------------------------
# MinHash similarity (Jaccard estimator)
# ----------------------------------------------------------------------
def minhash_signature(tokens: Iterable[str], num_hashes: int = 64) -> List[int]:
    """Return a MinHash signature for a set of tokens."""
    sig = [2**64 - 1] * num_hashes
    for token in tokens:
        token_bytes = token.encode("utf-8")
        for i in range(num_hashes):
            h = hashlib.blake2b(token_bytes + i.to_bytes(2, "little"), digest_size=8).digest()
            hv = int.from_bytes(h, "big")
            if hv < sig[i]:
                sig[i] = hv
    return sig

def minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

# ----------------------------------------------------------------------
# Hybrid similarity combining tropical, geometric and MinHash terms
# ----------------------------------------------------------------------
def hybrid_similarity(
    action_vec: np.ndarray,
    context_vec: np.ndarray,
    rho: float = 1.0,
    sigma: float = 1.0,
    tau: float = 1.0,
    num_minhash: int = 64,
) -> float:
    """
    Compute the fused similarity score:

        S = rho * ⟨MV(a) ⊙ MV(c)⟩₀
          + sigma * minhash_sim(tok(a), tok(c))
          + tau * ⟨x_a, x_c⟩⊕

    where ``tok`` are the character tokens of the identifiers.
    """
    # geometric part
    mv_a = vector_to_multivector(action_vec)
    mv_c = vector_to_multivector(context_vec)
    scalar_geo = mv_a.geometric_product(mv_c).scalar_part()

    # MinHash part
    tokens_a = set(action_vec.astype(str).tolist())
    tokens_c = set(context_vec.astype(str).tolist())
    sig_a = minhash_signature(tokens_a, num_hashes=num_minhash)
    sig_c = minhash_signature(tokens_c, num_hashes=num_minhash)
    mh_sim = minhash_similarity(sig_a, sig_c)

    # tropical part
    trop = tropical_dot(action_vec, context_vec)

    return rho * scalar_geo + sigma * mh_sim + tau * trop

# ----------------------------------------------------------------------
# Bandit decision rule that incorporates the hybrid similarity and
# the free‑energy update from Parent A.
# ----------------------------------------------------------------------
def select_hybrid_action(
    context_vec: np.ndarray,
    actions: List[BanditAction],
    action_features: Dict[str, np.ndarray],
    store: StoreState,
    rho: float = 1.0,
    sigma: float = 1.0,
    tau: float = 1.0,
    exploration_coef: float = 1.0,
) -> BanditAction:
    """
    Choose an action using a UCB‑style rule enriched with the hybrid
    similarity S(a,c) and a free‑energy‑based regulariser.

    The score for each action a is

        Q(a) = expected_reward_a + S(a,c) + exploration_a
               - λ * H(propensity_a)

    where H(p) = -p·log(p) is the Shannon entropy term (λ derived from
    the current StoreState.dance value).
    """
    lambda_reg = store.dance  # use the bounded gain as regularisation strength
    best_score = -math.inf
    best_action = None

    for act in actions:
        feat = action_features[act.action_id]
        # hybrid similarity term
        S = hybrid_similarity(feat, context_vec, rho, sigma, tau)

        # exploration (UCB) term
        exploration = exploration_coef * act.confidence_bound

        # entropy regulariser on propensity
        p = max(act.propensity, 1e-12)
        entropy = -p * math.log(p)

        total = act.expected_reward + S + exploration - lambda_reg * entropy

        if total > best_score:
            best_score = total
            best_action = act

    if best_action is None:
        raise RuntimeError("No actions available for selection")
    return best_action

# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a simple store (free‑energy tracker)
    store = StoreState(alpha=0.7, beta=0.3, dt=0.5, gain=0.2)

    # Dummy context vector (e.g. sensor readings)
    context = np.array([0.2, 1.5, -0.3, 0.7])

    # Define three actions with synthetic statistics
    actions = [
        BanditAction(
            action_id="a1",
            propensity=0.4,
            expected_reward=1.2,
            confidence_bound=0.5,
            algorithm="hybrid"
        ),
        BanditAction(
            action_id="a2",
            propensity=0.3,
            expected_reward=0.9,
            confidence_bound=0.7,
            algorithm="hybrid"
        ),
        BanditAction(
            action_id="a3",
            propensity=0.3,
            expected_reward=1.0,
            confidence_bound=0.6,
            algorithm="hybrid"
        ),
    ]

    # Random feature vectors for each action
    rng = np.random.default_rng(42)
    action_features = {
        act.action_id: rng.normal(loc=0.0, scale=1.0, size=context.shape)
        for act in actions
    }

    # Perform a selection
    chosen = select_hybrid_action(
        context_vec=context,
        actions=actions,
        action_features=action_features,
        store=store,
        rho=0.8,
        sigma=0.5,
        tau=0.3,
        exploration_coef=0.4,
    )
    print(f"Chosen action: {chosen.action_id}")

    # Simulate an update: pretend we observed a reward
    reward = rng.normal(loc=chosen.expected_reward, scale=0.2)
    update = BanditUpdate(
        context_id="ctx0",
        action_id=chosen.action_id,
        reward=reward,
        propensity=chosen.propensity,
    )

    # Update the free‑energy store with inflow/outflow derived from reward
    inflow = [max(0.0, reward)]
    outflow = [abs(min(0.0, reward))]
    level, delta = store.update(inflow, outflow)
    print(f"Store level={level:.3f}, delta={delta:.3f}, dance={store.dance:.3f}")

    # Simple sanity check that the code runs without exception
    assert isinstance(level, float) and isinstance(delta, float)