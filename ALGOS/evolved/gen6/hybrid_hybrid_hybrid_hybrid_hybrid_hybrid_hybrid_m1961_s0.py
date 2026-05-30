# DARWIN HAMMER — match 1961, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_indy_l_m1022_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_liquid_hybrid_hybrid_hybrid_m758_s1.py (gen5)
# born: 2026-05-29T23:40:03Z

"""Hybrid Bandit‑Geometric‑MinHash Algorithm
================================================
This module fuses the core topologies of the two parent algorithms:

* **Parent A** – a contextual bandit with ``select_action`` that uses
  reward averages and an exploration bonus (UCB‑style) together with a
  Gaussian‑beam weighting.
* **Parent B** – a MinHash similarity engine and a geometric‑algebra
  ``Multivector`` class that can combine feature vectors through the
  geometric product.

**Mathematical bridge**

Both parents operate on *feature representations*:

* The bandit treats a **context** as a vector of real numbers.
* The geometric algebra treats **actions** as multivectors whose
  components are indexed by basis blades (here encoded as frozensets of
  token characters).

The fusion creates a **hybrid similarity score** for each action:


S(a, c) = ρ·⟨MV(a) ⊙ MV(c)⟩₀ + σ·minhash_sim(tok(a), tok(c))


* ``⟨MV(a) ⊙ MV(c)⟩₀`` is the scalar (grade‑0) part of the geometric
  product of the action and context multivectors.
* ``minhash_sim`` is the MinHash Jaccard estimator between the token
  sets of the action and the context.
* ``ρ`` and ``σ`` are tunable scalars.

The bandit’s exploitation term becomes the ordinary empirical reward
plus the hybrid similarity ``S``; the exploration term remains the
UCB‑style bonus from Parent A, optionally weighted by a Gaussian beam
centered on a chosen context direction.

The result is a single unified decision rule that respects both
probabilistic bandit learning and geometric‑algebraic similarity."""

import json
import hashlib
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Iterable, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Data structures
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
# Global policy store (Parent A)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Clear the internal reward statistics."""
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    """Accumulate rewards for actions."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)   # cumulative reward
        stats[1] += 1.0                # count

def _reward(action_id: str) -> float:
    """Mean reward for an action; zero if never seen."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0

# ----------------------------------------------------------------------
# MinHash utilities (Parent B)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    """Hash a token with a 4‑byte seed using Blake2b (64‑bit output)."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(tokens: Iterable[str], num_perm: int = 64) -> np.ndarray:
    """Compute a MinHash signature for a set of tokens."""
    sig = np.full(num_perm, MAX64, dtype=np.uint64)
    for token in tokens:
        for i in range(num_perm):
            h = _hash(i, token)
            if h < sig[i]:
                sig[i] = h
    return sig

def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("Signature shapes must match")
    return float(np.mean(sig1 == sig2))

# ----------------------------------------------------------------------
# Geometric algebra utilities (Parent B)
# ----------------------------------------------------------------------
class Multivector:
    """A very lightweight implementation of a Clifford algebra element.

    Components are stored as a mapping ``blade -> coefficient`` where a
    blade is a ``frozenset`` of hashable identifiers (here we use token
    strings). The scalar (grade‑0) blade is the empty frozenset.
    """

    def __init__(self, components: Dict[FrozenSet[str], float] | None = None):
        self.components: Dict[FrozenSet[str], float] = {}
        if components:
            for blade, coeff in components.items():
                if coeff != 0.0:
                    self.components[frozenset(blade)] = float(coeff)

    # ------------------------------------------------------------------
    # Basic algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coeff in other.components.items():
            result[blade] = result.get(blade, 0.0) + coeff
        return Multivector(result)

    def __sub__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coeff in other.components.items():
            result[blade] = result.get(blade, 0.0) - coeff
        return Multivector(result)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (simplified).

        For this fusion we only need the scalar part of the product.
        The implementation therefore returns a multivector whose only
        non‑zero component is the scalar (grade‑0) part computed as the
        sum of products of matching blades.
        """
        scalar = 0.0
        for blade, a in self.components.items():
            b = other.components.get(blade)
            if b is not None:
                scalar += a * b
        return Multivector({frozenset(): scalar})

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) coefficient."""
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @staticmethod
    def from_tokens(tokens: Iterable[str], weight: float = 1.0) -> "Multivector":
        """Create a multivector where each token becomes a basis blade."""
        comps: Dict[FrozenSet[str], float] = {}
        for token in tokens:
            blade = frozenset([token])
            comps[blade] = comps.get(blade, 0.0) + weight
        return Multivector(comps)

# ----------------------------------------------------------------------
# Tokenisation utilities (shared)
# ----------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Very simple whitespace / punctuation tokeniser."""
    import re
    return [t for t in re.split(r"\W+", text.lower()) if t]

# ----------------------------------------------------------------------
# Hybrid similarity and scoring (core of the fusion)
# ----------------------------------------------------------------------
def hybrid_similarity(
    action_mv: Multivector,
    context_mv: Multivector,
    action_tokens: List[str],
    context_tokens: List[str],
    rho: float = 0.6,
    sigma: float = 0.4,
    num_perm: int = 64,
) -> float:
    """Compute the fused similarity S(a,c).

    Parameters
    ----------
    action_mv, context_mv
        Multivector representations of the action and the context.
    action_tokens, context_tokens
        Token lists used for MinHash.
    rho, sigma
        Weighting coefficients for the geometric and MinHash parts.
    num_perm
        Number of permutations for the MinHash signature.
    """
    # 1. Geometric scalar part
    geom_scalar = (action_mv * context_mv).scalar_part()

    # 2. MinHash similarity
    sig_a = minhash_signature(action_tokens, num_perm)
    sig_c = minhash_signature(context_tokens, num_perm)
    mh_sim = minhash_similarity(sig_a, sig_c)

    return rho * geom_scalar + sigma * mh_sim

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian weighting used as an optional context‑direction bias."""
    return math.exp(-((theta - center) ** 2) / (2 * width ** 2))

def hybrid_score(
    action_id: str,
    context_vec: Dict[str, float],
    context_tokens: List[str],
    rho: float = 0.6,
    sigma: float = 0.4,
    epsilon: float = 0.1,
) -> Tuple[float, float]:
    """Return (exploitation, exploration) terms for a given action.

    Exploitation = empirical reward + hybrid similarity.
    Exploration = UCB‑style bonus optionally modulated by a Gaussian beam.
    """
    # Empirical reward from the bandit
    base = _reward(action_id)

    # Build multivectors
    action_tokens = tokenize(action_id)
    action_mv = Multivector.from_tokens(action_tokens)
    context_mv = Multivector.from_tokens(context_tokens)

    # Hybrid similarity term
    sim = hybrid_similarity(
        action_mv, context_mv, action_tokens, context_tokens, rho, sigma
    )

    exploitation = base + sim

    # Exploration bonus (UCB)
    count = _POLICY.get(action_id, [0.0, 0.0])[1]
    scale = math.sqrt(sum(v * v for v in context_vec.values())) if context_vec else 1.0
    exploration = 0.1 * scale / math.sqrt(1.0 + count)

    # Optional Gaussian beam (using the norm of the context as angle proxy)
    beam = gaussian_beam(theta=scale, center=0.0, width=10.0)
    exploration *= beam

    return exploitation, exploration

# ----------------------------------------------------------------------
# Hybrid action selection (main entry point)
# ----------------------------------------------------------------------
def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
    rho: float = 0.6,
    sigma: float = 0.4,
) -> BanditAction:
    """Select an action using the fused bandit‑geometric‑MinHash rule.

    Parameters
    ----------
    context
        Mapping from feature names to real values.
    actions
        List of candidate action identifiers (strings).
    algorithm
        Currently only ``linucb`` and ``epsilon_greedy`` are supported.
    epsilon
        Exploration probability for epsilon‑greedy.
    seed
        Random seed for reproducibility.
    rho, sigma
        Weighting coefficients for the hybrid similarity.
    """
    if not actions:
        raise ValueError("At least one action must be supplied")

    rng = random.Random(seed)

    # Epsilon‑greedy shortcut
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen_id = rng.choice(actions)
        return BanditAction(
            action_id=chosen_id,
            propensity=epsilon,
            expected_reward=_reward(chosen_id),
            confidence_bound=0.0,
            algorithm=algorithm,
        )

    # Tokenise the context once
    context_tokens = [f"{k}:{v:.3f}" for k, v in context.items()]

    # Compute scores for each action
    best_id = None
    best_score = -math.inf
    best_exploit = best_explore = 0.0

    for a in actions:
        exploit, explore = hybrid_score(
            a, context, context_tokens, rho=rho, sigma=sigma, epsilon=epsilon
        )
        total = exploit + explore
        if total > best_score:
            best_score = total
            best_id = a
            best_exploit, best_explore = exploit, explore

    # Assemble result
    return BanditAction(
        action_id=best_id,
        propensity=1.0 - epsilon,
        expected_reward=best_exploit,
        confidence_bound=best_explore,
        algorithm=algorithm,
    )

# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic test
    reset_policy()

    # Define a tiny context vector
    ctx = {"temperature": 22.5, "humidity": 0.45, "pressure": 1013.25}

    # Candidate actions (could be any strings)
    actions = ["heat_up", "cool_down", "ventilate", "dehumidify"]

    # First selection (no prior data)
    chosen = select_action(ctx, actions, algorithm="linucb", seed=42)
    print("Chosen action:", chosen)

    # Simulate a reward and update the policy
    reward = random.random()  # mock reward
    upd = BanditUpdate(
        context_id="run1",
        action_id=chosen.action_id,
        reward=reward,
        propensity=chosen.propensity,
    )
    update_policy([upd])

    # Second selection – now the policy contains one observation
    chosen2 = select_action(ctx, actions, algorithm="linucb", seed=43)
    print("Chosen action after one update:", chosen2)

    # Verify that the code runs without raising exceptions
    sys.exit(0)