# DARWIN HAMMER — match 3154, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_perceptual_de_m828_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m39_s1.py (gen4)
# born: 2026-05-29T23:48:06Z

"""Hybrid Pheromone‑Bandit‑Fisher Algorithm
================================================

Parents
-------
* **hybrid_hybrid_hybrid_pherom_hybrid_perceptual_de_m828_s1.py** –  
  Provides a pheromone‑driven perceptual‑hash clustering where candidate
  uncertainty is measured by the expected entropy of pheromone signals and
  low‑quality candidates are pruned.

* **hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m39_s1.py** –  
  Supplies a Fisher‑information based angle selector together with a contextual
  multi‑armed bandit.  The bandit returns a *propensity* (confidence scalar) that
  can modulate any downstream computation.

Mathematical Bridge
-------------------
The bridge is the interpretation of the bandit **propensity** as a confidence
weight that scales the Fisher‑information contribution when updating pheromone
signals.  Concretely, for a candidate *c* we compute


I, F = compute_fisher_information(theta_c, mu, sigma, v)   # intensity, Fisher
Δφ = propensity_c * F                                      # confidence‑scaled update


The updated pheromone signal `phi_c` is then used to recompute the normalized
distribution `p_c = phi_c / Σ phi` whose entropy


H = - Σ p_c log(p_c)


guides the pruning probability: candidates whose contribution to the entropy
exceeds a threshold are discarded.  This fuses the matrix‑style Fisher
information, the bandit confidence, and the pheromone‑entropy machinery into a
single unified loop.

The module below implements this hybrid system with three public functions:
`compute_fisher_information`, `select_bandit_action`, and `hybrid_update`.
"""

import sys
import math
import random
import pathlib
import json
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Sequence, Callable

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Fisher core (Parent B)
# ----------------------------------------------------------------------
def compute_fisher_information(theta: float, mu: float, sigma: float, v: float) -> Tuple[float, float]:
    """
    Compute Gaussian intensity I and Fisher information F for a scalar angle.

    Returns
    -------
    I : float
        Intensity term (scaled Gaussian).
    F : float
        Fisher information term.
    """
    I = math.exp(-((theta - mu) / sigma) ** 2)          # Gaussian intensity
    # Avoid division by zero if I becomes extremely small
    if I == 0.0:
        I = sys.float_info.min
    F = (2 * (theta - mu) / sigma ** 2) ** 2 / I
    return v * I, v * F

# ----------------------------------------------------------------------
# Simple Bandit core (Parent B, trimmed)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "linucb"

# Global mutable policy store (simulates learning)
_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}

def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()

def select_bandit_action(
    context: Dict[str, float],
    actions: List[str],
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    ε‑greedy selection over provided actions.  The returned `propensity`
    encodes the confidence (1‑ε for the greedy choice, ε uniformly otherwise).
    """
    if not actions:
        raise ValueError("No actions provided")
    rng = random.Random(seed)
    if rng.random() < epsilon:
        chosen = rng.choice(actions)
        propensity = epsilon / len(actions)
    else:
        # Greedy: pick the action with highest stored value, fallback to random
        scores = [(action, _STORE.get(action, 0.0)) for action in actions]
        max_score = max(scores, key=lambda x: x[1])[1]
        best_actions = [a for a, s in scores if s == max_score]
        chosen = rng.choice(best_actions)
        propensity = 1.0 - epsilon
    # Dummy reward and confidence bound for illustration
    expected_reward = _STORE.get(chosen, 0.0)
    confidence_bound = math.sqrt(2 * math.log(1 + len(_POLICY)) / (1 + _STORE.get(chosen, 0.0) + 1e-9))
    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=expected_reward,
        confidence_bound=confidence_bound,
    )

def bandit_update(action: BanditAction, reward: float) -> None:
    """Simple update: increment stored value for the chosen action."""
    _STORE[action.action_id] = _STORE.get(action.action_id, 0.0) + reward

# ----------------------------------------------------------------------
# Pheromone‑Hash core (Parent A, simplified)
# ----------------------------------------------------------------------
def perceptual_hash(vector: Vector, bits: int = 64) -> int:
    """
    Very coarse perceptual hash: binarize the sign of each component,
    pad/truncate to `bits` and interpret as an integer.
    """
    signs = [1 if x >= 0 else 0 for x in vector[:bits]]
    # Pad with zeros if vector shorter than bits
    signs += [0] * (bits - len(signs))
    hash_int = 0
    for bit in signs:
        hash_int = (hash_int << 1) | bit
    return hash_int

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return bin(a ^ b).count("1")

def cluster_by_phash(vectors: List[Vector], radius: int = 3) -> List[List[int]]:
    """
    Naïve O(N²) clustering: each vector is assigned to a cluster whose
    representative hash is within `radius` Hamming distance.
    Returns a list of clusters, each a list of indices into `vectors`.
    """
    hashes = [perceptual_hash(v) for v in vectors]
    clusters: List[List[int]] = []
    assigned = set()
    for i, h_i in enumerate(hashes):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j, h_j in enumerate(hashes):
            if j in assigned:
                continue
            if hamming_distance(h_i, h_j) <= radius:
                cluster.append(j)
                assigned.add(j)
        clusters.append(cluster)
    return clusters

# ----------------------------------------------------------------------
# Hybrid System
# ----------------------------------------------------------------------
@dataclass
class Candidate:
    idx: int
    vector: Vector
    hash: int
    pheromone: float = 1.0  # initial signal strength

class HybridPheromoneBanditFisher:
    """
    Holds candidates, pheromone signals and performs hybrid updates.
    """
    def __init__(self, candidates: List[Candidate]):
        self.candidates: Dict[int, Candidate] = {c.idx: c for c in candidates}
        self.last_update: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)

    def expected_entropy(self) -> float:
        """Entropy of the normalized pheromone distribution."""
        signals = np.array([c.pheromone for c in self.candidates.values()], dtype=float)
        total = signals.sum()
        if total == 0:
            return 0.0
        probs = signals / total
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-12, 1.0)
        return -float(np.sum(probs * np.log(probs)))

    def prune_by_entropy(self, threshold: float = 0.8) -> None:
        """
        Remove candidates whose removal would lower the entropy below `threshold`.
        A simple heuristic: drop the candidate with the smallest pheromone
        contribution until entropy <= threshold or only one candidate remains.
        """
        while len(self.candidates) > 1 and self.expected_entropy() > threshold:
            # Find candidate with minimal pheromone
            min_idx = min(self.candidates, key=lambda k: self.candidates[k].pheromone)
            del self.candidates[min_idx]

    def hybrid_update(
        self,
        context: Dict[str, float],
        mu: float = 0.0,
        sigma: float = 1.0,
        v: float = 1.0,
        entropy_thresh: float = 0.85,
    ) -> None:
        """
        Perform a single hybrid iteration:
        1. Use bandit to select an action (candidate id).
        2. Compute Fisher information for the selected candidate's angle.
           The angle is taken as the arctangent of the first two components.
        3. Scale the Fisher information by the bandit propensity and update the
           candidate's pheromone signal.
        4. Re‑evaluate entropy and prune low‑impact candidates.
        """
        # 1. Bandit selection over candidate ids
        action = select_bandit_action(
            context=context,
            actions=list(self.candidates.keys()),
            epsilon=0.2,
        )
        cand = self.candidates.get(action.action_id)
        if cand is None:
            return  # selected candidate may have been pruned earlier

        # 2. Derive a scalar angle from the candidate vector (fallback to 0)
        if len(cand.vector) >= 2:
            theta = math.atan2(cand.vector[1], cand.vector[0])
        else:
            theta = 0.0

        # 3. Fisher info and pheromone update
        intensity, fisher = compute_fisher_information(theta, mu, sigma, v)
        delta_pheromone = action.propensity * fisher
        cand.pheromone = max(cand.pheromone + delta_pheromone, 0.0)

        # Simulate a reward proportional to intensity (for bandit learning)
        reward = intensity
        bandit_update(action, reward)

        # 4. Prune based on entropy
        self.prune_by_entropy(threshold=entropy_thresh)

        # Record time
        self.last_update = datetime.datetime.now(datetime.timezone.utc)

# ----------------------------------------------------------------------
# Public API Functions (demonstrating hybrid operation)
# ----------------------------------------------------------------------
def compute_fisher_information(theta: float, mu: float = 0.0, sigma: float = 1.0, v: float = 1.0) -> Tuple[float, float]:
    """Wrapper exposing the Fisher core (kept for API compatibility)."""
    return compute_fisher_information.__wrapped__(theta, mu, sigma, v) if hasattr(compute_fisher_information, "__wrapped__") else compute_fisher_information.__code__  # type: ignore

def select_bandit_action(
    context: Dict[str, float],
    actions: List[int],
    epsilon: float = 0.1,
    seed: int | str | None = None,
) -> BanditAction:
    """Convenient wrapper that forwards to the internal bandit selector."""
    return select_bandit_action(context, actions, epsilon, seed)

def hybrid_step(
    system: HybridPheromoneBanditFisher,
    context: Dict[str, float],
) -> None:
    """Execute one hybrid iteration on the provided system."""
    system.hybrid_update(context)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic candidates
    rng = np.random.default_rng(42)
    vectors = [rng.normal(size=8).tolist() for _ in range(20)]
    candidates = [
        Candidate(idx=i, vector=v, hash=perceptual_hash(v))
        for i, v in enumerate(vectors)
    ]

    # Initialise hybrid system
    hybrid_system = HybridPheromoneBanditFisher(candidates)

    # Dummy context (could be any feature dict)
    context = {"dummy_feature": 1.0}

    # Run a few hybrid steps
    for step in range(5):
        hybrid_step(hybrid_system, context)
        entropy = hybrid_system.expected_entropy()
        print(f"Step {step+1}: candidates={len(hybrid_system.candidates)}, entropy={entropy:.4f}")

    # Show remaining candidate ids
    print("Remaining candidate IDs:", list(hybrid_system.candidates.keys()))