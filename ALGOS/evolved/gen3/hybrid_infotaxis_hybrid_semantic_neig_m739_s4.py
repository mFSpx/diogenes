# DARWIN HAMMER — match 739, survivor 4
# gen: 3
# parent_a: infotaxis.py (gen0)
# parent_b: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py (gen2)
# born: 2026-05-29T23:30:39Z

"""Hybrid Infotaxis–Semantic Morphology System
Parents:
- infotaxis.py (entropy‑based action selection)
- hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py (morphology‑driven hybrid affinity)

Mathematical Bridge:
The recovery priority p∈[0,1] derived from a document’s morphology scales the cosine
similarity c∈[-1,1] of a candidate neighbor, yielding a hybrid affinity  

    h = c * p_other  

We treat the set of affinities for a given action as a probability‑like mass
and feed it into the infotaxis entropy machinery.  The expected entropy of an
action a is  

    E_a = p_hit(a)·H(hit_state_a) + (1−p_hit(a))·H(miss_state_a)

where p_hit(a) is the normalized hybrid affinity for the “hit” outcome.
Thus the morphology‑modulated semantic topology directly drives the
information‑theoretic action ranking."""


from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Parent B – Morphology & Recovery Priority
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Maps righting time index to a normalized priority in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Parent A – Entropy Helpers (adapted for numpy)
# ----------------------------------------------------------------------
def entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability vector (not necessarily normalized)."""
    total = probabilities.sum()
    if total <= 0:
        raise ValueError('positive probability mass required')
    probs = probabilities / total
    # Clip to avoid log(0)
    probs = np.clip(probs, eps, None)
    return -float(np.sum(probs * np.log(probs)))


def expected_entropy(p_hit: float,
                     hit_state: np.ndarray,
                     miss_state: np.ndarray) -> float:
    """Weighted entropy of hit/miss outcome distributions."""
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


def best_action(actions: Dict[str, Tuple[float, np.ndarray, np.ndarray]]) -> str:
    """Select action with minimal expected entropy; ties broken alphabetically."""
    if not actions:
        raise ValueError('actions required')
    return min(actions,
               key=lambda a: (expected_entropy(*actions[a]), a))


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def hybrid_affinity_matrix(cosine_sim: np.ndarray,
                           priorities: np.ndarray) -> np.ndarray:
    """
    Compute hybrid affinities h_ij = c_ij * p_j where:
        cosine_sim : (n_actions, n_candidates) matrix of cosine similarities
        priorities : (n_candidates,) vector of recovery priorities
    Returns an (n_actions, n_candidates) matrix of hybrid affinities.
    """
    if cosine_sim.shape[1] != priorities.shape[0]:
        raise ValueError('shape mismatch between cosine matrix and priorities')
    # Broadcast priorities over rows (actions)
    return cosine_sim * priorities[np.newaxis, :]


def build_action_distributions(
        hybrid_affinities: np.ndarray,
        hit_state_template: np.ndarray,
        miss_state_template: np.ndarray) -> Dict[str, Tuple[float, np.ndarray, np.ndarray]]:
    """
    For each action (row) produce a tuple (p_hit, hit_state, miss_state) suitable
    for the infotaxis expected_entropy function.
    The hit_state and miss_state are copies of the provided templates scaled by
    the hybrid affinity values, ensuring they remain valid probability-like vectors.
    """
    n_actions = hybrid_affinities.shape[0]
    actions: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}
    for i in range(n_actions):
        # p_hit is the normalized sum of affinities for this action
        raw = hybrid_affinities[i, :]
        p_hit = float(raw.sum())
        # Scale templates so that they reflect the distribution of individual candidates
        hit_state = hit_state_template * raw
        miss_state = miss_state_template * (1.0 - raw / (raw.max() if raw.max() != 0 else 1.0))
        actions[f'action_{i}'] = (p_hit, hit_state, miss_state)
    return actions


def hybrid_infotaxis_best_action(
        cosine_sim: np.ndarray,
        morphologies: List[Morphology],
        hit_state_template: np.ndarray,
        miss_state_template: np.ndarray) -> str:
    """
    End‑to‑end hybrid decision:
      1. Convert morphologies → priorities.
      2. Compute hybrid affinities.
      3. Build infotaxis action descriptors.
      4. Return the action with minimal expected entropy.
    """
    priorities = np.array([recovery_priority(m) for m in morphologies])
    h_aff = hybrid_affinity_matrix(cosine_sim, priorities)
    actions = build_action_distributions(h_aff, hit_state_template, miss_state_template)
    return best_action(actions)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy morphologies for 5 candidate neighbors
    candidates = [
        Morphology(1.0, 0.5, 0.3, 2.0),
        Morphology(0.8, 0.6, 0.4, 1.5),
        Morphology(1.2, 0.7, 0.5, 2.5),
        Morphology(0.9, 0.4, 0.2, 1.0),
        Morphology(1.1, 0.5, 0.3, 1.8),
    ]

    # Random cosine similarity matrix for 3 possible actions vs 5 candidates
    rng = np.random.default_rng(42)
    cosine_sim = rng.uniform(-1.0, 1.0, size=(3, len(candidates)))

    # Simple templates: uniform distributions over 5 bins
    hit_template = np.ones(len(candidates))
    miss_template = np.ones(len(candidates))

    best = hybrid_infotaxis_best_action(
        cosine_sim=cosine_sim,
        morphologies=candidates,
        hit_state_template=hit_template,
        miss_state_template=miss_template,
    )
    print(f"Chosen best action: {best}")
    # Ensure no exception and a valid action name is printed.