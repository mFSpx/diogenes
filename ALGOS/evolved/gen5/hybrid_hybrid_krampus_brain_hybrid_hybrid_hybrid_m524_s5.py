# DARWIN HAMMER — match 524, survivor 5
# gen: 5
# parent_a: hybrid_krampus_brainmap_bandit_router_m129_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_rbf_surrogate_m348_s0.py (gen4)
# born: 2026-05-29T23:29:22Z

"""Hybrid Krampus-RBF Bandit Router

This module fuses two parent algorithms:

* **Parent A** – a contextual bandit where Krampus brain‑map vibes are extracted as a feature
  vector and simple reward sums per action are maintained.
* **Parent B** – a radial‑basis‑function (RBF) surrogate that interprets Gaussian kernel
  matrices as coboundary operators of a sheaf, providing similarity measures between
  contexts and geometric descriptors (sphericity, flatness, righting time).

**Mathematical bridge** – The extracted vibe vector from Parent A is treated as a point
in a Hilbert space.  Parent B supplies a Gaussian kernel  
\(k(x, x') = \exp(-\varepsilon^2\|x-x'\|^2)\) that quantifies similarity between any two
vibe vectors.  By weighting historical rewards with this kernel we obtain a
*kernelised contextual bandit*: for a new context \(x\),

\[
\hat r_a(x)=\frac{\sum_{i: a_i=a} k(x, x_i)\, r_i}{\sum_{i: a_i=a} k(x, x_i)} ,
\qquad
U_a(x)=\hat r_a(x) + \beta\sqrt{\frac{1}{\sum_{i: a_i=a} k(x, x_i)}}
\]

where \(U_a\) is an Upper‑Confidence‑Bound used for action selection.  The
geometric descriptors from Parent B are optionally concatenated to the vibe vector,
allowing the kernel to incorporate both semantic and physical information.

The implementation below provides:
* `gaussian_kernel` – the RBF similarity.
* `HybridRouter.update_policy` – stores contexts, actions and rewards.
* `HybridRouter.select_action` – computes kernel‑weighted reward estimates and
  confidence bounds, returning a `BanditAction`.
* `HybridRouter.extract_full_features` – combines vibe extraction (stubs) with the
  geometry helpers from Parent B.

A lightweight smoke test demonstrates a full cycle without external data."""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

# ---------- Data structures -------------------------------------------------

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridKrampusRBF"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float
    feature_vector: np.ndarray  # pre‑computed context vector

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

# ---------- Parent‑B geometry helpers ---------------------------------------

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology,
                        b: float = 1.0 / 3.0,
                        k: float = 0.35,
                        neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck lever must be positive")
    inertia = (m.length ** 2 + m.width ** 2) / 12.0 * m.mass
    return (b * inertia) / (k * neck_lever * m.mass)

# ---------- Kernel utilities (Parent‑B core) ------------------------------

def gaussian_kernel(x: np.ndarray, y: np.ndarray, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel k(x,y)=exp(- (ε‖x−y‖)^2 )."""
    diff = x - y
    return math.exp(-((epsilon * np.linalg.norm(diff)) ** 2))

def kernel_vector(new_vec: np.ndarray,
                  stored_vecs: List[np.ndarray],
                  epsilon: float = 1.0) -> np.ndarray:
    """Return a column vector of kernel similarities between new_vec and each stored_vec."""
    if not stored_vecs:
        return np.array([])
    return np.array([gaussian_kernel(new_vec, v, epsilon) for v in stored_vecs])

# ---------- Hybrid router (fusion of A & B) --------------------------------

class HybridRouter:
    """Kernel‑augmented contextual bandit that uses Krampus vibes as context."""

    def __init__(self, epsilon: float = 1.0, beta: float = 1.0):
        self.epsilon = epsilon          # kernel bandwidth
        self.beta = beta                # confidence scaling
        self._contexts: List[np.ndarray] = []   # stored feature vectors
        self._actions: List[str] = []           # parallel list of actions
        self._rewards: List[float] = []         # parallel list of observed rewards

    # ----- Feature extraction (Parent‑A + geometry) -----------------------

    def extract_full_features(self, text: str,
                              morphology: Morphology | None = None) -> np.ndarray:
        """Combine stubbed vibe features with optional geometry descriptors."""
        feats: Dict[str, float] = {}
        feats.update(self._operator_vibes(text))
        feats.update(self._psyche_vibes(text))
        feats.update(self._resilience_vibes(text))
        feats.update(self._rainmaker_vibes(text))
        feats.update(self._operator_telemetry(text))

        if morphology is not None:
            feats["sphericity"] = sphericity_index(morphology.length,
                                                   morphology.width,
                                                   morphology.height)
            feats["flatness"] = flatness_index(morphology.length,
                                               morphology.width,
                                               morphology.height)
            feats["righting_time"] = righting_time_index(morphology)

        # Convert dict to ordered numpy array (sorted keys for determinism)
        ordered_keys = sorted(feats.keys())
        return np.array([feats[k] for k in ordered_keys], dtype=float)

    @staticmethod
    def _operator_vibes(text: str) -> Dict[str, float]:
        return {"operator_visceral_ratio": 0.5, "operator_tech_ratio": 0.3}

    @staticmethod
    def _psyche_vibes(text: str) -> Dict[str, float]:
        return {"psyche_forensic_shield_ratio": 0.2, "psyche_poetic_entropy": 0.1}

    @staticmethod
    def _resilience_vibes(text: str) -> Dict[str, float]:
        return {"resilience_bureaucratic_weaponization_index": 0.4,
                "resilience_resource_exhaustion": 0.35}

    @staticmethod
    def _rainmaker_vibes(text: str) -> Dict[str, float]:
        return {"rainmaker_threat_score": 0.6, "rainmaker_opportunity_factor": 0.2}

    @staticmethod
    def _operator_telemetry(text: str) -> Dict[str, float]:
        return {"operator_latency_ms": 120.0, "operator_throughput_ops": 2500.0}

    # ----- Policy update (store raw experiences) -------------------------

    def update_policy(self, updates: List[BanditUpdate]) -> None:
        """Append new observations to the internal kernel dataset."""
        for upd in updates:
            self._contexts.append(upd.feature_vector)
            self._actions.append(upd.action_id)
            self._rewards.append(upd.reward)

    # ----- Kernel‑based reward estimation ---------------------------------

    def _estimate_rewards(self, ctx_vec: np.ndarray) -> Tuple[Dict[str, float],
                                                            Dict[str, float]]:
        """
        For each action a, compute:
            μ_a = weighted average of past rewards (kernel similarity as weights)
            λ_a = sum of kernel weights (used for confidence)
        Returns two dictionaries: expected reward μ and weight sum λ.
        """
        if not self._contexts:
            return {}, {}

        k_vec = kernel_vector(ctx_vec, self._contexts, self.epsilon)  # shape (N,)
        # Convert to numpy for vectorised ops
        k_vec = k_vec.astype(float)

        reward_sums: Dict[str, float] = {}
        weight_sums: Dict[str, float] = {}

        for a, r, w in zip(self._actions, self._rewards, k_vec):
            reward_sums[a] = reward_sums.get(a, 0.0) + w * r
            weight_sums[a] = weight_sums.get(a, 0.0) + w

        expected: Dict[str, float] = {}
        for a in reward_sums:
            w = weight_sums[a]
            expected[a] = reward_sums[a] / w if w > 0 else 0.0

        return expected, weight_sums

    # ----- Action selection (UCB) -----------------------------------------

    def select_action(self, text: str,
                      morphology: Morphology | None = None,
                      propensity: float = 1.0) -> BanditAction:
        """Compute kernel‑augmented UCB for all actions and return the best."""
        ctx_vec = self.extract_full_features(text, morphology)

        exp_rewards, weight_sums = self._estimate_rewards(ctx_vec)

        # If no history, fall back to uniform random choice among stub actions
        if not exp_rewards:
            # define a minimal action set
            dummy_action = "default"
            return BanditAction(action_id=dummy_action,
                                propensity=propensity,
                                expected_reward=0.0,
                                confidence_bound=0.0)

        # Compute confidence bounds
        ucb_vals: Dict[str, float] = {}
        for a, mu in exp_rewards.items():
            lam = weight_sums.get(a, 0.0)
            conf = self.beta * (1.0 / math.sqrt(lam) if lam > 0 else float('inf'))
            ucb_vals[a] = mu + conf

        # Choose action with highest UCB
        best_action_id = max(ucb_vals, key=ucb_vals.get)
        best_mu = exp_rewards[best_action_id]
        best_conf = ucb_vals[best_action_id] - best_mu

        return BanditAction(action_id=best_action_id,
                            propensity=propensity,
                            expected_reward=best_mu,
                            confidence_bound=best_conf)

# ---------- Demonstration ---------------------------------------------------

def _demo_updates(router: HybridRouter) -> List[BanditUpdate]:
    """Create a few synthetic updates to seed the model."""
    texts = [
        "operator engaged, high threat level",
        "psyche stabilized, low entropy",
        "resilience metrics indicate resource strain",
        "rainmaker detected, opportunity rising"
    ]
    morphs = [
        Morphology(1.2, 0.8, 0.5, 2.3),
        Morphology(0.9, 1.0, 0.4, 1.8),
        Morphology(1.5, 1.2, 0.6, 3.0),
        Morphology(1.0, 0.9, 0.5, 2.0)
    ]
    actions = ["A", "B", "A", "C"]
    rewards = [1.0, 0.5, 0.8, 0.2]

    updates = []
    for i, txt in enumerate(texts):
        vec = router.extract_full_features(txt, morphs[i])
        upd = BanditUpdate(context_id=f"ctx_{i}",
                           action_id=actions[i],
                           reward=rewards[i],
                           propensity=1.0,
                           feature_vector=vec)
        updates.append(upd)
    return updates

if __name__ == "__main__":
    # Initialise router
    router = HybridRouter(epsilon=0.8, beta=0.7)

    # Seed with synthetic observations
    seed_updates = _demo_updates(router)
    router.update_policy(seed_updates)

    # Perform a prediction on a fresh context
    test_text = "operator under stress, resilience dropping"
    test_morph = Morphology(1.1, 0.9, 0.55, 2.2)

    chosen = router.select_action(test_text, test_morph)
    print("Chosen action:", asdict(chosen))
    # Verify that no exception is raised and output is sensible.