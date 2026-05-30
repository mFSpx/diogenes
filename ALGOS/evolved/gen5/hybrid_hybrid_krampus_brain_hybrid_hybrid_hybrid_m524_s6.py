# DARWIN HAMMER — match 524, survivor 6
# gen: 5
# parent_a: hybrid_krampus_brainmap_bandit_router_m129_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_rbf_surrogate_m348_s0.py (gen4)
# born: 2026-05-29T23:29:22Z

"""HybridRBFContextualBandit
Integrates:
- Parent A (krampus_brainmap + bandit router): extracts a high‑dimensional context vector from free‑form text and maintains per‑action reward statistics.
- Parent B (RBF surrogate + sheaf‑like coboundary): interprets Gaussian kernel weights as a coboundary operator Δ mapping contexts to reward estimates, and enriches the context with morphology‑derived geometric features (sphericity, flatness, righting‑time).

Mathematical bridge:
Given a context vector ϕ∈ℝ^d (text vibes ⊕ morphology indices) and a set of stored contexts {ϕ_i}_{i=1}^N with observed rewards r_i for actions a_i,
the Gaussian kernel k(ϕ,ϕ_i)=exp(−ε²‖ϕ−ϕ_i‖²) builds the kernel matrix K∈ℝ^{N×N}.
Treat K as the coboundary operator Δ of a sheaf whose node dimensions are the kernel weights.
Reward prediction for a candidate action a is
    μ_a(ϕ)=k_a^T (K+λI)^{-1} r_a,
where k_a is the vector of kernels between ϕ and contexts that previously selected a,
and r_a the corresponding reward vector.
Uncertainty (sheaf variance) is
    σ_a²(ϕ)=k(ϕ,ϕ)−k_a^T (K+λI)^{-1} k_a.
The hybrid policy selects a* = argmax_a μ_a(ϕ)+α·σ_a(ϕ), i.e. a LinUCB/Thompson‑style UCB built on the RBF surrogate.

The implementation below provides:
- feature extraction (text vibes + morphology indices),
- kernel‑based reward prediction with confidence bounds,
- online update of the context‑reward repository,
- three public functions demonstrating the hybrid operation.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

# -------------------- Parent A – Text vibe extraction -------------------- #

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float = 1.0  # unused in this hybrid but kept for compatibility

def extract_operator_vibes(text: str) -> Dict[str, float]:
    # placeholder – in practice this would parse the text
    return {"operator_visceral_ratio": 0.5, "operator_tech_ratio": 0.3}

def extract_psyche_vibes(text: str) -> Dict[str, float]:
    return {"psyche_forensic_shield_ratio": 0.2, "psyche_poetic_entropy": 0.1}

def extract_resilience_vibes(text: str) -> Dict[str, float]:
    return {"resilience_bureaucratic_weaponization_index": 0.4,
            "resilience_resource_exhaustion": 0.35}

def extract_rainmaker_vibes(text: str) -> Dict[str, float]:
    return {"rainmaker_influence_score": 0.6, "rainmaker_adaptability": 0.45}

def extract_operator_telemetry(text: str) -> Dict[str, float]:
    return {"operator_latency_ms": 120.0, "operator_throughput": 0.78}

def extract_full_features(text: str) -> Dict[str, float]:
    feats = {}
    feats.update(extract_operator_vibes(text))
    feats.update(extract_psyche_vibes(text))
    feats.update(extract_resilience_vibes(text))
    feats.update(extract_rainmaker_vibes(text))
    feats.update(extract_operator_telemetry(text))
    return feats

# -------------------- Parent B – Morphology & RBF utilities -------------------- #

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

def righting_time_index(m: Morphology,
                        b: float = 1.0 / 3.0,
                        k: float = 0.35,
                        neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and lever must be positive")
    # simplified physics‑based proxy
    inertia = (m.mass *
               (m.length ** 2 + m.width ** 2 + m.height ** 2) / 12.0)
    return (inertia ** b) * k / neck_lever

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

# -------------------- Hybrid Core -------------------- #

class HybridRBFContextualBandit:
    """
    Maintains a growing repository of (context, action, reward) triples.
    Uses an RBF kernel as a sheaf coboundary operator to produce
    LinUCB‑style upper‑confidence bounds for action selection.
    """
    def __init__(self,
                 epsilon: float = 0.5,
                 lambda_reg: float = 1e-3,
                 alpha: float = 1.0):
        self.epsilon = epsilon          # kernel bandwidth
        self.lambda_reg = lambda_reg    # regularisation for (K+λI)⁻¹
        self.alpha = alpha              # exploration weight
        # repository
        self.contexts: List[np.ndarray] = []          # shape (N, d)
        self.actions: List[str] = []                  # length N
        self.rewards: List[float] = []                # length N
        self._cached_K_inv: np.ndarray = None         # cached (K+λI)⁻¹
        self._dirty = True                            # flag to recompute inverse

    # ---------- kernel utilities ---------- #
    def _kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        return gaussian(np.linalg.norm(x - y), self.epsilon)

    def _kernel_vector(self, x: np.ndarray) -> np.ndarray:
        """k(x) = [k(x, φ_i)]_i"""
        if not self.contexts:
            return np.array([], dtype=float)
        stacked = np.stack(self.contexts)          # (N, d)
        dists = np.linalg.norm(stacked - x, axis=1)
        return np.vectorize(lambda r: gaussian(r, self.epsilon))(dists)

    def _ensure_inverse(self) -> None:
        """Recompute (K+λI)⁻¹ when repository changes."""
        if not self._dirty:
            return
        if not self.contexts:
            self._cached_K_inv = np.empty((0, 0))
            self._dirty = False
            return
        K = np.empty((len(self.contexts), len(self.contexts)), dtype=float)
        stacked = np.stack(self.contexts)
        for i in range(len(self.contexts)):
            diff = stacked - stacked[i]
            dists = np.linalg.norm(diff, axis=1)
            K[i, :] = np.vectorize(lambda r: gaussian(r, self.epsilon))(dists)
        # regularised inverse
        reg = self.lambda_reg * np.eye(K.shape[0])
        self._cached_K_inv = np.linalg.inv(K + reg)
        self._dirty = False

    # ---------- public API ---------- #
    def predict(self, context_vec: np.ndarray) -> Tuple[str, float]:
        """
        Returns (selected_action, upper_confidence_bound).
        If no prior data exists, falls back to a random action from a
        predefined small catalogue.
        """
        if not self.contexts:
            # no knowledge yet – random exploration
            dummy_actions = ["A", "B", "C"]
            return random.choice(dummy_actions), 0.0

        self._ensure_inverse()
        k_vec = self._kernel_vector(context_vec)           # (N,)

        # Group indices by action
        action_to_indices: Dict[str, List[int]] = {}
        for idx, a in enumerate(self.actions):
            action_to_indices.setdefault(a, []).append(idx)

        best_ucb = -float('inf')
        best_action = None

        for a, idxs in action_to_indices.items():
            k_a = k_vec[idxs]                              # (n_a,)
            r_a = np.array([self.rewards[i] for i in idxs])  # (n_a,)

            # μ_a = k_a^T (K+λI)^{-1}_aa r_a  (sub‑matrix extraction)
            K_inv_aa = self._cached_K_inv[np.ix_(idxs, idxs)]
            mu = k_a @ K_inv_aa @ r_a

            # σ_a² = k(x,x) - k_a^T (K+λI)^{-1}_aa k_a
            k_xx = 1.0  # gaussian(0)=1
            sigma2 = k_xx - k_a @ K_inv_aa @ k_a
            sigma = math.sqrt(max(sigma2, 0.0))

            ucb = mu + self.alpha * sigma
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = a

        return best_action, best_ucb

    def update(self, update: BanditUpdate, context_vec: np.ndarray) -> None:
        """
        Incorporates a new (context, action, reward) observation.
        """
        self.contexts.append(context_vec.copy())
        self.actions.append(update.action_id)
        self.rewards.append(update.reward)
        self._dirty = True

    # ---------- helper to build full feature vector ---------- #
    @staticmethod
    def build_feature_vector(text: str,
                             morphology: Morphology) -> np.ndarray:
        """
        Concatenates textual vibes with three morphology‑derived scalars.
        Returns a 1‑D numpy array suitable for kernel computation.
        """
        text_feats = extract_full_features(text)          # dict of floats
        # convert dict to ordered array (sorted keys for determinism)
        text_array = np.array([text_feats[k] for k in sorted(text_feats.keys())],
                              dtype=float)

        # morphology features
        sph = sphericity_index(morphology.length,
                               morphology.width,
                               morphology.height)
        flt = flatness_index(morphology.length,
                             morphology.width,
                             morphology.height)
        rgt = righting_time_index(morphology)

        morph_array = np.array([sph, flt, rgt], dtype=float)

        return np.concatenate([text_array, morph_array])

# -------------------- Demonstration Functions -------------------- #

def select_action(bandit: HybridRBFContextualBandit,
                  text: str,
                  morphology: Morphology) -> str:
    """
    High‑level wrapper that builds the feature vector, queries the bandit,
    and returns the chosen action identifier.
    """
    ctx = bandit.build_feature_vector(text, morphology)
    action, _ = bandit.predict(ctx)
    return action

def record_outcome(bandit: HybridRBFContextualBandit,
                   text: str,
                   morphology: Morphology,
                   chosen_action: str,
                   reward: float) -> None:
    """
    Records the observed reward for a previously selected action.
    """
    ctx = bandit.build_feature_vector(text, morphology)
    upd = BanditUpdate(context_id="ctx",
                       action_id=chosen_action,
                       reward=reward)
    bandit.update(upd, ctx)

def batch_training(bandit: HybridRBFContextualBandit,
                   data: List[Tuple[str, Morphology, str, float]]) -> None:
    """
    Accepts a list of (text, morphology, action, reward) tuples and
    incrementally trains the hybrid model.
    """
    for text, morph, act, rew in data:
        ctx = bandit.build_feature_vector(text, morph)
        upd = BanditUpdate(context_id="batch", action_id=act, reward=rew)
        bandit.update(upd, ctx)

# -------------------- Smoke Test -------------------- #

if __name__ == "__main__":
    # instantiate the hybrid bandit
    hybrid = HybridRBFContextualBandit(epsilon=0.7, lambda_reg=1e-2, alpha=0.5)

    # dummy morphology
    morph = Morphology(length=2.0, width=1.5, height=0.8, mass=3.4)

    # first decision (no history yet)
    txt1 = "Operator engaged in high‑risk maneuver, psyche stable."
    act1 = select_action(hybrid, txt1, morph)
    print(f"Selected action (first round): {act1}")

    # simulate a reward and record it
    reward1 = random.uniform(0, 1)
    record_outcome(hybrid, txt1, morph, act1, reward1)
    print(f"Recorded reward {reward1:.3f} for action {act1}")

    # second round with a different text
    txt2 = "Rainmaker influence surged, resilience metrics improving."
    act2 = select_action(hybrid, txt2, morph)
    print(f"Selected action (second round): {act2}")

    # batch training example
    batch = [
        ("Operator latency spikes", morph, "A", 0.2),
        ("Psychic shield active", morph, "B", 0.8),
        ("Resource exhaustion critical", morph, "C", 0.1),
    ]
    batch_training(hybrid, batch)
    # final query to ensure no exceptions
    final_action = select_action(hybrid, "System nominal, all metrics green.", morph)
    print(f"Final selected action after batch training: {final_action}")