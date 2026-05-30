# DARWIN HAMMER — match 2551, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_cockpi_m582_s4.py (gen4)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hard_truth_ma_m93_s0.py (gen3)
# born: 2026-05-29T23:42:50Z

"""Hybrid Bandit‑RBF Model
Parents:
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_cockpi_m582_s4.py (Bandit + Schoolfield temperature model)
- hybrid_hybrid_rbf_surrogate_hybrid_hard_truth_ma_m93_s0.py (RBF surrogate for stylometric similarity)

Mathematical bridge:
The RBF surrogate provides a similarity weight ϕ(i,j)=exp(−ε²‖v_i−v_j‖²) between
feature vectors v_i extracted from textual descriptions of actions.
The Bandit’s propensity and confidence bound are modulated by this weight and by
the temperature‑dependent developmental rate ρ(T) from the Schoolfield equation:

    adjusted_reward_i = expected_reward_i · ρ(T) · Σ_j ϕ(i,j)·propensity_j

Thus the two topologies are fused through a matrix‑vector product
S·p where S_ij=ϕ(i,j) (RBF similarity matrix) and p is the propensity vector,
scaled by the temperature factor ρ(T). The resulting value feeds the
Bandit’s update and selection logic.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures from Parent A
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


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


# ----------------------------------------------------------------------
# Functions from Parent B
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rbf_similarity(v1: np.ndarray, v2: np.ndarray, epsilon: float = 1.0) -> float:
    """RBF similarity based on Euclidean distance."""
    return gaussian(euclidean(v1.tolist(), v2.tolist()), epsilon)


def text_to_feature_vector(text: str) -> np.ndarray:
    """Simple stylometric feature vector: counts of FUNCTION_CATS tokens."""
    tokens = [t.lower().strip(".,!?;:") for t in text.split()]
    counts = []
    for cat in FUNCTION_CATS.values():
        counts.append(sum(1 for t in tokens if t in cat))
    return np.array(counts, dtype=float)


# ----------------------------------------------------------------------
# Hybrid Model
# ----------------------------------------------------------------------
class HybridBanditRBFModel:
    """Bandit model whose reward estimation is modulated by temperature
    dependent developmental rate and RBF similarity of action texts."""

    def __init__(self, epsilon: float = 1.0):
        self.policy: Dict[str, Tuple[float, int]] = {}  # action_id -> (total_reward, count)
        self.epsilon = epsilon
        self.action_features: Dict[str, np.ndarray] = {}  # action_id -> feature vector

    # ----- Bandit core (Parent A) -----
    def reset_policy(self) -> None:
        self.policy.clear()
        self.action_features.clear()

    def update_policy(self, updates: List[BanditUpdate]) -> None:
        for u in updates:
            total, cnt = self.policy.get(u.action_id, (0.0, 0))
            self.policy[u.action_id] = (total + float(u.reward), cnt + 1)

    def _average_reward(self, action_id: str) -> float:
        total, cnt = self.policy.get(action_id, (0.0, 0))
        return total / cnt if cnt else 0.0

    # ----- Schoolfield temperature model (Parent A) -----
    @staticmethod
    def c_to_k(celsius: float) -> float:
        return celsius + 273.15

    @staticmethod
    def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
        if temp_k <= 0 or params.rho_25 < 0:
            raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
        numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
            (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
        )
        low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
        high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
        return numerator / (1.0 + low + high)

    # ----- RBF similarity bridge (Parent B) -----
    def register_action_features(self, actions: List[BanditAction], texts: Dict[str, str]) -> None:
        """Map each action_id to a stylometric feature vector derived from its text."""
        for a in actions:
            txt = texts.get(a.action_id, "")
            self.action_features[a.action_id] = text_to_feature_vector(txt)

    def similarity_matrix(self, action_ids: List[str]) -> np.ndarray:
        """Compute symmetric similarity matrix S where S_ij = ϕ(i,j)."""
        n = len(action_ids)
        S = np.zeros((n, n), dtype=float)
        for i, aid_i in enumerate(action_ids):
            vi = self.action_features[aid_i]
            for j, aid_j in enumerate(action_ids):
                if j < i:
                    S[i, j] = S[j, i]
                else:
                    vj = self.action_features[aid_j]
                    S[i, j] = rbf_similarity(vi, vj, self.epsilon)
        return S

    # ----- Hybrid reward computation -----
    def adjusted_reward(self, action: BanditAction, temp_c: float, all_actions: List[BanditAction]) -> float:
        """Compute reward adjusted by temperature and RBF‑weighted propensities."""
        temp_k = self.c_to_k(temp_c)
        rho_T = self.developmental_rate(temp_k)

        # Build propensity vector aligned with all_actions order
        propensities = np.array([a.propensity for a in all_actions], dtype=float)

        # Similarity matrix
        S = self.similarity_matrix([a.action_id for a in all_actions])

        # Weighted sum of propensities for the target action
        idx = next(i for i, a in enumerate(all_actions) if a.action_id == action.action_id)
        weighted_prop = float(S[idx, :].dot(propensities))

        # Final adjusted reward
        return action.expected_reward * rho_T * weighted_prop

    # ----- Action selection using upper confidence bound -----
    def select_action(self, candidates: List[BanditAction], temp_c: float) -> BanditAction:
        """UCB‑style selection where the confidence term is scaled by similarity."""
        if not candidates:
            raise ValueError("candidate list empty")
        # Pre‑compute similarity matrix once
        S = self.similarity_matrix([a.action_id for a in candidates])
        temp_k = self.c_to_k(temp_c)
        rho_T = self.developmental_rate(temp_k)

        best_score = -float("inf")
        best_action = None
        for i, a in enumerate(candidates):
            avg_reward = self._average_reward(a.action_id)
            # similarity‑scaled exploration term
            sim_factor = S[i, :].mean()
            ucb = avg_reward + a.confidence_bound * math.sqrt(sim_factor) * rho_T
            if ucb > best_score:
                best_score = ucb
                best_action = a
        return best_action


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def compute_adjusted_rewards(model: HybridBanditRBFModel,
                             actions: List[BanditAction],
                             temp_c: float) -> Dict[str, float]:
    """Return a dict mapping action_id to its temperature‑ and similarity‑adjusted reward."""
    return {
        a.action_id: model.adjusted_reward(a, temp_c, actions)
        for a in actions
    }


def batch_update_with_texts(model: HybridBanditRBFModel,
                            updates: List[BanditUpdate],
                            actions: List[BanditAction],
                            texts: Dict[str, str]) -> None:
    """
    Register feature vectors, update the bandit policy with the supplied updates,
    and recompute internal structures.
    """
    model.register_action_features(actions, texts)
    model.update_policy(updates)


def run_selection_scenario(model: HybridBanditRBFModel,
                           actions: List[BanditAction],
                           temp_c: float) -> BanditAction:
    """Select an action using the hybrid UCB rule."""
    return model.select_action(actions, temp_c)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny set of actions with dummy texts
    actions = [
        BanditAction("A1", propensity=0.2, expected_reward=1.0, confidence_bound=0.5, algorithm="bandit"),
        BanditAction("A2", propensity=0.5, expected_reward=0.8, confidence_bound=0.4, algorithm="bandit"),
        BanditAction("A3", propensity=0.3, expected_reward=1.2, confidence_bound=0.6, algorithm="bandit"),
    ]

    texts = {
        "A1": "I think we should go to the market because it is close.",
        "A2": "You cannot be serious about that plan; it is impossible.",
        "A3": "The quick brown fox jumps over the lazy dog."
    }

    updates = [
        BanditUpdate(context_id="C1", action_id="A1", reward=1.0, propensity=0.2),
        BanditUpdate(context_id="C1", action_id="A2", reward=0.0, propensity=0.5),
        BanditUpdate(context_id="C2", action_id="A3", reward=1.0, propensity=0.3),
    ]

    model = HybridBanditRBFModel(epsilon=0.8)

    # Register features and apply updates
    batch_update_with_texts(model, updates, actions, texts)

    # Compute adjusted rewards at 25 °C
    adj = compute_adjusted_rewards(model, actions, temp_c=25.0)
    print("Adjusted rewards (°C=25):")
    for aid, val in adj.items():
        print(f"  {aid}: {val:.4f}")

    # Perform a selection at 30 °C
    chosen = run_selection_scenario(model, actions, temp_c=30.0)
    print(f"\nChosen action at 30 °C: {chosen.action_id}")

    # Verify that policy statistics are sensible
    print("\nPolicy statistics:")
    for aid in model.policy:
        total, cnt = model.policy[aid]
        print(f"  {aid}: total_reward={total}, count={cnt}, avg={total/cnt:.3f}")