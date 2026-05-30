# DARWIN HAMMER — match 1348, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_decisi_m260_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hdc_se_m1184_s1.py (gen5)
# born: 2026-05-29T23:35:28Z

"""Hybrid Lens‑Decision & Bandit‑RBF‑HDC‑WTA Fusion

Parents:
- **Algorithm A** (`hybrid_hybrid_hybrid_ternar_hybrid_hybrid_decisi_m260_s0.py`):
  Provides textual evidence extraction via regex patterns and maps these
  counts to a feature vector representing planning, delay, support, boundary,
  outcome and evidence signals.
- **Algorithm B** (`hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hdc_se_m1184_s1.py`):
  Implements a contextual LinUCB bandit whose expected reward is supplied by
  a radial‑basis‑function (RBF) surrogate, and augments the decision value
  with hyperdimensional computing (HDC) and a sparse winner‑take‑all (WTA)
  hypervector.

Mathematical Bridge
-------------------
The feature vector produced by *Algorithm A* is used as the **context**
for the LinUCB bandit of *Algorithm B*.  
The bandit’s expected reward `\hat r(c,a)` is replaced by an RBF surrogate
`g(c,a)` that maps the concatenated vector `[c, a_one_hot]` to a scalar.
The scalar `g` together with the **evidence strength** (sum of evidence‑related
regex matches) are encoded into two high‑dimensional bipolar vectors:


hv_morph = morphology_hv(evidence_strength)               # HDC encoding
hv_wta   = sparse_wta_hv([g])                               # WTA over the surrogate score


A single priority score is obtained by the dot product of these hypervectors,
which simultaneously captures (i) hyperdimensional similarity of the evidence
profile and (ii) the saliency of the surrogate‑predicted reward.

The module therefore fuses the textual‑analysis pipeline of A with the
contextual bandit‑RBF‑HDC‑WTA decision engine of B into a unified system.

"""

import math
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Sequence, Any

import numpy as np

# ----------------------------------------------------------------------
# Regex feature extraction (Algorithm A)
# ----------------------------------------------------------------------
import re
from datetime import datetime, timezone

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succ(?:eed|ess)|finished|closed)\b",
    re.I,
)

FEATURE_REGEXES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
}


def extract_features(text: str) -> Tuple[np.ndarray, float]:
    """
    Scan *text* with the regexes defined above.
    Returns:
        - a dense feature vector (order: evidence, planning, delay, support,
          boundary, outcome) containing the raw match counts,
        - a scalar *evidence_strength* defined as the evidence count
          multiplied by a small logarithmic factor to keep it in a comparable
          numeric range.
    """
    counts = []
    for name in ["evidence", "planning", "delay", "support", "boundary", "outcome"]:
        cnt = len(FEATURE_REGEXES[name].findall(text))
        counts.append(cnt)
    vec = np.array(counts, dtype=float)
    evidence_strength = vec[0] * (1.0 + math.log1p(vec[0]))
    return vec, evidence_strength


# ----------------------------------------------------------------------
# RBF surrogate (Algorithm B)
# ----------------------------------------------------------------------
def rbf_kernel(x: np.ndarray, c: np.ndarray, sigma: float) -> float:
    """Gaussian RBF kernel between vectors x and centre c."""
    diff = x - c
    return math.exp(-np.dot(diff, diff) / (2 * sigma ** 2))


def rbf_surrogate(
    context: np.ndarray,
    action_one_hot: np.ndarray,
    centres: List[np.ndarray],
    weights: List[float],
    sigma: float,
) -> float:
    """
    Predict a reward for a (context, action) pair using a weighted sum of
    RBF kernels centred at *centres* with corresponding *weights*.
    """
    x = np.concatenate([context, action_one_hot])
    return sum(w * rbf_kernel(x, c, sigma) for w, c in zip(weights, centres))


# ----------------------------------------------------------------------
# LinUCB Bandit core (Algorithm B)
# ----------------------------------------------------------------------
@dataclass
class LinUCBAction:
    action_id: str
    a_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    b_vector: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dim: int = 0

    def ensure_dim(self, d: int):
        """Resize a_matrix and b_vector lazily to dimension d."""
        if self.dim >= d:
            return
        if self.dim == 0:
            self.a_matrix = np.identity(d)
            self.b_vector = np.zeros(d)
        else:
            # enlarge existing matrices
            new_a = np.identity(d)
            new_a[: self.dim, : self.dim] = self.a_matrix
            self.a_matrix = new_a
            new_b = np.zeros(d)
            new_b[: self.dim] = self.b_vector
            self.b_vector = new_b
        self.dim = d

    def theta(self) -> np.ndarray:
        """Regularized least‑squares estimate of the reward parameters."""
        return np.linalg.solve(self.a_matrix, self.b_vector)


class LinUCBBandit:
    """
    Contextual bandit with LinUCB confidence bounds.
    Expected reward is supplied externally (here by the RBF surrogate).
    """

    def __init__(self, alpha: float, action_ids: List[str], context_dim: int):
        self.alpha = alpha
        self.actions: Dict[str, LinUCBAction] = {
            aid: LinUCBAction(aid) for aid in action_ids
        }
        self.context_dim = context_dim
        self.action_dim = len(action_ids)

        # initialise each action's matrices to the required dimension
        for act in self.actions.values():
            act.ensure_dim(context_dim + self.action_dim)

    def select_action(self, context: np.ndarray) -> str:
        """Return the action id with highest Upper Confidence Bound."""
        ucb_vals = {}
        for aid, act in self.actions.items():
            theta = act.theta()
            x = np.concatenate([context, self._one_hot(aid)])
            mean = float(np.dot(theta, x))
            var = float(np.sqrt(np.dot(x, np.linalg.solve(act.a_matrix, x))))
            ucb = mean + self.alpha * var
            ucb_vals[aid] = ucb
        return max(ucb_vals, key=ucb_vals.get)

    def update(
        self, context: np.ndarray, action_id: str, reward: float
    ) -> None:
        """Standard LinUCB update with observed *reward*."""
        act = self.actions[action_id]
        act.ensure_dim(self.context_dim + self.action_dim)
        x = np.concatenate([context, self._one_hot(action_id)])
        act.a_matrix += np.outer(x, x)
        act.b_vector += reward * x

    def _one_hot(self, action_id: str) -> np.ndarray:
        vec = np.zeros(self.action_dim)
        idx = list(self.actions.keys()).index(action_id)
        vec[idx] = 1.0
        return vec


# ----------------------------------------------------------------------
# Hyperdimensional encoding (Algorithm B)
# ----------------------------------------------------------------------
HV_DIM = 10000  # dimensionality of bipolar hypervectors


def morphology_hv(scalar: float) -> np.ndarray:
    """
    Encode a scalar (evidence_strength) into a bipolar hypervector.
    A deterministic pseudo‑random seed derived from the scalar ensures
    reproducibility.
    """
    rng = np.random.default_rng(int(scalar * 1e6) % (2 ** 32))
    bits = rng.integers(0, 2, size=HV_DIM, endpoint=False)
    hv = np.where(bits == 0, -1, 1).astype(np.int8)
    return hv


def sparse_wta_hv(scores: List[float], k: int = 1) -> np.ndarray:
    """
    Convert *scores* into a sparse WTA hypervector.
    The top‑k scores receive +1, all others -1.
    """
    hv = np.full(HV_DIM, -1, dtype=np.int8)
    if not scores:
        return hv
    # map each score to a random index; deterministic via hash
    indices = []
    for s in scores:
        rng = np.random.default_rng(hash(s) & 0xFFFFFFFF)
        idx = rng.integers(0, HV_DIM)
        indices.append((s, idx))
    # select top‑k by score
    top = sorted(indices, key=lambda t: t[0], reverse=True)[:k]
    for _, idx in top:
        hv[idx] = 1
    return hv


def hybrid_priority(
    context_vec: np.ndarray,
    evidence_strength: float,
    action_id: str,
    bandit: LinUCBBandit,
    rbf_params: Tuple[List[np.ndarray], List[float], float],
) -> float:
    """
    Compute a unified priority value:
      1. Predict surrogate reward `g` for (context, action).
      2. Encode evidence_strength via `morphology_hv`.
      3. Encode `g` via `sparse_wta_hv`.
      4. Return the normalized dot product between the two hypervectors.
    """
    centres, weights, sigma = rbf_params
    action_one_hot = bandit._one_hot(action_id)
    g = rbf_surrogate(context_vec, action_one_hot, centres, weights, sigma)

    hv_morph = morphology_hv(evidence_strength)
    hv_wta = sparse_wta_hv([g])

    # dot product in bipolar space ranges from -HV_DIM to +HV_DIM
    dot = int(np.dot(hv_morph, hv_wta))
    priority = (dot + HV_DIM) / (2 * HV_DIM)  # normalize to [0,1]
    return priority


# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_decision(text: str, bandit: LinUCBBandit, rbf_params: Tuple[List[np.ndarray], List[float], float]) -> Tuple[str, float]:
    """
    End‑to‑end pipeline:
      * Extract textual features.
      * Use the LinUCB bandit to pick an action.
      * Compute the fused priority.
    Returns the selected action id and its priority score.
    """
    ctx_vec, evidence_strength = extract_features(text)
    action_id = bandit.select_action(ctx_vec)
    priority = hybrid_priority(ctx_vec, evidence_strength, action_id, bandit, rbf_params)
    return action_id, priority


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic setup
    ACTION_IDS = ["accept", "reject", "review"]
    CONTEXT_DIM = 6  # six regex‑based features

    # Initialise bandit
    bandit = LinUCBBandit(alpha=1.0, action_ids=ACTION_IDS, context_dim=CONTEXT_DIM)

    # Create dummy RBF centres and weights
    rng = np.random.default_rng(42)
    num_centres = 5
    centres = [rng.normal(size=CONTEXT_DIM + len(ACTION_IDS)) for _ in range(num_centres)]
    weights = [rng.random() for _ in range(num_centres)]
    sigma = 1.5
    rbf_params = (centres, weights, sigma)

    # Example text
    sample = """
    The audit confirmed the source of the data. A detailed checklist was prepared,
    but the deployment will pause until the next review. Support from the team
    is available, and the boundary conditions have been respected.
    """

    action, prio = hybrid_decision(sample, bandit, rbf_params)
    print(f"Selected action: {action}")
    print(f"Hybrid priority: {prio:.4f}")

    # Simulate a reward and update the bandit
    reward = 1.0 if action == "accept" else 0.0
    ctx_vec, _ = extract_features(sample)
    bandit.update(ctx_vec, action, reward)

    # Run a second decision to show learning effect
    action2, prio2 = hybrid_decision(sample, bandit, rbf_params)
    print(f"After update -> action: {action2}, priority: {prio2:.4f}")