# DARWIN HAMMER — match 1203, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m468_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hard_t_ternary_router_m906_s1.py (gen4)
# born: 2026-05-29T23:34:26Z

"""Hybrid Algorithm: Regret‑Weighted Gini‑Modulated Hoeffding Tree + Bilinear Text‑Model Fusion

Parents
-------
- **Parent A** – hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m468_s1.py  
  Provides a regret‑weighted Hoeffding tree whose split decision uses a Gini‑weighted
  exploration term ε = base_ε·(1+λ_g·G) and a temperature‑dependent developmental rate ρ(T).

- **Parent B** – hybrid_hybrid_hybrid_hard_t_ternary_router_m906_s1.py  
  Generates high‑dimensional text feature vectors and projects them onto a low‑dimensional
  model space via a bilinear form; the projected representation drives routing decisions.

Mathematical Bridge
-------------------
Let  

* **x ∈ ℝⁿ** be the high‑dimensional text feature vector (Parent B).  
* **W ∈ ℝⁿˣᵐ** be a learned bilinear weight matrix; the projected model vector is  

      z = xᵀ W   ∈ ℝᵐ .

* The components of **z** are interpreted as *expected values* of **m** candidate actions.  
  From these values we compute the Gini coefficient **G(z)**, which quantifies inequality
  among the action expectations (Parent A).

* A temperature‑dependent developmental rate ρ(T) is obtained from a contextual
  dictionary (e.g., a “temperature” key).  

* The exploration term for each action becomes  

      ε = base_ε · (1 + λ_g·G(z))

  and the split‑gain estimate used by the Hoeffding tree is  

      gain_gap = ρ(T) · (max(z) – ε).

* Finally, a bandit‑style confidence bound for action *i* is  

      CB_i = z_i + confidence_i – ε·ρ(T),

  and the routing decision selects the action with the highest **CB_i**.

The code below implements this fused topology, exposing three core functions that
demonstrate the hybrid operation: feature extraction, bilinear projection with
Gini‑modulated confidence, and Hoeffding‑tree‑style split decision. A small smoke
test runs the pipeline on synthetic data."""

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action used in the regret‑weighted Hoeffding tree."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class BanditAction:
    """Bandit arm with propensity‑adjusted confidence bound."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


# ----------------------------------------------------------------------
# Parent‑B: Text feature extraction and bilinear projection
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
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def _tokenize(text: str) -> List[str]:
    """Very light tokeniser – splits on whitespace and strips punctuation."""
    punct = set("!?;:,.—-()[]{}\"'`/\\|@#$%^")
    tokens = []
    for raw in text.lower().split():
        token = "".join(ch for ch in raw if ch not in punct)
        if token:
            tokens.append(token)
    return tokens


def calculate_text_features(text: str) -> np.ndarray:
    """
    Produce a high‑dimensional feature vector whose components are the counts of
    words belonging to each FUNCTION_CATS category, plus a total‑word count.
    The resulting dimension is ``len(FUNCTION_CATS) + 1``.
    """
    tokens = _tokenize(text)
    total = len(tokens)
    counts = [total]
    for cat_words in FUNCTION_CATS.values():
        counts.append(sum(1 for t in tokens if t in cat_words))
    return np.array(counts, dtype=float)


def bilinear_projection(features: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    """
    Bilinear form ``z = xᵀ W`` where ``x`` is the high‑dimensional feature vector
    and ``W`` is a learned matrix.  The result ``z`` lives in the low‑dimensional
    model/action space and is interpreted as expected values of candidate actions.
    """
    if features.shape[0] != weight_matrix.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: features {features.shape} vs weight_matrix {weight_matrix.shape}"
        )
    return features @ weight_matrix


# ----------------------------------------------------------------------
# Parent‑A: Gini coefficient, developmental rate, and regret‑weighted confidence
# ----------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1‑D array of non‑negative numbers.
    Formula: G = (∑_i ∑_j |x_i – x_j|) / (2·n·∑_i x_i)
    """
    if values.ndim != 1:
        raise ValueError("Gini coefficient expects a 1‑D array")
    if np.any(values < 0):
        raise ValueError("Gini coefficient requires non‑negative values")
    n = values.size
    if n == 0:
        return 0.0
    sorted_vals = np.sort(values)
    cumulative = np.cumsum(sorted_vals)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_vals) / n
    return gini


def developmental_rate(context: Dict[str, Any]) -> float:
    """
    Temperature‑dependent scaling factor ρ(T).  If the context contains a numeric
    ``temperature`` entry we map it to a rate in (0, 1] via a simple exponential
    decay; otherwise a default of 1.0 is used.
    """
    temp = context.get("temperature")
    if isinstance(temp, (int, float)):
        # Clamp temperature to a reasonable range to avoid overflow.
        temp = max(min(float(temp), 100.0), 0.0)
        # Higher temperature → slower development (smaller ρ).
        return math.exp(-temp / 20.0)
    return 1.0


def regret_weighted_confidence(
    action: BanditAction,
    gini: float,
    base_eps: float,
    lambda_g: float,
    rho: float,
) -> float:
    """
    Compute the hybrid confidence bound for a single action:

        ε   = base_eps * (1 + λ_g * G)
        CB  = expected_reward + confidence_bound - ε * ρ

    The term ``ε`` inflates exploration when the action‑value distribution is
    highly unequal (large Gini).  The developmental rate ρ further modulates
    the exploration magnitude.
    """
    epsilon = base_eps * (1.0 + lambda_g * gini)
    return action.expected_reward + action.confidence_bound - epsilon * rho


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_hybrid_split_gain(
    projected_vals: np.ndarray,
    base_eps: float,
    lambda_g: float,
    context: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Implements the Hoeffding‑tree split decision used in Parent A, but with the
    projected model values ``projected_vals`` (from Parent B) as the gain
    candidates.

    Returns
    -------
    gain_gap : float
        The scaled gain gap ``ρ·(max_gain – ε)``.
    epsilon  : float
        The Gini‑modulated exploration term used in the calculation.
    """
    if projected_vals.size == 0:
        raise ValueError("No projected values to evaluate")
    gini = gini_coefficient(projected_vals)
    rho = developmental_rate(context)
    epsilon = base_eps * (1.0 + lambda_g * gini)
    max_gain = float(np.max(projected_vals))
    gain_gap = rho * (max_gain - epsilon)
    return gain_gap, epsilon


def select_action_via_routing(
    projected_vals: np.ndarray,
    bandit_actions: List[BanditAction],
    base_eps: float,
    lambda_g: float,
    context: Dict[str, Any],
) -> BanditAction:
    """
    Fuse the bilinear projection with the bandit confidence update.
    The ``projected_vals`` serve as the ``expected_reward`` component for each
    action (matched by order).  The action with the highest hybrid confidence
    bound is returned.
    """
    if len(projected_vals) != len(bandit_actions):
        raise ValueError(
            "Length mismatch between projected values and bandit actions"
        )
    rho = developmental_rate(context)
    gini = gini_coefficient(projected_vals)
    confidences = []
    for val, act in zip(projected_vals, bandit_actions):
        # Override the expected reward with the projected value.
        hybrid_action = BanditAction(
            action_id=act.action_id,
            propensity=act.propensity,
            expected_reward=val,
            confidence_bound=act.confidence_bound,
            algorithm=act.algorithm,
        )
        cb = regret_weighted_confidence(
            hybrid_action, gini, base_eps, lambda_g, rho
        )
        confidences.append(cb)
    best_idx = int(np.argmax(confidences))
    return bandit_actions[best_idx]


def hybrid_pipeline(
    text: str,
    weight_matrix: np.ndarray,
    bandit_actions: List[BanditAction],
    base_eps: float = 0.1,
    lambda_g: float = 0.5,
    context: Dict[str, Any] | None = None,
) -> Tuple[BanditAction, float, float]:
    """
    End‑to‑end hybrid operation:

    1. Extract high‑dimensional text features.
    2. Project them onto the model/action space via a bilinear form.
    3. Compute the Hoeffding‑tree split gain (for diagnostic purposes).
    4. Route to the best bandit action using the Gini‑modulated confidence.

    Returns
    -------
    chosen_action : BanditAction
        The action selected by the hybrid router.
    gain_gap      : float
        Scaled gain gap from the Hoeffding‑tree component.
    epsilon       : float
        Gini‑weighted exploration term.
    """
    if context is None:
        context = {}
    # 1. Feature extraction
    feats = calculate_text_features(text)  # shape (d,)
    # 2. Bilinear projection
    proj = bilinear_projection(feats, weight_matrix)  # shape (m,)
    # 3. Split‑gain computation
    gain_gap, epsilon = compute_hybrid_split_gain(
        proj, base_eps, lambda_g, context
    )
    # 4. Routing decision
    chosen = select_action_via_routing(
        proj, bandit_actions, base_eps, lambda_g, context
    )
    return chosen, gain_gap, epsilon


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic context with a temperature value.
    ctx = {"temperature": 25.0}

    # Example text.
    sample_text = (
        "The quick brown fox jumps over the lazy dog while I contemplate "
        "the meaning of existence and the role of auxiliary verbs."
    )

    # Create a random weight matrix.
    # Feature dimension = len(FUNCTION_CATS) + 1 (total word count)
    feat_dim = len(FUNCTION_CATS) + 1
    # Number of candidate actions (e.g., 4)
    num_actions = 4
    rng = np.random.default_rng(seed=42)
    W = rng.normal(loc=0.0, scale=1.0, size=(feat_dim, num_actions))

    # Define a set of bandit actions with placeholder confidence bounds.
    bandit_actions = [
        BanditAction(
            action_id=f"act_{i}",
            propensity=rng.random(),
            expected_reward=0.0,  # will be overwritten by projection
            confidence_bound=rng.random(),
        )
        for i in range(num_actions)
    ]

    # Run the hybrid pipeline.
    chosen_action, gain_gap, epsilon = hybrid_pipeline(
        text=sample_text,
        weight_matrix=W,
        bandit_actions=bandit_actions,
        base_eps=0.05,
        lambda_g=0.8,
        context=ctx,
    )

    print("Chosen action:", chosen_action)
    print(f"Gain gap: {gain_gap:.4f}")
    print(f"Epsilon (exploration term): {epsilon:.4f}")
    # Ensure no exception was raised and outputs are sensible.
    sys.exit(0)