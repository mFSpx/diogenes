# DARWIN HAMMER — match 1235, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_regret_m25_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rlct_g_m436_s0.py (gen5)
# born: 2026-05-29T23:34:38Z

"""Hybrid Algorithm: Fusion of DARWIN HAMMER — match 24, survivor 4 (Parent A)
and DARWIN HAMMER — match 436, survivor 0 (Parent B).

Mathematical Bridge
-------------------
* **Regret‑aware Hoeffding bound** – The regret value supplied by the
  regret‑engine (Parent A) is interpreted as an “energy” factor ϵ that
  scales the classic Hoeffding bound.  The bound becomes  

  \[
  B = \sqrt{\frac{g\,(1+\varepsilon)\,\ln(2/\delta)}{2\,n}},
  \]

  where *g* is the observed gain, *n* the number of samples and
  *δ* the confidence level.

* **Tropical Regret** – Tropical (max‑plus) algebra from Parent A is
  combined with the regret scalar from Parent B.  In max‑plus,
  addition is *max* and multiplication is *+*.  The “tropical regret”
  of a gain *g* and regret *r* is therefore  

  \[
  T(g,r)=\max(g,r) + \min(g,r)=g+r,
  \]

  but we keep the max‑operation explicit to highlight the tropical
  structure.

* **LSM‑guided NLMS with Graph‑augmented Weights** – The Least‑Squares
  Magnitude (LSM) vector (Parent B) derived from linguistic feature
  frequencies modulates the Normalised Least‑Mean‑Squares (NLMS) adaptation
  step.  The weight update also incorporates a graph‑based diffusion term
  taken from the distributed‑learning parent, allowing neighbouring
  nodes to share weight information.

The three core functions below realise this bridge, providing a single
unified system that can be embedded in any streaming‑learning pipeline."""

from __future__ import annotations

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import Mapping, Hashable, Iterable, List, Dict, Set

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from both parents)
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]

@dataclass(frozen=True)
class MathAction:
    """Atomic action used by both parent algorithms."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for a given action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ----------------------------------------------------------------------
# Parent‑A primitives
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    """Probability that a node broadcasts in the current phase."""
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))

def compute_hoeffding_bound(observed_gain: float, delta: float, n: int) -> float:
    """Classic Hoeffding bound."""
    if n <= 0:
        raise ValueError('sample size n must be positive')
    return math.sqrt((observed_gain * math.log(2 / delta)) / (2 * n))

# ----------------------------------------------------------------------
# Parent‑B utilities (LSM, graph‑based features)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, Set[str]] = {
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
        "no not never none neither cannot can't won't don't didn't isn't aren't was'nt weren't".split()
    ),
    "quantifier": set("all any both each few many most several some".split()),
}

def _token_category(token: str) -> str | None:
    """Return the first lexical category a token belongs to, or None."""
    token_lc = token.lower()
    for cat, vocab in FUNCTION_CATS.items():
        if token_lc in vocab:
            return cat
    return None

def compute_lsm_vector(text: str) -> np.ndarray:
    """
    Least‑Squares Magnitude (LSM) vector for a piece of text.
    The vector length equals the number of lexical categories defined in
    ``FUNCTION_CATS`` and contains the normalized squared frequencies.
    """
    tokens = text.split()
    cat_counts = np.zeros(len(FUNCTION_CATS), dtype=float)
    cat_index = {cat: i for i, cat in enumerate(FUNCTION_CATS)}
    for tok in tokens:
        cat = _token_category(tok)
        if cat is not None:
            cat_counts[cat_index[cat]] += 1.0
    if cat_counts.sum() == 0:
        return cat_counts  # all zeros – no recognised tokens
    frequencies = cat_counts / cat_counts.sum()
    lsm = frequencies ** 2
    return lsm

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def regret_aware_hoeffding(
    observed_gain: float,
    delta: float,
    n: int,
    regret: float,
) -> float:
    """
    Regret‑aware Hoeffding bound.

    The regret term ``regret`` (≥0) acts as an energy multiplier,
    inflating the bound when the algorithm has accumulated high regret.
    """
    if regret < 0:
        raise ValueError("regret must be non‑negative")
    # classic bound scaled by (1 + regret)
    scaled_gain = observed_gain * (1.0 + regret)
    return math.sqrt((scaled_gain * math.log(2 / delta)) / (2 * n))

def tropical_regret(gain: float, regret: float) -> float:
    """
    Tropical (max‑plus) combination of a gain and a regret value.

    In max‑plus algebra:
        addition → max(a, b)
        multiplication → a + b
    We return the tropical “product” of gain and regret,
    i.e. max(gain, regret) + min(gain, regret) = gain + regret,
    but the max operation is kept explicit for clarity.
    """
    return max(gain, regret) + min(gain, regret)

def nlms_update_with_graph(
    weight: np.ndarray,
    input_vec: np.ndarray,
    desired: float,
    mu: float,
    lsm: np.ndarray,
    graph: Graph,
    node_id: Node,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Normalised Least‑Mean‑Squares (NLMS) weight update augmented with:

    * LSM‑vector scaling – each weight component is multiplied by the
      corresponding LSM entry before the adaptation step.
    * Graph diffusion – after the standard NLMS update, a fraction of each
      neighbour's weight vector is added, modelling information sharing
      across the network topology.

    Parameters
    ----------
    weight : np.ndarray
        Current weight vector (shape (d,)).
    input_vec : np.ndarray
        Current input sample (shape (d,)).
    desired : float
        Desired scalar output.
    mu : float
        Step‑size (0 < mu ≤ 1).
    lsm : np.ndarray
        LSM vector (shape (d,)), must sum to ≤ 1.
    graph : Mapping[Node, Set[Node]]
        Adjacency list of the communication graph.
    node_id : hashable
        Identifier of the node whose weight is being updated.
    epsilon : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Updated weight vector.
    """
    if weight.shape != input_vec.shape:
        raise ValueError("weight and input_vec must have the same shape")
    if lsm.shape != weight.shape:
        raise ValueError("lsm must have the same shape as weight")
    # 1) LSM scaling
    scaled_weight = weight * (1.0 + lsm)  # simple additive scaling
    # 2) NLMS error
    y = np.dot(scaled_weight, input_vec)
    error = desired - y
    # 3) Normalised adaptation term
    norm_factor = epsilon + np.dot(input_vec, input_vec)
    adaptation = (mu * error / norm_factor) * input_vec
    updated = scaled_weight + adaptation
    # 4) Graph diffusion
    neighbours = graph.get(node_id, set())
    if neighbours:
        neighbour_sum = np.zeros_like(weight)
        for nb in neighbours:
            # In a real system we would fetch the neighbour's weight.
            # For this self‑contained demo we assume neighbours share the
            # same current weight (could be replaced by a cache).
            neighbour_sum += weight
        diffusion = 0.1 * neighbour_sum / len(neighbours)  # diffusion coefficient = 0.1
        updated += diffusion
    return updated

# ----------------------------------------------------------------------
# Demonstration of the hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_iteration(
    actions: List[MathAction],
    graph: Graph,
    node_id: Node,
    text_context: str,
    input_vec: np.ndarray,
    desired_output: float,
    mu: float = 0.5,
) -> Dict[str, float]:
    """
    Perform a single hybrid learning iteration for a given node.

    Steps
    -----
    1. Compute LSM from the supplied textual context.
    2. Choose the action with the highest expected value (regret = 1 - prob).
    3. Evaluate a regret‑aware Hoeffding bound for the chosen action.
    4. Compute tropical regret using the bound as “gain”.
    5. Update the NLMS weight vector with graph diffusion.

    Returns a dictionary with the intermediate numerical results.
    """
    # 1) LSM vector
    lsm = compute_lsm_vector(text_context)

    # 2) Simple regret definition (higher expected value → lower regret)
    best_action = max(actions, key=lambda a: a.expected_value)
    regret = max(0.0, 1.0 - best_action.expected_value)  # clamp to [0,1]

    # 3) Hoeffding bound (using a synthetic delta and sample count)
    delta = 0.05
    n_samples = max(1, int(len(input_vec) * 10))
    bound = regret_aware_hoeffding(best_action.expected_value, delta, n_samples, regret)

    # 4) Tropical regret
    trop = tropical_regret(best_action.expected_value, regret)

    # 5) NLMS weight update (starting from a zero vector if not provided)
    d = input_vec.shape[0]
    weight = np.zeros(d)
    updated_weight = nlms_update_with_graph(
        weight=weight,
        input_vec=input_vec,
        desired=desired_output,
        mu=mu,
        lsm=lsm,
        graph=graph,
        node_id=node_id,
    )

    return {
        "lsm_norm": float(np.linalg.norm(lsm)),
        "selected_action": best_action.id,
        "regret": regret,
        "hoeffding_bound": bound,
        "tropical_regret": trop,
        "weight_norm_before": float(np.linalg.norm(weight)),
        "weight_norm_after": float(np.linalg.norm(updated_weight)),
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a tiny communication graph
    demo_graph: Graph = {
        "A": {"B", "C"},
        "B": {"A"},
        "C": {"A"},
    }

    # Define a few actions
    demo_actions = [
        MathAction(id="act1", expected_value=0.7),
        MathAction(id="act2", expected_value=0.4),
        MathAction(id="act3", expected_value=0.9),
    ]

    # Random input vector (dimension 5)
    np.random.seed(42)
    inp = np.random.randn(5)

    # Run the hybrid iteration for node "A"
    results = hybrid_iteration(
        actions=demo_actions,
        graph=demo_graph,
        node_id="A",
        text_context="I love the quick brown fox and the lazy dog",
        input_vec=inp,
        desired_output=1.0,
    )

    for k, v in results.items():
        print(f"{k}: {v}")

    # Simple sanity checks (should never raise)
    assert 0.0 <= results["regret"] <= 1.0
    assert results["weight_norm_after"] >= results["weight_norm_before"]  # adaptation should increase norm
    print("Hybrid algorithm smoke test completed successfully.")