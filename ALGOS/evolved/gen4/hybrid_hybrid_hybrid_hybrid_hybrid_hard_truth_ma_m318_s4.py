# DARWIN HAMMER — match 318, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s1.py (gen3)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s0.py (gen2)
# born: 2026-05-29T23:28:17Z

"""Hybrid Minimum‑Cost Tree & Semantic‑Weighted VRAM Scheduler
================================================================

This module fuses two ancestral algorithms:

* **Parent A** – ``hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s1.py``  
  Provides ``tree_metrics`` that builds adjacency lists, Euclidean edge
  lengths and root‑to‑node distances for a geometric tree.

* **Parent B** – ``hybrid_hard_truth_math_hybrid_minimum_cost__m12_s0.py``  
  Supplies a lightweight semantic (LSM) vector derived from text and a set of
  lexical categories (``FUNCTION_CATS``).

--------------------------------------------------------------------
Mathematical Bridge
--------------------------------------------------------------------
The bridge is a *semantic weighting* of the geometric edge lengths.
For every edge ``e = (u, v)`` we compute a scalar weight ``w_e`` from the
LSM vector of a supplied description.  The weighted edge cost is

``c_e = ℓ_e · w_e``

where ``ℓ_e`` is the Euclidean length from ``tree_metrics``.  The total
tree cost ``C = Σ_e c_e`` is then treated as a Gaussian random variable.
A conjugate Gaussian‑Gaussian Bayesian update incorporates an observed VRAM
usage ``y`` (with observation variance ``σ_y²``) to produce a posterior
estimate of the true cost:

``μ_post = (σ_y²·μ_prior + σ_prior²·y) / (σ_prior² + σ_y²)``  
``σ_post² = (σ_prior²·σ_y²) / (σ_prior² + σ_y²)``

The posterior mean ``μ_post`` is used by the scheduler to decide whether
the current allocation fits within the available VRAM budget.

The resulting hybrid system therefore couples
* geometric topology (tree metrics) *
* semantic information (LSM vector) *
* probabilistic inference (Bayesian cost update) *
into a single, self‑contained decision engine.
"""

import math
import random
import sys
import pathlib
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Tree utilities
# ----------------------------------------------------------------------
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                # identify the directed edge that connects cur → nxt
                if (cur, nxt) in edge_len:
                    seg_len = edge_len[(cur, nxt)]
                else:
                    seg_len = edge_len[(nxt, cur)]
                dist[nxt] = dist[cur] + seg_len
                stack.append(nxt)

    return adj, edge_len, dist

# ----------------------------------------------------------------------
# Parent B – Semantic (LSM) utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

def words(text: str) -> List[str]:
    """Extract lower‑case alphabetic tokens from *text*."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

def lsm_vector(text: str) -> Dict[str, float]:
    """
    Compute a lightweight semantic (LSM) vector for *text*.
    The vector contains, for each lexical category, the normalized frequency
    of words belonging to that category.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    vec: Dict[str, float] = {}
    for cat, vocab in FUNCTION_CATS.items():
        cat_count = sum(cnt.get(w, 0) for w in vocab)
        vec[cat] = cat_count / total
    return vec

# ----------------------------------------------------------------------
# Hybrid core: semantic weighting of edges + Bayesian cost update
# ----------------------------------------------------------------------
def semantic_edge_weights(
    edge_len: Dict[Tuple[str, str], float],
    lsm_vec: Dict[str, float],
    cat_weights: Dict[str, float] | None = None,
) -> Dict[Tuple[str, str], float]:
    """
    Produce a weight for each edge by projecting the LSM vector onto a
    category‑weight vector.

    Parameters
    ----------
    edge_len : mapping edge → Euclidean length.
    lsm_vec  : semantic vector (category → frequency).
    cat_weights : optional explicit weights per category; if omitted,
                  each category contributes equally.

    Returns
    -------
    weighted_len : mapping edge → length * semantic_factor
    """
    if cat_weights is None:
        # Uniform contribution across all categories present in the LSM vector
        cat_weights = {cat: 1.0 for cat in lsm_vec}
    # Normalise cat_weights so that they sum to 1
    total_w = sum(cat_weights.values())
    norm_weights = {c: w / total_w for c, w in cat_weights.items()}

    # Compute a single scalar semantic factor from the vector
    semantic_factor = sum(lsm_vec.get(c, 0.0) * norm_weights.get(c, 0.0) for c in norm_weights)

    # Guard against degenerate factor (e.g., empty text)
    if semantic_factor <= 0.0:
        semantic_factor = 1.0

    weighted_len = {e: length * semantic_factor for e, length in edge_len.items()}
    return weighted_len

def total_tree_cost(weighted_edge_len: Dict[Tuple[str, str], float]) -> float:
    """Sum of all weighted edge lengths – the raw hybrid cost."""
    return sum(weighted_edge_len.values())

def gaussian_bayesian_update(
    prior_mean: float,
    prior_var: float,
    observation: float,
    obs_var: float,
) -> Tuple[float, float]:
    """
    Conjugate Gaussian‑Gaussian Bayesian update.

    Returns posterior mean and variance.
    """
    if prior_var <= 0 or obs_var <= 0:
        raise ValueError("Variances must be positive.")
    precision_prior = 1.0 / prior_var
    precision_obs = 1.0 / obs_var
    post_var = 1.0 / (precision_prior + precision_obs)
    post_mean = post_var * (precision_prior * prior_mean + precision_obs * observation)
    return post_mean, post_var

def hybrid_vram_decision(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    description: str,
    prior_mean: float,
    prior_var: float,
    observed_vram: float,
    obs_var: float,
    cat_weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    End‑to‑end hybrid operation:

    1. Compute geometric tree metrics.
    2. Derive an LSM vector from *description*.
    3. Weight edge lengths with the semantic factor.
    4. Obtain the raw hybrid cost.
    5. Perform a Bayesian update with the observed VRAM usage.

    Returns a dictionary summarising each stage.
    """
    # 1. Geometry
    adj, edge_len, dist = tree_metrics(nodes, edges, root)

    # 2. Semantics
    lsm_vec = lsm_vector(description)

    # 3. Weighting
    weighted_len = semantic_edge_weights(edge_len, lsm_vec, cat_weights)

    # 4. Raw cost
    raw_cost = total_tree_cost(weighted_len)

    # 5. Bayesian update
    post_mean, post_var = gaussian_bayesian_update(
        prior_mean, prior_var, observed_vram, obs_var
    )

    decision = {
        "adjacency": adj,
        "edge_lengths": edge_len,
        "distances_from_root": dist,
        "lsm_vector": lsm_vec,
        "semantic_factor": sum(
            lsm_vec.get(c, 0.0) * (cat_weights.get(c, 1.0) if cat_weights else 1.0)
            for c in (cat_weights or lsm_vec)
        ),
        "weighted_edge_lengths": weighted_len,
        "raw_hybrid_cost": raw_cost,
        "prior_mean": prior_mean,
        "prior_variance": prior_var,
        "observed_vram": observed_vram,
        "observation_variance": obs_var,
        "posterior_mean_cost": post_mean,
        "posterior_variance": post_var,
        "allocation_feasible": post_mean <= observed_vram,
    }
    return decision

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple tree: a root with two children and a leaf
    nodes = {
        "root": (0.0, 0.0),
        "A": (1.0, 0.0),
        "B": (0.0, 1.0),
        "C": (1.0, 1.0),
    }
    edges = [("root", "A"), ("root", "B"), ("A", "C")]
    root = "root"

    description = (
        "The model allocates VRAM efficiently using a hybrid scheduler that "
        "combines geometric cost with semantic awareness."
    )
    prior_mean = 10.0          # expected cost before observation
    prior_var = 4.0            # uncertainty in the prior
    observed_vram = 12.0       # measured VRAM consumption
    obs_var = 1.0              # measurement noise

    # Optional: give more importance to "auxiliary" and "preposition" categories
    cat_weights = {"auxiliary": 2.0, "preposition": 1.5, "article": 0.5}

    result = hybrid_vram_decision(
        nodes,
        edges,
        root,
        description,
        prior_mean,
        prior_var,
        observed_vram,
        obs_var,
        cat_weights,
    )

    print("Hybrid VRAM Decision Summary")
    print("-----------------------------")
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subk, subv in value.items():
                print(f"  {subk}: {subv}")
        else:
            print(f"{key}: {value}")