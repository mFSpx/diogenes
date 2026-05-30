# DARWIN HAMMER — match 318, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s1.py (gen3)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s0.py (gen2)
# born: 2026-05-29T23:28:17Z

"""Hybrid Minimum‑Cost Tree with LSM‑Weighted Bayesian Update
================================================================

This module fuses the core ideas of two parent algorithms:

* **Parent A** – ``hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s1.py``  
  Provides geometric tree metrics (adjacency, Euclidean edge lengths,
  root‑to‑node distances) that are used for estimating resource
  requirements.

* **Parent B** – ``hybrid_hard_truth_math_hybrid_minimum_cost__m12_s0.py``  
  Supplies a linguistic “LSM” (lexical‑semantic‑metric) vector derived
  from tokenised text and a taxonomy of word‑categories.  The LSM vector
  is employed here as a *probabilistic weight* for each node.

**Mathematical bridge**

For each edge *(u, v)* we compute a weight  

\[
w_{uv}= \frac{1}{2}\bigl(\mathbf{l}_u\cdot\mathbf{l}_v\bigr),
\]

where \(\mathbf{l}_u\) and \(\mathbf{l}_v\) are the LSM category‑frequency
vectors of the incident nodes and “·” denotes the dot‑product.  
The geometric length \(\ell_{uv}\) from Parent A is then multiplied by
\(w_{uv}\) to obtain a *hybrid edge cost*  

\[
c_{uv}= w_{uv}\,\ell_{uv}.
\]

The total hybrid tree cost \(C\) is the sum of all \(c_{uv}\).  Finally a
Gaussian Bayesian update incorporates an observed resource usage
\(y\) (e.g. measured VRAM consumption) to produce a posterior estimate
\(\hat C\):

\[
\hat C = \frac{\sigma_y^{2}\,C + \sigma_C^{2}\,y}
              {\sigma_y^{2}+\sigma_C^{2}},
\]

where \(\sigma_C^{2}\) and \(\sigma_y^{2}\) are prior and observation
variances, respectively.

The three public functions below implement this pipeline.
"""

import math
import random
import sys
import pathlib
import re
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – geometric tree utilities
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
    # adjacency list
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                # edge may be stored as (cur,nxt) or (nxt,cur)
                seg = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[seg]
                stack.append(nxt)

    return adj, edge_len, dist


# ----------------------------------------------------------------------
# Parent B – LSM (lexical‑semantic‑metric) utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
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
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def words(text: str) -> List[str]:
    """Tokenise a string into lower‑case alphabetic words."""
    return [w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())]


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Produce a normalised category‑frequency vector for *text*.

    The vector has an entry for each category in ``FUNCTION_CATS``; the
    value is the proportion of words belonging to that category.
    """
    ws = words(text)
    total = max(1, len(ws))
    # count words per category
    cat_counts: Dict[str, int] = {cat: 0 for cat in FUNCTION_CATS}
    for w in ws:
        for cat, vocab in FUNCTION_CATS.items():
            if w in vocab:
                cat_counts[cat] += 1
    return {cat: cnt / total for cat, cnt in cat_counts.items()}


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def edge_lsm_weight(
    vec_u: Dict[str, float],
    vec_v: Dict[str, float],
) -> float:
    """
    Compute a symmetric weight from two LSM vectors.

    The weight is the average dot‑product of the two vectors, guaranteeing
    a value in [0, 1] because each vector is a probability distribution.
    """
    # Convert dicts to ordered numpy arrays (consistent ordering)
    cats = sorted(FUNCTION_CATS.keys())
    u_arr = np.array([vec_u.get(c, 0.0) for c in cats])
    v_arr = np.array([vec_v.get(c, 0.0) for c in cats])
    return float(np.dot(u_arr, v_arr))


def hybrid_tree_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    node_texts: Dict[str, str],
) -> Tuple[float, Dict[Tuple[str, str], float]]:
    """
    Compute the hybrid cost of a tree.

    Returns
    -------
    total_cost : sum of LSM‑weighted edge lengths
    weighted_edges : mapping edge → weighted length (c_uv)
    """
    # 1️⃣ geometric metrics
    _, edge_len, _ = tree_metrics(nodes, edges, root)

    # 2️⃣ LSM vectors for every node
    lsm_cache: Dict[str, Dict[str, float]] = {
        n: lsm_vector(node_texts.get(n, "")) for n in nodes
    }

    # 3️⃣ weighted edges
    weighted_edges: Dict[Tuple[str, str], float] = {}
    total_cost = 0.0
    for (u, v) in edges:
        w = edge_lsm_weight(lsm_cache[u], lsm_cache[v])
        c = w * edge_len[(u, v)]
        weighted_edges[(u, v)] = c
        total_cost += c

    return total_cost, weighted_edges


def bayesian_update_cost(
    prior: float,
    observation: float,
    sigma_prior: float = 1.0,
    sigma_obs: float = 1.0,
) -> float:
    """
    Gaussian conjugate update for a scalar cost estimate.

    Parameters
    ----------
    prior : prior mean (e.g. hybrid tree cost)
    observation : measured resource usage
    sigma_prior : standard deviation of the prior
    sigma_obs : standard deviation of the observation

    Returns
    -------
    posterior mean (updated cost estimate)
    """
    var_prior = sigma_prior ** 2
    var_obs = sigma_obs ** 2
    posterior = (var_obs * prior + var_prior * observation) / (var_prior + var_obs)
    return posterior


def hybrid_cost_with_bayes(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    node_texts: Dict[str, str],
    observed_usage: float,
    sigma_prior: float = 1.0,
    sigma_obs: float = 1.0,
) -> float:
    """
    Full pipeline:
    1. Compute LSM‑weighted tree cost.
    2. Update the cost with a Bayesian observation.
    """
    raw_cost, _ = hybrid_tree_cost(nodes, edges, root, node_texts)
    return bayesian_update_cost(
        prior=raw_cost,
        observation=observed_usage,
        sigma_prior=sigma_prior,
        sigma_obs=sigma_obs,
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple tree: a triangle (root at A)
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "C")]
    root = "A"

    # Example texts attached to each node
    node_texts = {
        "A": "I am the root node, and I control the flow.",
        "B": "The child node B handles data processing.",
        "C": "C is another child, responsible for output generation.",
    }

    # Simulated observed VRAM usage
    observed_vram = 2.5  # arbitrary units

    # Run the hybrid pipeline
    posterior_cost = hybrid_cost_with_bayes(
        nodes,
        edges,
        root,
        node_texts,
        observed_usage=observed_vram,
        sigma_prior=0.8,
        sigma_obs=0.5,
    )

    print(f"Posterior hybrid cost estimate: {posterior_cost:.4f}")
    sys.exit(0)