# DARWIN HAMMER — match 1874, survivor 1
# gen: 5
# parent_a: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s0.py (gen2)
# parent_b: hybrid_hybrid_minhash_hybri_hybrid_hybrid_sparse_m1147_s0.py (gen4)
# born: 2026-05-29T23:39:23Z

"""
Hybrid module integrating:

- Parent A: hard_truth_math (LSM categorical vector extraction) and
  hybrid_minimum_cost_tree_bayes_update (minimum‑cost tree with Bayesian update).
- Parent B: hybrid_minhash_hybrid_rlct_grokking (MinHash signatures) and
  hybrid_hybrid_sparse_wta_hy_hybrid_minimum_cost (model pool & edge impedance handling).

Mathematical bridge:
The LSM categorical frequency vector `c` (size = number of function categories) is used
to scale edge impedances in a tree. A MinHash signature `h` (derived from tokenised
input) provides a prior probability distribution over edges. The hybrid edge weight
`w_e` is defined as

    w_e = ζ_e · (1 + α·mean(c)) · (1 + β·mean(h)/2⁶⁴)

where `ζ_e` is the geometric length (or given impedance) of edge `e`,
`α,β∈ℝ⁺` are tunable scaling factors.  
A Bayesian update combines this likelihood with the MinHash‑derived prior `π`,
yielding a posterior edge cost

    p_e ∝ w_e · π_e ,   Σ_e p_e = 1.

The sum of posterior costs gives the hybrid metric for the whole tree.
"""

from __future__ import annotations
import math
import random
import sys
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – categorical LSM vector utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

def _tokenise(text: str) -> List[str]:
    """Extract lowercase alphabetic tokens from text."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

def lsm_vector(text: str) -> Dict[str, float]:
    """
    Compute the normalized frequency of each function‑category in the given text.
    Returns a dict mapping category name → relative frequency.
    """
    tokens = _tokenise(text)
    total = max(1, len(tokens))
    cat_counts: Dict[str, int] = {cat: 0 for cat in FUNCTION_CATS}
    for cat, vocab in FUNCTION_CATS.items():
        cat_counts[cat] = sum(1 for t in tokens if t in vocab)
    return {cat: cnt / total for cat, cnt in cat_counts.items()}

def category_vector(text: str) -> np.ndarray:
    """Return LSM frequencies as a NumPy vector ordered by FUNCTION_CATS keys."""
    vec_dict = lsm_vector(text)
    return np.array([vec_dict[cat] for cat in sorted(FUNCTION_CATS.keys())], dtype=float)

# ----------------------------------------------------------------------
# Parent B – MinHash and model‑pool utilities
# ----------------------------------------------------------------------
NodeId = str
Edge = Tuple[NodeId, NodeId, float]  # (src, dst, impedance/length)

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

class ModelPool:
    """Simple RAM‑constrained model container."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)

def _hash(seed: int, token: str) -> int:
    """
    Deterministic 64‑bit hash using BLAKE2b.
    """
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(tokens: List[str], num_perm: int = 64, seed: int = 42) -> np.ndarray:
    """
    Compute a MinHash signature (array of minima) for a token list.
    """
    if not tokens:
        return np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    sig = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    for i in range(num_perm):
        h_vals = [_hash(seed + i, t) for t in tokens]
        sig[i] = min(h_vals)
    return sig

# ----------------------------------------------------------------------
# Hybrid core – combine LSM, MinHash, and tree cost with Bayesian update
# ----------------------------------------------------------------------
def hybrid_edge_weights(
    edges: List[Edge],
    cat_vec: np.ndarray,
    minhash_sig: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> Dict[Edge, float]:
    """
    Compute hybrid weights for each edge.

    Parameters
    ----------
    edges : list of Edge
        Tree edges with a geometric length or impedance as the third element.
    cat_vec : np.ndarray
        LSM categorical frequency vector.
    minhash_sig : np.ndarray
        MinHash signature (uint64 values).
    alpha, beta : float
        Scaling coefficients for the LSM and MinHash contributions.

    Returns
    -------
    dict mapping Edge → hybrid weight (float).
    """
    mean_cat = float(cat_vec.mean())          # 0 ≤ mean_cat ≤ 1
    mean_hash = float(minhash_sig.mean()) / float(2 ** 64)  # normalised to [0,1)

    scale = (1.0 + alpha * mean_cat) * (1.0 + beta * mean_hash)

    hybrid_weights: Dict[Edge, float] = {}
    for e in edges:
        base_impedance = e[2]
        hybrid_weights[e] = base_impedance * scale
    return hybrid_weights

def bayesian_posterior(
    hybrid_weights: Dict[Edge, float],
    prior_sig: np.ndarray,
) -> Dict[Edge, float]:
    """
    Perform a Bayesian update using the hybrid weights as likelihoods and a
    MinHash‑derived prior.

    The prior for edge e is taken as the normalised MinHash value of the
    token that most likely generated that edge (here approximated by the
    global signature mean).

    Returns a posterior probability distribution over edges (sums to 1).
    """
    # Approximate a uniform prior modulated by the global MinHash mean
    prior_mean = float(prior_sig.mean()) / float(2 ** 64)
    prior = {e: prior_mean for e in hybrid_weights}

    # Unnormalised posterior ∝ likelihood * prior
    unnorm = {e: hybrid_weights[e] * prior[e] for e in hybrid_weights}
    total = sum(unnorm.values()) or 1.0
    posterior = {e: v / total for e, v in unnorm.items()}
    return posterior

def hybrid_tree_cost(
    edges: List[Edge],
    text: str,
    tokens: List[str],
    alpha: float = 0.5,
    beta: float = 0.3,
) -> float:
    """
    End‑to‑end hybrid metric for a tree:
    1. Extract LSM category vector from `text`.
    2. Compute MinHash signature from `tokens`.
    3. Derive hybrid edge weights.
    4. Apply Bayesian update to obtain posterior edge probabilities.
    5. Return the expected cost = Σ_e posterior(e) * hybrid_weight(e).

    This function demonstrates the mathematical fusion of both parents.
    """
    cat_vec = category_vector(text)
    mh_sig = minhash_signature(tokens)

    hybrid_w = hybrid_edge_weights(edges, cat_vec, mh_sig, alpha=alpha, beta=beta)
    posterior = bayesian_posterior(hybrid_w, mh_sig)

    expected_cost = sum(posterior[e] * hybrid_w[e] for e in edges)
    return expected_cost

# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample textual input
    sample_text = "The quick brown fox jumps over the lazy dog while it thinks about the universe."
    sample_tokens = _tokenise(sample_text)

    # Construct a tiny tree with coordinates (used only for base impedance)
    node_pos: Dict[NodeId, Tuple[float, float]] = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
    }

    def edge_impedance(src: NodeId, dst: NodeId) -> float:
        return math.hypot(node_pos[src][0] - node_pos[dst][0],
                          node_pos[src][1] - node_pos[dst][1])

    tree_edges: List[Edge] = [
        ("A", "B", edge_impedance("A", "B")),
        ("A", "C", edge_impedance("A", "C")),
        ("B", "D", edge_impedance("B", "D")),
        ("C", "D", edge_impedance("C", "D")),
    ]

    cost = hybrid_tree_cost(tree_edges, sample_text, sample_tokens)
    print(f"Hybrid tree expected cost: {cost:.6f}")

    # Demonstrate ModelPool usage (unrelated to cost but shows integration)
    pool = ModelPool(ram_ceiling_mb=2000)
    model_small = ModelTier(name="tiny-gpt", ram_mb=500, tier="T1")
    model_large = ModelTier(name="mega-gpt", ram_mb=1800, tier="T3")
    pool.load(model_small)
    try:
        pool.load(model_large)  # should raise because T3 cannot coexist with any T2 (none present) but exceeds RAM
    except Exception as e:
        print(f"Model loading error (expected): {e}")
    print(f"Currently loaded models: {list(pool.loaded.keys())}")