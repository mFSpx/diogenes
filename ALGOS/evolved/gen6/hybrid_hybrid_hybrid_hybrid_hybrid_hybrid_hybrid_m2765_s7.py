# DARWIN HAMMER — match 2765, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_bayes_update__m1014_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m760_s0.py (gen5)
# born: 2026-05-29T23:45:52Z

"""Hybrid Decision‑Hygiene, Bayesian‑Ricci & VRAM‑Tree‑Fisher Fusion

Parents:
- hybrid_hybrid_hybrid_decisi_hybrid_bayes_update__m1014_s0.py (feature extraction, Shannon entropy, Bayesian marginal)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m760_s0.py (tree metrics, Euclidean edge lengths, VRAM‑budget style reasoning)

Mathematical bridge:
    • The Shannon entropy of the extracted textual features is interpreted as a *global prior* 𝑃₀ on the
      plausibility of any hypothesis (here: the hypothesis that a graph node satisfies a resource‑budget
      constraint).
    • The tree‑metric routine yields root‑to‑node distances 𝑑_i.  Treating these distances as observations,
      a Gaussian likelihood 𝓁_i = 𝒩(d_i | μ,σ²) is built, where the variance σ² is supplied by a *Fisher
      information* score computed from the whole distance distribution.
    • A hyper‑dimensional (HD) encoding maps each scalar distance into a high‑dimensional binary vector
      𝒙_i ∈ {−1,+1}^D.  The inner product ⟨𝒙_i,𝒙_j⟩ provides a similarity measure that can be used as an
      additional latent factor in the Bayesian update.
    • The posterior for node *i* is therefore
          P_i = (𝓁_i · P₀) / (𝓁_i·P₀ + (1−𝓁_i)·(1−P₀))
      optionally modulated by the HD similarity term.

The code below implements this fusion, exposing three core functions that demonstrate the combined
operations."""
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def extract_features(text: str) -> Dict[str, int]:
    """Extract evidence‑type tokens from free‑form text."""
    evidence_re = re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked)\b",
        flags=re.IGNORECASE,
    )
    feats: Dict[str, int] = defaultdict(int)
    for m in evidence_re.finditer(text):
        feats[m.group().lower()] += 1
    return dict(feats)


def compute_shannon_entropy(features: Dict[str, int]) -> float:
    """Shannon entropy of the empirical feature distribution."""
    total = sum(features.values())
    if total == 0:
        return 0.0
    probs = np.array(list(features.values())) / total
    return -float(np.sum(probs * np.log(probs + 1e-12)))


def bayes_marginal(prior: float, likelihood: float, false_positive: float = 0.0) -> float:
    """
    Return the marginal probability P(E) for a single hypothesis.
    Implements P(E) = L·π + FP·(1−π).
    """
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_posterior(prior: float, likelihood: float, false_positive: float = 0.0) -> float:
    """
    Posterior probability P(H|E) using Bayes rule with an optional false‑positive term.
    """
    marginal = bayes_marginal(prior, likelihood, false_positive)
    if marginal == 0.0:
        return 0.0
    return (likelihood * prior) / marginal


# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
def _euclidean_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
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
    dist : dict mapping node → cumulative distance from *root*
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = _euclidean_length(nodes[a], nodes[b])
        edge_len[(b, a)] = edge_len[(a, b)]  # undirected convenience

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    visited = {root}
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in visited:
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                visited.add(nxt)
                stack.append(nxt)
    return adj, edge_len, dist


def fisher_score(distances: Iterable[float]) -> float:
    """
    Simple Fisher‑information‑like score: variance of the distance distribution.
    Larger variance → higher information content.
    """
    arr = np.array(list(distances), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.var(arr) + 1e-12)  # epsilon to avoid zero


def hyperdimensional_encode(value: float, dim: int = 1024, seed: int = 0) -> np.ndarray:
    """
    Encode a scalar into a binary hyper‑dimensional vector (+1 / -1).
    The random projection matrix is seeded deterministically from ``seed`` and the value.
    """
    rng = np.random.default_rng(seed + int(abs(value) * 1e6) % (2 ** 32))
    return rng.choice([-1, 1], size=dim)


def hd_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine‑like similarity for binary HD vectors (range [-1, 1])."""
    return float(np.mean(vec_a * vec_b))


# ----------------------------------------------------------------------
# Hybrid core functions (fusion)
# ----------------------------------------------------------------------
def compute_global_prior(text: str) -> float:
    """
    Derive a normalized prior from textual evidence:
        prior = H / log(N)
    where H is Shannon entropy and N is the number of distinct tokens.
    The prior lies in (0,1].
    """
    feats = extract_features(text)
    H = compute_shannon_entropy(feats)
    N = max(len(feats), 1)
    max_entropy = math.log(N) if N > 1 else 1.0
    prior = H / max_entropy
    return min(max(prior, 1e-6), 1.0)


def node_likelihoods(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    fisher_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Compute a Gaussian‑like likelihood for each node based on its root distance.
    The variance is supplied by the Fisher score of the whole distance set.
    """
    _, _, distances = tree_metrics(nodes, edges, root)
    dist_vals = list(distances.values())
    var = fisher_score(dist_vals) * fisher_weight
    sigma = math.sqrt(var)
    mu = np.mean(dist_vals) if dist_vals else 0.0

    likelihoods: Dict[str, float] = {}
    for node, d in distances.items():
        # Gaussian pdf (unnormalised, clipped to [0,1])
        exponent = -0.5 * ((d - mu) / (sigma + 1e-12)) ** 2
        likelihood = float(math.exp(exponent))
        likelihoods[node] = min(max(likelihood, 0.0), 1.0)
    return likelihoods


def fused_node_posteriors(
    text: str,
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    dim: int = 1024,
) -> Dict[str, float]:
    """
    Full hybrid pipeline:
        1. Global prior from textual evidence (entropy‑based).
        2. Root‑to‑node distances → likelihoods (Gaussian with Fisher variance).
        3. Encode each distance into an HD vector; compute similarity to a global HD prototype.
           The similarity rescales the likelihood (acts as a latent factor).
        4. Bayesian posterior per node.
    Returns a mapping node → posterior probability.
    """
    prior = compute_global_prior(text)

    # Step 2: likelihoods from tree metrics
    likelihoods = node_likelihoods(nodes, edges, root)

    # Step 3: HD prototype (mean of all encoded vectors)
    _, _, distances = tree_metrics(nodes, edges, root)
    hd_vectors = {
        n: hyperdimensional_encode(d, dim=dim, seed=42) for n, d in distances.items()
    }
    prototype = np.mean(np.stack(list(hd_vectors.values())), axis=0)
    prototype = np.sign(prototype)  # keep binary sign

    # Step 4: combine and compute posterior
    posteriors: Dict[str, float] = {}
    for node, lik in likelihoods.items():
        sim = (hd_similarity(hd_vectors[node], prototype) + 1.0) / 2.0  # map [-1,1] → [0,1]
        adjusted_lik = lik * sim  # latent modulation
        post = bayes_posterior(prior, adjusted_lik, false_positive=0.0)
        posteriors[node] = post
    return posteriors


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The evidence was verified and documented. "
        "A screenshot and log were provided as proof. "
        "Further source citation is required."
    )

    # Simple geometric tree
    sample_nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
        "E": (2.0, 0.0),
    }
    sample_edges = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E")]
    root_node = "A"

    post = fused_node_posteriors(sample_text, sample_nodes, sample_edges, root_node)
    for n, p in sorted(post.items()):
        print(f"Node {n}: posterior = {p:.4f}")

    # Quick sanity checks (should not raise)
    assert 0.0 <= compute_global_prior(sample_text) <= 1.0
    assert isinstance(post, dict) and all(0.0 <= v <= 1.0 for v in post.values())
    print("Smoke test completed successfully.")