# DARWIN HAMMER — match 5054, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hoeffd_hybrid_endpoint_circ_m2160_s0.py (gen4)
# born: 2026-05-29T23:59:32Z

"""Hybrid Algorithm combining Fisher‑information driven infotaxis (Parent A) with
Hoeffding‑tree split testing, geometric morphology and circuit‑breaker constraints
(Parent B).

Mathematical bridge
-------------------
* From Parent A we obtain an *information‑gain* term

    IG(θ) = 𝓕(θ) · H(p_hit) · S_minhash

  where 𝓕(θ) is the Fisher information of a Gaussian beam, H(p) = −p·log p−(1−p)·log (1−p)
  is the binary entropy of the MinHash‑derived hit probability p_hit, and S_minhash is the
  Jaccard similarity estimated via MinHash.  IG(θ) quantifies the expected reduction of
  entropy when probing direction θ.

* From Parent B the Hoeffding split decision uses a gain

    g = α·sphericity + β·log ℓ + γ·log |C| + δ·IG(θ)

  where sphericity is a geometric quality of a morphology, ℓ is the likelihood of the
  observed reward stream, |C| is the cardinality of the candidate split, and IG(θ) is the
  information‑gain term defined above.  The Hoeffding bound ε = √(r²·ln(1/δ)/(2·n))
  provides a confidence interval for the gain gap.  A circuit‑breaker flag `open`
  acts as a hard constraint: a split is allowed only if the circuit is closed.

The code below implements this fused model, exposing three core functions:
`hybrid_information_gain`, `hybrid_split_decision` and `hybrid_tree_cost`,
together with a simple smoke test.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, field, asdict, frozen
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def minhash_signature(tokens: Iterable[str], seed: int = 0, n_hashes: int = 128) -> List[int]:
    """Very simple MinHash: hash each token with different salts and keep minima."""
    mins = [2 ** 64 - 1] * n_hashes
    for token in tokens:
        for i in range(n_hashes):
            h = hashlib.blake2b(token.encode('utf-8'), digest_size=8,
                               person=seed.to_bytes(8, 'little') + i.to_bytes(8, 'little')).digest()
            val = int.from_bytes(h, 'little')
            if val < mins[i]:
                mins[i] = val
    return mins


def jaccard_from_minhash(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("signatures must have same length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def binary_entropy(p: float, eps: float = 1e-12) -> float:
    """Binary entropy H(p) = -p log p - (1-p) log (1-p)."""
    p = min(max(p, eps), 1 - eps)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def hybrid_information_gain(
    theta: float,
    center: float,
    width: float,
    tokens_a: Iterable[str],
    tokens_b: Iterable[str],
) -> float:
    """
    Information‑gain term that fuses Fisher information, MinHash similarity and entropy.

    IG = Fisher(θ) * H(p_hit) * Jaccard
    where p_hit is taken as the Jaccard similarity.
    """
    fisher = fisher_score(theta, center, width)
    sig_a = minhash_signature(tokens_a, seed=42)
    sig_b = minhash_signature(tokens_b, seed=42)
    jaccard = jaccard_from_minhash(sig_a, sig_b)
    entropy = binary_entropy(jaccard)
    return fisher * entropy * jaccard


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound ε for range r, confidence δ and sample count n."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("invalid Hoeffding parameters")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑based split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


class CircuitBreaker:
    """Simple binary flag representing a hard constraint."""
    def __init__(self, open_: bool = False):
        self.open = open_

    def close(self):
        self.open = False

    def open_circuit(self):
        self.open = True


def sphericity_index(morphology: np.ndarray) -> float:
    """
    Placeholder geometric quality: ratio of volume to surface area.
    For a set of points we approximate volume by convex hull volume
    (here mocked by variance) and surface by mean pairwise distance.
    """
    if morphology.size == 0:
        return 0.0
    var = np.var(morphology, axis=0).sum()
    dists = np.linalg.norm(morphology[:, None, :] - morphology[None, :, :], axis=2)
    mean_dist = np.mean(dists)
    return var / (mean_dist + 1e-12)


def log_likelihood(rewards: Iterable[float]) -> float:
    """Simple log‑likelihood assuming Gaussian rewards with unit variance."""
    rewards = np.fromiter(rewards, dtype=float)
    if rewards.size == 0:
        return -np.inf
    mean = rewards.mean()
    # log p(x|μ) up to constant
    return -0.5 * np.sum((rewards - mean) ** 2)


def hybrid_split_decision(
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    circuit: CircuitBreaker,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta_ig: float = 0.1,
    info_gain: float = 0.0,
) -> SplitDecision:
    """
    Compute a combined gain that includes the information‑gain term from Parent A,
    then apply the Hoeffding bound together with the circuit‑breaker constraint.
    """
    combined_gain = (
        alpha * best_gain
        + beta * second_best_gain
        + gamma * math.log(max(best_gain, 1e-6))
        + delta_ig * info_gain
    )
    # For the bound we treat the range as the span of possible combined gains.
    gain_range = max(combined_gain, 1e-6) - min(best_gain, second_best_gain, 1e-6)
    eps = hoeffding_bound(gain_range, delta, n)
    gap = combined_gain - second_best_gain
    if circuit.open:
        reason = "circuit open – split forbidden"
        return SplitDecision(False, eps, gap, reason)
    if gap >= eps:
        reason = "gain gap exceeds Hoeffding bound"
        return SplitDecision(True, eps, gap, reason)
    else:
        reason = "gain gap below Hoeffding bound"
        return SplitDecision(False, eps, gap, reason)


# ----------------------------------------------------------------------
# Tree utilities from Parent A (used as cost term)
# ----------------------------------------------------------------------


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
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
    edge_len : dict mapping edge → length
    root_dist : dict mapping node → distance from root
    """
    adj: Dict[str, List[str]] = {k: [] for k in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length(nodes[u], nodes[v])
        edge_len[(v, u)] = edge_len[(u, v)]

    root_dist: Dict[str, float] = {root: 0.0}
    visited = {root}
    stack = [(root, 0.0)]
    while stack:
        node, dist = stack.pop()
        for nb in adj[node]:
            if nb not in visited:
                ndist = dist + edge_len[(node, nb)]
                root_dist[nb] = ndist
                visited.add(nb)
                stack.append((nb, ndist))
    return adj, edge_len, root_dist


def hybrid_tree_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    cost_factor: float = 0.05,
) -> float:
    """
    Compute a cost derived from the total root‑to‑leaf distance,
    scaled by ``cost_factor``.  This cost can be subtracted from the
    combined gain to penalise expensive sensing configurations.
    """
    _, _, root_dist = tree_metrics(nodes, edges, root)
    total = sum(root_dist.values())
    return cost_factor * total


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------


def generate_random_tokens(num: int, vocab_size: int = 1000) -> List[str]:
    """Create a list of pseudo‑tokens for MinHash."""
    return [f"tok{random.randint(0, vocab_size)}" for _ in range(num)]


def demo_hybrid_process() -> None:
    """Run a short demo that exercises the hybrid pipeline."""
    # --- Step 1: information gain ---
    theta = random.uniform(-math.pi, math.pi)
    center = 0.0
    width = 0.5
    tokens_a = generate_random_tokens(200)
    tokens_b = generate_random_tokens(200)
    ig = hybrid_information_gain(theta, center, width, tokens_a, tokens_b)

    # --- Step 2: geometric & statistical gain ---
    # synthetic morphology (10 points in 2‑D)
    morph = np.random.randn(10, 2)
    sph = sphericity_index(morph)
    rewards = np.random.randn(50)
    ll = log_likelihood(rewards)
    cardinality = len(rewards)

    best_gain = sph * 1.0 + ll * 0.5
    second_gain = sph * 0.8 + ll * 0.4

    # --- Step 3: split decision with circuit breaker ---
    circuit = CircuitBreaker(open_=False)
    decision = hybrid_split_decision(
        best_gain=best_gain,
        second_best_gain=second_gain,
        r=1.0,
        delta=0.05,
        n=1000,
        circuit=circuit,
        info_gain=ig,
    )
    print("Information Gain:", ig)
    print("Combined Gain:", best_gain, second_gain)
    print("Split Decision:", asdict(decision))

    # --- Step 4: tree cost penalty ---
    nodes = {"root": (0.0, 0.0), "a": (1.0, 0.0), "b": (0.0, 1.0), "c": (1.0, 1.0)}
    edges = [("root", "a"), ("root", "b"), ("a", "c")]
    cost = hybrid_tree_cost(nodes, edges, "root")
    print("Tree cost penalty:", cost)


if __name__ == "__main__":
    demo_hybrid_process()