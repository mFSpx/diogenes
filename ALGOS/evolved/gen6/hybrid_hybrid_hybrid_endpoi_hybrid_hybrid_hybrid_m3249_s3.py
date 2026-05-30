# DARWIN HAMMER — match 3249, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1129_s0.py (gen5)
# born: 2026-05-29T23:48:55Z

import math
import random
import sys
from pathlib import Path
import numpy as np
import hashlib
from collections import Counter, defaultdict
from typing import Iterable, List, Dict, Tuple, Set

MAX64 = (1 << 64) - 1


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """
    Per‑node circuit‑breaker that opens after a configurable number of failures.
    The failure threshold can be dynamically adjusted by external signals
    (e.g. similarity‑based risk scores).
    """

    def __init__(self, base_threshold: int = 3):
        if base_threshold <= 0:
            raise ValueError("base_threshold must be positive")
        self.base_threshold = base_threshold
        self.dynamic_factor = 1.0          # multiplicative factor applied to base_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    @property
    def threshold(self) -> int:
        """Current effective threshold after applying the dynamic factor."""
        return max(1, int(self.base_threshold * self.dynamic_factor))

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. the endpoint is usable)."""
        return not self.open

    def adjust_factor(self, factor: float) -> None:
        """Adjust the dynamic factor; must stay > 0."""
        if factor <= 0:
            raise ValueError("adjustment factor must be positive")
        self.dynamic_factor = factor


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of (seed, token) using Blake2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128) -> np.ndarray:
    """Return a MinHash signature (uint64 vector of length *k*) for a token set."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return np.full(k, MAX64, dtype=np.uint64)

    sig = np.empty(k, dtype=np.uint64)
    for i in range(k):
        sig[i] = min(_hash(i, t) for t in toks)
    return sig


def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if sig_a.shape != sig_b.shape:
        raise ValueError("signatures must have the same length")
    return float(np.mean(sig_a == sig_b))


def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    # Pad remaining bits with zeros if fewer than 64 values
    bits <<= max(0, 64 - len(values))
    return bits & ((1 << 64) - 1)


def _entropy_of_token_distribution(tokens: Iterable[str]) -> float:
    """Shannon entropy of the token frequency distribution."""
    toks = list(tokens)
    if not toks:
        return 0.0
    total = len(toks)
    freq = Counter(toks)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def hybrid_circuit_breaker(
    tokens: Dict[int, List[str]],
    node_features: Dict[int, List[float]],
    k: int = 128,
    alpha: float = 0.5,
    beta: float = 0.5,
    base_threshold: int = 3,
) -> Tuple[Dict[int, np.ndarray], Dict[int, EndpointCircuitBreaker]]:
    """
    Build per‑node MinHash signatures, compute adaptive circuit‑breaker objects,
    and expose a deeper mathematical coupling between similarity, entropy and
    failure risk.

    Parameters
    ----------
    tokens : mapping node → list of textual tokens
    node_features : mapping node → list of numeric features (used for perceptual hash)
    k : length of MinHash signatures
    alpha : weight of similarity‑driven risk (0 ≤ α ≤ 1)
    beta : weight of token‑entropy‑driven risk (0 ≤ β ≤ 1, α+β ≤ 1)
    base_threshold : baseline failure count before opening a circuit

    Returns
    -------
    Tuple[dict, dict]
        * node → MinHash signature (np.ndarray)
        * node → configured EndpointCircuitBreaker instance
    """
    if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
        raise ValueError("alpha and beta must be in [0, 1]")
    if alpha + beta > 1.0:
        raise ValueError("alpha + beta must not exceed 1")

    # 1️⃣ Compute MinHash signatures for every node
    minhash_sigs: Dict[int, np.ndarray] = {
        node: minhash_signature(tokens.get(node, []), k) for node in set(tokens) | set(node_features)
    }

    # 2️⃣ Compute pairwise Jaccard similarity matrix (symmetric)
    nodes = list(minhash_sigs.keys())
    n = len(nodes)
    sim_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            sim = jaccard_estimate(minhash_sigs[nodes[i]], minhash_sigs[nodes[j]])
            sim_matrix[i, j] = sim_matrix[j, i] = sim

    # 3️⃣ Derive a risk score per node:
    #    - similarity risk: 1 - average similarity to *other* nodes
    #    - entropy risk:   normalized Shannon entropy of its token set
    risk_scores: Dict[int, float] = {}
    for idx, node in enumerate(nodes):
        # similarity component (exclude self‑similarity)
        if n > 1:
            avg_sim = (np.sum(sim_matrix[idx]) - 1.0) / (n - 1)
        else:
            avg_sim = 1.0
        sim_risk = 1.0 - avg_sim

        # entropy component
        ent = _entropy_of_token_distribution(tokens.get(node, []))
        # Normalise entropy to [0,1] using log2(|vocab|) as max possible entropy
        vocab_size = len(set().union(*[set(t) for t in tokens.values()])) or 1
        max_ent = math.log2(vocab_size)
        ent_norm = ent / max_ent if max_ent > 0 else 0.0

        # Weighted combination
        risk = alpha * sim_risk + beta * ent_norm + (1 - alpha - beta) * 0.0
        risk_scores[node] = risk

    # 4️⃣ Initialise circuit‑breakers and adapt their dynamic factor.
    #    Higher risk → lower threshold (more aggressive opening).
    breakers: Dict[int, EndpointCircuitBreaker] = {}
    for node in nodes:
        cb = EndpointCircuitBreaker(base_threshold=base_threshold)
        # Map risk ∈ [0,1] to factor ∈ [0.5, 1.5] (example mapping)
        factor = 1.5 - risk_scores[node]  # risk 0 → 1.5 (lenient), risk 1 → 0.5 (strict)
        cb.adjust_factor(factor)
        breakers[node] = cb

    return minhash_sigs, breakers


def prune_probability(
    minhash_sigs: Dict[int, np.ndarray],
    breakers: Dict[int, EndpointCircuitBreaker],
) -> float:
    """
    Estimate the overall probability that a routing decision will be pruned.
    For each node we compare its signature against the global median signature;
    if the proportion of hash values below the median exceeds the node's
    dynamic threshold, we count it as a prune candidate.
    """
    # Global median signature (component‑wise)
    all_sigs = np.stack(list(minhash_sigs.values()))
    median_sig = np.median(all_sigs, axis=0)

    prune_flags = []
    for node, sig in minhash_sigs.items():
        below = np.mean(sig < median_sig)          # proportion of components smaller than median
        # Convert node's dynamic threshold to a comparable probability scale
        thresh_prob = 1.0 / breakers[node].threshold
        prune_flags.append(below > thresh_prob)

    return float(np.mean(prune_flags))


def apply_circuit_breaker(
    node_features: Dict[int, List[float]],
    breakers: Dict[int, EndpointCircuitBreaker],
) -> Dict[int, bool]:
    """
    Use perceptual hashes together with the adaptive circuit‑breaker to decide
    whether a node's routing decision should be pruned.

    Returns a mapping node → True (pruned) / False (kept).
    """
    decisions: Dict[int, bool] = {}
    for node, feats in node_features.items():
        phash = compute_phash(feats)
        # Normalise perceptual hash to [0,1] by dividing by 2**64‑1
        phash_norm = phash / ((1 << 64) - 1)

        # A node is pruned if its normalised hash exceeds the inverse of its
        # dynamic threshold (i.e. high hash → high risk) *or* the breaker is open.
        breaker = breakers.get(node)
        if breaker is None:
            # If no breaker exists for the node, fall back to a simple rule.
            decisions[node] = phash_norm > 0.5
            continue

        high_hash_risk = phash_norm > (1.0 / breaker.threshold)
        decisions[node] = high_hash_risk or not breaker.allow()
    return decisions


if __name__ == "__main__":
    # Example usage with synthetic data
    tokens_example = {
        1: ["apple", "banana", "cherry"],
        2: ["banana", "date", "fig"],
        3: ["grape", "apple", "fig", "honeydew"],
    }
    node_features_example = {
        1: [0.12, 0.34, 0.56, 0.78],
        2: [0.23, 0.45, 0.67, 0.89],
        3: [0.11, 0.22, 0.33, 0.44],
    }

    minhash_sigs, breakers = hybrid_circuit_breaker(
        tokens=tokens_example,
        node_features=node_features_example,
        k=128,
        alpha=0.4,
        beta=0.4,
        base_threshold=3,
    )

    prune_prob = prune_probability(minhash_sigs, breakers)
    pruned = apply_circuit_breaker(node_features_example, breakers)

    print("MinHash signatures (sample per node):")
    for node, sig in minhash_sigs.items():
        print(f"  Node {node}: {sig[:8]} ...")

    print("\nCircuit‑breaker thresholds per node:")
    for node, cb in breakers.items():
        print(f"  Node {node}: base={cb.base_threshold}, factor={cb.dynamic_factor:.2f}, effective={cb.threshold}")

    print(f"\nOverall prune probability: {prune_prob:.3f}")
    print("Pruned routing decisions:", pruned)