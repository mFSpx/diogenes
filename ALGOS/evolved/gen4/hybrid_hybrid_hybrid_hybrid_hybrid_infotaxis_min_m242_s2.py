# DARWIN HAMMER — match 242, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_epistemic_certainty_m2_s1.py (gen3)
# parent_b: hybrid_infotaxis_minhash_m63_s4.py (gen1)
# born: 2026-05-29T23:27:48Z

"""HybridSheafInfotaxis
Combines:
- hybrid_hybrid_hybrid_sketch_epistemic_certainty_m2_s1.py (sheaf cohomology over a graph where sections are
  Count‑Min / MinHash sketches)
- hybrid_infotaxis_minhash_m63_s4.py (MinHash signatures, similarity, and entropy‑based infotaxis)

Mathematical bridge:
Each node of the sheaf carries a MinHash signature σ∈ℤ^{k}.  The restriction maps on an edge (u,v) are taken
as the identity (or a projection) so that the coboundary δσ_{uv}=σ_u−σ_v measures the local disagreement.
The squared ℓ₂‑norm of δσ_{uv} is a surrogate for the information loss on that edge.
Summing over all edges yields a global “inconsistency” I = Σ_{(u,v)}‖δσ_{uv}‖².
The Real Log‑Canonical Threshold (RLRT) of a quadratic form is proportional to ½·log det(L) where L is the
graph Laplacian weighted by the edge disagreements.  We approximate the RLRT by
    RLRT ≈ ½·log(det(I·Iᵀ + ε·I_d)),
which can be computed from the matrix of edge disagreements.
Epistemic certainty from the infotaxis side provides an entropy H(σ) of the signature
distribution.  The hybrid objective combines the two:
    Φ = α·RLRT + β·H(σ)
with tunable weights α,β.

The module implements a concrete hybrid system that:
1. Builds a sheaf whose node sections are MinHash signatures of token sets.
2. Computes the coboundary‑based inconsistency matrix and its RLRT approximation.
3. Evaluates signature entropy and selects the action (node) that minimizes the hybrid
   expected entropy after a hypothetical token addition.

Only numpy and the Python standard library are used.
"""

import numpy as np
import hashlib
import math
import random
import sys
import pathlib
from collections import Counter, defaultdict
from typing import Iterable, List, Dict, Tuple, Set

MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128) -> np.ndarray:
    """Return a MinHash signature as a NumPy uint64 vector of length k."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return np.full(k, MAX64, dtype=np.uint64)
    sig = np.empty(k, dtype=np.uint64)
    for i in range(k):
        sig[i] = min(_hash(i, t) for t in toks)
    return sig


def signature_entropy(sig: np.ndarray) -> float:
    """Shannon entropy of the multiset of hash values in a signature."""
    if sig.size == 0:
        raise ValueError("signature must not be empty")
    counts = Counter(sig.tolist())
    probs = np.array(list(counts.values()), dtype=float)
    probs /= probs.sum()
    eps = 1e-12
    return -np.sum(probs * np.log(np.maximum(probs, eps)))


class HybridSheaf:
    """Sheaf whose sections are MinHash signatures."""

    def __init__(self, node_tokens: Dict[str, Iterable[str]], edges: List[Tuple[str, str]],
                 k: int = 128):
        """
        node_tokens: mapping node → iterable of tokens (the local data)
        edges: list of undirected edges (u, v)
        k: length of MinHash signatures
        """
        self.k = k
        self.edges = [(u, v) for u, v in edges]
        self.nodes = list(node_tokens.keys())
        # Sections are stored as numpy uint64 vectors
        self._sections: Dict[str, np.ndarray] = {
            n: minhash_signature(node_tokens[n], k=self.k) for n in self.nodes
        }
        # Identity restriction maps (could be extended)
        self._restrictions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        for u, v in self.edges:
            dim = self.k
            self._restrictions[(u, v)] = (np.eye(dim), np.eye(dim))
            self._restrictions[(v, u)] = (np.eye(dim), np.eye(dim))

    def set_section(self, node: str, tokens: Iterable[str]) -> None:
        """Replace the section at *node* with the MinHash of *tokens*."""
        self._sections[node] = minhash_signature(tokens, k=self.k)

    def get_section(self, node: str) -> np.ndarray:
        return self._sections[node]

    def coboundary_vector(self, edge: Tuple[str, str]) -> np.ndarray:
        """δσ_{uv} = σ_u - σ_v (identity restriction). Returns a float vector."""
        u, v = edge
        return self._sections[u].astype(np.float64) - self._sections[v].astype(np.float64)


def global_inconsistency_matrix(sheaf: HybridSheaf) -> np.ndarray:
    """
    Assemble the edge‑wise coboundary vectors into a matrix B (|E| × k)
    and return the Gram matrix G = Bᵀ B, which encodes the quadratic form
    Σ_{(u,v)}‖δσ_{uv}‖².
    """
    B = np.stack([sheaf.coboundary_vector(e) for e in sheaf.edges], axis=0)  # shape (E, k)
    G = B.T @ B  # shape (k, k)
    return G


def rlrt_approximation(G: np.ndarray, eps: float = 1e-6) -> float:
    """
    Approximate the Real Log‑Canonical Threshold for a quadratic form xᵀ G x
    by ½·log det(G + eps·I).  The small epsilon regularises singular matrices.
    """
    d = G.shape[0]
    reg = G + eps * np.eye(d)
    sign, logdet = np.linalg.slogdet(reg)
    if sign <= 0:
        # In degenerate cases fall back to a large penalty
        return float('inf')
    return 0.5 * logdet


def hybrid_expected_entropy(
    sheaf: HybridSheaf,
    node: str,
    new_token: str,
    alpha: float = 1.0,
    beta: float = 1.0
) -> float:
    """
    Expected hybrid objective after hypothetically adding *new_token* to *node*.
    The objective is Φ = α·RLRT + β·H(σ_all) where H is the average signature entropy.
    """
    # Preserve original section
    original = sheaf.get_section(node).copy()

    # Create a temporary token set: we do not store the full set, so we approximate
    # by recomputing the signature from the original tokens plus the new token.
    # For simplicity we assume the original token list is not available; we instead
    # recompute a signature that hashes the new token together with the old signature.
    # This yields a deterministic proxy signature.
    temp_tokens = set(original.tolist())
    temp_tokens.add(_hash(0, new_token) % MAX64)  # inject new token as a pseudo‑hash value
    sheaf.set_section(node, temp_tokens)  # type: ignore

    # Compute RLRT
    G = global_inconsistency_matrix(sheaf)
    rlrt = rlrt_approximation(G)

    # Compute average entropy over all node signatures
    entropies = [signature_entropy(sheaf.get_section(n)) for n in sheaf.nodes]
    avg_entropy = float(np.mean(entropies))

    # Restore original section
    sheaf._sections[node] = original

    return alpha * rlrt + beta * avg_entropy


def best_node_to_explore(
    sheaf: HybridSheaf,
    candidate_tokens: Dict[str, str],
    alpha: float = 1.0,
    beta: float = 1.0
) -> str:
    """
    Given a mapping node → candidate token (the token that would be observed
    if the agent moves to that node), return the node minimizing the hybrid
    expected entropy.
    """
    if not candidate_tokens:
        raise ValueError("candidate_tokens required")
    scores: Dict[str, float] = {}
    for node, token in candidate_tokens.items():
        scores[node] = hybrid_expected_entropy(sheaf, node, token, alpha, beta)
    # Choose minimal score, break ties alphabetically for determinism
    return min(scores, key=lambda n: (scores[n], n))


# ----------------------------------------------------------------------
# Demonstration functions ------------------------------------------------
# ----------------------------------------------------------------------


def build_hybrid_sheaf_example() -> HybridSheaf:
    """
    Construct a small example sheaf:
        A -- B -- C
    Each node receives a random set of words.
    """
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    node_tokens = {
        "A": random.sample(vocab, k=3),
        "B": random.sample(vocab, k=3),
        "C": random.sample(vocab, k=3),
    }
    edges = [("A", "B"), ("B", "C")]
    return HybridSheaf(node_tokens, edges, k=64)


def hybrid_action_demo(sheaf: HybridSheaf) -> str:
    """
    Simulate an infotaxis step: each neighboring node proposes a token
    (here we pick a random word).  The function returns the node that
    minimizes the hybrid expected entropy.
    """
    vocab = ["iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi"]
    candidate = {node: random.choice(vocab) for node in sheaf.nodes}
    return best_node_to_explore(sheaf, candidate, alpha=0.7, beta=0.3)


def report_sheaf_state(sheaf: HybridSheaf) -> None:
    """Print a concise summary of the sheaf's current signatures and entropy."""
    print("Node signatures (first 5 hashes):")
    for n in sheaf.nodes:
        sig = sheaf.get_section(n)
        print(f"  {n}: {sig[:5].tolist()} ...")
    ent = [signature_entropy(sheaf.get_section(n)) for n in sheaf.nodes]
    print(f"Average signature entropy: {np.mean(ent):.4f}")
    G = global_inconsistency_matrix(sheaf)
    print(f"RLRT approximation: {rlrt_approximation(G):.4f}")


if __name__ == "__main__":
    random.seed(42)
    sheaf = build_hybrid_sheaf_example()
    report_sheaf_state(sheaf)
    chosen = hybrid_action_demo(sheaf)
    print(f"\nBest node to explore next (hybrid infotaxis): {chosen}")