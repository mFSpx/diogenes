# DARWIN HAMMER — match 3785, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_decrea_hybrid_hybrid_hybrid_m1630_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1986_s2.py (gen5)
# born: 2026-05-29T23:52:58Z

"""Hybrid Fusion of Decreasing‑Pruning, Epistemic Certainty, Physarum Flux,
and Regret‑Weighted Similarity.

Parents:
- hybrid_hybrid_hybrid_decrea_hybrid_hybrid_hybrid_m1630_s1.py (decreasing
  pruning, epistemic certainty, virtual store)
- hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1986_s2.py (physarum
  flux, Span confidence, MinHash similarity, regret engine)

Mathematical Bridge:
The *confidence score* of a `Span` (parent B) is interpreted as an
*epistemic certainty* (parent A).  This scalar multiplies the physical flux
computed by the physarum dynamics.  The resulting quantity is further
scaled by a *MinHash similarity* between the current textual context and a
reference set (regret engine).  The product

    q = flux * span_score * similarity

drives both the conductance update of the physarum network and the adaptive
learning‑rate modulation of the weight matrix.  Simultaneously the virtual
store evolves according to a differential equation whose forcing term is
the same `q`, thereby coupling all three subsystems into a unified hybrid
engine."""

import math
import random
import sys
import pathlib
import hashlib
import numpy as np
from typing import List, Tuple, Callable

# ----------------------------------------------------------------------
# Core data structures (borrowed from Parent B)
# ----------------------------------------------------------------------
class Span:
    """Labeled text span with an epistemic confidence score."""
    def __init__(self, start: int, end: int, text: str, label: str, score: float):
        self.start = start
        self.end = end
        self.text = text
        self.label = label
        self.score = score

def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physical flux through an edge (Ohm‑like law)."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

# ----------------------------------------------------------------------
# Hybrid Engine
# ----------------------------------------------------------------------
class HybridEngine:
    """Unified system combining pruning, epistemic weighting, physarum flux,
    and regret‑weighted similarity."""
    def __init__(
        self,
        d_in: int,
        d_out: int,
        seed: int = 0,
        base_eta: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1.0,
        store_decay: float = 0.99,
        num_edges: int = None,
    ) -> None:
        self.d_in = d_in
        self.d_out = d_out
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.base_eta = base_eta          # baseline learning rate
        self.alpha = alpha                # store differential coefficient
        self.beta = beta                  # store differential coefficient
        self.dt = dt
        self.store_decay = store_decay

        # weight matrix (parent A)
        self.W = self.np_rng.random((d_in, d_out))

        # virtual store (vector of length d_in)
        self.store = np.zeros(d_in)

        # conductance vector for physarum edges (size = num_edges)
        self.num_edges = num_edges if num_edges is not None else d_in
        self.conductance = np.ones(self.num_edges)

    # ------------------------------------------------------------------
    # 1. Decreasing‑rate pruning (parent A)
    # ------------------------------------------------------------------
    def prune_probability(self, t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
        """Monotonically increasing probability that a weight is pruned.

        The functional form mirrors the parent implementation:
            p(t) = min(1, λ * (1 - exp(-α t)))
        """
        if t < 0 or lam < 0 or alpha < 0:
            raise ValueError('t, lam, and alpha must be non‑negative')
        p = lam * (1.0 - math.exp(-alpha * t))
        return min(1.0, p)

    # ------------------------------------------------------------------
    # 2. MinHash‑based similarity (parent B)
    # ------------------------------------------------------------------
    def minhash_similarity(self,
                           context: str,
                           reference_contexts: List[str],
                           num_perm: int = 64) -> float:
        """Approximate Jaccard similarity using simple MinHash.

        For each permutation we hash the shingle set and keep the minimum.
        The similarity is the fraction of permutations where the minima match.
        """
        if not reference_contexts:
            return 0.0

        # tokenise into word‑level shingles
        def shingles(text: str) -> set:
            words = text.lower().split()
            return set(words)

        ctx_set = shingles(context)
        ref_sets = [shingles(r) for r in reference_contexts]

        matches = 0
        for i in range(num_perm):
            # deterministic pseudo‑random hash function per permutation
            salt = f'salt{i}'.encode()
            def h(x: str) -> int:
                return int(hashlib.sha256(salt + x.encode()).hexdigest(), 16)

            min_ctx = min((h(w) for w in ctx_set), default=0)
            # check if any reference set shares the same min hash
            if any(min((h(w) for w in rs), default=0) == min_ctx for rs in ref_sets):
                matches += 1

        return matches / num_perm

    # ------------------------------------------------------------------
    # 3. Conductance update with epistemic and similarity weighting
    # ------------------------------------------------------------------
    def update_conductance(self,
                           edge_idx: int,
                           pressure_a: float,
                           pressure_b: float,
                           edge_length: float,
                           span: Span,
                           similarity: float,
                           dt: float = None,
                           gain: float = 1.0,
                           decay: float = 0.01) -> None:
        """Update a single conductance entry.

        q = flux * span_score * similarity
        dC/dt = gain * q - decay * C
        C(t+dt) = C + dt * dC/dt
        """
        if dt is None:
            dt = self.dt
        c = self.conductance[edge_idx]
        f = flux(c, edge_length, pressure_a, pressure_b)
        q = f * span.score * similarity
        dc = gain * q - decay * c
        self.conductance[edge_idx] = max(0.0, c + dt * dc)

    # ------------------------------------------------------------------
    # 4. Weight matrix adaptation using the same q as a learning‑rate factor
    # ------------------------------------------------------------------
    def adapt_weights(self,
                      input_vec: np.ndarray,
                      grad: np.ndarray,
                      t: float,
                      similarity: float) -> None:
        """Perform a single gradient‑descent step with adaptive η.

        η(t) = base_eta * (1 - prune_probability) * similarity
        """
        p = self.prune_probability(t)
        eta = self.base_eta * (1.0 - p) * similarity
        update = -eta * grad
        self.W += np.outer(input_vec, update)

        # optional pruning: zero‑out rows with probability p
        mask = self.rng.random() < p
        if mask:
            row = self.rng.randrange(self.d_in)
            self.W[row, :] = 0.0

    # ------------------------------------------------------------------
    # 5. Virtual store dynamics (parent A) driven by q
    # ------------------------------------------------------------------
    def update_store(self, q_vec: np.ndarray) -> None:
        """Euler integration of store differential equation:

        dS/dt = α * q_vec - β * S
        S(t+dt) = (1 - store_decay) * S + dt * (α * q_vec - β * S)
        """
        dq = self.alpha * q_vec - self.beta * self.store
        self.store = (1.0 - self.store_decay) * self.store + self.dt * dq

    # ------------------------------------------------------------------
    # 6. One unified hybrid step
    # ------------------------------------------------------------------
    def step(self,
             t: float,
             context: str,
             reference_contexts: List[str],
             spans: List[Span],
             pressures: Tuple[np.ndarray, np.ndarray],
             edge_lengths: np.ndarray,
             input_vec: np.ndarray,
             grad: np.ndarray) -> None:
        """Execute a full hybrid iteration.

        - Compute similarity between current and reference contexts.
        - For each edge update conductance using the associated Span.
        - Build a vector q (flux‑scaled) and feed it to the virtual store.
        - Adapt the weight matrix with the same similarity factor.
        """
        similarity = self.minhash_similarity(context, reference_contexts)

        # Ensure lengths match
        n = min(self.num_edges, len(spans), len(edge_lengths))
        q_vec = np.zeros(self.num_edges)

        for i in range(n):
            self.update_conductance(
                edge_idx=i,
                pressure_a=pressures[0][i],
                pressure_b=pressures[1][i],
                edge_length=edge_lengths[i],
                span=spans[i],
                similarity=similarity,
            )
            # recompute flux after conductance update for store forcing
            f = flux(self.conductance[i], edge_lengths[i],
                     pressures[0][i], pressures[1][i])
            q_vec[i] = f * spans[i].score * similarity

        self.update_store(q_vec)
        self.adapt_weights(input_vec, grad, t, similarity)

# ----------------------------------------------------------------------
# Helper functions exposing the hybrid operations (requirement: ≥3)
# ----------------------------------------------------------------------
def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Standalone wrapper for HybridEngine.prune_probability."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non‑negative')
    p = lam * (1.0 - math.exp(-alpha * t))
    return min(1.0, p)

def minhash_similarity(context: str,
                       reference_contexts: List[str],
                       num_perm: int = 64) -> float:
    """Standalone MinHash similarity (same algorithm as in HybridEngine)."""
    if not reference_contexts:
        return 0.0

    def shingles(text: str) -> set:
        return set(text.lower().split())

    ctx_set = shingles(context)
    ref_sets = [shingles(r) for r in reference_contexts]

    matches = 0
    for i in range(num_perm):
        salt = f'salt{i}'.encode()
        def h(x: str) -> int:
            return int(hashlib.sha256(salt + x.encode()).hexdigest(), 16)
        min_ctx = min((h(w) for w in ctx_set), default=0)
        if any(min((h(w) for w in rs), default=0) == min_ctx for rs in ref_sets):
            matches += 1
    return matches / num_perm

def hybrid_step_example() -> None:
    """Demonstrates a full hybrid iteration on synthetic data."""
    d_in, d_out = 4, 3
    engine = HybridEngine(d_in, d_out, seed=42, num_edges=5)

    t = 0.5
    context = "the quick brown fox jumps over the lazy dog"
    refs = [
        "a fast dark-colored animal leaps above a sleepy canine",
        "quick movement of a fox in the forest"
    ]
    spans = [
        Span(0, 3, "the quick brown", "NP", 0.9),
        Span(4, 5, "fox", "N", 0.8),
        Span(6, 9, "jumps over", "VP", 0.85),
        Span(10, 13, "the lazy dog", "NP", 0.95),
        Span(14, 15, ".", "PUNCT", 1.0)
    ]
    pressures_a = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    pressures_b = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    edge_lengths = np.array([1.0, 1.2, 0.9, 1.1, 1.0])
    input_vec = np.array([0.5, -0.3, 0.1, 0.7])
    grad = np.random.randn(d_out) * 0.01

    engine.step(
        t=t,
        context=context,
        reference_contexts=refs,
        spans=spans,
        pressures=(pressures_a, pressures_b),
        edge_lengths=edge_lengths,
        input_vec=input_vec,
        grad=grad,
    )

    # Simple sanity prints (can be removed)
    print("Conductance:", engine.conductance)
    print("Store:", engine.store)
    print("Weight matrix sample:", engine.W[:2, :2])

if __name__ == "__main__":
    hybrid_step_example()