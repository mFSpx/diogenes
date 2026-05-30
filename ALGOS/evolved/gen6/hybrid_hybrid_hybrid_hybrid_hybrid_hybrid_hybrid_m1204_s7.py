# DARWIN HAMMER — match 1204, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m504_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_tropical_maxp_m1163_s0.py (gen5)
# born: 2026-05-29T23:34:38Z

"""
Hybrid Module: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m504_s2 + hybrid_hybrid_hybrid_endpoi_hybrid_tropical_maxp_m1163_s0

Mathematical Bridge
-------------------
Parent A builds a **Sheaf** and extracts a graph Laplacian **L** (integer entries, zero diagonal,
off‑diagonal ±1).  
Parent B operates in the **tropical (max‑plus) algebra** where the “addition’’ is `max` and the
“multiplication’’ is ordinary `+`.  Tropical matrix multiplication therefore computes, for each
pair (i,j),

    (A ⊗ B)[i,j] = max_k ( A[i,k] + B[k,j] )

If we interpret the Laplacian **L** as a cost matrix, the tropical product naturally uses its
entries as the additive part of the max‑plus operation.  Thus **L** can be fed directly into the
tropical routines (`t_add`, `t_mul`, `t_matmul`) of Parent B.

The fusion therefore consists of:
1. Building the sheaf Laplacian from a textual input (Parent A).
2. Using that Laplacian as the weight matrix in a tropical max‑plus transformation
   (Parent B).
3. Guarding the whole pipeline with an **EndpointCircuitBreaker** that opens after a configurable
   number of failures (e.g., non‑finite results).

The three core functions below demonstrate this hybrid workflow.

"""

import math
import random
import sys
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Sheaf, stylometry and stable hash utilities
# ----------------------------------------------------------------------
FUNCTION_CATS = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Split text into lowercase alphabetic words."""
    return [w for w in (text or "").lower().split() if w.isalpha()]


def lsm_vector(text: str) -> Dict[str, float]:
    """Compute a simple lexical style‑matrix (LSM) vector for the given text."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stable_hash(text: str) -> int:
    """Deterministic 48‑bit hash used as an identifier."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


class Sheaf:
    """Graph‑like structure whose Laplacian will be used as tropical costs."""
    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)
        self.edges = list(edge_list)

    def compute_laplacian(self) -> np.ndarray:
        """Return L where L[i,i] = degree(i) and L[i,j] = -1 for an edge (i,j)."""
        n = len(self.node_dims)
        L = np.zeros((n, n), dtype=float)
        for u, v in self.edges:
            L[u, u] += 1
            L[v, v] += 1
            L[u, v] = -1
            L[v, u] = -1
        return L


def stylometry_features(text: str) -> np.ndarray:
    """Return a fixed‑size feature vector derived from the LSM."""
    vec = lsm_vector(text)
    return np.array(list(vec.values()), dtype=float)


# ----------------------------------------------------------------------
# Parent B – Tropical (max‑plus) algebra and circuit breaker
# ----------------------------------------------------------------------
def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition (max)."""
    return np.maximum(x, y)


def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication (ordinary addition)."""
    return np.add(x, y)


def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical matrix multiplication:
        (A ⊗ B)[i,j] = max_k ( A[i,k] + B[k,j] )
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # Broadcasting to compute all A[i,k] + B[k,j] for each (i,j)
    # Result shape: (i, j)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


@dataclass(frozen=True)
class Morphology:
    """Geometric description used only for demonstration."""
    length: float
    width: float
    height: float
    mass: float

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("width", self.width),
            ("height", self.height),
            ("mass", self.mass),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")


class EndpointCircuitBreaker:
    """Failure counter that opens after a threshold; used to protect the hybrid pipeline."""
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True

    def is_open(self) -> bool:
        return self.open


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def build_sheaf_from_text(text: str) -> Sheaf:
    """
    Construct a simple sheaf where each distinct word becomes a node.
    An edge is added between consecutive words (order‑preserving).
    """
    ws = words(text)
    unique = list(dict.fromkeys(ws))                     # preserve order, drop dupes
    idx = {w: i for i, w in enumerate(unique)}
    edges = [(idx[ws[i]], idx[ws[i + 1]]) for i in range(len(ws) - 1) if ws[i] != ws[i + 1]]
    node_dims = {i: 1 for i in range(len(unique))}      # dummy dimension = 1
    return Sheaf(node_dims, edges)


def tropical_transform_with_laplacian(text: str) -> np.ndarray:
    """
    1. Build a Sheaf from the text and obtain its Laplacian L.
    2. Compute stylometry feature vector f.
    3. Treat f as a row vector and apply tropical matrix multiplication:
           result = f ⊗ L
       (broadcasted so that each entry is max_k (f_k + L_{k,j}))
    The result lives in the same space as the number of nodes.
    """
    sheaf = build_sheaf_from_text(text)
    L = sheaf.compute_laplacian()
    f = stylometry_features(text)                       # shape (C,)
    # Align dimensions: pad f to length = number of nodes (if needed)
    n_nodes = L.shape[0]
    if f.size < n_nodes:
        f_padded = np.pad(f, (0, n_nodes - f.size), constant_values=-np.inf)
    else:
        f_padded = f[:n_nodes]
    # Tropical multiplication (row vector ⊗ matrix)
    result = t_matmul(f_padded[np.newaxis, :], L)       # shape (1, n_nodes)
    return result.squeeze()


def safe_hybrid_process(text: str, breaker: EndpointCircuitBreaker) -> Tuple[bool, np.ndarray]:
    """
    Run the hybrid pipeline under the protection of a circuit breaker.
    Returns (circuit_open, output). If the circuit is open, output is an empty array.
    """
    if breaker.is_open():
        return True, np.array([])

    try:
        out = tropical_transform_with_laplacian(text)
        # Simple sanity check – all entries must be finite
        if not np.all(np.isfinite(out)):
            raise ValueError("Non‑finite values in tropical output")
        breaker.record_success()
        return False, out
    except Exception:
        breaker.record_failure()
        return breaker.is_open(), np.array([])


def combine_morphology_and_output(morph: Morphology, vec: np.ndarray) -> np.ndarray:
    """
    Demonstrate a secondary hybrid operation: scale the tropical output by a
    factor derived from the morphology (e.g., volume / mass).
    """
    if vec.size == 0:
        return vec
    factor = (morph.length * morph.width * morph.height) / morph.mass
    return vec * factor


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "The quick brown fox jumps over the lazy dog while the dog watches."
    breaker = EndpointCircuitBreaker(failure_threshold=2)

    # Run the protected hybrid pipeline a few times, intentionally injecting a failure
    for i in range(4):
        # Introduce an artificial failure on the second iteration
        txt = sample_text if i != 1 else ""
        open_circuit, out_vec = safe_hybrid_process(txt, breaker)
        print(f"Iteration {i}: circuit_open={open_circuit}, output={out_vec}")

    # Demonstrate the morphology scaling
    morph = Morphology(length=2.0, width=1.5, height=0.5, mass=3.0)
    scaled = combine_morphology_and_output(morph, out_vec)
    print("Scaled output:", scaled)