# DARWIN HAMMER — match 1317, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s1.py (gen2)
# parent_b: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s2.py (gen2)
# born: 2026-05-29T23:35:10Z

"""
Hybrid Count‑Min / MinHash – Decision Hygiene Entropy Pruning

Parents:
- hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s1.py (Count‑Min sketch, MinHash LSH, cardinality estimate)
- hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s2.py (Shannon entropy of a feature vector, exponential pruning schedule)

Mathematical Bridge:
The Count‑Min sketch **C** is a linear map  x ↦ C = A · x, where *A* is a sparse binary matrix defined by hash functions.
Row‑sums of **C** give a non‑negative vector **c** whose normalised form **p = c / Σc** is a discrete probability distribution.
The Shannon entropy **H(p)** quantifies the information retained after the dimensionality‑reduction performed by the sketch.
We feed this entropy into the decision‑hygiene pruning schedule:

    p₀(t) = min(1, λ·exp(−α·t))                     # base exponential prune probability
    γ(p) = 1 + H(p) / H_max                         # entropy normalisation (H_max = log₂|rows|)
    p_hybrid(t, p) = p₀(t) / γ(p)                   # high‑entropy sketches are pruned less

The hybrid algorithm therefore:
1. Builds a Count‑Min sketch of the input multiset.
2. Derives an entropy measure from the sketch’s row‑sum distribution.
3. Applies entropy‑aware pruning to the sketch matrix.
4. Optionally groups documents with MinHash LSH using the pruned sketches.

The three public functions below illustrate this pipeline.
"""

import hashlib
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Core sketch utilities (from Parent A)
# ----------------------------------------------------------------------
def count_min_sketch(items, width=64, depth=4):
    """
    Build a Count‑Min sketch matrix C ∈ ℕ^{depth×width}.
    Each item updates one cell per row using a row‑specific hash.
    """
    table = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16)
            col = h % width
            table[d, col] += 1
    return table

def minhash_lsh_index(docs, shingle_size=5):
    """
    Very lightweight MinHash LSH: each document is represented by the
    minimum SHA‑1 hash (truncated) of its shingle set.
    Returns a dict {bucket_key: [doc_id, ...]}.
    """
    buckets = defaultdict(list)
    for doc_id, text in docs.items():
        shingles = {text[i:i+shingle_size] for i in range(0, len(text) - shingle_size + 1)}
        if not shingles:
            key = "empty"
        else:
            key = min(hashlib.sha1(s.encode()).hexdigest()[:6] for s in shingles)
        buckets[key].append(doc_id)
    return dict(buckets)

# ----------------------------------------------------------------------
# Entropy and pruning utilities (from Parent B)
# ----------------------------------------------------------------------
def shannon_entropy(probs):
    """
    Compute Shannon entropy (base 2) of a probability vector.
    Zero entries are ignored because 0·log₂0 = 0.
    """
    probs = np.asarray(probs, dtype=np.float64)
    nonzero = probs > 0
    return -np.sum(probs[nonzero] * np.log2(probs[nonzero]))

def exponential_prune_prob(t, lam=0.9, alpha=0.1):
    """
    Base pruning probability p₀(t) = min(1, λ·exp(−α·t)).
    """
    return min(1.0, lam * math.exp(-alpha * t))

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_sketch_with_entropy(items, width=64, depth=4):
    """
    1. Build a Count‑Min sketch from *items*.
    2. Derive a probability distribution from row‑sums.
    3. Return (sketch, entropy, H_max).
    """
    sketch = count_min_sketch(items, width, depth)
    row_sums = sketch.sum(axis=1).astype(np.float64)
    total = row_sums.sum()
    if total == 0:
        probs = np.full_like(row_sums, 1.0 / len(row_sums))
    else:
        probs = row_sums / total
    entropy = shannon_entropy(probs)
    H_max = math.log2(len(row_sums)) if len(row_sums) > 1 else 0.0
    return sketch, entropy, H_max

def entropy_aware_prune(sketch, t, lam=0.9, alpha=0.1, entropy=0.0, H_max=0.0):
    """
    Apply entropy‑modulated pruning to *sketch*.
    Each cell is zeroed independently with probability p_hybrid(t, entropy).
    Returns a new pruned sketch (numpy array).
    """
    base_p = exponential_prune_prob(t, lam, alpha)
    gamma = 1.0 + (entropy / H_max) if H_max > 0 else 1.0
    p_hybrid = base_p / gamma
    # Clip to [0,1] for safety
    p_hybrid = max(0.0, min(1.0, p_hybrid))

    mask = np.random.rand(*sketch.shape) > p_hybrid
    pruned = sketch * mask.astype(sketch.dtype)
    return pruned

def hybrid_lsh_pruned_index(docs, t, lam=0.9, alpha=0.1, width=64, depth=4):
    """
    End‑to‑end hybrid pipeline:
    * For each document, build a sketch from its characters (as items).
    * Compute entropy from the sketch.
    * Prune the sketch with entropy‑aware probability.
    * Insert the document id into a MinHash LSH bucket based on its (pruned) sketch.
    Returns:
        - lsh_index: dict {bucket_key: [doc_id, ...]}
        - sketches: dict {doc_id: pruned_sketch}
    """
    lsh_index = defaultdict(list)
    sketches = {}

    for doc_id, text in docs.items():
        # Use individual characters as items for the sketch
        items = list(text)
        sketch, entropy, H_max = hybrid_sketch_with_entropy(items, width, depth)
        pruned = entropy_aware_prune(sketch, t, lam, alpha, entropy, H_max)

        # Derive a deterministic bucket key from the pruned sketch:
        # flatten, hash, take first 6 hex digits.
        flat_bytes = pruned.tobytes()
        bucket_key = hashlib.sha1(flat_bytes).hexdigest()[:6]

        lsh_index[bucket_key].append(doc_id)
        sketches[doc_id] = pruned

    return dict(lsh_index), sketches

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic documents
    docs = {
        "doc1": "the quick brown fox jumps over the lazy dog",
        "doc2": "lorem ipsum dolor sit amet consectetur adipiscing elit",
        "doc3": "aaaaaa aaaaaa aaaaaa aaaaaa aaaaaa aaaaaa",  # low entropy
    }

    t = 5  # time step for pruning schedule
    lsh, sketches = hybrid_lsh_pruned_index(docs, t)

    print("LSH buckets and contained doc IDs:")
    for bucket, ids in lsh.items():
        print(f"  {bucket}: {ids}")

    print("\nSample pruned sketch shapes (should be (depth, width)):")
    for doc_id, sk in sketches.items():
        print(f"  {doc_id}: {sk.shape}, non‑zero cells = {np.count_nonzero(sk)}")