# DARWIN HAMMER — match 5622, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_decisi_m1317_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_physar_hybrid_temporal_moti_m1504_s1.py (gen6)
# born: 2026-05-30T00:03:32Z

"""Hybrid Count‑Min Sketch & Physarum‑Inspired Conductance Network

Parents:
- hybrid_hybrid_hybrid_sketch_hybrid_hybrid_decisi_m1317_s4.py (Count‑Min sketch, entropy‑aware pruning)
- hybrid_hybrid_hybrid_physar_hybrid_temporal_moti_m1504_s1.py (Physarum‑style conductance dynamics)

Mathematical Bridge:
Each row *r* of a Count‑Min sketch C ∈ ℕ^{depth×width} aggregates frequencies of streamed items.
Let s_r = Σ_j C[r, j] be the row‑sum and define a pressure vector p ∈ ℝ^{depth} by normalising s:
    p_r = s_r / Σ_k s_k .
Treat rows as nodes of a Physarum network. Conductance g_{uv} on edge (u,v) evolves according to
    g_{uv}(t+Δt) = max(0, g_{uv} + Δt·(γ·|Φ_{uv}| − δ·g_{uv}))
where Φ_{uv} = flux(g_{uv}, ℓ_{uv}=1, p_u, p_v) and γ = 1 + H/H_max scales gain with the Shannon
entropy H of the normalised row‑sum distribution (H_max = log₂(depth)). High‑entropy sketches
thus retain larger conductances, i.e. are pruned less aggressively. After a few dynamical steps
the conductance matrix is used as an entropy‑aware pruning mask for the original sketch.

The module implements this fused pipeline and provides three public functions:
    • count_min_sketch – builds the sketch.
    • physarum_conductance_update – performs one Physarum update step driven by sketch pressures.
    • hybrid_prune_sketch – runs the full hybrid process and returns a pruned sketch.
"""

import hashlib
import math
import random
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Count‑Min sketch utilities
# ----------------------------------------------------------------------
def count_min_sketch(items, width: int = 64, depth: int = 4) -> np.ndarray:
    """
    Build a Count‑Min sketch matrix C ∈ ℕ^{depth×width}.
    Each item updates one cell per row using a row‑specific SHA‑256 hash.
    """
    table = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16)
            col = h % width
            table[d, col] += 1
    return table

def row_sum_distribution(sketch: np.ndarray) -> np.ndarray:
    """
    Return the normalised row‑sum distribution p (probability vector).
    """
    row_sums = sketch.sum(axis=1).astype(float)
    total = row_sums.sum()
    if total == 0:
        return np.full_like(row_sums, 1.0 / row_sums.size)
    return row_sums / total

def shannon_entropy(p: np.ndarray) -> float:
    """
    Shannon entropy H(p) in bits.
    """
    mask = p > 0
    return -np.sum(p[mask] * np.log2(p[mask]))

# ----------------------------------------------------------------------
# Parent B – Physarum‑style flow utilities (adapted)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """
    Physical flux Φ = g/ℓ * (p_a - p_b).
    """
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float, gain: float, decay: float, dt: float) -> float:
    """
    Conductance dynamics: g ← max(0, g + dt·(gain·|q| - decay·g)).
    """
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def init_conductance_matrix(depth: int, seed: int = 0) -> np.ndarray:
    """
    Initialise a symmetric conductance matrix G ∈ ℝ^{depth×depth} with small positive values.
    Diagonal entries are zero (no self‑loops).
    """
    rng = np.random.default_rng(seed)
    G = rng.uniform(0.01, 0.05, size=(depth, depth))
    np.fill_diagonal(G, 0.0)
    # Symmetrise
    G = (G + G.T) / 2.0
    return G

def physarum_conductance_update(G: np.ndarray, pressures: np.ndarray,
                                gain_factor: float, decay: float,
                                dt: float = 1.0) -> np.ndarray:
    """
    Perform one Physarum update step over all unordered node pairs.
    - G: current conductance matrix (depth×depth, symmetric, zero diagonal).
    - pressures: node pressures derived from sketch row‑sums (length = depth).
    - gain_factor: γ = 1 + H/H_max (entropy‑scaled gain).
    Returns the updated conductance matrix.
    """
    depth = pressures.shape[0]
    new_G = G.copy()
    for i in range(depth):
        for j in range(i + 1, depth):
            g_ij = G[i, j]
            phi = flux(g_ij, edge_length=1.0, pressure_a=pressures[i], pressure_b=pressures[j])
            g_new = update_conductance(g_ij, phi, gain=gain_factor, decay=decay, dt=dt)
            new_G[i, j] = new_G[j, i] = g_new
    return new_G

def hybrid_prune_sketch(items,
                        width: int = 64,
                        depth: int = 4,
                        steps: int = 5,
                        dt: float = 1.0,
                        decay: float = 0.1,
                        prune_threshold: float = 0.02,
                        seed: int = 0) -> np.ndarray:
    """
    Full hybrid pipeline:
    1. Build Count‑Min sketch C.
    2. Compute row‑sum pressure vector p and its entropy H.
    3. Initialise conductance matrix G.
    4. Iterate Physarum dynamics with gain γ = 1 + H/H_max.
    5. Derive a row‑wise pruning mask from the final conductances.
    6. Apply mask to C (zeroing columns whose row conductance falls below threshold).
    Returns the pruned sketch matrix.
    """
    # Step 1
    C = count_min_sketch(items, width=width, depth=depth)

    # Step 2 – pressures and entropy
    p = row_sum_distribution(C)                     # length = depth
    H = shannon_entropy(p)
    H_max = math.log2(depth) if depth > 1 else 1.0
    gain_factor = 1.0 + H / H_max                    # γ

    # Step 3 – conductance matrix
    G = init_conductance_matrix(depth, seed=seed)

    # Step 4 – dynamics
    for _ in range(steps):
        G = physarum_conductance_update(G, p, gain_factor, decay, dt=dt)

    # Step 5 – row‑wise pruning mask
    # Row conductance is the sum of incident conductances.
    row_conductance = G.sum(axis=1)                  # shape (depth,)
    # Normalise to [0,1] for a probabilistic mask.
    max_conduct = row_conductance.max() if row_conductance.max() > 0 else 1.0
    row_mask = row_conductance / max_conduct          # higher = keep more

    # Step 6 – apply mask to each row of the sketch.
    # For each row r we keep a column j only if row_mask[r] >= prune_threshold.
    # This is a simple entropy‑aware hard pruning.
    pruned = C.copy()
    for r in range(depth):
        if row_mask[r] < prune_threshold:
            pruned[r, :] = 0
    return pruned

# ----------------------------------------------------------------------
# Auxiliary – MinHash LSH (kept from Parent A for completeness)
# ----------------------------------------------------------------------
def minhash_signature(item_set, num_perm: int = 64):
    """
    Compute a MinHash signature (array of length num_perm) for a set of hashable items.
    """
    sig = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    for token in item_set:
        token_bytes = str(token).encode()
        for i in range(num_perm):
            h = int(hashlib.sha256(f"{i}:{token_bytes}".decode()).hexdigest(), 16)
            if h < sig[i]:
                sig[i] = h
    return sig

def lsh_band_hash(signature: np.ndarray, band_size: int = 8) -> Tuple[int, ...]:
    """
    Split a MinHash signature into bands and hash each band to produce a bucket key.
    """
    if signature.size % band_size != 0:
        raise ValueError("signature length must be a multiple of band_size")
    bands = signature.reshape(-1, band_size)
    return tuple(int(hashlib.sha256(b.tobytes()).hexdigest(), 16) for b in bands)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic stream
    random.seed(42)
    items = [random.randint(0, 1000) for _ in range(500)]

    # Run hybrid pruning
    pruned_sketch = hybrid_prune_sketch(
        items,
        width=32,
        depth=4,
        steps=8,
        dt=0.5,
        decay=0.05,
        prune_threshold=0.1,
        seed=123,
    )

    # Basic sanity checks
    assert pruned_sketch.shape == (4, 32)
    total_counts = pruned_sketch.sum()
    print(f"Pruned sketch total count: {total_counts}")

    # Demonstrate MinHash on a small document collection
    docs = [
        {"the", "quick", "brown", "fox"},
        {"jumped", "over", "the", "lazy", "dog"},
        {"the", "quick", "blue", "hare"},
    ]
    signatures = [minhash_signature(d) for d in docs]
    bands = [lsh_band_hash(sig) for sig in signatures]
    print("LSH band hashes:", bands)