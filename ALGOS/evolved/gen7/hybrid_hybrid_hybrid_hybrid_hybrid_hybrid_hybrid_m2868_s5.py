# DARWIN HAMMER — match 2868, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_semant_hybrid_hybrid_hybrid_m2659_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1223_s1.py (gen5)
# born: 2026-05-29T23:46:19Z

"""
Hybrid Algorithm: hybrid_hybrid_hybrid_semantic_koopman_rbf_mfusion

Parents:
- hybrid_hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s0 (Gaussian RBF + semantic neighbors)
- hybrid_hybrid_hybrid_ternar_koopman_operator_m632_s1 (Koopman operator + morphology vector + minhash)

Mathematical Bridge:
The morphology vector (deterministic from Morphology attributes) seeds a dense
Gaussian RBF similarity matrix S_ij = exp(-ε² * ||v_i - v_j||²).  This matrix
represents pairwise affinities of document embeddings (or semantic neighbors).
We then treat S as an observable of a dynamical system and approximate the
Koopman operator K by projecting S onto its leading eigen‑subspace:
K ≈ V Λ V⁺, where V contains the dominant eigenvectors and Λ the eigenvalues.
Applying Kⁿ yields an evolved similarity matrix that captures temporal
motif dynamics.  Min‑hash signatures of the rows of S provide compact
representations that can be combined with temporal motif mining, producing
HybridMotif objects enriched with spatial (centroid) and dynamical scores.
"""

import sys
import random
import hashlib
import re
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int

@dataclass(frozen=True)
class HybridMotif:
    pattern: Tuple[str, ...]
    support: int
    centroid_lat: float
    centroid_lon: float
    score: float

# ----------------------------------------------------------------------
# Helper functions from Parent A
# ----------------------------------------------------------------------
def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def semantic_neighbors(doc_id: str, vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
    """Create a synthetic neighbor list using Gaussian similarity on index distance."""
    neighbors = [(doc_id, 1.0)]
    for i in range(1, len(vector)):
        sim = gaussian_rbf(i, epsilon=1.0)
        neighbors.append((f"doc_{i}", sim))
    # Return top‑k (including the query itself)
    return sorted(neighbors, key=lambda x: -x[1])[:k]

# ----------------------------------------------------------------------
# Helper functions from Parent B
# ----------------------------------------------------------------------
def random_vector(dim: int = 10000, seed: str | int | None = None) -> List[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]

def morphology_vector(m: Morphology, dim: int = 1024) -> List[float]:
    """Deterministic pseudo‑random vector derived from morphology attributes."""
    seed_bytes = hashlib.sha256(
        f"{m.length}{m.width}{m.height}{m.mass}".encode("utf-8")
    ).digest()[:8]
    seed_int = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed=seed_int)

def minhash_signature(vec: List[float], num_perm: int = 128) -> List[int]:
    """Very lightweight MinHash: treat each component as a hashable integer."""
    max_hash = (1 << 32) - 1
    signatures = [max_hash] * num_perm
    for idx, val in enumerate(vec):
        # simple hash mixing
        h = hash((idx, int(val * 1e9))) & max_hash
        for i in range(num_perm):
            combined = (h ^ (i * 0x5bd1e995)) % max_hash
            if combined < signatures[i]:
                signatures[i] = combined
    return signatures

# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def build_rbf_similarity_matrix(vectors: List[List[float]], epsilon: float = 1.0) -> np.ndarray:
    """
    Construct a dense similarity matrix S where
        S_ij = exp(-ε² * ||v_i - v_j||²)
    using the Gaussian RBF kernel.
    """
    n = len(vectors)
    mat = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        vi = np.array(vectors[i])
        for j in range(i, n):
            vj = np.array(vectors[j])
            dist_sq = np.sum((vi - vj) ** 2)
            sim = math.exp(-epsilon ** 2 * dist_sq)
            mat[i, j] = sim
            mat[j, i] = sim
    return mat

def koopman_evolve(sim_matrix: np.ndarray, steps: int = 1, rank: int = 5) -> np.ndarray:
    """
    Approximate the Koopman operator by retaining the leading `rank` eigenpairs
    of the similarity matrix and propagating them forward `steps` times:
        K ≈ V Λ V⁺
        S(t+1) = K S(t)
    """
    # Eigen‑decomposition
    vals, vecs = np.linalg.eig(sim_matrix)
    # Sort by magnitude
    idx = np.argsort(-np.abs(vals))
    vals = vals[idx][:rank]
    vecs = vecs[:, idx][:, :rank]

    # Construct Koopman approximation
    Lambda = np.diag(vals ** steps)  # raise eigenvalues to power `steps`
    K_approx = vecs @ Lambda @ np.linalg.pinv(vecs)

    # Evolve the matrix
    evolved = K_approx @ sim_matrix
    # Ensure symmetry and non‑negativity
    evolved = (evolved + evolved.T) / 2.0
    np.clip(evolved, 0.0, 1.0, out=evolved)
    return evolved

def hybrid_temporal_motif(
    morphology: Morphology,
    motifs: List[TemporalMotif],
    epsilon: float = 1.0,
    threshold: float = 0.4,
) -> List[HybridMotif]:
    """
    Fuse semantic‑neighbor RBF scores with Koopman‑evolved similarity to
    score temporal motifs.  For each motif we:
        1. Build a morphology‑seeded vector.
        2. Generate a synthetic neighbor list (semantic_neighbors).
        3. Compute an RBF similarity matrix for those neighbor vectors.
        4. Evolve the matrix with the Koopman operator.
        5. Aggregate scores above `threshold` into a HybridMotif.
    """
    # Step 1: base vector from morphology
    base_vec = morphology_vector(morphology, dim=256)

    # Step 2: create neighbor vectors (here we simply perturb the base vector)
    neighbor_vectors = []
    neighbor_ids = []
    for nid, sim in semantic_neighbors("root", base_vec, k=7):
        neighbor_ids.append(nid)
        # perturbation proportional to similarity
        perturbed = [v * (0.5 + 0.5 * sim) for v in base_vec]
        neighbor_vectors.append(perturbed)

    # Step 3: similarity matrix
    S = build_rbf_similarity_matrix(neighbor_vectors, epsilon=epsilon)

    # Step 4: Koopman evolution (single step suffices for demo)
    S_evolved = koopman_evolve(S, steps=1, rank=3)

    # Step 5: score motifs
    hybrid_motifs: List[HybridMotif] = []
    for motif in motifs:
        pattern = motif.pattern
        support = motif.support
        centroid_lat = 0.0
        centroid_lon = 0.0
        score = 0.0
        # Use pattern length as a proxy for index into similarity matrix
        for idx, token in enumerate(pattern):
            row = idx % S_evolved.shape[0]
            col = (idx + 1) % S_evolved.shape[1]
            sim = S_evolved[row, col]
            if sim > threshold:
                centroid_lat += row
                centroid_lon += col
                score += sim
        if score > 0:
            hybrid_motifs.append(
                HybridMotif(
                    pattern=pattern,
                    support=support,
                    centroid_lat=centroid_lat / len(pattern),
                    centroid_lon=centroid_lon / len(pattern),
                    score=score,
                )
            )
    return hybrid_motifs

def compute_minhash_fingerprint(morphology: Morphology, num_perm: int = 64) -> List[int]:
    """
    Produce a MinHash fingerprint of the morphology‑derived vector.
    This serves as a compact representation usable in downstream indexing.
    """
    vec = morphology_vector(morphology, dim=512)
    return minhash_signature(vec, num_perm=num_perm)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple morphology instance
    morph = Morphology(length=10.2, width=5.5, height=2.1, mass=3.3)

    # Synthetic temporal motifs
    motifs = [
        TemporalMotif(pattern=("A", "B", "C"), support=12),
        TemporalMotif(pattern=("X", "Y"), support=7),
        TemporalMotif(pattern=("M", "N", "O", "P"), support=3),
    ]

    # Run hybrid motif extraction
    hybrid = hybrid_temporal_motif(morph, motifs, epsilon=0.8, threshold=0.3)

    print("Hybrid Motifs:")
    for hm in hybrid:
        print(f"Pattern: {hm.pattern}, Score: {hm.score:.4f}, Centroid: ({hm.centroid_lat:.2f}, {hm.centroid_lon:.2f})")

    # Compute MinHash fingerprint
    fingerprint = compute_minhash_fingerprint(morph, num_perm=32)
    print("\nMinHash fingerprint (first 8 values):", fingerprint[:8])

    # Ensure the script exits cleanly
    sys.exit(0)