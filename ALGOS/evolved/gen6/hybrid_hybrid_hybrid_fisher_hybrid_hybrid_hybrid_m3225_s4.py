# DARWIN HAMMER — match 3225, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_hybrid_m1899_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hard_t_m1107_s2.py (gen4)
# born: 2026-05-29T23:48:41Z

import numpy as np
import random
import hashlib
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
FEATURE_DIM = 10          # dimension of raw text features
MASTER_DIM = 5            # dimension of the master (resource) space
PHASH_BITS = 64           # max bits for perceptual hash
ALPHA_CURV = 0.3          # learning rate for curvature updates
BETA_ADJ = 2.0            # scaling for adjacency conversion
SEED = 42                 # reproducibility

np.random.seed(SEED)
random.seed(SEED)

# ----------------------------------------------------------------------
# Core linear algebra objects (shared across the hybrid model)
# ----------------------------------------------------------------------
# Projection matrix P extracts the first MASTER_DIM components of v
P = np.zeros((MASTER_DIM, FEATURE_DIM))
for i in range(MASTER_DIM):
    P[i, i] = 1.0

# Fixed master resource vector m (learned once, then kept constant)
m_master = np.random.rand(MASTER_DIM)

# Curvature matrix C (symmetric, initialized to zero)
C = np.zeros((MASTER_DIM, MASTER_DIM))

# Adjacency weight matrix W (symmetric, initialized to small positive values)
W = np.full((MASTER_DIM, MASTER_DIM), 0.01)
np.fill_diagonal(W, 0.0)

# Pheromone levels per master node (non‑negative)
pheromone = np.zeros(MASTER_DIM)


# ----------------------------------------------------------------------
# Utility functions (more faithful to the original ideas)
# ----------------------------------------------------------------------
def compute_phash(values: np.ndarray) -> int:
    """Perceptual hash: 1 bit per value indicating >= median (max PHASH_BITS bits)."""
    if values.size == 0:
        return 0
    median = np.median(values)
    bits = 0
    for v in values[:PHASH_BITS]:
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two integers."""
    return (a ^ b).bit_count()


def broadcast_probability(phase: int, step: int) -> float:
    """A decaying broadcast probability that depends on phase and step."""
    base = 0.8
    decay = 0.95 ** (phase + step)
    return max(0.05, base * decay)


def extract_full_features(text: str) -> np.ndarray:
    """Deterministic pseudo‑feature extraction using a hash of the text."""
    # Produce a reproducible float vector in [0,1) from the SHA‑256 digest
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Split digest into 8‑byte chunks, convert to int, then normalise
    vals = []
    for i in range(FEATURE_DIM):
        chunk = digest[i*2:(i*2)+2]  # 2 bytes → 16 bits
        num = int.from_bytes(chunk, byteorder="big")
        vals.append(num / 65535.0)
    return np.array(vals, dtype=np.float64)


def compatibility_score(v: np.ndarray, m: np.ndarray) -> float:
    """
    Compatibility score s = (P @ v)ᵀ m.
    This projects the high‑dimensional feature onto the master space before
    measuring alignment with the master resource vector.
    """
    projected = P @ v                # shape (MASTER_DIM,)
    return float(projected @ m)      # scalar


def bayesian_curvature_update(C: np.ndarray, projected: np.ndarray, score: float) -> np.ndarray:
    """
    Bayesian‑style update of the curvature matrix.
    Treat the outer product of the projected vector as evidence.
    C_new = (1 - α) * C + α * score * (proj ⊗ proj)
    """
    outer = np.outer(projected, projected)
    return (1.0 - ALPHA_CURV) * C + ALPHA_CURV * score * outer


def update_pheromone(phash: int) -> None:
    """
    Convert the perceptual hash into a pheromone increment vector.
    Each bit set adds a small constant to the corresponding node.
    """
    global pheromone
    for i in range(min(MASTER_DIM, PHASH_BITS)):
        if (phash >> (PHASH_BITS - 1 - i)) & 1:
            pheromone[i] += 0.01
    # Optional decay to keep pheromone bounded
    pheromone *= 0.99


def curvature_to_adjacency(C: np.ndarray) -> np.ndarray:
    """
    Map curvature values to edge weights.
    Use a normalized exponential decay: w_ij = exp(-β * (1 - C_ij_norm))
    """
    # Normalise curvature to [0,1] (avoid division by zero)
    max_abs = np.max(np.abs(C)) if np.max(np.abs(C)) != 0 else 1.0
    C_norm = (C / max_abs + 1.0) / 2.0  # now in [0,1]
    adj = np.exp(-BETA_ADJ * (1.0 - C_norm))
    np.fill_diagonal(adj, 0.0)
    return adj


# ----------------------------------------------------------------------
# Hybrid model encapsulating the full update cycle
# ----------------------------------------------------------------------
class HybridGraphPheromoneModel:
    def __init__(self):
        self.P = P
        self.m = m_master
        self.C = C.copy()
        self.W = W.copy()
        self.pheromone = pheromone.copy()

    def step(self, text: str, phase: int, step: int) -> dict:
        """
        Perform a full update:
        1. Extract deterministic features.
        2. Compute perceptual hash and update pheromone.
        3. Compute compatibility score.
        4. Update curvature matrix.
        5. Re‑compute adjacency matrix with pheromone bias.
        Returns a dict with diagnostics.
        """
        # 1. Feature extraction
        v = extract_full_features(text)                     # (FEATURE_DIM,)

        # 2. Pheromone update via perceptual hash
        phash = compute_phash(v)
        update_pheromone(phash)

        # 3. Compatibility score
        projected = self.P @ v                               # (MASTER_DIM,)
        score = compatibility_score(v, self.m)

        # 4. Curvature update
        self.C = bayesian_curvature_update(self.C, projected, score)

        # 5. Adjacency recomputation
        base_adj = curvature_to_adjacency(self.C)
        # Additive pheromone bias (broadcast probability scales the bias)
        prob = broadcast_probability(phase, step)
        pheromone_bias = np.outer(self.pheromone, self.pheromone) * prob
        self.W = base_adj + pheromone_bias
        np.clip(self.W, 0.0, 1.0, out=self.W)  # keep within [0,1]
        np.fill_diagonal(self.W, 0.0)

        return {
            "features": v,
            "phash": phash,
            "score": score,
            "curvature_norm": np.linalg.norm(self.C),
            "adjacency_mean": np.mean(self.W),
            "pheromone": self.pheromone.copy(),
        }

    def get_curvature(self) -> np.ndarray:
        return self.C.copy()

    def get_adjacency(self) -> np.ndarray:
        return self.W.copy()

    def get_pheromone(self) -> np.ndarray:
        return self.pheromone.copy()


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = HybridGraphPheromoneModel()
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence blends statistics with logic.",
        "Graph curvature can guide diffusion processes."
    ]
    for i, txt in enumerate(texts, 1):
        diagnostics = model.step(txt, phase=i, step=i)
        print(f"--- Step {i} ---")
        print(f"Score: {diagnostics['score']:.4f}")
        print(f"Curvature norm: {diagnostics['curvature_norm']:.4f}")
        print(f"Adjacency mean weight: {diagnostics['adjacency_mean']:.4f}")
        print(f"Pheromone vector: {diagnostics['pheromone']}")
        print()
    # Final matrices (optional inspection)
    # print("Final Curvature Matrix:\n", model.get_curvature())
    # print("Final Adjacency Matrix:\n", model.get_adjacency())