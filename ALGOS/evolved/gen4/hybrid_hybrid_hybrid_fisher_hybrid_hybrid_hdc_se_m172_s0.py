# DARWIN HAMMER — match 172, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_jepa_energy_m52_s2.py (gen2)
# parent_b: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s1.py (gen3)
# born: 2026-05-29T23:27:25Z

"""
Hybrid JEPA-HDC algorithm.

Parents:
- **hybrid_fisher_localization_krampus_chrono_m17_s1.py** (Algorithm A) – extracts
  candidate timestamps from text and scores them with a Fisher-information
  based “information density” using a Gaussian beam.
- **hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s1.py** (Algorithm B) – defines
  a sparse neural architecture using Hyperdimensional Computing (HDC) and a
  Winner-Takes-All (WTA) mechanism. The latent variable *z* encodes hidden
  causes.

Mathematical bridge:
The Fisher score *F(θ)* is a scalar that quantifies how informative a particular
timestamp is. In HDC, the latent variable *z* is bound to a hyperdimensional
vector *h* using a Winner-Takes-All (WTA) mechanism. We treat the Fisher score
of a timestamp candidate as the latent *z* bound to the hyperdimensional vector
*h*. Thus each date candidate yields a JEPA-like energy function in the
representation space.

    E(candidate) = ‖ encoder(t) – predictor( encoder(t_prev), F(θ) ) ‖²
                   = ‖ h – wta( F(θ), h ) ‖²
"""
import math
import random
import sys
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Algorithm A – Fisher-based date extraction
# ---------------------------------------------------------------------------

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def parse_loose_datetime(raw: str) -> datetime | None:
    text = raw.strip().strip("'\"`[]()")
    if not text:
        return None

# ----------------------------------------------------------------------
# Hyperdimensional primitives
# ----------------------------------------------------------------------
Vector = List[int]  # bipolar hypervector


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big"
    )
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: Iterable[Vector]) -> Vector:
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    dim = len(vecs[0])
    for v in vecs:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    # element-wise sum then binarize by sign
    summed = [sum(comp) for comp in zip(*vecs)]
    return [1 if s >= 0 else -1 for s in summed]


def similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity for bipolar vectors (identical to normalized dot)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / len(a)  # because |a| = |b| = sqrt(dim) for bipolar vectors


# ----------------------------------------------------------------------
# Sparse WTA primitives
# ----------------------------------------------------------------------
def wta(z: float, h: Vector) -> Vector:
    """Winner-Takes-All mechanism."""
    max_idx = np.argmax(np.abs(h))
    return [1 if i == max_idx else -1 for i in range(len(h))]


def jepa_energy(candidate: datetime, reference: datetime, fisher_score: float, h: Vector) -> float:
    """JEPA-like energy function."""
    encoder_candidate = symbol_vector(candidate.strftime("%Y-%m-%d %H:%M:%S"), len(h))
    encoder_reference = symbol_vector(reference.strftime("%Y-%m-%d %H:%M:%S"), len(h))
    predictor_input = bind(encoder_reference, [fisher_score])
    predictor_output = wta(fisher_score, h)
    return np.linalg.norm(np.array(encoder_candidate) - np.array(predictor_output))


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------

def hybrid_fisher_jepa_energy(candidate: datetime, reference: datetime, fisher_score: float, h: Vector) -> float:
    """Hybrid Fisher-JEPA energy function."""
    return jepa_energy(candidate, reference, fisher_score, h)


def hybrid_fisher_wta_energy(candidate: datetime, reference: datetime, fisher_score: float, h: Vector) -> float:
    """Hybrid Fisher-WTA energy function."""
    return np.linalg.norm(np.array(symbol_vector(candidate.strftime("%Y-%m-%d %H:%M:%S"), len(h))) - np.array(wta(fisher_score, h)))


def hybrid_jepa_wta_energy(candidate: datetime, reference: datetime, fisher_score: float, h: Vector) -> float:
    """Hybrid JEPA-WTA energy function."""
    encoder_candidate = symbol_vector(candidate.strftime("%Y-%m-%d %H:%M:%S"), len(h))
    encoder_reference = symbol_vector(reference.strftime("%Y-%m-%d %H:%M:%S"), len(h))
    predictor_input = bind(encoder_reference, [fisher_score])
    predictor_output = wta(fisher_score, h)
    return np.linalg.norm(np.array(encoder_candidate) - np.array(predictor_output))


if __name__ == "__main__":
    # Smoke test
    candidate = datetime.now(timezone.utc)
    reference = datetime.now(timezone.utc) - datetime.timedelta(days=1)
    fisher_score = fisher_score(1643723400)
    h = random_vector(len(h))
    print(hybrid_fisher_jepa_energy(candidate, reference, fisher_score, h))
    print(hybrid_fisher_wta_energy(candidate, reference, fisher_score, h))
    print(hybrid_jepa_wta_energy(candidate, reference, fisher_score, h))