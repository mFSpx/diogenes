# DARWIN HAMMER — match 1084, survivor 1
# gen: 2
# parent_a: hybrid_privacy_sketches_m15_s2.py (gen1)
# parent_b: hybrid_fractional_hdc_counterfactual_effec_m38_s1.py (gen1)
# born: 2026-05-29T23:32:42Z

"""Hybrid CMS‑HDC module.

Parents:
- **hybrid_privacy_sketches_m15_s2.py** – provides a Count‑Min Sketch (CMS) for
  compact frequency estimation and a reconstruction‑risk score based on the
  ratio *unique_quasi_identifiers / total_records*.
- **hybrid_fractional_hdc_counterfactual_effec_m38_s1.py** – supplies hyperdimensional
  computing (HDC) primitives: random hypervectors, binding, fractional‑power
  binding and similarity measures for causal effect encoding.

Mathematical bridge:
The CMS matrix is interpreted as a weighted collection of (row, column) tokens.
Each token is hashed to a *random complex hypervector* (unit‑magnitude).  The
CMS → hypervector conversion aggregates these token hypervectors weighted by the
cell counts, yielding a single high‑dimensional representation `cms_hv`.  This
hypervector can then be *bound* to a causal hypervector (e.g. representing a
treatment or policy) using the HDC binding operator (element‑wise multiplication).
Fractional‑power binding (`fractional_power`) modulates the strength of the causal
relationship by raising the phase of the bound hypervector to a real exponent.
The resulting bound hypervector is finally used to adjust the reconstruction‑risk
score, producing a hybrid risk estimate that accounts for both frequency‑based
privacy leakage and encoded causal influence.

The module demonstrates three hybrid operations:
1. `cms_to_hv` – convert a Count‑Min Sketch into a complex hypervector.
2. `bind_causal_to_cms` – bind a causal hypervector to the sketch hypervector,
   optionally with fractional power.
3. `hybrid_risk_with_causal_effect` – compute a risk score that blends the
   privacy‑risk ratio with a causal effect derived from the bound hypervector.
"""

import hashlib
import random
import math
import sys
from pathlib import Path
from collections import defaultdict
from typing import Iterable, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – Count‑Min Sketch utilities
# ---------------------------------------------------------------------------

def _cms_hash(item: str, depth: int, width: int) -> List[int]:
    """Return a list of column indices, one per hash row."""
    return [
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ]

def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> np.ndarray:
    """Build a Count‑Min Sketch matrix as a NumPy int array."""
    cms = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        cols = _cms_hash(item, depth, width)
        for d, c in enumerate(cols):
            cms[d, c] += 1
    return cms

def _estimate_cardinality_from_cms(cms: np.ndarray) -> int:
    """Coarse cardinality estimator: distinct non‑zero cells divided by depth."""
    nonzero = np.count_nonzero(cms)
    depth = cms.shape[0]
    return max(1, nonzero // depth)

def hyperloglog_cardinality(items: Iterable[str]) -> int:
    """Exact distinct count (placeholder for a real HLL implementation)."""
    return len(set(items))

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Ratio‑based risk, clipped to [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

# ---------------------------------------------------------------------------
# Parent B – Hyperdimensional Computing primitives (complex variant)
# ---------------------------------------------------------------------------

def random_hv(d: int = 10000, seed: int | None = None) -> np.ndarray:
    """Generate a random unit‑magnitude complex hypervector of dimension *d*."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
    return np.exp(1j * theta)

def bind(hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
    """Binding operator for complex hypervectors – element‑wise multiplication."""
    return hv1 * hv2

def fractional_power(hv: np.ndarray, exponent: float) -> np.ndarray:
    """
    Fractional‑power binding: raise the phase of each component to *exponent*.
    For unit‑magnitude complex numbers this is equivalent to
    e^{i * theta * exponent}.
    """
    angles = np.angle(hv)
    return np.exp(1j * angles * exponent)

def similarity(hv1: np.ndarray, hv2: np.ndarray) -> float:
    """Cosine‑like similarity for complex hypervectors (real part of inner product)."""
    inner = np.vdot(hv1, hv2)  # conjugate of hv1 times hv2
    return inner.real / (np.linalg.norm(hv1) * np.linalg.norm(hv2))

# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def cms_to_hv(cms: np.ndarray, hv_dim: int = 10000, seed: int | None = None) -> np.ndarray:
    """
    Convert a Count‑Min Sketch matrix into a single complex hypervector.

    Each (row, col) cell is hashed to a deterministic random hypervector;
    the cell count weights that hypervector before aggregation.
    """
    depth, width = cms.shape
    agg = np.zeros(hv_dim, dtype=np.complex128)

    # deterministic seed for reproducibility
    base_seed = seed if seed is not None else 0

    for d in range(depth):
        for w in range(width):
            cnt = cms[d, w]
            if cnt == 0:
                continue
            # Derive a per‑cell seed from its coordinates and the base seed
            cell_seed = hash((base_seed, d, w)) & ((1 << 31) - 1)
            token_hv = random_hv(d=hv_dim, seed=cell_seed)
            agg += cnt * token_hv
    # Normalize to unit magnitude per component (optional but stabilises later ops)
    norm = np.linalg.norm(agg)
    return agg / norm if norm != 0 else agg

def bind_causal_to_cms(
    cms_hv: np.ndarray,
    causal_factor: str,
    exponent: float = 1.0,
    hv_dim: int = 10000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Encode a causal factor as a hypervector, optionally apply fractional power,
    and bind it to the sketch hypervector.

    Parameters
    ----------
    cms_hv : complex hypervector derived from a CMS.
    causal_factor : string identifier of the causal variable (e.g. "treatment").
    exponent : fractional power exponent modelling strength (default 1.0 = full binding).
    hv_dim : dimension of hypervectors.
    seed : optional base seed for reproducibility.
    """
    # Hash the causal factor to obtain a deterministic seed
    factor_seed = hash((seed, causal_factor)) & ((1 << 31) - 1)
    factor_hv = random_hv(d=hv_dim, seed=factor_seed)

    if exponent != 1.0:
        factor_hv = fractional_power(factor_hv, exponent)

    return bind(cms_hv, factor_hv)

def hybrid_risk_with_causal_effect(
    items: Iterable[str],
    causal_factor: str,
    cms_width: int = 64,
    cms_depth: int = 4,
    hv_dim: int = 10000,
    exponent: float = 1.0,
    seed: int | None = None,
) -> Tuple[float, float]:
    """
    Compute a privacy reconstruction risk score and adjust it using a causal
    effect encoded via HDC binding.

    Returns
    -------
    (raw_risk, adjusted_risk)
    """
    # 1️⃣ Build CMS and estimate cardinalities
    cms = count_min_sketch(items, width=cms_width, depth=cms_depth)
    unique_qi_est = _estimate_cardinality_from_cms(cms)
    total_records = hyperloglog_cardinality(items)

    raw_risk = reconstruction_risk_score(unique_qi_est, total_records)

    # 2️⃣ Convert CMS to hypervector
    cms_hv = cms_to_hv(cms, hv_dim=hv_dim, seed=seed)

    # 3️⃣ Bind causal factor (with optional fractional power)
    bound_hv = bind_causal_to_cms(
        cms_hv,
        causal_factor,
        exponent=exponent,
        hv_dim=hv_dim,
        seed=seed,
    )

    # 4️⃣ Derive a scalar causal influence from similarity to a reference vector.
    #    For the demo we use a random reference hypervector.
    ref_hv = random_hv(d=hv_dim, seed=seed if seed is not None else 42)
    causal_influence = similarity(bound_hv, ref_hv)  # in [-1, 1]

    # Map influence to a scaling factor in [0.5, 1.5] (neutral at 0)
    scale = 1.0 + 0.5 * causal_influence

    adjusted_risk = max(0.0, min(1.0, raw_risk * scale))
    return raw_risk, adjusted_risk

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Small synthetic dataset
    sample_items = [
        "alice|age=34|zip=12345",
        "bob|age=28|zip=12345",
        "carol|age=34|zip=67890",
        "dave|age=40|zip=67890",
        "eve|age=28|zip=12345",
    ]

    # Causal factor (e.g., presence of a privacy‑preserving policy)
    factor = "policy_enabled"

    raw, adj = hybrid_risk_with_causal_effect(
        items=sample_items,
        causal_factor=factor,
        cms_width=32,
        cms_depth=3,
        hv_dim=2048,
        exponent=0.7,
        seed=12345,
    )

    print(f"Raw reconstruction risk: {raw:.4f}")
    print(f"Adjusted risk after causal binding: {adj:.4f}")