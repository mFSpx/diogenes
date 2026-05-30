# DARWIN HAMMER — match 1075, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_krampu_m196_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m50_s0.py (gen4)
# born: 2026-05-29T23:32:41Z

"""Hybrid Krampus‑Hoeffding Allocation Algorithm

Parents:
- **Parent A** – Hybrid Krampus Count‑Min Sketch with Ollivier‑Ricci curvature.
- **Parent B** – Health‑score driven workshare allocation using Hoeffding bound
  and Gini coefficient.

Mathematical Bridge:
The curvature κᵢ computed from the Krampus semantic graph is injected as an
additional scalar feature of each text node. After projecting the node into a
3‑D coordinate pᵢ = (xᵢ, yᵢ, zᵢ) the coordinate is hashed into a Count‑Min Sketch,
producing a compact frequency estimate f(pᵢ).  These frequency estimates are
interpreted as *model‑specific demand signals* (one key per model name).  The
demand vector f is combined with the health‑score vector h of the deployed
models.  The combined vector is evaluated with the Gini coefficient to measure
fairness, while a Hoeffding bound on the range of the combined vector decides
whether a re‑allocation of workshare is statistically justified.  Thus the
geometric information of Parent A directly influences the adaptive allocation
logic of Parent B, yielding a single unified hybrid system.
"""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent‑B utilities (health score, Hoeffding bound, Gini)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def health_score(reconstruction_risk: float, recovery_priority: float) -> float:
    """Higher health means lower risk and lower recovery priority."""
    return (1.0 - reconstruction_risk) * (1.0 - recovery_priority)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a range r, confidence 1‑δ, and n observations."""
    if r <= 0 or not (0.0 < delta < 1.0) or n <= 0:
        raise ValueError("Invalid parameters for Hoeffding bound")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non‑negative iterable."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0.0:
        raise ValueError("Values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


# ----------------------------------------------------------------------
# Parent‑A utilities (feature extraction, curvature, projection)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Placeholder deterministic feature extractor."""
    # In a real system this would parse the text; here we return a fixed vector.
    return {
        "visceral_ratio": 0.5,
        "tech_ratio": 0.3,
        "legal_osint_ratio": 0.2,
        "ledger_density": 0.1,
        "recursion_score": 0.4,
        "directive_ratio": 0.6,
        "target_density": 0.7,
        "forensic_shield_ratio": 0.8,
        "poetic_entropy": 0.9,
        "dissociative_index": 0.1,
        "wrath_velocity": 0.2,
        "bureaucratic_weaponization_index": 0.3,
        "resource_exhaustion_metric": 0.4,
        "swarm_orchestration_de": 0.5,
    }


def compute_curvature(features: Dict[str, float]) -> float:
    """A surrogate for Ollivier‑Ricci curvature: mean minus std."""
    vals = np.array(list(features.values()))
    return float(vals.mean() - vals.std())


def project_to_3d(features: Dict[str, float], curvature: float) -> Tuple[float, float, float]:
    """Map three primary features plus curvature to a 3‑D point."""
    vals = list(features.values())
    x = vals[0] + curvature
    y = vals[1] + curvature
    z = vals[2] + curvature
    return (x, y, z)


# ----------------------------------------------------------------------
# Count‑Min Sketch (the bridge structure)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Simple Count‑Min Sketch with pairwise‑independent hash functions."""

    def __init__(self, depth: int = 5, width: int = 2000):
        self.depth = depth
        self.width = width
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.seeds = [random.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item: str, i: int) -> int:
        # Combine the item with a seed to obtain a deterministic hash per row.
        return (hash((item, self.seeds[i])) % self.width)

    def add(self, item: str, count: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.table[i, idx] += count

    def estimate(self, item: str) -> int:
        return min(self.table[i, self._hash(item, i)] for i in range(self.depth))


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def build_semantic_curvatures(texts: List[str]) -> List[float]:
    """Compute curvature κᵢ for each text using Parent‑A pipeline."""
    curvatures = []
    for txt in texts:
        feats = extract_full_features(txt)
        κ = compute_curvature(feats)
        curvatures.append(κ)
    return curvatures


def update_sketch_with_texts(sketch: CountMinSketch, texts: List[str]) -> None:
    """Project each text to 3‑D, hash the coordinate, and update the sketch."""
    for txt in texts:
        feats = extract_full_features(txt)
        κ = compute_curvature(feats)
        coord = project_to_3d(feats, κ)
        key = f"{coord[0]:.4f}_{coord[1]:.4f}_{coord[2]:.4f}"
        sketch.add(key)


def allocate_workshare(
    models: List["ModelTier"],
    health_scores: List[float],
    sketch: CountMinSketch,
    delta: float = 0.05,
) -> Tuple[np.ndarray, float, float]:
    """
    Produce a workshare vector across `models`.

    1. Frequency demand `f_m` = sketch estimate of the model name.
    2. Normalise demand to obtain `d_m`.
    3. Combine demand with health: `c_m = d_m * h_m`.
    4. Compute Gini of `c_m` (fairness metric).
    5. Use Hoeffding bound on the range of `c_m` to decide whether to
       re‑normalise (`bound > 0.1` triggers a re‑allocation).
    """
    # 1‑2. Demand from sketch
    demand = np.array([sketch.estimate(m.name) for m in models], dtype=float)
    if demand.sum() == 0.0:
        demand_norm = np.ones_like(demand) / len(demand)
    else:
        demand_norm = demand / demand.sum()

    # 3. Combine with health scores
    health_arr = np.array(health_scores, dtype=float)
    combined = demand_norm * health_arr

    # 4. Fairness metric
    gini = gini_coefficient(combined)

    # 5. Hoeffding‑bound driven decision
    r = combined.max() - combined.min()
    bound = hoeffding_bound(r, delta, len(models))
    if bound > 0.1:
        allocation = combined / combined.sum()  # re‑normalise
    else:
        allocation = demand_norm  # keep demand‑only allocation

    return allocation, gini, bound


# ----------------------------------------------------------------------
# Model definition (Parent‑B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample texts (could be any strings)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Quantum entanglement defies classical intuition.",
        "Economic models often rely on equilibrium assumptions.",
        "Neural networks approximate complex functions.",
    ]

    # Initialise sketch
    cms = CountMinSketch(depth=4, width=1024)

    # Build curvature‑aware sketch
    update_sketch_with_texts(cms, sample_texts)

    # Define a few model tiers
    models = [
        ModelTier("qwen-0.5b", 512, "T1", 1024),
        ModelTier("reasoning-t2", 3000, "T2", 2048),
        ModelTier("tool-t2", 2600, "T2", 2048),
        ModelTier("qwen-7b", 7000, "T3", 4096),
    ]

    # Mock reconstruction risk / recovery priority per model
    healths = []
    for i, m in enumerate(models):
        # Arbitrary deterministic numbers for the demo
        uq = (i + 1) * 10          # unique quasi‑identifiers
        total = 1000 + i * 100    # total records
        risk = reconstruction_risk_score(uq, total)
        priority = 0.1 * i        # increasing priority
        healths.append(health_score(risk, priority))

    # Allocate workshare using the hybrid algorithm
    allocation, gini, bound = allocate_workshare(models, healths, cms)

    # Output results
    print("Workshare allocation per model:")
    for m, w in zip(models, allocation):
        print(f"  {m.name}: {w:.4f}")

    print(f"Gini coefficient of allocation: {gini:.4f}")
    print(f"Hoeffding bound (δ=0.05): {bound:.4f}")

    # Simple sanity checks
    assert math.isclose(allocation.sum(), 1.0, rel_tol=1e-6), "Allocation must sum to 1"
    sys.exit(0)