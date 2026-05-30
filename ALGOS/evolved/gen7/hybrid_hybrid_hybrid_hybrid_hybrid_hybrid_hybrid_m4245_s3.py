# DARWIN HAMMER — match 4245, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2731_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2439_s3.py (gen6)
# born: 2026-05-29T23:54:28Z

"""Hybrid Endpoint‑Voronoi‑Liquid‑Decision Algorithm
===================================================

Parent A: *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2731_s1.py* – provides
`EndpointCircuitBreaker` whose `failure_rate()` is a normalised privacy‑load.

Parent B: *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2439_s3.py* – builds a
Voronoi tessellation, maintains a per‑region hidden state `h_r` that follows a
liquid‑time‑constant ODE

    dh_r/dt = -h_r / τ_r + f_r

with the time constant `τ_r` derived from a hyper‑dimensional binding between the
region centroid and a sketch of the feature vectors that fall inside the region.

**Mathematical bridge**  
The bridge is the *privacy‑load* of the circuit breaker.  For each Voronoi region
we instantiate an `EndpointCircuitBreaker`.  Its current `failure_rate()` is used
as a multiplicative factor on the ODE time‑constant:

    τ_r = τ_base * (1 + λ * cb.failure_rate())

where `λ` is a tunable sensitivity.  Consequently, a region that experiences many
failures evolves more slowly, which in turn makes the decision‑hygiene score
more conservative.  The ODE update can also trigger the circuit breaker: if the
updated hidden state exceeds a threshold we record a failure, otherwise a success.

The module below implements the fused mathematics, provides three core functions
that demonstrate the hybrid operation, and includes a tiny smoke‑test.
"""

import math
import random
import sys
import pathlib
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Endpoint circuit breaker (unchanged)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple circuit‑breaker that tracks consecutive failures.

    The normalised failure rate in ``[0, 1]`` is interpreted as a *privacy‑load*.
    """

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        """Reset the failure counter and close the breaker."""
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        """Increment the failure counter and open the breaker if needed."""
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        """Return ``True`` if the circuit is closed (i.e. work may proceed)."""
        return not self.open

    def failure_rate(self) -> float:
        """Normalised failure rate in ``[0, 1]``."""
        return min(self.failures / self.failure_threshold, 1.0)


# ----------------------------------------------------------------------
# Parent B – Voronoi + liquid‑time‑constant ODE (trimmed & adapted)
# ----------------------------------------------------------------------
Point = Tuple[float, float]          # 2‑D coordinate
FeatureVec = np.ndarray              # 1‑D float array
HyperVector = np.ndarray              # binary hyper‑vector (uint8)

def _euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _nearest_region(p: Point, seeds: List[Point]) -> int:
    """Return the index of the seed closest to point ``p``."""
    distances = [_euclidean(p, s) for s in seeds]
    return int(np.argmin(distances))

def voronoi_assign(points: List[Point], seeds: List[Point]) -> List[int]:
    """Assign each point to the nearest Voronoi seed.

    Returns a list ``region_ids`` parallel to ``points``.
    """
    return [_nearest_region(p, seeds) for p in points]

# ----------------------------------------------------------------------
# Simple Count‑Min Sketch (deterministic for reproducibility)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Very small deterministic count‑min sketch using two hash functions."""
    def __init__(self, width: int = 64):
        self.width = width
        self.table = np.zeros(width, dtype=np.int32)

    def _hash(self, key: bytes, seed: int) -> int:
        h = hashlib.blake2b(key, digest_size=4, person=seed.to_bytes(4, "little"))
        return int.from_bytes(h.digest(), "little") % self.width

    def add(self, item: bytes, count: int = 1) -> None:
        for seed in (0xA5A5A5A5, 0x5A5A5A5A):
            idx = self._hash(item, seed)
            self.table[idx] += count

    def estimate(self, item: bytes) -> int:
        return min(self.table[self._hash(item, seed)] for seed in (0xA5A5A5A5, 0x5A5A5A5A))

# ----------------------------------------------------------------------
# Hyper‑dimensional binding utilities
# ----------------------------------------------------------------------
def random_hypervector(dim: int = 1024) -> HyperVector:
    """Generate a random binary hyper‑vector."""
    return np.random.randint(0, 2, size=dim, dtype=np.uint8)

def bind_vectors(a: HyperVector, b: HyperVector) -> HyperVector:
    """Element‑wise XOR binding (binary hyper‑vectors)."""
    return np.bitwise_xor(a, b)

def cosine_similarity(u: HyperVector, v: HyperVector) -> float:
    """Cosine similarity for binary vectors treated as 0/1."""
    dot = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return dot / (norm_u * norm_v)

# ----------------------------------------------------------------------
# Region state handling (liquid‑time‑constant ODE)
# ----------------------------------------------------------------------
@dataclass
class RegionState:
    """Hidden state h_r and supporting structures for a Voronoi region."""
    h: float = 0.0                         # hidden state
    centroid: Point = (0.0, 0.0)           # geometric centre of the region
    sketch: CountMinSketch = CountMinSketch()
    cb: EndpointCircuitBreaker = EndpointCircuitBreaker()
    centroid_hv: HyperVector = random_hypervector()
    sketch_hv: HyperVector = random_hypervector()

def _tau_from_binding(centroid: Point,
                     region_state: RegionState,
                     tau_base: float,
                     lambda_priv: float) -> float:
    """
    Compute the region‑specific time constant τ_r.

    τ_r = τ_base * (1 + λ * failure_rate) * (1 / (1 + sim))
    where ``sim`` is the cosine similarity between the bound centroid‑sketch vector
    and the centroid hyper‑vector.  Higher similarity => smaller denominator => larger τ.
    """
    # Bind centroid hyper‑vector with sketch hyper‑vector
    bound = bind_vectors(region_state.centroid_hv, region_state.sketch_hv)
    sim = cosine_similarity(bound, region_state.centroid_hv)  # in [0,1]
    priv_factor = 1.0 + lambda_priv * region_state.cb.failure_rate()
    tau = tau_base * priv_factor * (1.0 / (1.0 + sim))
    return max(tau, 1e-6)   # avoid division by zero later

def update_region_state(region_state: RegionState,
                       features: List[FeatureVec],
                       tau_base: float = 1.0,
                       lambda_priv: float = 2.0,
                       dt: float = 0.1,
                       failure_threshold: float = 5.0) -> None:
    """
    Perform one ODE step for a region.

    * ``features`` – list of feature vectors that fell inside the region during the
      current batch.
    * ``tau_base`` – baseline time constant.
    * ``lambda_priv`` – sensitivity to the circuit‑breaker privacy load.
    * ``dt`` – integration step.
    * ``failure_threshold`` – if the updated hidden state exceeds this value we
      record a failure in the circuit breaker.
    """
    if not region_state.cb.allow():
        # Circuit open – skip update but still record that we tried
        region_state.cb.record_failure()
        return

    # ---- sketch update -------------------------------------------------
    for fv in features:
        # Hash the feature vector (bytes) into the sketch
        key = fv.tobytes()
        region_state.sketch.add(key)

    # Re‑compute a lightweight sketch‑derived hyper‑vector:
    # we hash the whole sketch table to obtain a deterministic binary vector.
    sketch_bytes = region_state.sketch.table.tobytes()
    region_state.sketch_hv = np.frombuffer(
        hashlib.blake2b(sketch_bytes, digest_size=128).digest(),
        dtype=np.uint8
    ) % 2  # binary

    # ---- ODE step -------------------------------------------------------
    # For illustration we set f_r as the mean L2 norm of the incoming features.
    if features:
        f_r = float(np.mean([np.linalg.norm(fv) for fv in features]))
    else:
        f_r = 0.0

    tau_r = _tau_from_binding(region_state.centroid, region_state, tau_base, lambda_priv)
    dh = (-region_state.h / tau_r + f_r) * dt
    region_state.h += dh

    # ---- circuit‑breaker interaction ------------------------------------
    if region_state.h > failure_threshold:
        region_state.cb.record_failure()
    else:
        region_state.cb.record_success()

# ----------------------------------------------------------------------
# Decision‑hygiene scoring (entropy + Hoeffding bound)
# ----------------------------------------------------------------------
def hygiene_score_with_sketch(features: List[FeatureVec],
                              region_state: RegionState,
                              delta: float = 0.05) -> float:
    """
    Compute a Shannon‑entropy weighted score for a batch of feature vectors.

    Feature weights are derived from the sketch’s empirical frequencies and
    tightened with a Hoeffding bound:

        w_i = (p_i + ε) / Σ_j (p_j + ε)

    where ``p_i`` is the estimated probability from the sketch and ``ε`` is the
    Hoeffding confidence interval.
    The final score is the weighted sum of per‑feature entropies.
    """
    if not features:
        return 0.0

    # Build a frequency map from the sketch (deterministic hashing of each feature)
    freq: Dict[bytes, int] = defaultdict(int)
    total = 0
    for fv in features:
        key = fv.tobytes()
        cnt = region_state.sketch.estimate(key)
        freq[key] += cnt
        total += cnt

    if total == 0:
        # No sketch information – fall back to uniform weighting
        probs = np.full(len(features[0]), 1.0 / len(features[0]))
    else:
        # Hoeffding bound ε = sqrt( (1/(2n)) * ln(2/δ) )
        n = total
        eps = math.sqrt((1.0 / (2 * n)) * math.log(2.0 / delta))
        # Build probability vector per dimension by averaging across items
        # For simplicity we treat each dimension equally and use eps as additive smoothing.
        probs = np.full(len(features[0]), eps)
        # Add empirical probability (average over all items)
        avg_empirical = n / (len(features) * len(features[0]))
        probs += avg_empirical
        probs /= probs.sum()

    # Compute per‑feature entropy and aggregate
    entropies = []
    for fv in features:
        # Normalise feature vector to a probability distribution
        fv_sum = fv.sum()
        if fv_sum == 0:
            continue
        p = fv / fv_sum
        # Clip to avoid log(0)
        p = np.clip(p, 1e-12, 1.0)
        ent = -np.sum(p * np.log2(p))
        entropies.append(ent)

    if not entropies:
        return 0.0

    entropies = np.array(entropies)
    weighted_score = float(np.dot(entropies, probs))
    return weighted_score

# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_process(points: List[Point],
                   features: List[FeatureVec],
                   seeds: List[Point],
                   tau_base: float = 1.0,
                   lambda_priv: float = 2.0) -> Tuple[Dict[int, RegionState], List[float]]:
    """
    End‑to‑end hybrid processing:

    1. Assign each point to a Voronoi region.
    2. Group features by region.
    3. Update each region's hidden state via the liquid‑time‑constant ODE,
       using the circuit‑breaker privacy‑load to modulate τ.
    4. Compute a hygiene score for each region.

    Returns a mapping ``region_id -> RegionState`` and a list of hygiene scores
    ordered by region id.
    """
    region_ids = voronoi_assign(points, seeds)

    # Initialise region states lazily
    regions: Dict[int, RegionState] = {}
    region_features: Dict[int, List[FeatureVec]] = defaultdict(list)

    for rid, pt, fv in zip(region_ids, points, features):
        if rid not in regions:
            # Compute an approximate centroid from the seed (could be refined later)
            regions[rid] = RegionState(centroid=seeds[rid])
        region_features[rid].append(fv)

    scores: List[float] = []
    for rid, state in regions.items():
        batch = region_features.get(rid, [])
        update_region_state(state, batch, tau_base, lambda_priv)
        score = hygiene_score_with_sketch(batch, state)
        scores.append(score)

    return regions, scores

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Generate 20 random 2‑D points and corresponding 5‑dim feature vectors
    points = [(random.random(), random.random()) for _ in range(20)]
    features = [np.random.rand(5).astype(np.float32) for _ in range(20)]

    # Choose 4 Voronoi seeds (could be any subset of points)
    seeds = points[:4]

    regions, scores = hybrid_process(points, features, seeds)

    print("Region states:")
    for rid, state in regions.items():
        print(f"  Region {rid}: h={state.h:.4f}, failure_rate={state.cb.failure_rate():.2f}, open={not state.cb.allow()}")

    print("\nHygiene scores per region:")
    for idx, sc in enumerate(scores):
        print(f"  Region {idx}: score={sc:.4f}")

    sys.exit(0)