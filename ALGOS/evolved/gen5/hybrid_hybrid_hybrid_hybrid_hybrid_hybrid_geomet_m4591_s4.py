# DARWIN HAMMER — match 4591, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1260_s1.py (gen4)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_percyphon_hyb_m1442_s2.py (gen3)
# born: 2026-05-29T23:56:57Z

import numpy as np
import math
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional

# ----------------------------------------------------------------------
# Parent A – Model pool & entropy utilities (enhanced)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """Immutable descriptor for a loaded entity."""
    name: str
    ram_mb: int
    tier: str  # e.g., "T1", "T2", "T3"


class ModelPool:
    """
    LRU‑based pool that stores arbitrary payloads (e.g., Morphology objects)
    while respecting a RAM ceiling.  The pool is an OrderedDict mapping a
    unique key to a tuple ``(ModelTier, payload)``; the order reflects recent
    usage (most‑recently‑used at the end).  Eviction removes the least‑recently‑used
    entries until the new item fits.
    """
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self._store: "OrderedDict[str, Tuple[ModelTier, Any]]" = OrderedDict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _used(self) -> int:
        return sum(tier.ram_mb for tier, _ in self._store.values())

    def _ensure_fits(self, ram_mb: int) -> None:
        """Raise if a single item cannot ever fit."""
        if ram_mb > self.ram_ceiling_mb:
            raise RuntimeError(
                f"Item requires {ram_mb} MB > pool ceiling {self.ram_ceiling_mb} MB"
            )

    def _evict_until_fits(self, required_mb: int) -> None:
        """Evict LRU entries until ``required_mb`` can be accommodated."""
        while self._store and self._used() + required_mb > self.ram_ceiling_mb:
            evicted_key, (evicted_tier, _) = self._store.popitem(last=False)
            # Debug hook – in production replace with proper logging
            # print(f"Evicted {evicted_key} ({evicted_tier.ram_mb} MB)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_loaded(self, name: str) -> bool:
        return name in self._store

    def load(self, model: ModelTier, payload: Any = None) -> None:
        """
        Load a model (and optional payload) respecting tier‑exclusion rules
        and RAM constraints.  If the model already exists it is refreshed
        as most‑recently‑used.
        """
        self._ensure_fits(model.ram_mb)

        # Tier‑exclusion: T3 cannot coexist with any T2 entry.
        if model.tier == "T3" and any(t.tier == "T2" for t, _ in self._store.values()):
            raise RuntimeError("T3 mutually exclusive with any resident T2 model")

        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._evict_until_fits(model.ram_mb)

        # Refresh if already present – move to end (MRU)
        if model.name in self._store:
            self._store.pop(model.name)

        self._store[model.name] = (model, payload)

    def get(self, name: str) -> Tuple[ModelTier, Any]:
        """Retrieve a stored entry and promote it to MRU."""
        tier, payload = self._store.pop(name)
        self._store[name] = (tier, payload)  # re‑insert as MRU
        return tier, payload

    def random_entry(self) -> Tuple[str, ModelTier, Any]:
        """Pick a random entry; also promotes it to MRU."""
        if not self._store:
            raise RuntimeError("ModelPool is empty")
        name = random.choice(list(self._store.keys()))
        tier, payload = self.get(name)
        return name, tier, payload

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        used = self._used()
        return f"<ModelPool {len(self)} items, {used}/{self.ram_ceiling_mb} MB used>"


def calculate_entropy(feature_counts: Dict[str, int]) -> float:
    """Shannon entropy H = - Σ p_i log₂ p_i (bits)."""
    total = sum(feature_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for cnt in feature_counts.values():
        if cnt <= 0:
            continue
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy


def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Proportion of records that are uniquely identifiable."""
    if total_records == 0:
        return 0.0
    return unique_quasi_identifiers / total_records


# ----------------------------------------------------------------------
# Parent B – Geometry & Clifford‑style operations (enhanced)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Physical description of an object."""
    length: float
    width: float
    height: float
    mass: float = 1.0  # default mass if not supplied

    def volume(self) -> float:
        return self.length * self.width * self.height

    def surface_area(self) -> float:
        l, w, h = self.length, self.width, self.height
        return 2 * (l * w + w * h + h * l)


def sphericity_index(morph: Morphology) -> float:
    """
    Classic sphericity: Φ = (π^{1/3} (6V)^{2/3}) / A,
    where V is volume and A is surface area.
    Returns a value in (0, 1]; 1 for a perfect sphere.
    """
    V = morph.volume()
    A = morph.surface_area()
    if V <= 0 or A <= 0:
        raise ValueError("Morphology dimensions must be positive")
    phi = (math.pi ** (1.0 / 3.0) * (6 * V) ** (2.0 / 3.0)) / A
    return min(max(phi, 0.0), 1.0)


def flatness_index(morph: Morphology) -> float:
    """
    Flatness defined as the ratio of the smallest to the largest dimension.
    Values close to 0 indicate a very flat object, 1 indicates a cube‑like shape.
    """
    dims = sorted([morph.length, morph.width, morph.height])
    return dims[0] / dims[-1] if dims[-1] > 0 else 0.0


def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Simple Clifford‑like product: a·b (dot) + a⊙b (element‑wise).
    Works for 1‑D arrays of equal length.
    """
    if a.shape != b.shape:
        raise ValueError("Arrays must share shape for geometric_product")
    return np.dot(a, b) + np.multiply(a, b)


def gaussian_noise(shape: Tuple[int, ...], scale: float) -> np.ndarray:
    """Zero‑mean Gaussian noise with standard deviation = ``scale``."""
    return np.random.normal(loc=0.0, scale=scale, size=shape)


def ttt_ga_forward(W: np.ndarray,
                   R: np.ndarray,
                   x: np.ndarray,
                   eta_w: float,
                   eta_r: float) -> np.ndarray:
    """
    Three‑tier GA forward pass.
    - Additive Gaussian noise (more realistic than uniform).
    - Supports broadcasting where appropriate.
    """
    noise_w = gaussian_noise(W.shape, eta_w)
    noise_r = gaussian_noise(R.shape, eta_r)
    return np.dot(W + noise_w, np.dot(R + noise_r, x))


# ----------------------------------------------------------------------
# Fusion – Information‑Geometric Transformation (deepened)
# ----------------------------------------------------------------------
def _entropy_weighted_factor(base: float, entropy: float, eps: float = 1e-6) -> float:
    """
    Produce a bounded scaling factor.
    - ``base`` is a geometric scalar (sphericity or flatness) in [0,1].
    - ``entropy`` is in bits; we map it via a softplus to keep positivity.
    - Result is clamped to a reasonable interval to avoid exploding matrices.
    """
    # Softplus ensures smooth positivity even for zero entropy.
    soft_entropy = math.log1p(math.exp(entropy))  # ≈ log(1+e^H)
    factor = base * (soft_entropy + eps)
    # Clamp to [eps, 10] – 10 is an empirical ceiling for stability.
    return max(min(factor, 10.0), eps)


def hybrid_ttt_ga_entropy(x_seq: np.ndarray,
                          W: np.ndarray,
                          R: np.ndarray,
                          eta_w: float,
                          eta_r: float,
                          morphology: Morphology,
                          feature_counts: Dict[str, int]) -> np.ndarray:
    """
    Core hybrid operation with deeper integration:
    1. Compute Shannon entropy ``H``.
    2. Derive geometry scalars ``S`` (sphericity) and ``F`` (flatness).
    3. Convert them into bounded, entropy‑aware scaling factors.
    4. Apply *matrix exponentiation* to embed the scaling into a smooth
       Lie‑group transformation (avoids abrupt magnitude changes).
    5. Run the GA forward pass.
    """
    H = calculate_entropy(feature_counts)
    S = sphericity_index(morphology)
    F = flatness_index(morphology)

    scale_s = _entropy_weighted_factor(S, H)
    scale_f = _entropy_weighted_factor(F, H)

    # Use matrix exponential to embed scaling into a continuous transformation.
    # This is more mathematically faithful than simple element‑wise scaling.
    exp_W = scipy_expm(W * scale_s) if scale_s != 1.0 else W
    exp_R = scipy_expm(R * scale_f) if scale_f != 1.0 else R

    return ttt_ga_forward(exp_W, exp_R, x_seq, eta_w, eta_r)


# ----------------------------------------------------------------------
# Helper: matrix exponential (fallback if SciPy unavailable)
# ----------------------------------------------------------------------
def scipy_expm(A: np.ndarray) -> np.ndarray:
    """
    Compute the matrix exponential using a Padé approximant.
    This implementation avoids a hard SciPy dependency while staying
    numerically stable for moderate‑sized matrices (≤ 100×100).
    """
    from numpy.linalg import norm

    # Scaling‑and‑squaring parameters (from Higham 2005)
    maxnorm = 5.371920351148152
    n_squarings = max(0, int(np.ceil(np.log2(norm(A, ord=1) / maxnorm))))
    A_scaled = A / (2 ** n_squarings)

    # Padé coefficients for order 13
    c = [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ]

    A2 = A_scaled @ A_scaled
    A4 = A2 @ A2
    A6 = A4 @ A2

    U = A_scaled @ (
        c[13] * A6 + c[11] * A4 + c[9] * A2 + c[7] * np.eye(A.shape[0])
    )
    V = c[12] * A6 + c[10] * A4 + c[8] * A2 + c[6] * np.eye(A.shape[0])

    P = U + V
    Q = -U + V

    R = np.linalg.solve(Q, P)

    # Undo scaling‑and‑squaring
    for _ in range(n_squarings):
        R = R @ R
    return R


# ----------------------------------------------------------------------
# Integration utilities
# ----------------------------------------------------------------------
def load_morphology_into_pool(pool: ModelPool,
                              morphology: Morphology,
                              name: str,
                              tier: str = "T1") -> None:
    """
    Convert a ``Morphology`` into a ``ModelTier`` whose RAM cost is proportional
    to its physical volume (rounded up).  The morphology object itself is stored
    as the payload for later retrieval.
    """
    volume = morphology.volume()                     # cubic units
    ram_mb = max(1, int(math.ceil(volume * 8.0)))    # 8 MB per unit³ (tunable)
    model = ModelTier(name=name, ram_mb=ram_mb, tier=tier)
    pool.load(model, payload=morphology)


def evaluate_hybrid_system(pool: ModelPool,
                           x_seq: np.ndarray,
                           W: np.ndarray,
                           R: np.ndarray,
                           eta_w: float,
                           eta_r: float,
                           feature_counts: Dict[str, int],
                           unique_quasi_identifiers: int,
                           total_records: int) -> Dict[str, Any]:
    """
    Full pipeline demonstration:
    1. Randomly (LRU‑aware) pick a loaded morphology.
    2. Compute the hybrid transformation.
    3. Compute reconstruction risk.
    Returns a dictionary with all intermediate artefacts for inspection.
    """
    if not len(pool):
        raise RuntimeError("ModelPool is empty – load at least one Morphology.")

    name, tier, morph = pool.random_entry()

    transformed = hybrid_ttt_ga_entropy(
        x_seq=x_seq,
        W=W,
        R=R,
        eta_w=eta_w,
        eta_r=eta_r,
        morphology=morph,
        feature_counts=feature_counts,
    )

    risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)

    return {
        "selected_morphology": name,
        "tier": tier.tier,
        "morphology": morph,
        "entropy_bits": calculate_entropy(feature_counts),
        "sphericity": sphericity_index(morph),
        "flatness": flatness_index(morph),
        "output_vector": transformed,
        "reconstruction_risk": risk,
        "pool_state": repr(pool),
    }