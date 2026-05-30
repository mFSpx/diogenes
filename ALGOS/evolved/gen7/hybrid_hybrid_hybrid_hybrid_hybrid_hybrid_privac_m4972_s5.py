# DARWIN HAMMER — match 4972, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1896_s2.py (gen6)
# parent_b: hybrid_hybrid_privacy_model_hybrid_hybrid_fisher_m33_s0.py (gen4)
# born: 2026-05-29T23:59:14Z

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 format with trailing “Z”."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian kernel value.  Width must be > 0."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.
    Used as a *sensitivity* weight – higher curvature → higher weight.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Structural Similarity Index for 1‑D signals.
    The implementation follows the classic definition but works for any
    1‑D NumPy array of equal length.
    """
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")

    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class Model:
    """Simple container for a model's meta‑information."""
    name: str
    ram_mb: int
    tier: str = "standard"


@dataclass
class ModelPool:
    """Manages a collection of models with a global RAM ceiling."""
    ram_ceiling_mb: int = 6000
    _models: List[Model] = field(default_factory=list)

    def add(self, model: Model) -> None:
        """Add a model, ensuring the RAM ceiling is not exceeded."""
        total = sum(m.ram_mb for m in self._models) + model.ram_mb
        if total > self.ram_ceiling_mb:
            raise RuntimeError(
                f"Adding {model.name} would exceed RAM ceiling "
                f"({total} MB > {self.ram_ceiling_mb} MB)."
            )
        self._models.append(model)

    @property
    def names(self) -> List[str]:
        return [m.name for m in self._models]

    @property
    def ram_vector(self) -> np.ndarray:
        """Return a column vector of RAM usage for all models."""
        return np.array([m.ram_mb for m in self._models], dtype=float)

    def index_of(self, name: str) -> int:
        """Lookup index of a model by name – raises KeyError if missing."""
        for i, m in enumerate(self._models):
            if m.name == name:
                return i
        raise KeyError(f"Model {name!r} not found in pool.")


# ----------------------------------------------------------------------
# Hybrid operations – deeper mathematical integration
# ----------------------------------------------------------------------
def privacy_risk_vector(quasi_identifiers: List[int], records: int) -> np.ndarray:
    """
    Normalised reconstruction risk for each quasi‑identifier column.
    Returns a vector of length ``len(quasi_identifiers)``.
    """
    if records <= 0:
        raise ValueError("records must be > 0")
    risks = np.array(
        [max(0.0, min(1.0, q / records)) for q in quasi_identifiers],
        dtype=float,
    )
    return risks


def resource_similarity_matrix(pool: ModelPool) -> np.ndarray:
    """
    Pairwise similarity of model RAM footprints using SSIM.
    The diagonal is forced to 1.0 (perfect self‑similarity).
    """
    ram = pool.ram_vector
    n = ram.size
    sim = np.eye(n, dtype=float)

    # SSIM expects a signal; we treat each RAM value as a 1‑element signal.
    # For a single element the formula collapses to a value in [0,1]
    # that depends only on the two scalars.
    for i in range(n):
        for j in range(i + 1, n):
            # Build tiny 1‑D arrays to feed the SSIM routine.
            xi = np.array([ram[i]], dtype=float)
            xj = np.array([ram[j]], dtype=float)
            sim_ij = ssim(xi, xj, dynamic_range=pool.ram_ceiling_mb)
            sim[i, j] = sim_ij
            sim[j, i] = sim_ij
    return sim


def fisher_weighted_risk(risk_vec: np.ndarray,
                         ram_vec: np.ndarray,
                         center: float = 0.0,
                         width: float = 6000.0) -> np.ndarray:
    """
    Apply Fisher‑score weighting to the privacy‑risk vector.
    The weight for each model is proportional to the curvature of a
    Gaussian centred at ``center`` with scale ``width`` evaluated at the
    model's RAM usage.
    """
    weights = np.array(
        [fisher_score(r, center, width) for r in ram_vec], dtype=float
    )
    # Normalise to keep the scale comparable to the original risk.
    if weights.sum() == 0:
        return risk_vec
    return risk_vec * (weights / weights.sum())


def compute_model_scores(pool: ModelPool,
                         quasi_identifiers: List[int],
                         records: int,
                         reg_lambda: float = 1e-3) -> np.ndarray:
    """
    Solve a regularised linear system that blends privacy risk,
    Fisher‑derived sensitivity, and resource similarity.
    """
    # 1️⃣ Privacy risk (per column) → broadcast to model space.
    risk_vec = privacy_risk_vector(quasi_identifiers, records)

    # 2️⃣ Fisher weighting based on each model's RAM.
    weighted_risk = fisher_weighted_risk(risk_vec, pool.ram_vector)

    # 3️⃣ Resource similarity matrix (positive‑definite by construction).
    S = resource_similarity_matrix(pool)

    # 4️⃣ Regularised inversion (ridge) to avoid singularities.
    A = S + reg_lambda * np.eye(S.shape[0])

    # Solve A · w = weighted_risk  (least‑squares if dimensions differ)
    # If risk vector length ≠ number of models we project it.
    if weighted_risk.size != A.shape[0]:
        # Broadcast by averaging risk across columns.
        avg_risk = weighted_risk.mean()
        b = np.full(A.shape[0], avg_risk, dtype=float)
    else:
        b = weighted_risk

    w = np.linalg.solve(A, b)
    return w


def select_models(pool: ModelPool,
                  quasi_identifiers: List[int],
                  records: int,
                  threshold: float = 0.5) -> List[str]:
    """
    Return the names of models whose computed score exceeds ``threshold``.
    """
    scores = compute_model_scores(pool, quasi_identifiers, records)
    selected = [
        name for name, score in zip(pool.names, scores) if score > threshold
    ]
    return selected


# ----------------------------------------------------------------------
# Smoke test (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a modest pool respecting the RAM ceiling.
    pool = ModelPool(ram_ceiling_mb=6000)
    pool.add(Model(name="model1", ram_mb=1000))
    pool.add(Model(name="model2", ram_mb=2000))
    pool.add(Model(name="model3", ram_mb=3500))

    # Example quasi‑identifiers: counts of distinct values per column.
    quasi_ids = [10, 20, 5]
    total_records = 10_000

    chosen = select_models(pool, quasi_ids, total_records, threshold=0.4)
    print("Selected models:", chosen)