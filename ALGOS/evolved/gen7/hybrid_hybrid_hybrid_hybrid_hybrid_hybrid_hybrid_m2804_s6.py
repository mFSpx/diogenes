# DARWIN HAMMER — match 2804, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1963_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_route_m1803_s2.py (gen6)
# born: 2026-05-29T23:46:08Z

import numpy as np
import random
from dataclasses import dataclass, asdict
from typing import Any, Tuple, Dict, List, Optional

# ----------------------------------------------------------------------
# Global ModelPool (singleton) ------------------------------------------------
# ----------------------------------------------------------------------
class ModelPool:
    """
    A singleton pool that manages loaded models respecting RAM limits and tier
    exclusivity.  The pool is shared across all fused functions so that eviction
    policies have a global effect.
    """
    _instance: Optional["ModelPool"] = None

    def __new__(cls, ram_ceiling_mb: int = 6000):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(ram_ceiling_mb)
        return cls._instance

    def _init(self, ram_ceiling_mb: int) -> None:
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.tier_hierarchy = {"T1": 0, "T2": 1, "T3": 2}

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _can_load(self, model: "ModelTier") -> bool:
        """Check if a model can be loaded without violating constraints."""
        if model.tier not in self.tier_hierarchy:
            raise ValueError(f"Invalid tier '{model.tier}'.")
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 models cannot coexist with any T2 model.")
        return model.ram_mb + self._used() <= self.ram_ceiling_mb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def load(self, model: "ModelTier") -> None:
        """Load a model, raising if constraints would be violated."""
        if not self._can_load(model):
            raise RuntimeError("Cannot load model without eviction.")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: "ModelTier") -> None:
        """
        Load a model, evicting the lowest‑priority models until there is enough
        RAM.  Eviction respects tier hierarchy: T1 < T2 < T3.
        """
        while not self._can_load(model):
            if not self.loaded:
                raise RuntimeError("Insufficient RAM even after full eviction.")
            # Choose the model with the smallest hierarchy value (lowest priority)
            evict_name = min(
                self.loaded,
                key=lambda n: self.tier_hierarchy[self.loaded[n].tier],
            )
            del self.loaded[evict_name]
        self.loaded[model.name] = model

    def current_state(self) -> Dict[str, Any]:
        """Return a snapshot of the pool for debugging / introspection."""
        return {
            "ram_ceiling_mb": self.ram_ceiling_mb,
            "used_mb": self._used(),
            "loaded_models": [asdict(m) for m in self.loaded.values()],
        }

# ----------------------------------------------------------------------
# Data structures (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """
    Simple immutable descriptor for a model.
    """
    name: str
    ram_mb: int
    tier: str  # Expected values: "T1", "T2", "T3"


# ----------------------------------------------------------------------
# Path‑signature utilities (Parent A)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Convert a path into its lead‑lag representation.
    Input shape: (T, d)
    Output shape: (2*T-1, 2*d)
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("Path must be a 2‑D array (time, dimension).")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[-1], path[-1]])
    return out


def signature_level2(path: np.ndarray) -> np.ndarray:
    """
    Compute the level‑2 (matrix) signature of a path.
    Returns a (d, d) matrix.
    """
    path = np.asarray(path, dtype=float)
    if path.shape[0] < 2:
        raise ValueError("Path must contain at least two points.")
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running = path[:-1] - path[0]               # (T‑1, d)
    return running.T @ increments               # (d, d)


def bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Evaluate B‑spline basis functions of given degree at points x.
    Returns an (len(x), len(knots) - degree - 1) array.
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)

    n_basis = len(knots) - degree - 1
    if n_basis <= 0:
        raise ValueError("Invalid knot vector for the requested degree.")

    # Initial piecewise constant basis (degree 0)
    N = np.zeros((len(x), n_basis), dtype=float)
    for i in range(n_basis):
        left = knots[i]
        right = knots[i + 1]
        N[:, i] = np.where((x >= left) & (x < right), 1.0, 0.0)
    # Include the rightmost knot
    N[:, -1] = np.where(x == knots[-1], 1.0, N[:, -1])

    # Recursive Cox–de Boor formula
    for p in range(1, degree + 1):
        for i in range(n_basis):
            denom1 = knots[i + p] - knots[i]
            term1 = 0.0 if denom1 == 0 else ((x - knots[i]) / denom1) * N[:, i]
            denom2 = knots[i + p + 1] - knots[i + 1]
            term2 = 0.0 if denom2 == 0 else ((knots[i + p + 1] - x) / denom2) * N[:, i + 1]
            N[:, i] = term1 + term2
    return N[:, :n_basis]


def evaluate_bspline(control_points: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Evaluate a B‑spline curve defined by control_points at uniformly spaced
    parameter values in [0, 1].
    Returns an array of shape (n_samples, d).
    """
    control_points = np.asarray(control_points, dtype=float)
    if control_points.ndim != 2:
        raise ValueError("Control points must be a 2‑D array (n_ctrl, d).")
    n_ctrl, d = control_points.shape

    # Build a clamped knot vector if the user supplied a simple uniform vector
    if knots.ndim == 1 and len(knots) == n_ctrl:
        # Create a clamped knot vector of appropriate length
        knots = np.concatenate((
            np.zeros(degree),
            np.linspace(0, 1, n_ctrl - degree + 1),
            np.ones(degree)
        ))
    elif len(knots) != n_ctrl + degree + 1:
        raise ValueError("Knot vector length must be n_ctrl + degree + 1.")

    t_vals = np.linspace(0, 1, 200)  # 200 samples give a smooth curve
    B = bspline_basis(t_vals, knots, degree)          # (200, n_ctrl)
    curve = B @ control_points                         # (200, d)
    return curve


# ----------------------------------------------------------------------
# Deterministic dummy response generator (replaces random placeholders)
# ----------------------------------------------------------------------
def _deterministic_response(seed: int, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Produce a reproducible pseudo‑random array based on a model‑specific seed.
    This mimics a model's output while keeping the code deterministic for testing.
    """
    rng = np.random.default_rng(seed)
    return rng.random(shape)


# ----------------------------------------------------------------------
# Fusion functions (deepened integration)
# ----------------------------------------------------------------------
def fused_signature_similarity(input_path: np.ndarray, model: ModelTier) -> np.ndarray:
    """
    Load the model (with eviction), generate a deterministic response, and
    compute a level‑2 signature similarity between the *lead‑lag transformed*
    input and response.
    """
    pool = ModelPool()
    pool.load_with_eviction(model)

    # Deterministic response shaped like the input
    response = _deterministic_response(hash(model.name) % (2**32), input_path.shape)

    # Apply lead‑lag transform to both paths before signature comparison
    input_ll = lead_lag_transform(input_path)
    response_ll = lead_lag_transform(response)

    # Compute signatures and then a similarity matrix (difference of signatures)
    sig_input = signature_level2(input_ll)
    sig_response = signature_level2(response_ll)
    similarity = sig_response - sig_input
    return similarity


def fused_bezier_model(model: ModelTier, control_points: np.ndarray) -> np.ndarray:
    """
    Load the model (with eviction) and evaluate a B‑spline curve defined by the
    supplied control points.  The curve points are returned rather than the raw
    basis coefficients, providing a more useful geometric object.
    """
    pool = ModelPool()
    pool.load_with_eviction(model)

    # Use deterministic response to optionally perturb control points (optional)
    perturb = _deterministic_response(hash(model.name) % (2**32), control_points.shape) * 0.01
    perturbed_ctrl = control_points + perturb

    # Build a uniform clamped knot vector
    degree = 3
    n_ctrl = perturbed_ctrl.shape[0]
    knots = np.concatenate((
        np.zeros(degree),
        np.linspace(0, 1, n_ctrl - degree + 1),
        np.ones(degree)
    ))

    curve = evaluate_bspline(perturbed_ctrl, knots, degree)
    return curve


def fused_risk_assessment(input_path: np.ndarray, model: ModelTier) -> float:
    """
    Load the model (with eviction), generate a deterministic response, compute a
    level‑2 signature similarity, and summarise risk as the Frobenius norm of the
    similarity matrix normalised by path dimensionality.
    """
    pool = ModelPool()
    pool.load_with_eviction(model)

    response = _deterministic_response(hash(model.name) % (2**32), input_path.shape)

    # Lead‑lag transform before signature to capture temporal ordering
    sim_matrix = signature_level2(lead_lag_transform(response) - lead_lag_transform(input_path))

    # Normalised Frobenius norm as a scalar risk indicator
    risk = np.linalg.norm(sim_matrix, ord="fro") / sim_matrix.shape[0]
    return float(risk)


# ----------------------------------------------------------------------
# Simple sanity‑check entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a synthetic 2‑D path
    T, d = 15, 2
    input_path = np.linspace(0, 1, T).reshape(-1, 1) * np.array([1.0, -0.5])

    # Define a model tier
    test_model = ModelTier(name="alpha", ram_mb=1200, tier="T1")

    # Run fused operations
    sig_sim = fused_signature_similarity(input_path, test_model)
    print("Signature similarity matrix (d×d):")
    print(sig_sim)

    bezier_curve = fused_bezier_model(test_model, input_path)
    print("\nEvaluated B‑spline curve shape:", bezier_curve.shape)

    risk_score = fused_risk_assessment(input_path, test_model)
    print("\nRisk assessment score:", risk_score)

    # Show pool state for debugging
    print("\nModelPool state after operations:")
    print(ModelPool().current_state())