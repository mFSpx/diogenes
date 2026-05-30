# DARWIN HAMMER — match 3331, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1335_s3.py (gen5)
# parent_b: serpentina_self_righting.py (gen0)
# born: 2026-05-29T23:49:30Z

import math
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Utility: feature scaling
# ----------------------------------------------------------------------
def _scale_features(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute z‑score scaling parameters (mean, std) and return scaled vectors.
    Zero std is replaced by 1 to avoid division by zero.
    """
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    std[std == 0.0] = 1.0
    scaled = (vectors - mean) / std
    return scaled, mean, std


def _apply_scaling(vector: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Scale a single vector using pre‑computed mean/std."""
    return (vector - mean) / std


# ----------------------------------------------------------------------
# Parent A building blocks (unchanged core kernels)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have same shape")
    return float(np.linalg.norm(a - b))


def rbf_similarity(a: np.ndarray, b: np.ndarray, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel based on Euclidean distance."""
    return gaussian(euclidean(a, b), epsilon)


# ----------------------------------------------------------------------
# Parent B building blocks (unchanged)
# ----------------------------------------------------------------------
class Morphology:
    """Immutable container for shape and mass."""
    __slots__ = ('length', 'width', 'height', 'mass')

    def __init__(self, length: float, width: float, height: float, mass: float):
        if min(length, width, height, mass) <= 0:
            raise ValueError("All morphology parameters must be positive")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity: cube‑root of volume divided by longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness: average of length & width relative to height."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Estimated time to self‑right, derived from biology."""
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority (0‥1) for external assistance."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Deeply integrated hybrid model
# ----------------------------------------------------------------------
class HybridModel:
    """
    Encapsulates a fused similarity‑priority metric with:
      * Z‑score normalisation of raw morphology dimensions.
      * Enrichment of the feature space by sphericity & flatness.
      * Fisher‑information‑inspired weight derived from the gradient of the RBF kernel.
      * Convex combination of biomechanical priority and information weight.
    """

    def __init__(self,
                 reference_morphologies: list[Morphology],
                 epsilon: float = 1.0,
                 info_alpha: float = 0.6):
        """
        Parameters
        ----------
        reference_morphologies: list of Morphology objects used to build the reference vector.
        epsilon: bandwidth for the Gaussian RBF.
        info_alpha: mixing coefficient (0‥1) controlling the influence of the information weight.
        """
        if not reference_morphologies:
            raise ValueError("At least one reference morphology required")
        self.epsilon = epsilon
        self.info_alpha = float(info_alpha)
        if not (0.0 <= self.info_alpha <= 1.0):
            raise ValueError("info_alpha must be in [0,1]")

        # Build the raw 6‑D vectors (length, width, height, mass, sphericity, flatness)
        raw_vectors = np.stack([self._morphology_to_raw_vec(m) for m in reference_morphologies])
        self.scaled_vectors, self.mean_vec, self.std_vec = _scale_features(raw_vectors)

        # Reference is the centroid of the scaled space
        self.ref_vec = self.scaled_vectors.mean(axis=0)

        # Pre‑compute a rough bound for the information term (max gradient norm)
        # This is used to normalise the Fisher‑information estimate to [0,1].
        grads = np.apply_along_axis(self._rbf_gradient, 1, self.scaled_vectors)
        self._info_max = np.linalg.norm(grads, axis=1).max()
        if self._info_max == 0.0:
            self._info_max = 1.0  # avoid division by zero

    @staticmethod
    def _morphology_to_raw_vec(m: Morphology) -> np.ndarray:
        """Create a 6‑D raw feature vector (including shape indices)."""
        sph = sphericity_index(m.length, m.width, m.height)
        flt = flatness_index(m.length, m.width, m.height)
        return np.array([m.length, m.width, m.height, m.mass, sph, flt], dtype=float)

    def _scaled_vec(self, m: Morphology) -> np.ndarray:
        """Scale a morphology into the same space as the reference."""
        raw = self._morphology_to_raw_vec(m)
        return _apply_scaling(raw, self.mean_vec, self.std_vec)

    # ------------------------------------------------------------------
    # Fisher‑information approximation via kernel gradient
    # ------------------------------------------------------------------
    def _rbf_gradient(self, vec: np.ndarray) -> np.ndarray:
        """
        Gradient of the Gaussian RBF w.r.t. the input vector evaluated at the reference.
        ∇_x K(x, μ) = -2·ε²·(x-μ)·K(x, μ)
        """
        diff = vec - self.ref_vec
        k = gaussian(np.linalg.norm(diff), self.epsilon)
        return -2.0 * (self.epsilon ** 2) * diff * k

    def _information_weight(self, vec: np.ndarray) -> float:
        """
        Approximate Fisher information by the squared norm of the kernel gradient,
        then normalise to [0,1] using the pre‑computed maximum.
        """
        grad = self._rbf_gradient(vec)
        info = np.linalg.norm(grad) ** 2
        return min(1.0, info / (self._info_max ** 2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def hybrid_priority(self, m: Morphology) -> float:
        """
        Compute the fused metric:
          p  = biomechanical recovery priority (Parent B)
          s  = RBF similarity to the reference (Parent A)
          i  = normalised information weight derived from ∇K
          hp = (1‑α)·p + α·i·s
        The product i·s ensures that high similarity boosts the influence of
        the information term, while the convex combination guarantees the result
        stays in [0,1].
        """
        p = recovery_priority(m)                     # biomechanical component
        vec = self._scaled_vec(m)                    # bring m into the scaled space
        s = rbf_similarity(vec, self.ref_vec, self.epsilon)  # raw similarity (0‥1)
        i = self._information_weight(vec)           # information weight (0‥1)

        hp = (1.0 - self.info_alpha) * p + self.info_alpha * (i * s)
        return max(0.0, min(1.0, hp))

    def similarity(self, m: Morphology) -> float:
        """Convenient accessor for the pure RBF similarity."""
        return rbf_similarity(self._scaled_vec(m), self.ref_vec, self.epsilon)


# ----------------------------------------------------------------------
# Bandit update (multi‑arm aware, retains probability simplex)
# ----------------------------------------------------------------------
def bandit_update(weights: np.ndarray, reward: float, alpha: float = 0.1) -> np.ndarray:
    """
    Exponential‑gradient style update for a probability vector.
    Guarantees non‑negativity and unit sum.
    """
    if weights.ndim != 1:
        raise ValueError("weights must be a 1‑D array")
    if not (0.0 <= reward <= 1.0):
        raise ValueError("reward must be in [0,1]")
    if np.any(weights < 0):
        raise ValueError("weights must be non‑negative")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    # multiplicative update then renormalise
    new_weights = weights * np.exp(alpha * reward)
    return new_weights / new_weights.sum()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample population
    population = [
        Morphology(30.0, 20.0, 10.0, 5.0),
        Morphology(25.0, 22.0, 12.0, 4.5),
        Morphology(28.0, 18.0, 9.0, 5.5),
        Morphology(32.0, 24.0, 11.0, 5.2),
    ]

    # Initialise hybrid model with the whole population as reference set
    model = HybridModel(reference_morphologies=population,
                        epsilon=0.8,
                        info_alpha=0.55)

    # Single‑arm bandit (weight always 1) – kept for API compatibility
    weight = np.array([1.0])

    for idx, morph in enumerate(population, start=1):
        sim = model.similarity(morph)
        hp = model.hybrid_priority(morph)
        weight = bandit_update(weight, hp, alpha=0.15)
        print(f"Sample {idx}: similarity={sim:.4f}, hybrid_priority={hp:.4f}, bandit_weight={weight[0]:.4f}")

    sys.exit(0)