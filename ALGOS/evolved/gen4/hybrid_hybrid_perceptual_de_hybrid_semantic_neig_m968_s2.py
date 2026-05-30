# DARWIN HAMMER — match 968, survivor 2
# gen: 4
# parent_a: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s0.py (gen3)
# parent_b: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s4.py (gen2)
# born: 2026-05-29T23:32:08Z

import math
import numpy as np
from typing import List, Tuple

# ----------------------------------------------------------------------
# Parent A utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()

def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: median‑based 64‑bit signature."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

class RBFSurrogate:
    """Radial‑basis‑function surrogate model."""
    def __init__(self, centers: List[Tuple[float, ...]], weights: np.ndarray, epsilon: float = 1.0):
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def predict(self, x: List[float]) -> float:
        """Predict a scalar value for input vector x."""
        return float(
            sum(
                w * gaussian(euclidean(x, c), self.epsilon)
                for w, c in zip(self.weights, self.centers)
            )
        )

def fit_rbf(points: List[List[float]],
            values: List[float],
            epsilon: float = 1.0,
            ridge: float = 1e-9) -> RBFSurrogate:
    """
    Fit an RBF surrogate by solving (Φ + λI)w = y,
    where Φ_{ij}=gaussian(||p_i-p_j||, ε).
    """
    n = len(points)
    if n == 0:
        raise ValueError("No points to fit.")
    # Build kernel matrix Φ
    phi = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            phi[i, j] = gaussian(euclidean(points[i], points[j]), epsilon)
    # Regularisation
    phi += ridge * np.eye(n)
    # Solve for weights
    y = np.asarray(values, dtype=float)
    weights, *_ = np.linalg.lstsq(phi, y, rcond=None)
    centers = [tuple(map(float, p)) for p in points]
    return RBFSurrogate(centers, weights, epsilon)

# ----------------------------------------------------------------------
# Parent B utilities
# ----------------------------------------------------------------------
class Morphology:
    """Geometric description of an object."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        if min(length, width, height, mass) <= 0:
            raise ValueError("All morphology parameters must be positive.")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean to the longest dimension."""
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness as (length+width)/(2·height)."""
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology,
                        b: float = 1.0 / 3.0,
                        k: float = 0.35,
                        neck_lever: float = 1.0) -> float:
    """Estimated time for a self‑righting object to recover."""
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """
    Normalised priority in [0,1] that reflects how quickly the object can
    recover; higher priority means more tolerant circuit behaviour.
    """
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if den == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / den

# ----------------------------------------------------------------------
# Hybrid structures
# ----------------------------------------------------------------------
class HybridCircuitRBF:
    """
    Combines an RBF surrogate with morphology‑driven circuit‑breaker logic.
    """
    def __init__(self,
                 surrogate: RBFSurrogate,
                 morphology: Morphology,
                 base_failure_threshold: int = 3,
                 cluster_tolerance: float = 0.1):
        self.surrogate = surrogate
        self.morphology = morphology
        self.base_failure_threshold = base_failure_threshold
        self.cluster_tolerance = cluster_tolerance
        self.failure_counter = 0
        self.cluster_hashes = {}

    def predict(self, x: List[float]) -> float:
        """Raw surrogate prediction."""
        return self.surrogate.predict(x)

    def _dynamic_threshold(self) -> float:
        """
        Threshold that shrinks when recovery priority is high,
        i.e. the system tolerates more failures.
        """
        rho = recovery_priority(self.morphology)
        return self.base_failure_threshold * (1.0 - rho)

    def _get_cluster_hash(self, x: List[float]) -> int:
        """Get perceptual hash for input x."""
        return compute_phash(x)

    def _check_cluster_membership(self, x: List[float]) -> bool:
        """Check if x belongs to an existing cluster."""
        x_hash = self._get_cluster_hash(x)
        for cluster_hash in self.cluster_hashes:
            if hamming_distance(x_hash, cluster_hash) <= self.cluster_tolerance:
                return True
        return False

    def _add_to_cluster(self, x: List[float]) -> None:
        """Add x to a new cluster."""
        x_hash = self._get_cluster_hash(x)
        self.cluster_hashes[x_hash] = True

    def check_and_update(self, x: List[float]) -> Tuple[float, bool, int]:
        """
        Evaluate input x, update the failure counter and report circuit status.

        Returns
        -------
        pred : float
            Surrogate prediction.
        tripped : bool
            True if the failure counter exceeds the dynamic threshold.
        counter : int
            Current failure counter value.
        """
        if not self._check_cluster_membership(x):
            self._add_to_cluster(x)
            # Refit surrogate model for new cluster
            # NOTE: This is a simplified example; in practice, you would need to handle
            #       refitting the surrogate model and updating the circuit state.
            pass

        pred = self.predict(x)
        # Simple heuristic: negative predictions indicate a “failure”.
        if pred < 0.0:
            self.failure_counter += 1
        else:
            self.failure_counter = max(0, self.failure_counter - 1)

        tripped = self.failure_counter >= self._dynamic_threshold()
        return pred, tripped, self.failure_counter

# ----------------------------------------------------------------------
# Public hybrid API
# ----------------------------------------------------------------------
def fit_hybrid(points: List[List[float]],
               values: List[float],
               morphology: Morphology,
               epsilon: float = 1.0,
               ridge: float = 1e-9,
               base_failure_threshold: int = 3,
               cluster_tolerance: float = 0.1) -> HybridCircuitRBF:
    """
    Fit a hybrid circuit model.

    Parameters
    ----------
    points : List[List[float]]
        Input data points.
    values : List[float]
        Corresponding output values.
    morphology : Morphology
        Geometric description of the object.
    epsilon : float, optional
        RBF kernel parameter (default: 1.0).
    ridge : float, optional
        Regularisation parameter (default: 1e-9).
    base_failure_threshold : int, optional
        Base failure threshold (default: 3).
    cluster_tolerance : float, optional
        Cluster tolerance (default: 0.1).

    Returns
    -------
    HybridCircuitRBF
        Fitted hybrid circuit model.
    """
    surrogate = fit_rbf(points, values, epsilon, ridge)
    return HybridCircuitRBF(surrogate, morphology, base_failure_threshold, cluster_tolerance)

def predict_hybrid(circuit: HybridCircuitRBF, x: List[float]) -> Tuple[float, bool, int]:
    """
    Evaluate input x using the hybrid circuit model.

    Parameters
    ----------
    circuit : HybridCircuitRBF
        Fitted hybrid circuit model.
    x : List[float]
        Input data point.

    Returns
    -------
    pred : float
        Surrogate prediction.
    tripped : bool
        True if the failure counter exceeds the dynamic threshold.
    counter : int
        Current failure counter value.
    """
    return circuit.check_and_update(x)

def update_circuit(circuit: HybridCircuitRBF, x: List[float], value: float) -> None:
    """
    Update the circuit model with new data.

    Parameters
    ----------
    circuit : HybridCircuitRBF
        Fitted hybrid circuit model.
    x : List[float]
        Input data point.
    value : float
        Corresponding output value.
    """
    # NOTE: This is a simplified example; in practice, you would need to handle
    #       updating the circuit state and refitting the surrogate model.
    pass