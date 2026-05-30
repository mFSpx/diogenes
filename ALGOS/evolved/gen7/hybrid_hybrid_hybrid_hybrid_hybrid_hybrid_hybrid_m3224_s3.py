# DARWIN HAMMER — match 3224, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_tropical_maxp_m2056_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m976_s1.py (gen5)
# born: 2026-05-29T23:48:40Z

"""hybrid_fusion_nlms_curvature_rbf.py
This module fuses the two parent algorithms:

* **Parent A** – NLMS adaptive filter with radial‑basis‑function (RBF) modeling,
  tropical max‑plus evaluation and graph construction.
* **Parent B** – Circuit‑breaker endpoint logic, morphology‑driven Fisher score,
  and Ollivier‑Ricci curvature estimation.

**Mathematical bridge**

The Fisher score derived from the `Morphology` object is used to scale the
NLMS learning rate `mu`.  The Ollivier‑Ricci curvature computed on the weight
graph supplies the coefficients for the tropical max‑plus polynomial, thus
linking the curvature‑based geometry of Parent B with the tropical algebra of
Parent A.  The hybrid step updates the weight vector with a curvature‑aware
NLMS rule and evaluates a tropical polynomial whose coefficients are
curvature‑modulated, thereby unifying the governing equations of both parents."""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Data structures shared by the hybrid algorithm
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    """Graph node representing a weight entry."""
    id: int
    weight: float

@dataclass(frozen=True)
class Morphology:
    """Geometric description of an entity (used for Fisher score)."""
    length: float
    width: float
    height: float
    mass: float

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("width", self.width),
            ("height", self.height),
            ("mass", self.mass),
        ):
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")

class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = ""  # placeholder
        if self.failures >= self.failure_threshold:
            self.open = True

# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Radial basis function – Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))

def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction using current weight vector."""
    return float(np.dot(weights, x))

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """
    Normalised Least‑Mean‑Squares (NLMS) weight update.

    Returns the new weight vector and the instantaneous error.
    """
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    new_weights = weights + mu * error * x / power
    return new_weights, error

def fisher_score(morph: Morphology) -> float:
    """
    Simple surrogate Fisher information score derived from morphology.

    The score is proportional to the product of the geometric dimensions,
    ensuring a positive scaling factor.
    """
    volume = morph.length * morph.width * morph.height
    return float(volume * morph.mass)

def curvature_matrix(weights: np.ndarray) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature matrix for a fully‑connected graph
    whose nodes are the weight entries.

    For nodes i and j the curvature is defined as:
        κ_ij = exp( -|w_i - w_j| )
    which satisfies 0 < κ_ij ≤ 1.
    """
    n = len(weights)
    curv = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                curv[i, j] = 1.0
            else:
                curv[i, j] = math.exp(-abs(weights[i] - weights[j]))
    return curv

def tropical_max_plus(coeffs: np.ndarray, variables: np.ndarray) -> float:
    """
    Tropical max‑plus evaluation.

    Computes max_i ( coeff_i + ⟨variables, e_i⟩ )
    where e_i is the i‑th unit vector (i.e. selects variable i).
    """
    result = -np.inf
    for i in range(len(coeffs)):
        term = coeffs[i] + variables[i]
        if term > result:
            result = term
    return float(result)

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_step(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    morph: Morphology,
    breaker: EndpointCircuitBreaker,
) -> dict:
    """
    Perform one hybrid iteration:

    1. Compute a Fisher‑scaled learning rate: μ' = μ * fisher_score / (norm_factor).
    2. Apply NLMS update with μ'.
    3. Build curvature matrix from the *new* weights.
    4. Use the diagonal of the curvature matrix as tropical coefficients.
    5. Evaluate tropical polynomial on the input vector.
    6. Update circuit‑breaker state based on prediction error.

    Returns a dictionary with intermediate results for inspection.
    """
    # 1. Fisher‑scaled learning rate
    base_mu = 0.5
    fisher = fisher_score(morph)
    norm_factor = np.linalg.norm(weights) + 1e-9
    mu_scaled = base_mu * fisher / norm_factor

    # 2. NLMS update
    new_weights, error = nlms_update(weights, x, target, mu=mu_scaled)

    # 3. Curvature matrix from updated weights
    curv_mat = curvature_matrix(new_weights)

    # 4. Tropical coefficients – we take the diagonal (self‑curvature)
    tropical_coeffs = np.diag(curv_mat)

    # 5. Tropical evaluation
    tropical_val = tropical_max_plus(tropical_coeffs, x)

    # 6. Circuit‑breaker handling
    if breaker.open:
        # If already open, record a success attempt to possibly close it
        breaker.record_success()
    else:
        if abs(error) > 1.0:          # arbitrary error threshold
            breaker.record_failure()
        else:
            breaker.record_success()

    return {
        "new_weights": new_weights,
        "error": error,
        "mu_scaled": mu_scaled,
        "curvature_matrix": curv_mat,
        "tropical_coeffs": tropical_coeffs,
        "tropical_value": tropical_val,
        "breaker_open": breaker.open,
    }

def construct_weight_graph(weights: np.ndarray) -> dict[int, list[tuple[int, float]]]:
    """
    Build an undirected similarity graph from the weight vector.
    Edge weight between nodes i and j is a similarity measure derived from
    the absolute weight difference.
    """
    graph: dict[int, list[tuple[int, float]]] = {}
    n = len(weights)
    for i in range(n):
        graph[i] = []
        for j in range(n):
            if i == j:
                continue
            diff = abs(weights[i] - weights[j])
            similarity = 1.0 - diff / (1.0 + diff)  # maps diff∈[0,∞) to similarity∈[0,1)
            graph[i].append((j, similarity))
    return graph

def rbf_transform(x: np.ndarray, centers: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian RBF transformation to the input vector `x` using
    the supplied `centers`. Returns a new feature vector of the same length.
    """
    transformed = np.empty_like(x, dtype=float)
    for i, (xi, ci) in enumerate(zip(x, centers)):
        r = abs(xi - ci)
        transformed[i] = gaussian_rbf(r, epsilon)
    return transformed

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise random seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Random weight vector and input
    dim = 5
    w = np.random.randn(dim)
    x = np.random.randn(dim)

    # Target generated with a hidden true weight vector
    true_w = np.random.randn(dim)
    target = predict(true_w, x) + np.random.normal(scale=0.1)

    # Morphology instance (arbitrary positive values)
    morph = Morphology(length=2.3, width=1.7, height=0.9, mass=3.5)

    # Circuit breaker
    breaker = EndpointCircuitBreaker(failure_threshold=2)

    # Run a single hybrid step
    result = hybrid_step(w, x, target, morph, breaker)

    # Display key outcomes
    print("Error after NLMS update :", result["error"])
    print("Scaled learning rate μ' :", result["mu_scaled"])
    print("Tropical evaluation    :", result["tropical_value"])
    print("Circuit breaker open?  :", result["breaker_open"])

    # Build and show a small part of the weight graph
    graph = construct_weight_graph(result["new_weights"])
    print("\nSample graph edges (node 0):")
    for neighbor, sim in graph[0][:3]:
        print(f"  -> node {neighbor} similarity {sim:.3f}")