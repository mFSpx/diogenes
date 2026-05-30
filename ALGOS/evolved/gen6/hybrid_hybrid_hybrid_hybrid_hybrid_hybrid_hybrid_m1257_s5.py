# DARWIN HAMMER — match 1257, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m538_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_endpoi_epistemic_certainty_m59_s2.py (gen3)
# born: 2026-05-29T23:34:48Z

"""hybrid_fusion_endpoint_certainty.py
Integrates:

* Parent A (HybridFusion) – resource‑vector formulation with Gaussian radial‑basis‑function (RBF) surrogate,
* Parent B (Endpoint Epistemic Certainty) – dataclasses for EngineEndpoint, Morphology, CertaintyFlag and Bayesian
  uncertainty propagation through a linear state‑space model.

Mathematical bridge:
The RBF surrogate of Parent A predicts a *score* component of the resource vector.  
The remaining components (distance, privacy‑load) are derived from the same input vector.
The resulting 3‑dimensional resource vector is then interpreted as a *measurement* in a linear
Gaussian state‑space model (SSM) of Parent B:

    x_{k+1} = A·x_k + B·u_k + w_k,   w_k ∼ N(0,Q)
    y_k     = H·x_k + v_k,           v_k ∼ N(0,R)

where `y_k` is the resource vector, `H` selects the observable dimensions and the
measurement covariance `R` is built from the epistemic certainty flags attached to an
`EngineEndpoint`.  The SSM equations are solved with the linear system solver from
Parent A, thereby fusing the two topologies into a single hybrid estimator.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
class HybridFusion:
    def __init__(
        self,
        d_in: int,
        d_out: int,
        seed: int = 0,
        base_eta: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1.0,
        store_decay: float = 0.99,
    ) -> None:
        self.d_in = d_in
        self.d_out = d_out
        self.seed = seed
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.store_decay = store_decay
        self.rng = random.Random(seed)

    @staticmethod
    def gaussian(r: float, epsilon: float = 1.0) -> float:
        """Isotropic Gaussian RBF."""
        return math.exp(-((epsilon * r) ** 2))

    @staticmethod
    def euclidean(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            raise ValueError("vectors must have same dimension")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
        """Solve A·x = b with simple Gaussian elimination (no pivoting beyond max row)."""
        n = len(b)
        # Build augmented matrix
        M = [row[:] + [rhs] for row, rhs in zip(A, b)]

        for col in range(n):
            # Pivot
            pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
            if abs(M[pivot][col]) < 1e-12:
                raise ValueError("singular system")
            M[col], M[pivot] = M[pivot], M[col]

            # Normalize pivot row
            div = M[col][col]
            M[col] = [v / div for v in M[col]]

            # Eliminate other rows
            for row in range(n):
                if row == col:
                    continue
                factor = M[row][col]
                M[row] = [v_row - factor * v_col for v_row, v_col in zip(M[row], M[col])]

        # Extract solution
        return [M[i][-1] for i in range(n)]

# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"
    certainty_flag: CertaintyFlag = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ----------------------------------------------------------------------
# Hybrid functions (the new algorithm)
# ----------------------------------------------------------------------
def rbf_predict(
    x: List[float],
    centers: List[List[float]],
    weights: List[float],
    epsilon: float = 1.0,
) -> float:
    """Predict a scalar using a weighted sum of Gaussian RBFs."""
    if len(centers) != len(weights):
        raise ValueError("centers and weights length mismatch")
    total = 0.0
    for c, w in zip(centers, weights):
        r = HybridFusion.euclidean(x, c)
        total += w * HybridFusion.gaussian(r, epsilon)
    return total


def compute_resource_vector(
    x: List[float],
    centers: List[List[float]],
    weights: List[float],
    epsilon: float,
    alpha: float,
    beta: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build the 3‑component resource vector:
        distance      = α·‖x‖_2
        privacy_load = β·mean(x)
        score        = RBF surrogate (parent A)
    Returns both the numpy vector and a dict of the raw components.
    """
    norm = math.sqrt(sum(v * v for v in x))
    distance = alpha * norm
    privacy_load = beta * (sum(x) / len(x))
    score = rbf_predict(x, centers, weights, epsilon)

    vec = np.array([distance, privacy_load, score], dtype=float)
    raw = {"distance": distance, "privacy_load": privacy_load, "score": score}
    return vec, raw


def build_measurement_covariance(flag: CertaintyFlag) -> np.ndarray:
    """
    Convert an epistemic certainty flag into a diagonal measurement covariance R.
    Higher confidence → lower variance.
    """
    # Map confidence (basis points) to variance in [1e-4, 1.0]
    conf = flag.confidence_bps / 10000.0
    var = (1.0 - conf) ** 2 + 1e-4
    R = np.diag([var, var, var])
    return R


def propagate_state(
    x_prev: np.ndarray,
    P_prev: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    u: np.ndarray,
    Q: np.ndarray,
    y_meas: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One step of Kalman‑filter style propagation using the linear system solver from
    HybridFusion (solve_linear) to compute the Kalman gain.
    """
    # Predict
    x_pred = A @ x_prev + B @ u
    P_pred = A @ P_prev @ A.T + Q

    # Innovation
    y_pred = H @ x_pred
    S = H @ P_pred @ H.T + R  # innovation covariance

    # Solve for Kalman gain K = P_pred·H.T·S^{-1}
    # Instead of a direct inverse we solve S·K.T = (P_pred·H.T).T
    S_np = S.tolist()
    rhs = (P_pred @ H.T).T.tolist()
    K_T = [HybridFusion.solve_linear(S_np, col) for col in rhs]
    K = np.array(K_T).T

    # Update
    x_upd = x_pred + K @ (y_meas - y_pred)
    P_upd = (np.eye(len(x_prev)) - K @ H) @ P_pred
    return x_upd, P_upd


def fuse_endpoint_with_resource(
    endpoint: EngineEndpoint,
    resource_vec: np.ndarray,
    state: np.ndarray,
    cov: np.ndarray,
) -> Dict[str, Any]:
    """
    Create a unified representation that merges endpoint metadata, epistemic certainty,
    and the hybrid estimator state.
    """
    if endpoint.certainty_flag is None:
        raise ValueError("EngineEndpoint must carry a CertaintyFlag")
    R = build_measurement_covariance(endpoint.certainty_flag)

    # Simple linear SSM matrices (identity dynamics, direct observation)
    dim = state.shape[0]
    A = np.eye(dim)
    B = np.zeros((dim, dim))
    u = np.zeros(dim)
    Q = np.eye(dim) * 1e-3
    H = np.eye(dim)[:3]  # map first three state dimensions to the resource vector

    # Perform one Kalman update using the resource vector as measurement
    new_state, new_cov = propagate_state(state, cov, A, B, u, Q, resource_vec, H, R)

    fused = {
        "endpoint_id": endpoint.engine_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resource_vector": resource_vec.tolist(),
        "state_estimate": new_state.tolist(),
        "state_covariance": new_cov.tolist(),
        "certainty": endpoint.certainty_flag.as_dict(),
        "morphology": asdict(endpoint.morphology),
    }
    return fused


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy data for the hybrid estimator
    d = 5
    x_input = [0.5, -1.2, 3.3, 0.0, 2.1]

    # RBF centres and weights (random but reproducible)
    rng = random.Random(42)
    centers = [[rng.uniform(-2, 2) for _ in range(d)] for _ in range(4)]
    weights = [rng.uniform(-1, 1) for _ in range(4)]

    # Compute resource vector
    vec, raw = compute_resource_vector(
        x_input,
        centers,
        weights,
        epsilon=0.8,
        alpha=1.5,
        beta=0.7,
    )
    print("Resource vector:", vec)
    print("Raw components:", raw)

    # Build a dummy endpoint with a certainty flag
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=10.0)
    flag = CertaintyFlag(
        label="PROBABLE",
        confidence_bps=7500,
        authority_class="internal",
        rationale="simulated test",
    )
    endpoint = EngineEndpoint(
        engine_id="engine-001",
        channel="alpha",
        residency="us-east",
        runtime="python3.11",
        resource_class="compute",
        always_on=True,
        endpoint="http://localhost/test",
        capabilities=["predict", "estimate"],
        morphology=morph,
        certainty_flag=flag,
    )

    # Initial state (3‑dim for simplicity) and covariance
    init_state = np.zeros(3)
    init_cov = np.eye(3) * 0.1

    fused_output = fuse_endpoint_with_resource(endpoint, vec, init_state, init_cov)
    print("\nFused output:")
    for k, v in fused_output.items():
        print(f"{k}: {v}")