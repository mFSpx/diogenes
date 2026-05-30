# DARWIN HAMMER — match 3867, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_model__m591_s2.py (gen5)
# parent_b: hybrid_hybrid_endpoint_circ_state_space_duality_m1_s1.py (gen2)
# born: 2026-05-29T23:52:13Z

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional


# ----------------------------------------------------------------------
# Morphology and health metrics
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def _check_positive(*values: float) -> None:
    for v in values:
        if v <= 0:
            raise ValueError("all dimensions and mass must be positive")


def sphericity_index(length: float, width: float, height: float) -> float:
    _check_positive(length, width, height)
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    _check_positive(length, width, height)
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    _check_positive(m.mass, neck_lever)
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized health score in [0,1] based on righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Rotor (geometric algebra inspired)
# ----------------------------------------------------------------------
@dataclass
class Rotor:
    """Simple 2‑D rotor represented by an angle (radians)."""
    angle: float = 0.0

    def matrix(self) -> np.ndarray:
        """Return the 2‑D rotation matrix."""
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.array([[c, -s], [s, c]])


def rotate(rotor: Rotor, x: np.ndarray) -> np.ndarray:
    """Apply rotor to vector x (supports any dimensionality by embedding)."""
    if x.ndim != 1:
        raise ValueError("rotate expects a 1‑D vector")
    # For dimensions >2 we rotate in the first two components, leaving the rest untouched.
    if x.size < 2:
        raise ValueError("vector must have at least two components for rotation")
    rot_mat = rotor.matrix()
    y = x.copy()
    y[:2] = rot_mat @ x[:2]
    return y


def update_rotor(rotor: Rotor, grad: float, lr: float) -> Rotor:
    """Update rotor angle using a scalar gradient (e.g., from a geometric product)."""
    new_angle = rotor.angle - lr * grad
    return Rotor(new_angle)


# ----------------------------------------------------------------------
# State‑Space Model (SSM) abstraction
# ----------------------------------------------------------------------
@dataclass
class StateSpaceModel:
    """Discrete‑time linear state‑space model."""
    A: np.ndarray  # state transition matrix
    B: np.ndarray  # control input matrix
    C: np.ndarray  # output matrix
    D: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))  # feed‑through

    def step(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One time‑step update: returns (next_state, output)."""
        x_next = self.A @ x + self.B @ u
        y = self.C @ x_next + self.D @ u
        return x_next, y


def init_random_ssm(dim_state: int, dim_input: int, dim_output: int, scale: float = 0.01,
                    seed: int = 0) -> StateSpaceModel:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim_state, dim_state)) * scale
    B = rng.standard_normal((dim_state, dim_input)) * scale
    C = rng.standard_normal((dim_output, dim_state)) * scale
    D = rng.standard_normal((dim_output, dim_input)) * scale
    return StateSpaceModel(A, B, C, D)


# ----------------------------------------------------------------------
# Core hybrid operation with deeper integration
# ----------------------------------------------------------------------
def geometric_product(a: float, b: np.ndarray) -> np.ndarray:
    """Scalar‑vector geometric product (simple scaling)."""
    return a * b


def hybrid_operation(
    x: np.ndarray,
    W: np.ndarray,
    rotor: Rotor,
    lr: float,
    morphology: Morphology,
    ssm: StateSpaceModel,
    control: np.ndarray,
) -> Tuple[float, np.ndarray, Rotor, np.ndarray, float]:
    """
    Perform one hybrid step:
      1. Linear projection via W.
      2. Compute loss against the SSM output (target).
      3. Gradient descent on W.
      4. Rotor update using a geometric product‑derived gradient.
      5. Rotate the input vector.
      6. Compute health‑weighted score.
    Returns loss, updated W gradient, updated rotor, rotated vector, health score.
    """
    # 1. Linear projection
    pred = W @ x

    # 2. SSM forward pass (use current state = x, control = control)
    _, ssm_out = ssm.step(x, control)

    # 3. Simple squared‑error loss against SSM output (broadcast if needed)
    target = ssm_out.squeeze()
    residual = pred - target
    loss = float(residual @ residual)

    # 4. Gradient w.r.t. W (outer product)
    grad_W = 2.0 * np.outer(residual, x)

    # 5. Geometric product between rotor angle and prediction, then derive scalar grad
    gp = geometric_product(rotor.angle, pred)
    rotor_grad = float(np.mean(gp))  # scalar summary for angle update

    # 6. Update rotor
    rotor = update_rotor(rotor, rotor_grad, lr)

    # 7. Rotate the original input
    x_rotated = rotate(rotor, x)

    # 8. Health score (used later for weighting)
    health_score = recovery_priority(morphology)

    return loss, grad_W, rotor, x_rotated, health_score


def improved_hybrid_operation(
    x: np.ndarray,
    W: np.ndarray,
    rotor: Rotor,
    lr: float,
    alpha: float = 0.1,
    morphology: Optional[Morphology] = None,
    ssm: Optional[StateSpaceModel] = None,
    control: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, Rotor, np.ndarray, float]:
    """
    Enhanced hybrid step that:
      * Applies health‑dependent learning‑rate scaling.
      * Returns the updated weight matrix directly.
    """
    if morphology is None:
        morphology = Morphology(1.0, 1.0, 1.0, 1.0)
    if ssm is None:
        # default 4‑dim state, 4‑dim input, 4‑dim output
        ssm = init_random_ssm(dim_state=x.size, dim_input=control.size if control is not None else x.size,
                              dim_output=W.shape[0], seed=42)
    if control is None:
        control = np.zeros_like(x)

    loss, grad_W, rotor, x_rotated, health_score = hybrid_operation(
        x, W, rotor, lr, morphology, ssm, control
    )

    # health‑aware learning rate scaling (more healthy → larger step)
    effective_lr = lr * (0.5 + 0.5 * health_score)

    # weight update with additional momentum‑like term alpha
    W_update = W - alpha * grad_W * effective_lr

    return loss, W_update, rotor, x_rotated, health_score


def improved_route_command(
    x: np.ndarray,
    W: np.ndarray,
    rotor: Rotor,
    lr: float,
    threshold: float,
    alpha: float = 0.1,
    morphology: Optional[Morphology] = None,
    ssm: Optional[StateSpaceModel] = None,
    control: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Decide whether to use the rotated input directly or to apply the updated linear map,
    based on loss relative to a threshold.
    Returns the selected output vector and the health score.
    """
    loss, W_updated, rotor, x_rotated, health_score = improved_hybrid_operation(
        x, W, rotor, lr, alpha, morphology, ssm, control
    )
    if loss < threshold:
        return x_rotated, health_score
    else:
        return W_updated @ x_rotated, health_score


# ----------------------------------------------------------------------
# Example usage (executed only when run as script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dimensions
    dim = 8

    # Initialise components
    W = init_random_ssm(dim_state=dim, dim_input=dim, dim_output=dim).C  # use C as a random projection matrix
    rotor = Rotor(angle=0.1)
    x = np.random.rand(dim)
    lr = 0.01
    threshold = 0.05
    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.0)

    # Simple control vector (could be sensor readings, etc.)
    control_vec = np.random.rand(dim)

    # Run the improved routing
    out_vec, health = improved_route_command(
        x,
        W,
        rotor,
        lr,
        threshold,
        alpha=0.1,
        morphology=morph,
        control=control_vec,
    )
    print("Output vector:", out_vec)
    print("Health score :", health)