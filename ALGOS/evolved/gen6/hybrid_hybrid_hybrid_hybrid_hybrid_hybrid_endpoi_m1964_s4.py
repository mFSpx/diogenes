# DARWIN HAMMER — match 1964, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m635_s0.py (gen5)
# parent_b: hybrid_hybrid_endpoint_circ_state_space_duality_m1_s1.py (gen2)
# born: 2026-05-29T23:40:12Z

import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Callable, Any

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Geometric description of an engine component."""

    length: float
    width: float
    height: float
    mass: float

    def __post_init__(self) -> None:
        if any(v <= 0 for v in (self.length, self.width, self.height, self.mass)):
            raise ValueError("All morphology dimensions and mass must be positive.")


@dataclass(frozen=True)
class EngineEndpoint:
    """Immutable description of an engine endpoint."""

    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    health_score: float
    morphology: Morphology

    def __post_init__(self) -> None:
        if not (0.0 <= self.health_score <= 1.0):
            raise ValueError("health_score must be in the interval [0, 1].")
        if not self.capabilities:
            raise ValueError("capabilities list cannot be empty.")


# ----------------------------------------------------------------------
# Epistemic flag handling
# ----------------------------------------------------------------------


EPISTEMIC_FLAGS: Tuple[str, ...] = (
    "FACT",
    "PROBABLE",
    "POSSIBLE",
    "UNCERTAIN",
    "UNKNOWN",
)

def epistemic_flag_vector(endpoint: EngineEndpoint) -> np.ndarray:
    """Binary vector indicating presence of each epistemic flag."""
    return np.array(
        [1.0 if flag in endpoint.capabilities else 0.0 for flag in EPISTEMIC_FLAGS],
        dtype=float,
    )


# ----------------------------------------------------------------------
# Geometry‑derived indices (used later for system matrices)
# ----------------------------------------------------------------------


def sphericity_index(m: Morphology) -> float:
    """Ratio of the geometric mean of the dimensions to the longest side."""
    gm = (m.length * m.width * m.height) ** (1.0 / 3.0)
    return gm / max(m.length, m.width, m.height)


def flatness_index(m: Morphology) -> float:
    """Measure of how flat the object is."""
    return (m.length + m.width) / (2.0 * m.height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """Physical proxy for recovery speed."""
    fi = flatness_index(m)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised priority in [0,1] based on righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# State‑space model utilities
# ----------------------------------------------------------------------


def build_state_space(endpoint: EngineEndpoint) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a simple continuous‑time state‑space representation (A, B, C, D)
    that embeds morphological health and epistemic information.

    The construction is deliberately lightweight but captures:
      * A: stability governed by recovery priority (eigenvalues in left half‑plane)
      * B: input scaling by health_score
      * C: observation matrix that extracts epistemic flags
      * D: direct feed‑through of feature‑count information
    """
    # Dimensionality: we keep a 2‑state system for simplicity.
    n_states = 2

    # A – diagonal with negative real parts proportional to recovery priority.
    # Higher priority (i.e. faster righting) yields faster decay.
    priority = recovery_priority(endpoint.morphology)
    a_val = - (0.5 + 0.5 * priority)  # ensures eigenvalues in (-1, -0.5)
    A = np.diag([a_val, a_val])

    # B – scales the external input (here the health_score) into the state.
    B = np.full((n_states, 1), endpoint.health_score)

    # C – maps state to epistemic flag space.
    # We project the 2‑dimensional state onto the flag vector direction.
    flag_vec = epistemic_flag_vector(endpoint)
    # Normalise to avoid scaling explosion.
    flag_norm = np.linalg.norm(flag_vec) if np.linalg.norm(flag_vec) > 0 else 1.0
    C = (flag_vec / flag_norm).reshape(1, n_states)

    # D – direct feed‑through of the feature‑count (scalar) into the output.
    feature_count = len(endpoint.capabilities)
    D = np.array([[float(feature_count)]])

    return A, B, C, D


def semiseparable_causal_matrix(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, dt: float = 1.0
) -> np.ndarray:
    """
    Discretise the continuous‑time state‑space model using a simple
    zero‑order hold (ZOH) and return the resulting 2×2 causal matrix.
    The matrix is semiseparable because it can be expressed as
    U·Vᵀ + diag(d) with low‑rank U, V.
    """
    # Discrete‑time A_d = exp(A·dt)
    A_d = scipy_expm(A * dt)

    # Discrete‑time B_d = ∫₀^{dt} exp(A·τ) dτ · B
    # For diagonal A this integral has a closed form.
    if np.allclose(A, np.diag(np.diag(A))):
        # element‑wise exponential integral
        B_d = np.empty_like(B)
        for i in range(A.shape[0]):
            a_i = A[i, i]
            if np.isclose(a_i, 0.0):
                B_d[i, :] = dt * B[i, :]
            else:
                B_d[i, :] = (1 - math.exp(a_i * dt)) / (-a_i) * B[i, :]
    else:
        # fallback to numerical integration for non‑diagonal A
        B_d = _numeric_B_integral(A, B, dt)

    # Assemble the discrete‑time state‑space matrix
    # Output y = C·x + D·u ; we are interested in the mapping from u to y after one step.
    causal_matrix = C @ A_d @ B_d + D
    return causal_matrix.squeeze()


def scipy_expm(M: np.ndarray) -> np.ndarray:
    """Fallback implementation of matrix exponential using scaling‑and‑squaring."""
    # For 2×2 matrices a closed‑form solution exists; we use it for speed.
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    tr = a + d
    det = a * d - b * c
    # Compute eigenvalues λ₁, λ₂
    disc = math.sqrt(tr * tr - 4 * det)
    λ1 = (tr + disc) / 2.0
    λ2 = (tr - disc) / 2.0
    # Exponential of diagonalizable matrix
    if not math.isclose(λ1, λ2):
        expM = (
            np.exp(λ1) * (M - λ2 * np.eye(2)) - np.exp(λ2) * (M - λ1 * np.eye(2))
        ) / (λ1 - λ2)
    else:
        # Repeated eigenvalue – use series expansion up to 2nd order (sufficient for 2×2)
        expM = np.exp(λ1) * (np.eye(2) + (M - λ1 * np.eye(2)))
    return expM


def _numeric_B_integral(A: np.ndarray, B: np.ndarray, dt: float, steps: int = 32) -> np.ndarray:
    """Numerical approximation of ∫₀^{dt} exp(A·τ) dτ · B."""
    ts = np.linspace(0, dt, steps)
    integral = np.zeros_like(B, dtype=float)
    for τ in ts:
        integral += scipy_expm(A * τ) @ B
    integral *= dt / steps
    return integral


# ----------------------------------------------------------------------
# Hybrid SSM step implementations
# ----------------------------------------------------------------------


def hybrid_ssm_step(endpoint: EngineEndpoint, state: np.ndarray) -> np.ndarray:
    """
    Perform a single hybrid state‑space update for *one* endpoint.
    The update respects the underlying continuous‑time dynamics,
    discretises them, and then blends the result with the current state.
    """
    if state.ndim != 1 or state.shape[0] != 2:
        raise ValueError("state must be a 1‑D array of length 2.")

    # Build the continuous‑time model for this endpoint.
    A, B, C, D = build_state_space(endpoint)

    # Discrete causal matrix (2×2) that maps the previous state to the new state.
    causal = semiseparable_causal_matrix(A, B, C, D, dt=1.0)

    # Apply the causal matrix to the state.
    new_state = causal @ state

    # Blend with epistemic flags and feature‑count information.
    flags = epistemic_flag_vector(endpoint)
    feature_count = np.array([len(endpoint.capabilities)], dtype=float)

    # The flags vector length equals len(EPISTEMIC_FLAGS); we project it onto the state space.
    # Use the same C matrix (already normalised) as the projection operator.
    flag_projection = C @ flags.reshape(-1, 1)  # shape (1,1)
    # Broadcast to state dimension.
    flag_projection = np.full_like(new_state, flag_projection.squeeze())

    # Final combination – health_score already influences the dynamics via B.
    blended = new_state + flag_projection + feature_count

    return blended


def hybrid_ssm_sequential(endpoints: List[EngineEndpoint], initial_state: np.ndarray) -> np.ndarray:
    """
    Apply ``hybrid_ssm_step`` sequentially over a list of endpoints.
    The state is propagated forward, allowing each endpoint to influence the next.
    """
    state = np.asarray(initial_state, dtype=float)
    for ep in endpoints:
        state = hybrid_ssm_step(ep, state)
    return state


def hybrid_ssm_parallel(endpoints: List[EngineEndpoint], initial_state: np.ndarray) -> np.ndarray:
    """
    Parallel version: compute the contribution of each endpoint independently,
    then aggregate by a health‑score weighted average.
    """
    if initial_state.ndim != 1 or initial_state.shape[0] != 2:
        raise ValueError("initial_state must be a 1‑D array of length 2.")

    contributions = []
    weights = []

    for ep in endpoints:
        # Independent propagation from the same initial state.
        contrib = hybrid_ssm_step(ep, np.asarray(initial_state, dtype=float))
        contributions.append(contrib)
        weights.append(ep.health_score)

    contributions = np.stack(contributions)  # shape (N, 2)
    weights = np.array(weights, dtype=float).reshape(-1, 1)  # shape (N, 1)

    # Normalise weights to avoid shrinking the magnitude.
    if np.isclose(weights.sum(), 0.0):
        normalized_weights = np.full_like(weights, 1.0 / len(weights))
    else:
        normalized_weights = weights / weights.sum()

    # Weighted average across endpoints.
    parallel_state = (normalized_weights * contributions).sum(axis=0)
    return parallel_state


# ----------------------------------------------------------------------
# Example usage (guarded by __main__)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Sample morphology objects.
    morph1 = Morphology(length=2.0, width=1.0, height=1.5, mass=3.0)
    morph2 = Morphology(length=1.2, width=0.9, height=0.8, mass=2.1)

    # Sample engine endpoints.
    endpoint1 = EngineEndpoint(
        engine_id="1",
        channel="channel1",
        residency="residency1",
        runtime="runtime1",
        resource_class="rc1",
        always_on=True,
        endpoint="ep1",
        capabilities=["FACT", "PROBABLE"],
        health_score=0.6,
        morphology=morph1,
    )

    endpoint2 = EngineEndpoint(
        engine_id="2",
        channel="channel2",
        residency="residency2",
        runtime="runtime2",
        resource_class="rc2",
        always_on=False,
        endpoint="ep2",
        capabilities=["POSSIBLE", "UNCERTAIN"],
        health_score=0.9,
        morphology=morph2,
    )

    # Initial 2‑dimensional state.
    init_state = np.zeros(2, dtype=float)

    # Run the three core functions.
    print("Sequential result:", hybrid_ssm_sequential([endpoint1, endpoint2], init_state))
    print("Parallel result:  ", hybrid_ssm_parallel([endpoint1, endpoint2], init_state))
    print("Single step (ep1):", hybrid_ssm_step(endpoint1, init_state))