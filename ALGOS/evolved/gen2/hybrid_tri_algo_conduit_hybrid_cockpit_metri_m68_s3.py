# DARWIN HAMMER — match 68, survivor 3
# gen: 2
# parent_a: tri_algo_conduit.py (gen0)
# parent_b: hybrid_cockpit_metrics_rectified_flow_m10_s1.py (gen1)
# born: 2026-05-29T23:26:48Z

import numpy as np
import math
from dataclasses import dataclass
from typing import Callable, Tuple

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ConduitDecision:
    """Immutable container for a decision made by the hybrid system."""
    action: str
    confidence_gap: float
    epsilon: float
    signal_score: float
    noise_score: float
    dormancy_probability: float
    recovery_priority: float
    reason: str


# ----------------------------------------------------------------------
# Entropy utilities
# ----------------------------------------------------------------------
def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    """Shannon entropy of a byte sequence, normalized to the range [0, 1].

    The raw entropy is measured in bits (base‑2).  Dividing by 8 converts the
    maximum possible entropy (8 bits per byte) to a unit interval, which
    matches the scaling used elsewhere in the algorithm.
    """
    if not data:
        return 0.0
    chunk = data[:sample]
    return shannon_entropy(chunk) / 8.0


def shannon_entropy(sequence: bytes) -> float:
    """Classic Shannon entropy (bits) for a byte sequence."""
    if not sequence:
        return 0.0
    # Frequency count – using a 256‑length array is faster than a dict.
    freq = np.bincount(np.frombuffer(sequence, dtype=np.uint8), minlength=256)
    prob = freq[freq > 0] / len(sequence)
    return -np.sum(prob * np.log2(prob))


# ----------------------------------------------------------------------
# Scoring functions (original “tri_algo_conduit” side)
# ----------------------------------------------------------------------
def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> Tuple[float, float]:
    """Compute a signal‑to‑noise pair in the unit interval.

    The formulation mirrors the original implementation but adds a few
    safeguards:
      * ``status_code`` handling is now explicit about falsy values.
      * ``mime`` is lower‑cased once for efficiency.
      * The entropy contribution is capped to avoid runaway values.
    """
    size = len(data)
    entropy = _byte_entropy(data)

    # --- Bonuses -------------------------------------------------------
    status_bonus = 0.18 if status_code is not None and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(tok in (mime or "").lower() for tok in ("html", "json", "text", "xml")) else 0.02
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)

    # --- Signal --------------------------------------------------------
    raw_signal = (
        0.20
        + status_bonus
        + mime_bonus
        + size_bonus
        + keyword_bonus
        + structure_bonus
        + 0.12 * entropy
    )
    signal = np.clip(raw_signal, 0.0, 1.0)

    # --- Noise ---------------------------------------------------------
    raw_noise = (
        0.58
        - 0.22 * entropy
        - keyword_bonus
        - structure_bonus
        + (0.12 if size < 64 else 0.0)
    )
    noise = np.clip(raw_noise, 0.0, 1.0)

    return float(signal), float(noise)


def cockpit_honesty(signal_score: float, noise_score: float) -> float:
    """Proportion of signal in the total (signal + noise)."""
    total = signal_score + noise_score
    return 1.0 if total <= 0 else np.clip(signal_score / total, 0.0, 1.0)


def anti_slop_ratio(signal_score: float, total_score: float) -> float:
    """Weight that penalises low‑signal trajectories.

    The original code passed ``total_score = 1.0`` which nullified the
    denominator.  Here we use the genuine total (signal + noise) to keep
    the ratio meaningful.
    """
    if total_score <= 0:
        return 0.0
    return np.clip(signal_score / total_score, 0.0, 1.0)


# ----------------------------------------------------------------------
# Flow utilities (original “hybrid_cockpit_metrics_rectified_flow” side)
# ----------------------------------------------------------------------
def interpolant(x0, x1, t):
    """Linear interpolation that works for scalars, vectors or tensors."""
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if x0.ndim > t.ndim:
        t = t[..., np.newaxis]
    return t * x1 + (1.0 - t) * x0


def flow_target(x0, x1):
    """Desired displacement from ``x0`` to ``x1``."""
    return np.asarray(x1, dtype=np.float64) - np.asarray(x0, dtype=np.float64)


# ----------------------------------------------------------------------
# Regularisation – the weakest point of the original hybrid
# ----------------------------------------------------------------------
def _curvature_penalty(trajectory: np.ndarray) -> float:
    """Second‑order smoothness penalty (discrete curvature).

    ``trajectory`` is of shape (T, …).  The penalty is the mean squared
    second‑difference, encouraging straight lines without excessive
    oscillation.
    """
    # First differences (velocities)
    v = np.diff(trajectory, axis=0)
    # Second differences (accelerations / curvature)
    a = np.diff(v, axis=0)
    return float(np.mean(a ** 2))


def signal_straightness_regularization(
    signal_score: float,
    noise_score: float,
    x0: np.ndarray,
    x1: np.ndarray,
    v_pred: np.ndarray,
) -> float:
    """Weighted L2 deviation from the ideal straight flow.

    The weight is derived from the *honesty* metric (signal / (signal+noise)),
    ensuring that low‑quality signals receive a stronger penalty.
    """
    target = flow_target(x0, x1)
    diff = v_pred - target
    honesty = cockpit_honesty(signal_score, noise_score)
    # Inverse honesty: the worse the honesty, the larger the penalty.
    weight = 1.0 - honesty
    return float(np.mean(diff ** 2) * weight)


def flow_loss(
    v_pred: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    signal_score: float,
    noise_score: float,
    trajectory: np.ndarray | None = None,
    λ_reg: float = 0.1,
    λ_curv: float = 0.05,
) -> float:
    """Composite loss for a single step of the flow.

    Parameters
    ----------
    v_pred : np.ndarray
        Predicted velocity field.
    x0, x1 : np.ndarray
        Start and target positions.
    signal_score, noise_score : float
        Scores from ``signal_scores``.
    trajectory : np.ndarray, optional
        Full trajectory up to the current step.  If supplied, a curvature
        penalty is added.
    λ_reg, λ_curv : float
        Hyper‑parameters controlling the strength of the regularisation
        and curvature terms respectively.
    """
    target = flow_target(x0, x1)
    diff = v_pred - target
    base = np.mean(diff ** 2)

    # Adaptive regularisation based on honesty
    reg = signal_straightness_regularization(
        signal_score, noise_score, x0, x1, v_pred
    )
    loss = base + λ_reg * reg

    # Optional curvature term – deepens the mathematical coupling
    if trajectory is not None:
        loss += λ_curv * _curvature_penalty(trajectory)

    return float(loss)


# ----------------------------------------------------------------------
# Solver – now returns the full trajectory for curvature evaluation
# ----------------------------------------------------------------------
def hybrid_solve(
    v_fn: Callable[[np.ndarray, float, float], np.ndarray],
    x0: np.ndarray,
    signal_score: float,
    noise_score: float,
    steps: int = 10,
) -> np.ndarray:
    """Integrate the velocity field while respecting the hybrid loss.

    The solver records the entire trajectory, enabling curvature‑based
    regularisation without requiring a second pass.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    dt = 1.0 / steps
    ts = np.linspace(0.0, 1.0 - dt, steps)

    traj = np.empty((steps + 1,) + x0.shape, dtype=np.float64)
    traj[0] = x0
    z = x0.copy()

    for k, t in enumerate(ts):
        v = v_fn(z, float(t), signal_score)
        # Simple explicit Euler step
        z = z + dt * np.asarray(v, dtype=np.float64)
        traj[k + 1] = z

    return traj


# ----------------------------------------------------------------------
# Example / smoke test
# ----------------------------------------------------------------------
def _example_velocity(z: np.ndarray, t: float, signal_score: float) -> np.ndarray:
    """Toy velocity field used by the smoke test.

    It points directly toward the origin, scaled by the signal score.
    """
    direction = -z
    norm = np.linalg.norm(direction) + 1e-9
    return (signal_score / norm) * direction


def smoke_test() -> None:
    data = b"Hello, World!"
    signal, noise = signal_scores(data, status_code=200, mime="text/plain")
    honesty = cockpit_honesty(signal, noise)
    anti_slop = anti_slop_ratio(signal, signal + noise)

    print(
        f"Signal: {signal:.4f}, Noise: {noise:.4f}, Honesty: {honesty:.4f}, "
        f"Anti‑Slop: {anti_slop:.4f}"
    )

    # Simple 2‑D start/target
    start = np.array([2.0, 3.0])
    target = np.array([0.0, 0.0])

    traj = hybrid_solve(_example_velocity, start, signal, noise, steps=20)

    # Evaluate loss on the final step (illustrative)
    final_v = _example_velocity(traj[-2], 0.95, signal)
    loss = flow_loss(
        final_v,
        traj[-2],
        target,
        signal,
        noise,
        trajectory=traj,
    )
    print(f"Final loss (incl. curvature): {loss:.6f}")


if __name__ == "__main__":
    smoke_test()