# DARWIN HAMMER — match 2853, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_hybrid_m1864_s2.py (gen5)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_hybrid_diffus_m1445_s2.py (gen5)
# born: 2026-05-29T23:46:14Z

"""hybrid_fisher_pheromone_morphology.py
Hybrid algorithm merging:

- Parent A (hybrid_fisher_localization...): Gaussian beam, Fisher score, SSIM similarity, fractional decay kernel.
- Parent B (hybrid_endpoint_circuit...): Morphology based recovery priority, pheromone signaling, diffusion‑style update.

Mathematical bridge:
1. The Fisher score computed from a Gaussian beam is used as a *weight* for the SSIM
   similarity between two signal vectors (pheromone traces).  This creates a
   *Fisher‑weighted similarity* that quantifies how structurally similar two
   pheromone patterns are while emphasizing regions of high information content.
2. The fractional‑decay kernel modulates the *edge weights* of a routing graph.
   Those modulated weights are multiplied by the *recovery priority* derived from
   the morphology (righting‑time index).  The product is then employed in a
   minimum‑cost decision (e.g., selecting the next routing hop).
3. The pheromone update itself blends a decayed old pheromone level with a new
   signal scaled by the morphology‑derived priority, yielding a diffusion‑like
   dynamics that respects both physical decay and structural recovery ability.

The three core functions below demonstrate this fused behaviour.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural similarity index (1‑D version)."""
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


def fractional_decay(alpha: float, t: float) -> float:
    """Exponential decay kernel used in the second parent."""
    return math.exp(-alpha * t)


# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Physical description of the agent."""
    length: float
    width: float
    height: float
    mass: float


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology,
                        b: float = 1.0 / 3.0,
                        k: float = 0.35,
                        neck_lever: float = 1.0) -> float:
    """Time needed for the morphology to self‑right."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised priority in [0,1] based on righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def fisher_weighted_ssim(theta: float,
                         center: float,
                         width: float,
                         signal_a: np.ndarray,
                         signal_b: np.ndarray) -> float:
    """
    Compute SSIM between two pheromone/feature signals and weight it by the
    Fisher score evaluated at ``theta``.  The result lives in [0,1] (SSIM) scaled
    by a positive Fisher factor, providing a similarity measure that favours
    regions of high information content.
    """
    base_ssim = ssim(signal_a, signal_b)
    weight = fisher_score(theta, center, width)
    # Normalise weight to a sensible range to keep the final value bounded.
    # Empirically, Fisher scores for typical Gaussian parameters are <= 1.
    norm_weight = min(weight, 1.0)
    return base_ssim * norm_weight


def update_pheromone(pheromone: float,
                     t: float,
                     alpha: float,
                     morphology: Morphology,
                     incoming_signal: float) -> float:
    """
    Diffusion‑style pheromone update:
        new = decay(old) + priority * incoming
    where decay is the fractional decay kernel and priority comes from the
    morphology recovery priority.
    """
    decay_factor = fractional_decay(alpha, t)
    priority = recovery_priority(morphology)
    return pheromone * decay_factor + incoming_signal * priority


def route_via_morphology(edge_weights: np.ndarray,
                         t: float,
                         alpha: float,
                         morphology: Morphology) -> int:
    """
    Choose the index of the minimum‑cost edge after modulating the raw
    ``edge_weights`` with both fractional decay and morphology‑derived priority.
    Returns the index of the selected edge (i.e., next hop).
    """
    if edge_weights.ndim != 1:
        raise ValueError("edge_weights must be a 1‑D array")
    decay = fractional_decay(alpha, t)
    priority = recovery_priority(morphology)
    modulated = edge_weights * decay * (1.0 - priority)  # lower priority → higher cost
    return int(np.argmin(modulated))


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Dummy signals for SSIM
    sig1 = np.random.randint(0, 256, size=100).astype(float)
    sig2 = np.random.randint(0, 256, size=100).astype(float)

    # Fisher‑weighted similarity
    theta_val = 0.3
    sim = fisher_weighted_ssim(theta_val, center=0.0, width=1.0, signal_a=sig1, signal_b=sig2)
    print(f"Fisher‑weighted SSIM: {sim:.4f}")

    # Morphology instance
    morph = Morphology(length=0.5, width=0.3, height=0.2, mass=2.0)

    # Pheromone dynamics
    old_pheromone = 0.8
    t = 2.0
    alpha = 0.1
    incoming = 0.4
    new_pheromone = update_pheromone(old_pheromone, t, alpha, morph, incoming)
    print(f"Updated pheromone level: {new_pheromone:.4f}")

    # Routing decision
    edges = np.array([1.2, 0.9, 1.5, 0.7])
    chosen = route_via_morphology(edges, t, alpha, morph)
    print(f"Chosen edge index: {chosen}")

    # Timestamp sanity check
    print(f"Timestamp: {now_z()}")

    sys.exit(0)