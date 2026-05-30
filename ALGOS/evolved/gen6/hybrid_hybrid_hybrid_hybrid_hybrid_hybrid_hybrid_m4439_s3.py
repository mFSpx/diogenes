# DARWIN HAMMER — match 4439, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m976_s5.py (gen5)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_model__m584_s1.py (gen4)
# born: 2026-05-29T23:55:40Z

"""Hybrid Morphology‑Circuit‑Flow ↔ Fisher‑Geometric Fusion
================================================================

This module fuses the two DARWIN‑HAMMER parents:

* **Parent A** – ``hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s2.py``  
  Provides a :class:`Morphology` description, an ``EndpointCircuitBreaker`` and a
  *Fisher* weight that rescales a transport flow.

* **Parent B** – ``hybrid_hybrid_hybrid_fisher_hybrid_hybrid_model__m584_s1.py``  
  Supplies a Gaussian‑beam based Fisher information routine, a geometric‑product
  implementation and a VRAM‑scheduler.

**Mathematical bridge**

1. Let ``m = [ℓ, w, h, μ]`` be the 4‑dimensional morphology vector.
2. Let ``s ∈ ℝ^{p}`` be a stylometric embedding and ``b ∈ ℝ^{q}`` a brain‑map
   embedding.  Concatenating yields the *extended state*

   ``v̂ = [s; b; m] ∈ ℝ^{p+q+4}``.

3. For each component of ``m`` we treat it as the mean ``θ`` of a Gaussian
   beam with a fixed width ``σ = 1.0`` and centre ``c = 0``.  The scalar
   Fisher information from Parent B is

   ``I_i = fisher_score(θ_i, c, σ)``

   and the **Fisher weight** is the geometric mean of the four entries:

   ``w_f = (∏_{i=1}^4 I_i)^{1/4}``.

4. The raw transport is realised as a *geometric product* between the
   stylometric and brain parts of the extended state:

   ``Φ(v̂) = geometric_product(s, b)``   (Parent B).

   The Fisher weight rescales it:

   ``Φ_f = w_f · Φ(v̂)``.

5. Parent A contributes an Ollivier‑Ricci curvature ``κ`` defined on the
   edge ``(v̂_src, v̂_tgt)``.  We approximate it by the cosine similarity of the
   two extended states and map it to ``[-1, 1]``:

   ``κ = 2·cos_sim(v̂_src, v̂_tgt) - 1``.

   The curvature modulates the flow:

   ``v_hybrid = (1 + κ) · Φ_f``.

6. The resulting scalar drives an ``EndpointCircuitBreaker``; its failure
   probability is pruned proportionally to the absolute hybrid flow magnitude.

The three core functions below implement this pipeline, and a tiny smoke‑test
exercises the whole chain without external dependencies."""

import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – morphology and circuit‑breaker primitives
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


class EndpointCircuitBreaker:
    """Simple breaker whose failure probability can be pruned."""

    def __init__(self, base_failure: float = 0.5):
        if not 0.0 <= base_failure <= 1.0:
            raise ValueError("base_failure must be in [0, 1]")
        self.base_failure = base_failure

    def prune_probability(self, factor: float) -> float:
        """
        Reduce the failure probability by ``factor`` (clamped to [0,1]).
        Larger ``factor`` → smaller failure chance.
        """
        factor = max(0.0, min(1.0, factor))
        pruned = self.base_failure * (1.0 - factor)
        return max(0.0, min(1.0, pruned))


# ----------------------------------------------------------------------
# Parent B – Fisher, geometric product and VRAM scheduler
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity G(θ) with centre `center` and std‑dev `width`."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam w.r.t. its mean.
    Implements (∂G/∂θ)² / G, which equals 1/width² for a pure Gaussian.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width ** 2))
    return (derivative ** 2) / intensity


def geometric_product(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Geometric product between two vectors.
    For the purposes of this hybrid we use the ordinary dot product,
    which is the scalar part of the full Clifford product.
    """
    return float(np.dot(vector1, vector2))


def vram_scheduler(memory_required: float, gpu_memory: float) -> float:
    """
    Very small VRAM scheduler: returns the fraction of GPU memory that will be used.
    """
    if gpu_memory <= 0:
        raise ValueError("gpu_memory must be positive")
    return min(1.0, memory_required / gpu_memory)


# ----------------------------------------------------------------------
# Hybrid core – mathematical bridge
# ----------------------------------------------------------------------


def build_extended_state(
    stylometric: np.ndarray,
    brain: np.ndarray,
    morph: Morphology,
) -> np.ndarray:
    """
    Concatenate stylometric, brain and morphology vectors into a single state.
    """
    m_vec = np.array([morph.length, morph.width, morph.height, morph.mass], dtype=float)
    return np.concatenate([stylometric, brain, m_vec])


def fisher_weight_from_morph(morph: Morphology, width: float = 1.0, center: float = 0.0) -> float:
    """
    Compute a scalar Fisher weight from the four morphology components.
    Each component is interpreted as the mean of a Gaussian beam.
    The weight is the geometric mean of the four Fisher informations.
    """
    components = np.array([morph.length, morph.width, morph.height, morph.mass], dtype=float)
    infos = [fisher_score(theta=float(c), center=center, width=width) for c in components]
    # geometric mean
    prod = 1.0
    for i in infos:
        prod *= i
    return float(prod ** (1.0 / len(infos)))


def curvature_cosine(v_src: np.ndarray, v_tgt: np.ndarray) -> float:
    """
    Approximate Ollivier‑Ricci curvature by a rescaled cosine similarity.
    Cosine similarity ∈ [-1, 1]; we map it to κ ∈ [-1, 1] directly.
    """
    dot = float(np.dot(v_src, v_tgt))
    norm_src = float(np.linalg.norm(v_src))
    norm_tgt = float(np.linalg.norm(v_tgt))
    if norm_src == 0 or norm_tgt == 0:
        return 0.0
    cos_sim = dot / (norm_src * norm_tgt)
    # Clamp to avoid numerical drift
    return max(-1.0, min(1.0, cos_sim))


def hybrid_transport(
    stylometric_src: np.ndarray,
    brain_src: np.ndarray,
    stylometric_tgt: np.ndarray,
    brain_tgt: np.ndarray,
    morph_src: Morphology,
    morph_tgt: Morphology,
) -> Tuple[float, float]:
    """
    Perform the full hybrid computation.

    Returns
    -------
    hybrid_value : float
        The curvature‑adjusted, Fisher‑weighted transport scalar.
    pruned_failure : float
        Failure probability of an ``EndpointCircuitBreaker`` after pruning.
    """
    # 1. Build extended states
    v_src = build_extended_state(stylometric_src, brain_src, morph_src)
    v_tgt = build_extended_state(stylometric_tgt, brain_tgt, morph_tgt)

    # 2. Fisher weight from source morphology (any consistent choice works)
    w_f = fisher_weight_from_morph(morph_src)

    # 3. Raw transport via geometric product of stylometric & brain parts
    phi_raw = geometric_product(stylometric_src, brain_tgt)

    # 4. Apply Fisher scaling
    phi_fisher = w_f * phi_raw

    # 5. Curvature modulation
    kappa = curvature_cosine(v_src, v_tgt)
    hybrid_val = (1.0 + kappa) * phi_fisher

    # 6. Circuit breaker pruning
    breaker = EndpointCircuitBreaker(base_failure=0.5)
    # Use magnitude of hybrid_val (abs) normalised to [0,1] as pruning factor
    factor = max(0.0, min(1.0, abs(hybrid_val) / (abs(hybrid_val) + 1.0)))
    pruned = breaker.prune_probability(factor)

    return hybrid_val, pruned


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Random but reproducible seeds
    random.seed(42)
    np.random.seed(42)

    # Generate dummy embeddings
    dim_styl = 8
    dim_brain = 6
    s_src = np.random.randn(dim_styl)
    b_src = np.random.randn(dim_brain)
    s_tgt = np.random.randn(dim_styl)
    b_tgt = np.random.randn(dim_brain)

    # Random morphology instances
    morph_src = Morphology(
        length=abs(random.gauss(10, 2)),
        width=abs(random.gauss(5, 1)),
        height=abs(random.gauss(2, 0.5)),
        mass=abs(random.gauss(20, 4)),
    )
    morph_tgt = Morphology(
        length=abs(random.gauss(9, 2)),
        width=abs(random.gauss(4.5, 1)),
        height=abs(random.gauss(2.2, 0.5)),
        mass=abs(random.gauss(22, 4)),
    )

    # Run hybrid transport
    hybrid_val, pruned_failure = hybrid_transport(
        stylometric_src=s_src,
        brain_src=b_src,
        stylometric_tgt=s_tgt,
        brain_tgt=b_tgt,
        morph_src=morph_src,
        morph_tgt=morph_tgt,
    )

    print(f"{now_z()} – Hybrid transport value: {hybrid_val:.6f}")
    print(f"{now_z()} – Pruned failure probability: {pruned_failure:.4f}")

    # Demonstrate VRAM scheduler (auxiliary)
    mem_req = 2.3  # GB
    gpu_mem = 8.0  # GB
    usage = vram_scheduler(mem_req, gpu_mem)
    print(f"{now_z()} – VRAM usage fraction: {usage:.2%}")