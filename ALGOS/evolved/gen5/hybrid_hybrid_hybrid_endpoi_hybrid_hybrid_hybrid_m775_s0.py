# DARWIN HAMMER — match 775, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s2.py (gen3)
# born: 2026-05-29T23:30:58Z

"""Hybrid algorithm combining:
- Parent A: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s0.py
- Parent B: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s2.py

Mathematical bridge:
Both parents expose an SSIM similarity measure.  Parent A provides a Fisher information‑derived
scalar `fisher_score` that quantifies the curvature of a Gaussian beam; this scalar is used
as a *learning‑rate modifier*.  Parent B updates a linear weight matrix `W` of a ternary
router using an error signal derived from an SSIM comparison between the router’s input
and output.  The hybrid therefore:
1. Computes a geometry‑driven Fisher factor `γ = fisher_score(θ, μ, σ)`.
2. Routes a feature vector `x` through a linear ternary router `y = softmax(W·x)`.
3. Evaluates similarity `ρ = ssim(x, y)`.
4. Forms an error `e = 1‑ρ` and updates the router matrix with a Fisher‑scaled step:
   `ΔW = - η·γ·e·(y·xᵀ)`  (outer product of output and input).

The resulting system adapts its routing matrix according to both statistical curvature
(Fisher) and perceptual similarity (SSIM)."""

import math
import random
import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timezone
import json
from typing import Any, Dict

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian envelope centred at *center* with standard‑deviation *width*."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam (scalar curvature)."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index Measure (SSIM) for 1‑D signals."""
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


# ----------------------------------------------------------------------
# Simple geometric container (Parent A)
# ----------------------------------------------------------------------
class Morphology:
    """Geometric description of an entity; supplies a Fisher‑scaled learning rate."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        if length <= 0 or width <= 0 or height <= 0:
            raise ValueError("geometric dimensions must be positive")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

    def fisher_learning_rate(self, theta: float) -> float:
        """Map a scalar *theta* to a learning‑rate factor via Fisher information."""
        # Use the entity’s geometric width as the Gaussian width; centre at half‑length.
        center = self.length / 2.0
        width = self.width
        return fisher_score(theta, center, width)


# ----------------------------------------------------------------------
# Ternary linear router (Parent B)
# ----------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z_max = np.max(z)
    e = np.exp(z - z_max)
    return e / e.sum()


class TernaryRouter:
    """Linear router with a 3‑state (ternary) output space."""
    def __init__(self, input_dim: int, output_dim: int = 3, seed: int | None = None):
        rng = np.random.default_rng(seed)
        # Initialise weight matrix with small random values.
        self.W = rng.normal(loc=0.0, scale=0.01, size=(output_dim, input_dim))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute router output probabilities."""
        logits = self.W @ x
        return softmax(logits)

    def update(self, x: np.ndarray, y: np.ndarray,
               error: float, lr_factor: float, base_lr: float = 0.1) -> None:
        """Fisher‑scaled gradient descent on the weight matrix.

        ΔW = - η·γ·e·(y·xᵀ)
        where η is a base learning rate, γ is the Fisher factor, e is the error.
        """
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1‑D vectors")
        outer = np.outer(y, x)               # shape (output_dim, input_dim)
        delta = - base_lr * lr_factor * error * outer
        self.W += delta


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_fisher_scaled_lr(morph: Morphology, theta: float) -> float:
    """Return a learning‑rate scaling factor derived from morphology."""
    return morph.fisher_learning_rate(theta)


def route_and_score(packet: Dict[str, Any],
                    router: TernaryRouter,
                    morph: Morphology) -> Dict[str, Any]:
    """
    Extract a numeric feature vector from *packet*, run the ternary router,
    compute SSIM similarity between input and output, and return a enriched
    routing dict.
    """
    # 1️⃣ Feature extraction – for demo purposes we hash textual content into a fixed‑size vector.
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    # Simple deterministic pseudo‑random vector based on text hash.
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    input_vec = rng.random(morph.length)  # length used as dimensionality

    # 2️⃣ Router forward pass.
    output_vec = router.forward(input_vec)

    # 3️⃣ Similarity assessment via SSIM (scale vectors to 0‑255 range).
    scale = 255.0
    ssim_score = ssim(input_vec * scale, output_vec * scale)

    # 4️⃣ Package result.
    result = {
        "input_vector": input_vec.tolist(),
        "output_vector": output_vec.tolist(),
        "ssim": float(ssim_score),
        "engine_channel": "hybrid_fisher_ternary",
        "outbound_state": "draft_only",
    }
    return result


def hybrid_training_step(packet: Dict[str, Any],
                         router: TernaryRouter,
                         morph: Morphology,
                         theta: float = 0.5) -> None:
    """
    Perform one training iteration:
    - compute Fisher‑scaled learning rate,
    - forward‑propagate,
    - evaluate SSIM,
    - update router weights.
    """
    # Extract same deterministic input vector as in route_and_score.
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    x = rng.random(morph.length)

    y = router.forward(x)

    # SSIM similarity (scaled to 0‑255).
    ρ = ssim(x * 255.0, y * 255.0)
    error = 1.0 - ρ

    # Fisher factor.
    γ = compute_fisher_scaled_lr(morph, theta)

    # Update router.
    router.update(x, y, error, lr_factor=γ)


# ----------------------------------------------------------------------
# Helper utilities (mirroring Parent B)
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z‑format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_context(text: str | None) -> dict[str, Any]:
    """Parse optional JSON context."""
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a simple morphology (dimensionality = 8).
    morph = Morphology(length=8.0, width=2.0, height=1.5, mass=3.0)

    # Initialise router with matching input dimension.
    router = TernaryRouter(input_dim=int(morph.length), seed=42)

    # Dummy packet.
    packet = {
        "text_surface": "example payload",
        "raw_command": None,
        "text": None,
        "source": "unit_test",
        "payload": {"value": 123},
    }

    # Run a forward pass and display SSIM.
    result = route_and_score(packet, router, morph)
    print("Forward pass result:", result)

    # Perform a few training steps.
    for i in range(5):
        hybrid_training_step(packet, router, morph, theta=0.7)
        updated = route_and_score(packet, router, morph)
        print(f"Step {i+1} SSIM:", updated["ssim"])

    print("Smoke test completed at", now_z())