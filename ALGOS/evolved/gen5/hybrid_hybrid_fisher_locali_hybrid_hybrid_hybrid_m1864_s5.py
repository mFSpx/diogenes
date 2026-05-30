# DARWIN HAMMER — match 1864, survivor 5
# gen: 5
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m149_s5.py (gen4)
# born: 2026-05-29T23:39:27Z

"""Hybrid Algorithm: fisher_pheromone_fractional_router
Parents:
- hybrid_fisher_localization_hybrid_ternary_route_m40_s1.py (Gaussian beam, Fisher score, SSIM similarity routing)
- hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m149_s5.py (StoreState pheromone dynamics with Caputo fractional decay)

Mathematical Bridge:
The Fisher information derived from a packet’s textual surface is used as a *time‑like* argument
for the Caputo‑fractional decay kernel that modulates pheromone strength.  The similarity
between packet text and a reference (via SSIM) supplies a raw pheromone signal; this signal is
scaled by the fractional decay evaluated at the current StoreState level.  The resulting
pheromone inflow updates the StoreState, whose “dance” value (a bounded signal derived from
the last level change) is fed back as an additional weighting factor for the routing decision.
Thus the Fisher‑weighted similarity and the fractional pheromone dynamics are tightly coupled
into a single hybrid loop."""
import sys
import math
import random
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

# ----------------------------------------------------------------------
# Parent A – Gaussian beam, Fisher score, SSIM
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
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
# Parent B – StoreState and fractional decay
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    level: float = 0.0          # current pheromone store level
    alpha: float = 1.0          # scaling for inflow (also used as fractional order)
    beta: float = 1.0           # scaling for outflow
    dt: float = 1.0             # integration step
    base: float = 1.0           # base offset for dance signal
    gain: float = 1.0           # gain for dance signal
    limit: float = 10.0         # upper bound for dance signal
    _last_delta: float = 0.0    # internal bookkeeping

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Euler integration of the store level."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded signal derived from the most recent level change."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))

# ----------------------------------------------------------------------
# Hybrid building blocks
# ----------------------------------------------------------------------
def fractional_decay(alpha: float, t: float, eps: float = 1e-12) -> float:
    """
    Simple Caputo‑type fractional decay kernel.
    For 0 < alpha <= 1 the kernel behaves like exp(-t**alpha).
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError('alpha must be in (0, 1]')
    return math.exp(- (max(t, 0.0) ** alpha))

def fisher_weighted_similarity(packet: Dict[str, Any],
                               reference_text: str,
                               center: float,
                               width: float) -> float:
    """
    Compute a similarity score between packet text and a reference,
    weighted by the Fisher information of the packet’s “theta”.
    """
    text = str(packet.get("text_surface") or
               packet.get("raw_command") or
               packet.get("text") or "")
    # Encode characters as integer codes for SSIM
    x = np.fromiter((ord(c) for c in text), dtype=np.float64, count=len(text))
    y = np.fromiter((ord(c) for c in reference_text), dtype=np.float64, count=len(reference_text))
    # Pad the shorter array to match lengths (required for SSIM)
    if x.size < y.size:
        x = np.pad(x, (0, y.size - x.size), constant_values=0)
    elif y.size < x.size:
        y = np.pad(y, (0, x.size - y.size), constant_values=0)

    similarity = ssim(x, y)

    # Derive a theta value from packet length (normalized to [0, 2π])
    theta = (len(text) % 360) * (2 * math.pi / 360.0)
    fisher = fisher_score(theta, center, width)

    # Combine: Fisher acts as a multiplicative confidence weight
    return fisher * similarity

def hybrid_pheromone_inflow(packet: Dict[str, Any],
                            reference_text: str,
                            store: StoreState,
                            center: float = 0.0,
                            width: float = 1.0,
                            half_life: float = 10.0) -> float:
    """
    Produce a pheromone inflow value for the StoreState.
    The raw signal is the Fisher‑weighted SSIM similarity; it is
    attenuated by a classic exponential half‑life and by the
    fractional decay evaluated at the current store level.
    """
    raw_signal = fisher_weighted_similarity(packet, reference_text, center, width)

    # Classic discrete half‑life decay factor
    delta_t = store.dt
    exponential_factor = 0.5 ** (delta_t / half_life)

    # Fractional decay factor using the store level as the “time” argument
    frac_factor = fractional_decay(store.alpha, store.level)

    inflow = raw_signal * exponential_factor * frac_factor
    return inflow

def hybrid_route(packet: Dict[str, Any],
                 reference_text: str,
                 store: StoreState,
                 center: float = 0.0,
                 width: float = 1.0) -> Dict[str, Any]:
    """
    Perform a routing decision that incorporates:
    1. Fisher‑weighted SSIM similarity (produces pheromone inflow)
    2. StoreState update (modifies internal level)
    3. The StoreState “dance” signal as an additional routing weight
    Returns a new packet dict enriched with routing metadata.
    """
    inflow = hybrid_pheromone_inflow(packet, reference_text, store, center, width)
    # No explicit outflow in this simplified hybrid; could be extended later
    store.update(inflow=[inflow], outflow=[])

    # Compute final routing weight: similarity * (1 + normalized dance)
    base_similarity = fisher_weighted_similarity(packet, reference_text, center, width)
    dance_factor = 1.0 + (store.dance / store.limit)  # normalized to [1,2]
    routing_score = base_similarity * dance_factor

    routed_packet = {
        "original": packet,
        "routing_score": routing_score,
        "store_level": store.level,
        "dance": store.dance,
        "inflow_used": inflow,
        "timestamp": None  # placeholder for future extensions
    }
    return routed_packet

# ----------------------------------------------------------------------
# Demonstration / Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic packet
    pkt = {
        "text_surface": "Hello world! This is a test packet.",
        "source": "sensor_A",
        "payload": {"value": 42}
    }
    reference = "Hello world! This is a reference message for similarity."

    # Initialise StoreState with a non‑trivial alpha (fractional order)
    store = StoreState(level=0.5, alpha=0.7, beta=0.5, dt=1.0,
                       base=0.2, gain=0.8, limit=5.0)

    # Run a few hybrid steps to show dynamics
    for step in range(3):
        routed = hybrid_route(pkt, reference, store,
                              center=math.pi, width=0.5)
        print(f"Step {step+1}:")
        print(f"  Store level   = {routed['store_level']:.4f}")
        print(f"  Dance signal  = {routed['dance']:.4f}")
        print(f"  Routing score = {routed['routing_score']:.6f}")
        print("-" * 40)