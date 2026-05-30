# DARWIN HAMMER — match 1864, survivor 4
# gen: 5
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m149_s5.py (gen4)
# born: 2026-05-29T23:39:27Z

"""Hybrid Algorithm integrating Fisher‑SSIM routing with fractional pheromone decay.

Parents:
- PARENT ALGORITHM A: fisher_localization + SSIM similarity routing.
- PARENT ALGORITHM B: StoreState pheromone dynamics with Caputo fractional decay.

Mathematical Bridge:
The SSIM similarity between a packet's text surface and a reference string is first
converted into a Fisher information weight (via `fisher_score`). This weight modulates
the *fractional decay kernel* that governs pheromone signal attenuation in the
StoreState dynamics. Consequently, the inflow to the pheromone store is

    inflow = base_signal * 0.5^{Δt/half_life} * fractional_decay(α, level) * fisher_weight

where `fractional_decay(α, level) = level^{α-1} / Γ(α)`. The store level then feeds back
into the routing decision through the StoreState `dance` signal, creating a closed
loop that fuses both topologies.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Utilities from Parent A
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
# Fractional decay from Parent B (Caputo kernel approximation)
# ----------------------------------------------------------------------
def fractional_decay(alpha: float, t: float) -> float:
    """Caputo‑type decay kernel: t^{α-1} / Γ(α)  (t ≥ 0)."""
    if t < 0:
        return 0.0
    if alpha <= 0:
        raise ValueError('alpha must be positive')
    return (t ** (alpha - 1)) / math.gamma(alpha)

def pheromone_decay_signal(signal0: float,
                           dt: float,
                           half_life: float,
                           store_level: float,
                           alpha: float) -> float:
    """Base signal attenuated by exponential half‑life and fractional decay."""
    exponential = 0.5 ** (dt / half_life) if half_life > 0 else 1.0
    frac = fractional_decay(alpha, store_level)
    return signal0 * exponential * frac

# ----------------------------------------------------------------------
# StoreState from Parent B
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0      # inflow scaling
    beta: float = 1.0       # outflow scaling
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Euler integration of the store level."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded signal derived from the last delta; used for routing bias."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_fisher_weight(packet_text: str,
                          reference_text: str,
                          center: float,
                          width: float) -> float:
    """Compute Fisher information weight from SSIM similarity."""
    x = np.fromiter((ord(c) for c in packet_text), dtype=np.float64)
    y = np.fromiter((ord(c) for c in reference_text), dtype=np.float64)
    # Pad the shorter array to match lengths
    if x.size < y.size:
        x = np.pad(x, (0, y.size - x.size), constant_values=0)
    elif y.size < x.size:
        y = np.pad(y, (0, x.size - y.size), constant_values=0)
    similarity = ssim(x, y)
    # Map similarity ([-1,1] approx) to a theta‑like quantity
    theta = (similarity + 1.0) * 0.5  # now in [0,1]
    return fisher_score(theta, center, width)

def packet_inflow(packet: Dict,
                  reference_text: str,
                  store: StoreState,
                  base_signal: float,
                  half_life: float,
                  center: float,
                  width: float,
                  frac_alpha: float) -> float:
    """Calculate the pheromone inflow contributed by a single packet."""
    text = str(packet.get("text_surface") or
               packet.get("raw_command") or
               packet.get("text") or "")
    fisher_w = compute_fisher_weight(text, reference_text, center, width)
    raw_signal = pheromone_decay_signal(base_signal,
                                        store.dt,
                                        half_life,
                                        store.level,
                                        frac_alpha)
    return raw_signal * fisher_w

def hybrid_route(packet: Dict,
                 reference_text: str,
                 store: StoreState,
                 base_signal: float = 1.0,
                 half_life: float = 5.0,
                 center: float = 0.5,
                 width: float = 0.15,
                 frac_alpha: float = 0.8) -> Dict:
    """
    Perform a routing decision that couples SSIM/Fisher similarity with
    fractional pheromone dynamics.

    Returns a dict containing:
        - inflow: computed inflow value
        - new_level: updated store level
        - route: chosen path ('high_priority' if dance > threshold else 'standard')
        - metadata: auxiliary information for debugging
    """
    inflow = packet_inflow(packet, reference_text, store,
                           base_signal, half_life,
                           center, width, frac_alpha)
    # No explicit outflow in this simplified hybrid; can be extended later.
    new_level, _ = store.update([inflow], [])
    route = "high_priority" if store.dance > (store.limit * 0.6) else "standard"
    metadata = {
        "dance_signal": store.dance,
        "store_level": new_level,
        "inflow": inflow,
        "fisher_center": center,
        "fisher_width": width,
        "fractional_alpha": frac_alpha
    }
    return {"inflow": inflow, "new_level": new_level, "route": route, "metadata": metadata}

def simulate_hybrid_flow(packets: List[Dict],
                         reference_text: str,
                         steps: int = 10) -> List[Dict]:
    """Run a short simulation over a list of packets, updating the StoreState."""
    store = StoreState(level=0.5, alpha=1.2, beta=0.9, dt=1.0, base=1.0, gain=2.0, limit=12.0)
    results = []
    for i in range(min(steps, len(packets))):
        res = hybrid_route(packets[i], reference_text, store)
        results.append(res)
    return results

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal packet collection
    packets = [
        {"text_surface": "activate sensor array", "source": "nodeA"},
        {"text_surface": "deactivate shield", "source": "nodeB"},
        {"text_surface": "run diagnostic", "source": "nodeC"},
        {"text_surface": "update firmware", "source": "nodeD"},
        {"text_surface": "reset subsystem", "source": "nodeE"},
    ]
    reference = "activate sensor array"
    simulation = simulate_hybrid_flow(packets, reference, steps=5)
    for idx, entry in enumerate(simulation, 1):
        print(f"Step {idx}: route={entry['route']}, level={entry['new_level']:.4f}, inflow={entry['inflow']:.6f}")
    sys.exit(0)