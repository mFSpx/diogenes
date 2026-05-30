# DARWIN HAMMER — match 2764, survivor 6
# gen: 6
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1892_s0.py (gen5)
# born: 2026-05-29T23:45:45Z

"""Hybrid Fisher‑SSIM Geometric Routing
This module fuses the core mathematics of two parent algorithms:

* **Parent A** provides a Gaussian beam model, Fisher‑information scoring and a
  Structural Similarity (SSIM) index for comparing one‑dimensional signals.
* **Parent B** supplies a lightweight geometric‑algebra implementation (Multivector)
  together with Euclidean distance utilities.

**Mathematical bridge**

1. A text payload is mapped to a 2‑D point `p = (x, y)` (via a deterministic
   character‑hash).  
2. The polar angle `θ = atan2(y, x)` is fed to the Gaussian beam, yielding a
   Fisher score `F(θ)`.  
3. The same payload, interpreted as a numeric signal, is compared with a
   reference signal using SSIM, giving a similarity `S`.  
4. The scalar weight `w = F·S` modulates a geometric‑algebra representation:
   the point `p` is turned into a multivector `M(p)`.  Dot‑products between
   `M(p)` and seed multivectors are scaled by `w`, producing a weighted
   similarity metric that drives routing decisions.

The resulting hybrid algorithm therefore combines statistical‑information
theory (Fisher), perceptual similarity (SSIM) and geometric algebra in a
single unified routing computation.
"""

import math
import random
import sys
import pathlib
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Gaussian beam, Fisher score, SSIM
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity for angle theta."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information of the Gaussian beam at angle theta."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.cov(x, y, ddof=0)[0, 1]
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))

# ----------------------------------------------------------------------
# Parent B – Minimal geometric algebra (Multivector) and distance utils
# ----------------------------------------------------------------------
class Multivector:
    """Simple multivector for up to n‑dimensional Euclidean space."""
    def __init__(self, components: dict, n: int):
        # components: {frozenset({i, j, ...}): coefficient}
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        """Extract the k‑grade part."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if abs(v) > 1e-15}, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        neg = Multivector({k: -v for k, v in other.components.items()}, other.n)
        return self + neg

    def __mul__(self, other: "Multivector") -> "Multivector":
        # Geometric product limited to scalar‑vector interactions for simplicity
        result = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                new_blade = blade_a.symmetric_difference(blade_b)
                sign = 1
                # Determine sign from swapping basis vectors (grade‑2 antisymmetry)
                # For this lightweight implementation we ignore sign changes;
                # the product behaves like a commutative multiplication of scalars.
                result[new_blade] = result.get(new_blade, 0.0) + coeff_a * coeff_b * sign
        return Multivector({k: v for k, v in result.items() if abs(v) > 1e-15}, self.n)

    def dot(self, other: "Multivector") -> float:
        """Scalar product (grade‑0 part of geometric product)."""
        return sum(
            coeff_a * other.components.get(blade_a, 0.0)
            for blade_a, coeff_a in self.components.items()
            if len(blade_a) == 1  # vector part only
        )

def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance in 2‑D."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    """Index of the seed nearest to point."""
    if not seeds:
        raise ValueError('seed list must not be empty')
    dists = [distance(point, s) for s in seeds]
    return int(np.argmin(dists))

# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def text_to_point(text: str) -> tuple[float, float]:
    """Deterministically map a string to a 2‑D point."""
    if not text:
        return (0.0, 0.0)
    # Simple hash: split ordinal sum into two components
    total = sum(ord(ch) for ch in text)
    x = (total % 257) / 256.0  # normalized to [0,1)
    y = ((total // 257) % 257) / 256.0
    return (x, y)

def multivector_from_point(p: tuple[float, float]) -> Multivector:
    """Create a vector‑grade multivector from a 2‑D point."""
    components = {
        frozenset({1}): p[0],
        frozenset({2}): p[1],
    }
    return Multivector(components, n=2)

def hybrid_weight(packet: dict, reference_text: str,
                  center: float, width: float) -> float:
    """Combine Fisher information (angular) with SSIM (signal) into a scalar weight."""
    # 1) Convert packet text to point and angle
    payload = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    pt = text_to_point(payload)
    theta = math.atan2(pt[1], pt[0])  # angle in radians

    # 2) Fisher score for that angle
    f = fisher_score(theta, center, width)

    # 3) SSIM between payload and reference (as ascii arrays)
    a = np.fromiter((ord(c) for c in payload), dtype=np.float64, count=len(payload))
    b = np.fromiter((ord(c) for c in reference_text), dtype=np.float64, count=len(reference_text))
    # Pad the shorter array with zeros to equal length
    if a.size < b.size:
        a = np.pad(a, (0, b.size - a.size), constant_values=0)
    elif b.size < a.size:
        b = np.pad(b, (0, a.size - b.size), constant_values=0)
    s = ssim(a, b)

    # 4) Combine multiplicatively (both >0)
    return f * s

def route_packet(packet: dict,
                 reference_text: str,
                 center: float,
                 width: float,
                 seed_points: list[tuple[tuple[float, float], str]]) -> dict:
    """
    Route a packet to the most appropriate seed.

    Parameters
    ----------
    packet : dict
        Incoming packet containing a textual payload.
    reference_text : str
        Text used as SSIM reference.
    center, width : float
        Parameters of the Gaussian beam for Fisher scoring.
    seed_points : list of (point, label)
        Candidate routing destinations.

    Returns
    -------
    dict with keys ``selected_label`` and ``weight``.
    """
    weight = hybrid_weight(packet, reference_text, center, width)

    # Build multivector for the packet
    payload = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    pkt_point = text_to_point(payload)
    pkt_mv = multivector_from_point(pkt_point)

    # Compute weighted dot similarity with each seed
    best_idx = None
    best_score = -math.inf
    for idx, (seed_pt, label) in enumerate(seed_points):
        seed_mv = multivector_from_point(seed_pt)
        # Simple similarity: dot product scaled by weight
        sim = pkt_mv.dot(seed_mv) * weight
        if sim > best_score:
            best_score = sim
            best_idx = idx

    selected_label = seed_points[best_idx][1] if best_idx is not None else None
    return {"selected_label": selected_label, "weight": weight, "similarity": best_score}

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example packet
    packet_example = {
        "text_surface": "Hello, world!",
        "source": "sensor_A",
    }

    # Reference text for SSIM
    reference = "Hello, universe!"

    # Gaussian beam parameters
    beam_center = 0.0          # radians
    beam_width = 1.0           # radians

    # Define a few seed points with human‑readable labels
    seeds = [
        ((0.2, 0.3), "Node_A"),
        ((0.7, 0.1), "Node_B"),
        ((0.4, 0.8), "Node_C"),
    ]

    result = route_packet(packet_example, reference, beam_center, beam_width, seeds)
    print("Routing decision:", result)