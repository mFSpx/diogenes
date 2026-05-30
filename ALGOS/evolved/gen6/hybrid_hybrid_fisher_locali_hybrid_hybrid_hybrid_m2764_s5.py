# DARWIN HAMMER — match 2764, survivor 5
# gen: 6
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1892_s0.py (gen5)
# born: 2026-05-29T23:45:45Z

"""Hybrid Fisher‑Geometric Similarity Router

This module fuses two parent algorithms:

* **Parent A** – Fisher‑localization and SSIM‑based routing.  It provides
  `gaussian_beam`, `fisher_score` and `ssim` to quantify the information
  content of a packet’s “text surface” and its structural similarity to a
  reference sample.

* **Parent B** – Geometric‑algebra utilities (`Multivector`, `distance`,
  `nearest`).  It supplies a compact algebraic representation for scalar
  and vector‑grade quantities and geometric operations such as the dot
  product.

**Mathematical bridge** – The Fisher score (a scalar) is embedded as the
scalar part of a multivector together with first‑order (mean) and second‑
order (variance) statistical moments of the packet’s text histogram.  The
resulting multivector lives in a 3‑dimensional geometric algebra and can
be compared to a reference multivector by the algebraic dot product.
The dot product yields a *geometric similarity* which is then fused with
the SSIM value to obtain a unified routing metric.  The metric is finally
used together with Euclidean distances to a set of seed coordinates to
select the optimal routing destination.

The implementation below provides three public hybrid functions and a
smoke‑test that runs without external data.
"""

import math
import random
import sys
import pathlib
import numpy as np

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural similarity index (grayscale, 1‑D)."""
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x.size:
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
# Parent B components – a minimal geometric algebra implementation
# ----------------------------------------------------------------------
class Multivector:
    """Simple multivector for a 3‑dimensional Euclidean algebra.

    Blades are represented by frozensets of basis indices, e.g.
    scalar → frozenset(), e1 → frozenset({1}), e2 → frozenset({2}), etc.
    """
    def __init__(self, components: dict, n: int = 3):
        self.n = int(n)
        # prune near‑zero coefficients
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    def grade(self, k: int) -> "Multivector":
        """Return the grade‑k part of the multivector."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Scalar (grade‑0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), tuple(x[0]))):
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
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) - coef
        return Multivector(result, self.n)

    def dot(self, other: "Multivector") -> float:
        """Geometric dot product restricted to scalar result."""
        # The dot product of two multivectors is the sum of products of
        # matching blades that collapse to grade‑0.
        total = 0.0
        for blade, coef in self.components.items():
            # Only blades that are identical contribute to the scalar part.
            if blade in other.components:
                total += coef * other.components[blade]
        return total

def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance in 2‑D."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    """Index of the seed closest to *point*."""
    if not seeds:
        raise ValueError("seed list is empty")
    dists = [distance(point, s) for s in seeds]
    return int(np.argmin(dists))

# ----------------------------------------------------------------------
# Hybrid constructions
# ----------------------------------------------------------------------
def encode_packet_multivector(packet: dict,
                              center: float,
                              width: float) -> Multivector:
    """Encode a packet into a multivector.

    Grade‑0 : Fisher information of the packet’s angle (theta).
    Grade‑1 : Mean intensity of the text histogram (e1) and variance (e2).
    """
    # Extract a numeric “theta” – for demo we map the first character code.
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    theta = float(ord(text[0]) if text else 0.0)

    # Fisher scalar
    f = fisher_score(theta, center, width)

    # Histogram of character codes as a proxy for intensity.
    codes = np.fromiter((ord(ch) for ch in text), dtype=np.uint8, count=len(text))
    if codes.size == 0:
        mean_intensity = 0.0
        variance = 0.0
    else:
        mean_intensity = float(np.mean(codes))
        variance = float(np.var(codes))

    components = {
        frozenset(): f,                 # scalar (grade‑0)
        frozenset({1}): mean_intensity, # e1
        frozenset({2}): variance,       # e2
    }
    return Multivector(components, n=3)

def reference_multivector(reference_text: str,
                          center: float,
                          width: float) -> Multivector:
    """Create a reference multivector from a plain string."""
    # Use the same encoding strategy as for packets.
    dummy_packet = {"text_surface": reference_text}
    return encode_packet_multivector(dummy_packet, center, width)

def hybrid_metric(packet_mv: Multivector,
                  reference_mv: Multivector,
                  ssim_score: float,
                  alpha: float = 0.6) -> float:
    """Combine geometric (dot) similarity with SSIM.

    The metric is a convex combination:
        M = α * (dot / (|p|·|r| + ε)) + (1‑α) * SSIM
    where the dot product is normalised by the product of Euclidean norms
    of the grade‑1 parts to keep the scale comparable.
    """
    eps = 1e-12
    # Geometric part – only grade‑1 components are used for direction.
    p_vec = np.array([packet_mv.components.get(frozenset({1}), 0.0),
                      packet_mv.components.get(frozenset({2}), 0.0)])
    r_vec = np.array([reference_mv.components.get(frozenset({1}), 0.0),
                      reference_mv.components.get(frozenset({2}), 0.0)])

    norm_p = np.linalg.norm(p_vec) + eps
    norm_r = np.linalg.norm(r_vec) + eps
    dot_norm = packet_mv.dot(reference_mv) / (norm_p * norm_r)

    return alpha * dot_norm + (1.0 - alpha) * ssim_score

def similarity_based_hybrid_routing(packet: dict,
                                    reference_text: str,
                                    center: float,
                                    width: float,
                                    seeds: list[tuple[float, float]]) -> dict:
    """Route a packet to the nearest seed according to the hybrid metric.

    Returns a dictionary with routing information.
    """
    # 1️⃣ Encode packet and reference as multivectors.
    pkt_mv = encode_packet_multivector(packet, center, width)
    ref_mv = reference_multivector(reference_text, center, width)

    # 2️⃣ Compute SSIM between the raw byte arrays of packet and reference.
    pkt_bytes = np.frombuffer(str(packet).encode('utf-8'), dtype=np.uint8)
    ref_bytes = np.frombuffer(reference_text.encode('utf-8'), dtype=np.uint8)
    # Truncate to the shorter length for a fair comparison.
    min_len = min(pkt_bytes.size, ref_bytes.size)
    ssim_val = ssim(pkt_bytes[:min_len].astype(float), ref_bytes[:min_len].astype(float))

    # 3️⃣ Hybrid similarity score.
    metric = hybrid_metric(pkt_mv, ref_mv, ssim_val)

    # 4️⃣ Choose seed: we treat the metric as a pseudo‑distance (higher is better),
    #    so we invert it and add the Euclidean distance to the seed.
    best_idx = None
    best_score = -math.inf
    for idx, seed in enumerate(seeds):
        # Combine metric with spatial proximity.
        spatial = -distance((center, width), seed)  # negative because closer is better
        combined = metric + 0.1 * spatial
        if combined > best_score:
            best_score = combined
            best_idx = idx

    selected_seed = seeds[best_idx] if best_idx is not None else None
    return {
        "selected_seed_index": best_idx,
        "selected_seed": selected_seed,
        "hybrid_metric": metric,
        "ssim": ssim_val,
        "packet_mv": pkt_mv,
        "reference_mv": ref_mv,
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a dummy packet.
    packet_example = {
        "text_surface": "Hello, world!",
        "source": "sensor_A",
        "raw_command": None,
        "intent": "greeting",
    }

    # Reference text to compare against.
    reference = "Hello, universe!"

    # Parameters for the Fisher Gaussian.
    center_angle = 50.0
    beam_width = 10.0

    # Seed coordinates (arbitrary 2‑D points).
    seed_points = [(45.0, 9.0), (60.0, 12.0), (30.0, 5.0)]

    routing = similarity_based_hybrid_routing(
        packet=packet_example,
        reference_text=reference,
        center=center_angle,
        width=beam_width,
        seeds=seed_points,
    )

    print("Routing result:")
    for k, v in routing.items():
        if k.endswith("_mv"):
            # Avoid dumping large multivector internals unless needed.
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v}")