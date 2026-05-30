# DARWIN HAMMER — match 1792, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_korpus_text_h_gliner_zero_shot_ext_m118_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_caputo_serpentina_self_righ_m444_s1.py (gen4)
# born: 2026-05-29T23:38:51Z

"""Hybrid Text‑Caputo‑Geometric (HTCG) algorithm.

This module fuses the two parent algorithms:

* **Parent A** – `hybrid_hybrid_korpus_text_h_gliner_zero_shot_ext_m118_s0.py`  
  Provides a *minhash* signature for a text string and the `Span` dataclass that
  encapsulates a textual segment.

* **Parent B** – `hybrid_hybrid_hybrid_caputo_serpentina_self_righ_m444_s1.py`  
  Supplies a *Caputo fractional‑derivative* kernel and a *geometric rotor* that
  rotates a four‑dimensional morphology vector.

**Mathematical bridge**

1. The minhash signature (a discrete integer sequence) and the auxiliary
   `vector_literal` representation are concatenated to form a numeric *signal*
   `x(t)` derived from the input text.

2. The Caputo fractional‑derivative `D^α x(t)` (order `0<α<1`) is applied to this
   signal, producing a weight vector that captures long‑range memory of the
   textual pattern.

3. The aggregated magnitude of the derivative weights determines a rotation
   angle `θ`.  This angle parametrises a geometric‑algebra rotor `R(θ)=exp(θ B)`
   that rotates a morphology vector `v = (length, width, entropy·100,
   mean(signature))` in the `(length, width)` plane.

4. The rotated morphology is attached to a `Span` instance, yielding a single
   unified object that carries both the linguistic representation and the
   history‑dependent geometric transformation.

The three core functions below demonstrate the hybrid operation:
`text_to_signal`, `caputo_weighted_angle`, and `rotate_morphology_span`."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
import re
import numpy as np

# ----------------------------------------------------------------------
# Parent‑A components (minhash, entropy, Span)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """Immutable container for a textual span together with auxiliary data."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str
    metadata: dict  # additional hybrid information


def minhash_for_text(text: str, k: int = 64) -> list[int]:
    """Return a min‑hash signature of length *k* for *text*."""
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    if len(text) < 5:
        # degenerate case – return a zero‑filled signature
        return [0] * k
    shingles = [text[i:i + 5] for i in range(len(text) - 4)]
    signature = np.random.randint(0, 1000000, size=k)
    for s in shingles:
        h = hash(s) % k
        signature[h] = min(signature[h], hash(s) % 1000000)
    return signature.tolist()


def entropy_for_text(text: str) -> float:
    """Shannon‑like entropy approximated by the ratio of distinct characters."""
    text = text or ""
    text = text[:10000]  # limit to avoid pathological lengths
    return float(len(set(text))) / len(text) if text else 0.0


def vector_literal(text: str) -> np.ndarray:
    """Deterministic 16‑dimensional float vector derived from hashed substrings."""
    hashes = np.array([hash(text + str(i)) for i in range(16)], dtype=np.float64)
    # Normalise to the interval (0,1)
    return hashes / float(2 ** 31 - 1)


def generate_span(text: str, label: str, score: float, metadata: dict) -> Span:
    """Create a Span covering the whole *text* and embed *metadata*."""
    return Span(start=0, end=len(text), text=text, label=label,
                score=score, backend="HTCG", metadata=metadata)


# ----------------------------------------------------------------------
# Parent‑B components (Caputo derivative, geometric rotor)
# ----------------------------------------------------------------------
def lanczos_gamma(z: float) -> float:
    """Lanczos approximation of the Gamma function for positive *z*."""
    if z < 0.5:
        # Reflection formula for stability
        return math.pi / (math.sin(math.pi * z) * lanczos_gamma(1 - z))
    g = 7
    z += g + 0.5
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    a = p[0]
    for i in range(1, len(p)):
        a += p[i] / (z - i)
    t = z + 0.5
    return math.sqrt(2 * math.pi) * (t ** (z - 0.5)) * math.exp(-t) * a


def caputo_derivative(signal: np.ndarray, alpha: float, dt: float = 1.0) -> np.ndarray:
    """
    Discrete Grünwald‑Letnikov approximation of the Caputo fractional derivative
    of order *alpha* (0 < α < 1) for a 1‑D signal sampled with uniform spacing *dt*.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0,1)")
    n = signal.shape[0]
    deriv = np.zeros_like(signal, dtype=float)
    gamma_alpha1 = math.gamma(alpha + 1)

    # Pre‑compute binomial‑type coefficients C(α,k)
    coeffs = np.empty(n, dtype=float)
    for k in range(n):
        coeffs[k] = ((-1) ** k) * gamma_alpha1 / (
            math.gamma(k + 1) * math.gamma(alpha - k + 1)
        )

    # Convolution‑like accumulation
    for i in range(n):
        deriv[i] = np.sum(coeffs[: i + 1] * signal[i::-1])
    deriv /= dt ** alpha
    return deriv


def geometric_rotor(v: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate a 4‑D morphology vector *v* in the (0,1) plane by angle *theta*.
    The rotation matrix is:
        [[cosθ, -sinθ, 0, 0],
         [sinθ,  cosθ, 0, 0],
         [   0,     0, 1, 0],
         [   0,     0, 0, 1]]
    """
    c, s = math.cos(theta), math.sin(theta)
    R = np.array(
        [[c, -s, 0, 0],
         [s,  c, 0, 0],
         [0,  0, 1, 0],
         [0,  0, 0, 1]],
        dtype=float,
    )
    return R @ v


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def text_to_signal(text: str) -> np.ndarray:
    """
    Convert *text* into a numeric signal by concatenating the minhash signature
    and the 16‑dimensional literal vector.
    """
    mh = np.array(minhash_for_text(text), dtype=float)
    vl = vector_literal(text)
    return np.concatenate([mh, vl])


def caputo_weighted_angle(signal: np.ndarray, alpha: float) -> float:
    """
    Compute a rotation angle from the Caputo derivative of *signal*.
    The angle is proportional to the L1‑norm of the derivative and wrapped to
    the interval [0, 2π).
    """
    deriv = caputo_derivative(signal, alpha)
    magnitude = np.sum(np.abs(deriv))
    # Scale to a full circle; the scaling factor is arbitrary but keeps θ bounded.
    theta = (magnitude % (2 * math.pi))
    return theta


def rotate_morphology_span(text: str, label: str, score: float,
                          alpha: float = 0.4) -> Span:
    """
    Full hybrid pipeline:

    1. Build a numeric signal from the text.
    2. Derive a Caputo‑weighted rotation angle.
    3. Assemble a morphology vector using length, width (entropy‑scaled) and
       statistics of the signal.
    4. Rotate the morphology vector with the geometric rotor.
    5. Pack everything into a :class:`Span` with rich metadata.
    """
    # --- Step 1: signal -------------------------------------------------
    signal = text_to_signal(text)

    # --- Step 2: fractional‑derivative → angle -------------------------
    theta = caputo_weighted_angle(signal, alpha)

    # --- Step 3: morphology vector --------------------------------------
    length = float(len(text))
    width = entropy_for_text(text) * 100.0          # scale for numeric stability
    height = np.mean(signal) * 1e-3                 # modest scaling
    mass = np.std(signal) * 1e-3
    v = np.array([length, width, height, mass], dtype=float)

    # --- Step 4: rotate -------------------------------------------------
    v_rot = geometric_rotor(v, theta)

    # --- Step 5: create Span with metadata -----------------------------
    metadata = {
        "minhash_signature": minhash_for_text(text),
        "vector_literal": vector_literal(text).tolist(),
        "caputo_alpha": alpha,
        "rotation_angle_rad": theta,
        "original_morphology": v.tolist(),
        "rotated_morphology": v_rot.tolist(),
    }
    return generate_span(text, label, score, metadata)


def hybrid_operation(text: str, label: str = "generic", score: float = 1.0,
                     alpha: float = 0.4) -> Span:
    """
    Convenience wrapper that mirrors the original ``hybrid_operation`` from
    Parent A while embedding the full HTCG pipeline.
    """
    return rotate_morphology_span(text, label, score, alpha)


def extract_hybrid_vector(text: str) -> dict:
    """
    Produce a flat dictionary summarising the hybrid numeric representation.
    This mirrors the incomplete ``extract_master_vector`` from Parent A but
    now incorporates the Caputo‑derived angle.
    """
    signal = text_to_signal(text)
    theta = caputo_weighted_angle(signal, alpha=0.4)
    return {
        "signal_mean": float(np.mean(signal)),
        "signal_std": float(np.std(signal)),
        "caputo_angle": theta,
        "entropy": entropy_for_text(text),
        "length": len(text),
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "In the quiet dawn, the river whispered secrets of the mountains, "
        "carrying them downstream where they dissolved into mist."
    )
    span = hybrid_operation(sample, label="nature", score=0.92, alpha=0.35)
    print("Hybrid Span:")
    print(asdict(span))
    vec = extract_hybrid_vector(sample)
    print("\nHybrid vector summary:")
    for k, v in vec.items():
        print(f"  {k}: {v}")