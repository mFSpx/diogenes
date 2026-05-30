# DARWIN HAMMER — match 5305, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1673_s1.py (gen5)
# born: 2026-05-30T00:01:07Z

"""
Hybrid Morphology‑Fisher Cockpit (HMFC)

Parents:
- hybrid_hybrid_hard_truth_ma_rectified_flow_m91_s1.py (Morphology metrics, stylometry constraints)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1673_s1.py (Gaussian beam, Fisher information, reconstruction risk, cockpit honesty)

Mathematical bridge:
The sphericity (S) and flatness (F) indices of a Morphology are used as the *center* and *width* of a Gaussian
intensity ℓ(θ)=exp[-½((θ‑S)/F)²].  From this intensity we obtain the Fisher score I(θ)= (∂ℓ/∂θ)²/ℓ.
A stylometric *honesty* metric H (ratio of function‑word tokens to total tokens) modulates the pruning
probability P = H·I(θ).  The reconstruction risk R = unique_quasi_identifiers / total_records scales the
diffusion intensity that drives a right‑ing‑time correction Δt = P·R·volume⁻¹.  The final right‑ing time is

    t_right = base_time·exp(‑Δt).

Thus the two parent topologies are fused through shared scalar quantities (S,F,H,R) that
interlink morphology geometry with Fisher‑based information‑theoretic pruning.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A structures – Morphology and geometric indices
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity: (V)^{1/3} / length where V = l·w·h."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    volume = length * width * height
    return volume ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness: min(l,w,h) / max(l,w,h)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return min(length, width, height) / max(length, width, height)


def morphology_features(morph: Morphology) -> Dict[str, float]:
    """Collect geometric descriptors used downstream."""
    vol = morph.length * morph.width * morph.height
    spher = sphericity_index(morph.length, morph.width, morph.height)
    flat = flatness_index(morph.length, morph.width, morph.height)
    surface = 2 * (morph.length * morph.width + morph.length * morph.height + morph.width * morph.height)
    density = morph.mass / vol if vol > 0 else 0.0
    return {
        "volume": vol,
        "sphericity": spher,
        "flatness": flat,
        "surface_area": surface,
        "density": density,
    }


# ----------------------------------------------------------------------
# Parent‑B structures – Gaussian beam, Fisher score, reconstruction risk
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity ℓ(θ) with centre and width."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single‑parameter Gaussian model."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def reconstruction_risk(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk R = uqid / total, clipped to (0,1]."""
    if total_records <= 0:
        raise ValueError("total_records must be positive")
    return min(max(unique_quasi_identifiers / total_records, 1e-9), 1.0)


# ----------------------------------------------------------------------
# Hybrid components
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}


def honesty_metric(text: str) -> float:
    """
    Stylometric honesty H = (#function‑word tokens) / (total tokens).
    Returns a value in (0,1].
    """
    tokens = [t.lower().strip(".,!?;:") for t in text.split()]
    if not tokens:
        return 1.0
    func_count = sum(
        1 for tok in tokens if any(tok in cat for cat in FUNCTION_CATS.values())
    )
    return max(min(func_count / len(tokens), 1.0), 1e-9)


def pruning_probability(
    morph: Morphology,
    theta: float,
    unique_qi: int,
    total_records: int,
    text: str,
) -> float:
    """
    Core hybrid operation.
    1. Extract sphericity (S) and flatness (F) from the morphology.
    2. Use S as Gaussian centre, F as width → compute Fisher score I(θ).
    3. Compute honesty H from the supplied text.
    4. Compute reconstruction risk R.
    5. Return pruning probability P = H * I(θ) * R.
    """
    feats = morphology_features(morph)
    center = feats["sphericity"]
    width = max(feats["flatness"], 1e-6)  # avoid zero
    I = fisher_score(theta, center, width)
    H = honesty_metric(text)
    R = reconstruction_risk(unique_qi, total_records)
    return H * I * R


def corrected_righting_time(
    base_time: float,
    morph: Morphology,
    theta: float,
    unique_qi: int,
    total_records: int,
    text: str,
) -> float:
    """
    Adjust a baseline right‑ing time using the hybrid pruning probability.
    Δt = P * R * (1 / volume)
    t_right = base_time * exp(‑Δt)
    """
    if base_time <= 0:
        raise ValueError("base_time must be positive")
    feats = morphology_features(morph)
    vol = feats["volume"]
    if vol <= 0:
        raise ValueError("morphology volume must be positive")
    P = pruning_probability(morph, theta, unique_qi, total_records, text)
    R = reconstruction_risk(unique_qi, total_records)
    delta = P * R / vol
    return base_time * math.exp(-delta)


def hybrid_summary(
    morph: Morphology,
    theta: float,
    unique_qi: int,
    total_records: int,
    text: str,
) -> Dict[str, float]:
    """
    Demonstrates three distinct hybrid functions and aggregates their results.
    Returns a dictionary with:
        - pruning_probability
        - corrected_righting_time
        - honesty_metric
    """
    prob = pruning_probability(morph, theta, unique_qi, total_records, text)
    t_corr = corrected_righting_time(1.0, morph, theta, unique_qi, total_records, text)
    H = honesty_metric(text)
    return {
        "pruning_probability": prob,
        "corrected_righting_time": t_corr,
        "honesty_metric": H,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a representative morphology
    demo_morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Sample parameters
    theta_val = 0.8
    unique_qi = 42
    total_records = 1000
    sample_text = (
        "I think therefore I am, and the quick brown fox jumps over the lazy dog."
    )

    # Run hybrid functions
    prob = pruning_probability(
        demo_morph, theta_val, unique_qi, total_records, sample_text
    )
    t_right = corrected_righting_time(
        base_time=2.5,
        morph=demo_morph,
        theta=theta_val,
        unique_qi=unique_qi,
        total_records=total_records,
        text=sample_text,
    )
    summary = hybrid_summary(
        demo_morph, theta_val, unique_qi, total_records, sample_text
    )

    # Simple verification prints (no assertions required)
    print(f"Pruning probability: {prob:.6g}")
    print(f"Corrected righting time (base=2.5): {t_right:.6g}")
    print("Hybrid summary:", summary)

    sys.exit(0)