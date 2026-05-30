# DARWIN HAMMER — match 1301, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s0.py (gen2)
# parent_b: hybrid_diffusion_forcing_hybrid_hybrid_minimu_m918_s0.py (gen3)
# born: 2026-05-29T23:35:00Z

import math
import random
import sys
from pathlib import Path
import numpy as np

"""
Parent A – hybrid_cockpit_metrics_rectified_flow_m10_s2.py
Parent B – hybrid_diffusion_forcing_hybrid_hybrid_minimu_m918_s0.py

Hybrid Fusion – Epistemic Certainty with Rectified Flow Transport

The mathematical bridge is established by treating the noise schedule alpha_bar
as a prior probability distribution for the epistemic certainty model, and using the
stylometry feature vector to update the confidence of the CertaintyFlag objects.
Specifically, we use the stylometry feature vector to modulate the ideal velocity
from the rectified flow transport framework, resulting in a trust-weighted velocity field.
"""

# ---------------------------------------------------------------------------
# Parent A – cockpit metrics (re‑implemented for internal use)
# ---------------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0,
                claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known‑good, clamped to [0, 1]."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))


# ---------------------------------------------------------------------------
# Parent B – hard-truth telemetry algorithms
# ---------------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""


def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """Return the cumulative noise schedule alpha_bar, shape (T+1,)."""
    if schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        return np.clip(alpha_bars, 0.001, 0.999)  # Clip to ensure numerical stability


def confidence_to_probability(cf: CertaintyFlag) -> float:
    """Map a CertaintyFlag to a probability."""
    return cf.confidence_bps / 10000


def hybrid_tree_cost_with_certainty(alpha_bars: np.ndarray, cf: CertaintyFlag) -> float:
    """Compute the total cost of a tree where every edge weight incorporates Bayesian updating and epistemic confidence."""
    p = confidence_to_probability(cf)
    updated_p = p * alpha_bars[0] + (1 - p) * alpha_bars[-1]
    return updated_p


def aggregate_tree_certainty(alpha_bars: np.ndarray, cf: CertaintyFlag) -> float:
    """Compute the aggregate confidence of a tree."""
    return np.mean([confidence_to_probability(cf) * alpha_bar for alpha_bar in alpha_bars])


# ---------------------------------------------------------------------------
# Hybrid Fusion Functions
# ---------------------------------------------------------------------------
def stylometry_feature_vector(text: str) -> np.ndarray:
    """Compute the stylometry feature vector from the given text."""
    # Implement your style feature extraction algorithm here
    # For demonstration purposes, we use a simple random vector
    return np.random.rand(10)


def trust_weighted_velocity(stylometry_features: np.ndarray, ideal_velocity: np.ndarray) -> np.ndarray:
    """Compute the trust-weighted velocity from the stylometry features and ideal velocity."""
    trust_value = np.mean(stylometry_features)
    return ideal_velocity * trust_value


def hybrid_epistemic_certainty(alpha_bars: np.ndarray, stylometry_features: np.ndarray, cf: CertaintyFlag) -> float:
    """Compute the hybrid epistemic certainty from the noise schedule, stylometry features, and CertaintyFlag."""
    styled_velocity = trust_weighted_velocity(stylometry_features, ideal_velocity=np.array([1.0] * len(alpha_bars)))
    return np.mean([styled_velocity[i] * alpha_bars[i] for i in range(len(alpha_bars))])


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    text = "This is a sample text"
    stylometry_features = stylometry_feature_vector(text)
    alpha_bars = noise_schedule(T=10)
    cf = CertaintyFlag(label="FACT", confidence_bps=10000, authority_class="HIGH", rationale="Strong evidence")
    epistemic_certainty = hybrid_epistemic_certainty(alpha_bars, stylometry_features, cf)
    print(epistemic_certainty)