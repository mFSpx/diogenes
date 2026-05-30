# DARWIN HAMMER — match 4998, survivor 0
# gen: 6
# parent_a: hybrid_xgboost_objective_hybrid_hybrid_hybrid_m2130_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_sketches_hybr_m2194_s2.py (gen5)
# born: 2026-05-29T23:59:23Z

"""
Hybrid XGBoost‑Krampus‑Ricci‑Sketch Algorithm
================================================

This module fuses the core mathematics of **Parent A** (XGBoost objective with
Ollivier‑Ricci curvature) and **Parent B** (epistemic certainty flags with a
Count‑Min sketch).  

The mathematical bridge is the *curvature‑weighted gradient/hessian* matrix.
In XGBoost the first‑order (g) and second‑order (h) statistics drive tree
splitting.  In the Krampus‑Ricci framework a curvature scalar κ(·) modulates
these statistics per feature.  We use κ as a multiplicative weight that is
then streamed into a Count‑Min sketch, which aggregates the weighted statistics
over many updates while respecting an epistemic confidence decay derived from
the `CertaintyFlag`.  The sketch therefore stores a compact, curvature‑aware
summary of the XGBoost learning signal.

The three primary hybrid functions are:

* ``curved_grad_hess`` – compute g, h and apply feature‑wise curvature.
* ``update_sketch`` – feed curvature‑weighted statistics into a Count‑Min
  sketch, scaling decay by epistemic confidence.
* ``predict_from_sketch`` – retrieve the aggregated statistics for a feature
  and produce a probability estimate via the logistic sigmoid.

All components are pure‑Python and depend only on the standard library and
NumPy.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
from typing import Tuple, Dict, List, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (XGBoost + Ollivier‑Ricci)
# ----------------------------------------------------------------------


def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    """Logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-margin))


def binary_logistic_grad_hess(y_true: np.ndarray, margin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classic XGBoost binary logistic gradient and Hessian.

    Returns
    -------
    g : np.ndarray
        Gradient vector (p - y).
    h : np.ndarray
        Hessian vector (p * (1-p)).
    """
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Stub feature extractor – in practice this would parse *text* and return
    a dense feature map.  Here we synthesize random values for demonstration.
    """
    features: Dict[str, float] = {}
    features.update({
        "operator_visceral_ratio": random.random(),
        "operator_tech_ratio": random.random(),
        "operator_legal_osint_ratio": random.random(),
        "psyche_forensic_shield_ratio": random.random(),
        "psyche_poetic_entropy": random.random(),
        "psyche_dissociative_index": random.random(),
        "resilience_bureaucratic_weaponization_index": random.random(),
        "resilience_resource_exhaustion_metric": random.random(),
        "resilience_swarm_orchestration_density": random.random(),
        "rainmaker_corporate_grit_tension": random.random(),
        "rainmaker_countdown_density": random.random(),
        "rainmaker_asset_structuring_weight": random.random(),
        "telemetry_agent_symmetry_ratio": random.random(),
        "telemetry_protocol_discipline": random.random(),
        "telemetry_manic_velocity": random.random(),
    })
    return features


def calculate_oric_curvature(features: Dict[str, float]) -> Dict[str, float]:
    """
    Very simplified Ollivier‑Ricci curvature estimator.
    Each feature receives a curvature κ ∈ [0, 0.2] proportional to its raw
    magnitude and a domain‑specific scaling factor.
    """
    oric: Dict[str, float] = {}
    for name, value in features.items():
        if "operator" in name:
            scale = 0.12
        elif "psyche" in name:
            scale = 0.08
        elif "resilience" in name:
            scale = 0.10
        elif "rainmaker" in name:
            scale = 0.07
        elif "telemetry" in name:
            scale = 0.09
        else:
            scale = 0.05
        # Clip curvature to a modest range.
        curvature = max(0.0, min(0.2, value * scale))
        oric[name] = curvature
    return oric


def curved_grad_hess(
    y_true: np.ndarray,
    margin: np.ndarray,
    feature_map: Dict[str, float],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute curvature‑weighted gradient and Hessian per feature.

    For each feature *f* we multiply the global gradient/hessian vectors by the
    scalar curvature κ_f, producing a feature‑specific statistic that can be
    aggregated downstream.

    Returns
    -------
    weighted_g : dict mapping feature name → weighted gradient array
    weighted_h : dict mapping feature name → weighted Hessian array
    """
    g, h = binary_logistic_grad_hess(y_true, margin)
    curvature = calculate_oric_curvature(feature_map)

    weighted_g: Dict[str, np.ndarray] = {}
    weighted_h: Dict[str, np.ndarray] = {}
    for fname, kappa in curvature.items():
        weighted_g[fname] = g * kappa
        weighted_h[fname] = h * kappa
    return weighted_g, weighted_h


# ----------------------------------------------------------------------
# Parent B utilities (Epistemic flags + Count‑Min Sketch)
# ----------------------------------------------------------------------


EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True, slots=True)
class CertaintyFlag:
    """
    Immutable container for an epistemic certainty flag.
    """
    label: str
    confidence_bps: int  # basis points, 0..10000 → 0%..100%
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class CountMinSketch:
    """
    Fixed‑size Count‑Min sketch with optional exponential decay.
    Decay is applied lazily on each ``add`` call.
    """

    def __init__(self, width: int = 2048, depth: int = 4, decay_rate: float = 0.0):
        if not (0.0 <= decay_rate < 1.0):
            raise ValueError("decay_rate must be in [0, 1)")
        self.width = width
        self.depth = depth
        self.decay_rate = decay_rate
        self.tables = np.zeros((depth, width), dtype=np.float64)
        self._seeds = [random.randint(0, 2**31 - 1) for _ in range(depth)]

    def _hashes(self, key: str) -> List[int]:
        """Return a list of column indices, one per row."""
        hashes = []
        key_bytes = key.encode("utf-8")
        for seed in self._seeds:
            h = hashlib.sha256(seed.to_bytes(4, "little") + key_bytes).digest()
            idx = int.from_bytes(h[:4], "little") % self.width
            hashes.append(idx)
        return hashes

    def add(self, key: str, value: float) -> None:
        """Add *value* to the sketch under *key*, applying decay first."""
        if self.decay_rate > 0.0:
            self.tables *= (1.0 - self.decay_rate)
        cols = self._hashes(key)
        for row, col in enumerate(cols):
            self.tables[row, col] += value

    def estimate(self, key: str) -> float:
        """Return the minimum count estimate for *key*."""
        cols = self._hashes(key)
        return min(self.tables[row, col] for row, col in enumerate(cols))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def update_sketch(
    sketch: CountMinSketch,
    weighted_g: Dict[str, np.ndarray],
    weighted_h: Dict[str, np.ndarray],
    flag: CertaintyFlag,
) -> None:
    """
    Stream curvature‑weighted gradient/Hessian statistics into the sketch.

    The epistemic confidence (basis points) modulates the decay rate:
        effective_decay = base_decay * (1 - confidence)

    For each feature we add the L1 norm of its weighted gradient and Hessian
    (a scalar) to the sketch under the feature name.
    """
    base_decay = sketch.decay_rate
    confidence = flag.confidence_bps / 10000.0  # 0..1
    effective_decay = base_decay * (1.0 - confidence)
    # Temporarily adjust decay for this batch.
    original_decay = sketch.decay_rate
    sketch.decay_rate = effective_decay

    for fname in weighted_g.keys():
        g_norm = float(np.linalg.norm(weighted_g[fname], ord=1))
        h_norm = float(np.linalg.norm(weighted_h[fname], ord=1))
        contribution = g_norm + h_norm
        sketch.add(fname, contribution)

    # Restore original decay.
    sketch.decay_rate = original_decay


def predict_from_sketch(sketch: CountMinSketch, feature_name: str, bias: float = 0.0) -> float:
    """
    Retrieve the aggregated statistic for *feature_name* and map it to a
    probability via the logistic sigmoid.  The bias term allows a global shift.
    """
    estimate = sketch.estimate(feature_name)
    margin = estimate + bias
    return float(sigmoid(margin))


def hybrid_step(
    y_true: np.ndarray,
    margin: np.ndarray,
    raw_text: str,
    sketch: CountMinSketch,
    flag: CertaintyFlag,
) -> Dict[str, float]:
    """
    Perform a full hybrid iteration:
      1. Extract features from *raw_text*.
      2. Compute curvature‑weighted gradient/Hessian.
      3. Update the sketch with epistemic decay.
      4. Return probability predictions for all extracted features.

    Returns a mapping feature → probability.
    """
    features = extract_full_features(raw_text)
    wg, wh = curved_grad_hess(y_true, margin, features)
    update_sketch(sketch, wg, wh, flag)

    predictions = {
        fname: predict_from_sketch(sketch, fname, bias=0.0) for fname in features.keys()
    }
    return predictions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic binary classification data
    n_samples = 128
    y = np.random.randint(0, 2, size=n_samples).astype(np.float64)
    margins = np.random.randn(n_samples)  # model outputs before sigmoid

    # Initialise hybrid components
    cms = CountMinSketch(width=1024, depth=5, decay_rate=0.02)
    flag = CertaintyFlag(
        label="PROBABLE",
        confidence_bps=7500,
        authority_class="research",
        rationale="synthetic test run",
        evidence_refs=("urn:example:test",),
    )

    # Run a few hybrid steps with dummy text payloads
    for step in range(3):
        txt = f"sample payload {step}"
        probs = hybrid_step(y, margins, txt, cms, flag)
        # Print a couple of probabilities to verify execution
        sample_feat = random.choice(list(probs.keys()))
        print(f"Step {step+1}: feature '{sample_feat}' → prob {probs[sample_feat]:.4f}")

    sys.exit(0)