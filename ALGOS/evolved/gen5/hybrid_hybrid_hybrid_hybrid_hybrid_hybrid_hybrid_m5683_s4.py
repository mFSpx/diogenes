# DARWIN HAMMER — match 5683, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_regret_engine_m1650_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hard_t_m1107_s1.py (gen4)
# born: 2026-05-30T00:04:11Z

"""Hybrid Regret‑Curvature Engine

This module fuses the two parent algorithms:

* **Parent A** – epistemic certainty flags and a regret‑engine that produces a
  weight‑scaled Gini coefficient.
* **Parent B** – extraction of a high‑dimensional feature map and computation of
  an Ollivier‑Ricci curvature matrix that is updated by a Bayesian rule.

The mathematical bridge is the **hybrid score** `S` produced by Parent A:

S = w * Gini(values) * (1 – H/ log2|A|)

where `w` is the epistemic weight from a `CertaintyFlag`,
`Gini` is the Gini coefficient of the regret‑weighted action values and `H`
is the Shannon entropy of the normalised action distribution.
`S` is a scalar that serves as Bayesian evidence for the curvature update in
Parent B.  The curvature entries are therefore modulated by epistemic certainty
and by the inequality of the regret‑engine, yielding a single unified system."""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Iterable, Tuple, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – epistemic certainty flag and regret‑engine utilities
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # 0 .. 10000
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class Action:
    """Simple regret‑engine action."""
    name: str
    expected_value: float
    cost: float
    risk: float  # 0..1 probability of adverse outcome

    def regret_weighted_value(self) -> float:
        """Regret‑weighted scalar used for the Gini computation."""
        # Higher cost and risk increase regret; we subtract a penalty.
        penalty = self.cost * (1 + self.risk)
        return self.expected_value - penalty


def gini_coefficient(x: np.ndarray) -> float:
    """Return the Gini coefficient of a 1‑D array."""
    if x.size == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = x.size
    cumulative = np.cumsum(sorted_x, dtype=float)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy (base‑2) of a probability vector."""
    p = p[p > 0]
    return -float(np.sum(p * np.log2(p)))


def epistemic_weight(flag: CertaintyFlag) -> float:
    """Convert confidence basis points to a scalar weight w ∈ [0,1]."""
    return flag.confidence_bps / 10000.0


def hybrid_score(flag: CertaintyFlag, actions: List[Action]) -> float:
    """
    Compute the hybrid scalar S = w * Gini * (1 - H / log2|A|).

    - w : epistemic weight from the certainty flag
    - Gini : Gini coefficient of regret‑weighted action values
    - H : Shannon entropy of the normalised regret‑weighted values
    - |A| : number of actions
    """
    if not actions:
        return 0.0
    w = epistemic_weight(flag)
    values = np.array([a.regret_weighted_value() for a in actions], dtype=float)
    # Ensure non‑negative for Gini; shift if necessary.
    if np.any(values < 0):
        values = values - values.min() + 1e-9
    gini = gini_coefficient(values)
    probs = values / values.sum() if values.sum() != 0 else np.full_like(values, 1 / len(values))
    H = shannon_entropy(probs)
    max_entropy = math.log2(len(actions))
    entropy_factor = 1.0 - (H / max_entropy if max_entropy > 0 else 0.0)
    return w * gini * entropy_factor


# ----------------------------------------------------------------------
# Parent B – feature extraction, Ollivier‑Ricci curvature and Bayesian update
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Mock feature extraction – returns a dict of pseudo‑random floats."""
    rnd = random.Random(hash(text))
    features: Dict[str, float] = {}
    for name in (
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight",
        "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline",
        "telemetry_manic_velocity",
    ):
        features[name] = rnd.random()
    return features


def calculate_oric_curvature(features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute a toy Ollivier‑Ricci curvature for each feature.

    Operator‑related features receive a scaling of 0.1, all others keep the
    original magnitude.  This mirrors the original parent implementation.
    """
    curvature: Dict[str, float] = {}
    for key, val in features.items():
        if "operator" in key:
            curvature[key] = val * 0.1
        else:
            curvature[key] = val
    return curvature


def bayesian_update_curvature(curv: Dict[str, float], evidence: float) -> Dict[str, float]:
    """
    Perform a simple scalar Bayesian update on each curvature entry.

    Prior: N(curv_i, σ²_prior) with σ²_prior = 1.
    Likelihood: N(evidence, σ²_like) with σ²_like = 1.
    Posterior mean: (curv_i + evidence) / 2.
    """
    updated: Dict[str, float] = {}
    for key, val in curv.items():
        posterior = (val + evidence) / 2.0
        updated[key] = posterior
    return updated


# ----------------------------------------------------------------------
# Hybrid operations – three public functions
# ----------------------------------------------------------------------
def compute_hybrid_regret_curvature(flag: CertaintyFlag, actions: List[Action]) -> Dict[str, float]:
    """
    1️⃣ Compute the hybrid scalar S from the certainty flag and actions.
    2️⃣ Extract features from a dummy text (the flag label is used as seed).
    3️⃣ Compute curvature and update it with S as Bayesian evidence.
    Returns the updated curvature dictionary.
    """
    s = hybrid_score(flag, actions)
    features = extract_full_features(flag.label)
    curvature = calculate_oric_curvature(features)
    updated_curvature = bayesian_update_curvature(curvature, s)
    return updated_curvature


def evaluate_hybrid_system(flag: CertaintyFlag, actions: List[Action]) -> Tuple[float, float]:
    """
    Returns a tuple (S, C_norm) where:
    * S – hybrid scalar from the regret‑engine.
    * C_norm – Euclidean norm of the curvature vector after Bayesian update.
    """
    updated_curvature = compute_hybrid_regret_curvature(flag, actions)
    vec = np.fromiter(updated_curvature.values(), dtype=float)
    c_norm = float(np.linalg.norm(vec))
    s = hybrid_score(flag, actions)
    return s, c_norm


def summarize_hybrid(flag: CertaintyFlag, actions: List[Action]) -> Dict[str, Any]:
    """
    Produce a human‑readable summary containing:
    * original certainty flag fields,
    * hybrid scalar,
    * curvature norm,
    * top‑3 curvature entries (by absolute magnitude).
    """
    s, c_norm = evaluate_hybrid_system(flag, actions)
    updated_curvature = compute_hybrid_regret_curvature(flag, actions)
    top3 = sorted(updated_curvature.items(), key=lambda kv: -abs(kv[1]))[:3]
    summary: Dict[str, Any] = {
        "certainty_flag": asdict(flag),
        "hybrid_score": s,
        "curvature_norm": c_norm,
        "top_curvature_entries": {k: v for k, v in top3},
    }
    return summary


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a certainty flag with moderate confidence.
    flag = CertaintyFlag(
        label="PROBABLE",
        confidence_bps=7300,
        authority_class="research",
        rationale="preliminary statistical evidence",
        evidence_refs=("ref1", "ref2"),
    )

    # Define a small set of actions.
    actions = [
        Action(name="A1", expected_value=120.0, cost=30.0, risk=0.2),
        Action(name="A2", expected_value=80.0, cost=20.0, risk=0.1),
        Action(name="A3", expected_value=50.0, cost=10.0, risk=0.05),
    ]

    # Run the hybrid pipeline and print a concise summary.
    summary = summarize_hybrid(flag, actions)
    print("Hybrid Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")