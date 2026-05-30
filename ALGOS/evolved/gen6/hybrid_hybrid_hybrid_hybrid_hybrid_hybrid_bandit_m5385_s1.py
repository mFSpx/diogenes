# DARWIN HAMMER — match 5385, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m1911_s2.py (gen5)
# parent_b: hybrid_hybrid_bandit_router_hybrid_hybrid_bayes__m2202_s0.py (gen4)
# born: 2026-05-30T00:01:32Z

"""Hybrid algorithm combining:
- Parent A: reconstruction risk scoring, causal effect structures, geometric utilities.
- Parent B: temperature‑dependent activity gate, Bayesian update, bandit action selection, deterministic feature extraction.

Mathematical bridge:
The temperature activity `A(T)` from the Schoolfield model is used as a scaling factor
for both (i) the reconstruction‑risk probability `R` from Parent A and (ii) the
likelihood term in the Bayesian update of a feature‑weight vector.  The product
`A(T)·(1‑R)` therefore modulates exploration‑exploitation in the bandit router
while simultaneously tempering the posterior over causal‑effect‑derived features.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Type aliases (shared)
# ----------------------------------------------------------------------
Point = Tuple[float, float]          # 2‑D coordinates of a node
Edge = Tuple[str, str]               # connection between node identifiers
Morphology = Tuple[float, float, float]  # (length, width, height)


# ----------------------------------------------------------------------
# Parent A structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


@dataclass(frozen=True)
class CausalEffect:
    """Container for a causal effect estimate."""
    effect_id: str
    treatment: str
    outcome: str
    confounders: Tuple[str, ...]
    ate_estimate: float | None
    ate_confidence_interval: Tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: Tuple[str, ...]
    heterogeneous_effects: Dict[str, float]


def euclidean_length(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def reconstruction_risk_score(
    unique_quasi_identifiers: int,
    total_records: int,
    laplace_smoothing: float = 1.0,
) -> float:
    """
    Probability that a record can be re‑identified.
    """
    if total_records <= 0:
        return 0.0
    numer = unique_quasi_identifiers + laplace_smoothing
    denom = total_records + 2 * laplace_smoothing
    return max(0.0, min(1.0, numer / denom))


# ----------------------------------------------------------------------
# Parent B structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # prior probability of being chosen
    expected_reward: float     # modelled mean reward


def temperature_activity(celsius: float) -> float:
    """
    Normalized activity gate from Celsius using a Gaussian centered at 25 °C.
    """
    T_ref = 25.0
    T = celsius + T_ref
    sigma = 5.0
    A = np.exp(-((T - T_ref) ** 2) / (2 * sigma ** 2))
    return float(A)


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def hybrid_feature_vector(
    record: Dict[str, Any],
    causal_effects: List[CausalEffect],
) -> Dict[str, float]:
    """
    Deterministic extraction of a master feature vector.

    The vector combines:
    * Simple statistics derived from the record (e.g. normalized numeric fields).
    * Aggregated heterogeneous ATE estimates from supplied CausalEffect objects.
    * A geometric proxy derived from any supplied morphology tuple.
    """
    vec: Dict[str, float] = {}

    # 1. Record‑level numeric normalization (0‑1 scaling per key)
    numeric_keys = [k for k, v in record.items() if isinstance(v, (int, float))]
    for k in numeric_keys:
        val = float(record[k])
        # naive min‑max using a fixed plausible range [0, 1000]
        vec[f"rec_{k}"] = max(0.0, min(1.0, val / 1000.0))

    # 2. Aggregate heterogeneous effects (mean ATE per confounder)
    agg: Dict[str, List[float]] = {}
    for ce in causal_effects:
        for conf, ate in ce.heterogeneous_effects.items():
            agg.setdefault(conf, []).append(ate)
    for conf, ates in agg.items():
        vec[f"ate_{conf}"] = sum(ates) / len(ates)

    # 3. Optional morphology handling
    morph: Morphology | None = record.get("morphology")
    if morph:
        length, width, height = morph
        vec["morph_volume"] = length * width * height

    return vec


def hybrid_bayes_update(
    prior: Dict[str, float],
    reward: float,
    temperature_celsius: float,
) -> Dict[str, float]:
    """
    Temperature‑aware Bayesian update of a feature weight vector.

    For each feature `i` we treat `prior[i]` as a prior probability.
    The likelihood is modelled as a Bernoulli trial whose success probability
    is scaled by the temperature activity `A(T)`.  The posterior is then:

        posterior[i] ∝ prior[i] * (A(T) * reward + (1‑A(T)) * (1‑reward))

    The result is renormalized to sum to 1.
    """
    A = temperature_activity(temperature_celsius)
    likelihood = A * reward + (1.0 - A) * (1.0 - reward)

    unnorm: Dict[str, float] = {}
    for k, p in prior.items():
        unnorm[k] = max(0.0, p * likelihood)

    total = sum(unnorm.values())
    if total == 0.0:
        # fallback to uniform distribution over existing keys
        n = len(unnorm)
        return {k: 1.0 / n for k in unnorm}
    return {k: v / total for k, v in unnorm.items()}


def hybrid_select_action(
    actions: List[BanditAction],
    master_vector: Dict[str, float],
    record: Dict[str, Any],
    temperature_celsius: float,
) -> BanditAction:
    """
    Temperature‑aware bandit action selection.

    Each action receives a composite score:

        score = propensity * expected_reward *
                A(T) * (1 - R) * Σ_i w_i·f_i

    where:
    * `A(T)` is the temperature activity,
    * `R` is the reconstruction risk score derived from the record,
    * `w_i` are the weights from `master_vector`,
    * `f_i` are the corresponding feature values extracted from the record
      (mirroring `hybrid_feature_vector` but using the raw values).

    The action with the highest score is returned.
    """
    A = temperature_activity(temperature_celsius)

    # reconstruction risk based on quasi‑identifier counts in the record
    uq = record.get("unique_quasi_identifiers", 0)
    total = record.get("total_records", 1)
    R = reconstruction_risk_score(int(uq), int(total))

    # compute dot‑product between master_vector and record features
    # (use the same deterministic extraction as in `hybrid_feature_vector`)
    feature_vec = hybrid_feature_vector(record, [])
    dot = sum(master_vector.get(k, 0.0) * v for k, v in feature_vec.items())

    best: BanditAction | None = None
    best_score = -math.inf
    for act in actions:
        score = (
            act.propensity
            * act.expected_reward
            * A
            * (1.0 - R)
            * (dot if dot > 0 else 1.0)   # avoid zeroing out when vector is empty
        )
        if score > best_score:
            best_score = score
            best = act
    assert best is not None, "No actions provided"
    return best


def combined_risk_temperature_metric(
    record: Dict[str, Any],
    temperature_celsius: float,
) -> float:
    """
    A simple fused metric that multiplies the reconstruction risk score
    with the temperature activity gate, yielding a value in [0, 1].
    """
    uq = record.get("unique_quasi_identifiers", 0)
    total = record.get("total_records", 1)
    R = reconstruction_risk_score(int(uq), int(total))
    A = temperature_activity(temperature_celsius)
    return R * A


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic record
    record_example = {
        "age": 34,
        "income": 72000,
        "unique_quasi_identifiers": 150,
        "total_records": 10000,
        "morphology": (2.1, 0.8, 1.5),
    }

    # dummy causal effects
    ce1 = CausalEffect(
        effect_id="ce1",
        treatment="drug",
        outcome="recovery",
        confounders=("age", "sex"),
        ate_estimate=0.12,
        ate_confidence_interval=(0.05, 0.19),
        refutation_passed=True,
        refutation_methods=("placebo",),
        heterogeneous_effects={"age": 0.10, "sex": 0.15},
    )
    ce2 = CausalEffect(
        effect_id="ce2",
        treatment="exercise",
        outcome="weight_loss",
        confounders=("diet",),
        ate_estimate=-0.08,
        ate_confidence_interval=(-0.12, -0.04),
        refutation_passed=False,
        refutation_methods=("random",),
        heterogeneous_effects={"diet": -0.07},
    )
    causal_list = [ce1, ce2]

    # build master vector
    master_vec = hybrid_feature_vector(record_example, causal_list)

    # initial uniform prior for Bayesian update demonstration
    prior = {k: 1.0 / len(master_vec) for k in master_vec}
    posterior = hybrid_bayes_update(prior, reward=1.0, temperature_celsius=22.0)

    # define a few bandit actions
    actions = [
        BanditAction(action_id="a1", propensity=0.4, expected_reward=0.6),
        BanditAction(action_id="a2", propensity=0.3, expected_reward=0.8),
        BanditAction(action_id="a3", propensity=0.3, expected_reward=0.5),
    ]

    chosen = hybrid_select_action(
        actions,
        master_vector=posterior,
        record=record_example,
        temperature_celsius=22.0,
    )

    print("Master vector (first 5):", dict(list(master_vec.items())[:5]))
    print("Posterior (first 5):", dict(list(posterior.items())[:5]))
    print("Chosen action:", chosen)
    print(
        "Combined risk‑temperature metric:",
        combined_risk_temperature_metric(record_example, 22.0),
    )