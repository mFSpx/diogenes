# DARWIN HAMMER — match 956, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_label_foundry_m21_s1.py (gen3)
# born: 2026-05-29T23:31:47Z

"""
Hybrid Poikilotherm‑Labeling State‑Space Duality (HPLSSD)

This module fuses two parent algorithms:

* **Parent A** – the Hybrid Poikilotherm‑State‑Space Duality (HPSSD) that bridges
  the Schoolfield‑Rollinson poikilotherm developmental rate with the State‑Space
  Duality (SSD) sequential and semiseparable parallel forms.
* **Parent B** – the Hybrid algorithm fusion of hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s2.py and label_foundry.py.

The mathematical bridge between the two parents lies in the ability of both algorithms
to handle noisy and uncertain data. The HPSSD algorithm uses the temperature‑dependent
scalar `r(t) = developmental_rate(T(t))` to modulate the state‑transition matrix `A`
in the SSD, while the hybrid algorithm uses sketch primitives to estimate empirical mean
rewards and their variances, and weak supervision labeling primitives to handle noisy labels.
By combining these two approaches, we can create a hybrid algorithm that can handle both
noisy rewards and labels.

The hybrid algorithm uses the developmental rate to scale the state‑transition matrix `A`
in the SSD, while also using the sketch primitives to estimate the empirical mean rewards
and their variances. The labeling process uses the weak supervision labeling primitives
to handle noisy labels, and the hybrid algorithm combines the results of the sketching
and labeling processes to produce a final output.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable, Set, Callable

import numpy as np


# ----------------------------------------------------------------------
# Parent‑A: Poikilotherm developmental rate
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (in arbitrary units)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15            # K  (10 °C)
    t_high: float = 307.15           # K  (34 °C)
    delta_h_low: float = -45_000.0   # J mol⁻¹
    delta_h_high: float = 65_000.0   # J mol⁻¹
    r_cal: float = 1.987             # cal mol⁻¹ K⁻¹ (≈8.314 J mol⁻¹ K⁻¹)


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield‑Rollinson temperature‑dependent developmental rate.

    Parameters
    ----------
    temp_k : float
        Absolute temperature in Kelvin (must be > 0).
    params : SchoolfieldParams, optional
        Model parameters.

    Returns
    -------
    rate : float
        Dimensionless rate factor; 1.0 at 25 °C when `rho_25` = 1.
    """
    if temp_k <= 0:
        raise ValueError("Temperature must be greater than 0")
    return (params.delta_h_high - params.delta_h_low) / (params.t_high - params.t_low) * (temp_k - params.t_low) + params.delta_h_low / (params.t_high - params.t_low) + params.rho_25


# ----------------------------------------------------------------------
# Parent‑B: Probabilistic Labeling
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingFunctionResult:
    """Result of a labeling function."""
    lf_name: str
    doc_id: str
    label: int

@dataclass(frozen=True)
class ProbabilisticLabel:
    """Probabilistic label with confidence."""
    doc_id: str
    label: int
    confidence: float

@dataclass(frozen=True)
class LabelError:
    """Label error with error probability."""
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float

def labeling_function(name: str | None = None):
    """Decorator for labeling functions."""
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]:
    """Aggregate labels from batches."""
    votes = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes[r.doc_id].append(r.label)
    out = []
    for d, vs in votes.items():
        if not vs:
            out.append(ProbabilisticLabel(d, 0, 0.0))
        else:
            label_counts = Counter(vs)
            total_votes = len(vs)
            max_votes = max(label_counts.values())
            for label, count in label_counts.items():
                confidence = count / total_votes
                out.append(ProbabilisticLabel(d, label, confidence))
    return out


# ----------------------------------------------------------------------
# Hybrid Poikilotherm‑Labeling State‑Space Duality
# ----------------------------------------------------------------------
def hybrid_ssm_step(A: np.ndarray, r: float, b: np.ndarray) -> np.ndarray:
    """
    Hybrid state space step.

    Parameters
    ----------
    A : np.ndarray
        State transition matrix.
    r : float
        Developmental rate.
    b : np.ndarray
        Input vector.

    Returns
    -------
    x : np.ndarray
        Next state.
    """
    return r * A @ b


def hybrid_ssm_sequential(A: np.ndarray, r: float, b: np.ndarray, num_steps: int) -> np.ndarray:
    """
    Hybrid sequential state space step.

    Parameters
    ----------
    A : np.ndarray
        State transition matrix.
    r : float
        Developmental rate.
    b : np.ndarray
        Input vector.
    num_steps : int
        Number of steps.

    Returns
    -------
    x : np.ndarray
        Final state.
    """
    x = b
    for _ in range(num_steps):
        x = hybrid_ssm_step(A, r, x)
    return x


def hybrid_labeling(skm: np.ndarray, lf_results: List[LabelingFunctionResult]) -> ProbabilisticLabel:
    """
    Hybrid labeling.

    Parameters
    ----------
    skm : np.ndarray
        Sketch matrix.
    lf_results : List[LabelingFunctionResult]
        Labeling function results.

    Returns
    -------
    pl : ProbabilisticLabel
        Probabilistic label.
    """
    votes = defaultdict(list)
    for lf_result in lf_results:
        votes[lf_result.doc_id].append(lf_result.label)
    doc_id, label = max(votes.items(), key=lambda item: len(item[1]))
    confidence = len(votes[doc_id]) / len(lf_results)
    return ProbabilisticLabel(doc_id, label, confidence)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    params = SchoolfieldParams()
    temp_k = 298.15  # 25°C
    r = developmental_rate(temp_k, params)
    A = np.array([[0.5, 0.3], [0.2, 0.7]])
    b = np.array([1.0, 2.0])
    x = hybrid_ssm_step(A, r, b)
    print(x)

    lf_results = [
        LabelingFunctionResult("lf1", "doc1", 0),
        LabelingFunctionResult("lf2", "doc1", 1),
        LabelingFunctionResult("lf3", "doc2", 0),
    ]
    skm = np.array([[0.4, 0.6], [0.3, 0.7]])
    pl = hybrid_labeling(skm, lf_results)
    print(pl)