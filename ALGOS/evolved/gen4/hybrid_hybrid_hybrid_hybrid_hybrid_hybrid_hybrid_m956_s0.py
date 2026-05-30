# DARWIN HAMMER — match 956, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_label_foundry_m21_s1.py (gen3)
# born: 2026-05-29T23:31:47Z

"""
Hybrid Algorithm Fusion of hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s3.py and 
hybrid_hybrid_hybrid_bandit_label_foundry_m21_s1.py.

The mathematical bridge between the two parents lies in their ability to handle 
time-varying state-space and noisy labels. The 
hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s3.py algorithm uses 
temperature-dependent scalar to modulate the state-transition matrix, while 
the hybrid_hybrid_hybrid_bandit_label_foundry_m21_s1.py algorithm uses weak 
supervision labeling primitives to handle noisy labels. By combining these two 
approaches, we can create a hybrid algorithm that can handle both time-varying 
state-space and noisy labels.

The hybrid algorithm uses the temperature-dependent scalar to modulate the 
state-transition matrix, and then uses the modulated matrix to inform the 
labeling process. The labeling process uses the weak supervision labeling 
primitives to handle noisy labels, and the hybrid algorithm combines the 
results of the state-space and labeling processes to produce a final output.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np

__all__ = [
    "SchoolfieldParams",
    "developmental_rate",
    "temperature_dependent_state_transition",
    "hybrid_ssm_step",
    "labeling_function",
    "aggregate_labels",
    "hybrid_operation",
]

# ----------------------------------------------------------------------
# Parent-A: Poikilotherm developmental rate
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
    if temp_k <= 0:
        raise ValueError("Temperature must be greater than 0")
    rate = params.rho_25 * math.exp((params.delta_h_activation / params.r_cal) * 
                                     (1 / 298.15 - 1 / temp_k))
    if temp_k < params.t_low:
        rate *= math.exp((params.delta_h_low / params.r_cal) * 
                        (1 / params.t_low - 1 / temp_k))
    elif temp_k > params.t_high:
        rate *= math.exp((params.delta_h_high / params.r_cal) * 
                        (1 / params.t_high - 1 / temp_k))
    return rate

def temperature_dependent_state_transition(temp_k: float, 
                                          state_transition_matrix: np.ndarray, 
                                          params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    rate = developmental_rate(temp_k, params)
    return state_transition_matrix * rate

def hybrid_ssm_step(state: np.ndarray, 
                    state_transition_matrix: np.ndarray, 
                    temp_k: float, 
                    params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    modulated_matrix = temperature_dependent_state_transition(temp_k, state_transition_matrix, params)
    return np.dot(modulated_matrix, state)

# ----------------------------------------------------------------------
# Parent-B: Label Foundry
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int

@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float

def labeling_function(name: str | None = None):
    def deco(fn):
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]:
    votes = {}
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                if r.doc_id not in votes:
                    votes[r.doc_id] = []
                votes[r.doc_id].append(r.label)
    out = []
    for d, vs in votes.items():
        if not vs:
            out.append(ProbabilisticLabel(d, 0, 0.0))
        else:
            label = max(set(vs), key=vs.count)
            confidence = vs.count(label) / len(vs)
            out.append(ProbabilisticLabel(d, label, confidence))
    return out

# ----------------------------------------------------------------------
# Hybrid Operation
# ----------------------------------------------------------------------
def hybrid_operation(state: np.ndarray, 
                     state_transition_matrix: np.ndarray, 
                     temp_k: float, 
                     batches: list[list[LabelingFunctionResult]], 
                     params: SchoolfieldParams = SchoolfieldParams()) -> list[ProbabilisticLabel]:
    next_state = hybrid_ssm_step(state, state_transition_matrix, temp_k, params)
    labels = aggregate_labels(batches)
    return labels

if __name__ == "__main__":
    # Smoke test
    state = np.array([1.0, 2.0, 3.0])
    state_transition_matrix = np.array([[0.9, 0.1, 0.0], [0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    temp_k = 298.15
    batches = [[LabelingFunctionResult("lf1", "doc1", 1), LabelingFunctionResult("lf2", "doc2", 0)]]
    labels = hybrid_operation(state, state_transition_matrix, temp_k, batches)
    print(labels)