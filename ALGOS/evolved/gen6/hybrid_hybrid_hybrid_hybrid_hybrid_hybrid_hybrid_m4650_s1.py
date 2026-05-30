# DARWIN HAMMER — match 4650, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_label__hybrid_hybrid_bandit_m415_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_liquid_m961_s1.py (gen5)
# born: 2026-05-29T23:57:15Z

import numpy as np
from dataclasses import dataclass, asdict
from math import exp, sqrt
from pathlib import Path
from random import Random
from typing import Any, Callable, Dict, List, Tuple

@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # binary 0/1


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # in [0,1]


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}  # virtual VRAM store per key


def labeling_function(name: str | None = None):
    """Decorator that annotates a labeling function with a name."""
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco


def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """Pure A-logic: majority vote with confidence = proportion of votes."""
    votes: Dict[str, List[int]] = {}
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes.setdefault(r.doc_id, []).append(r.label)
    out = []
    for doc_id, labels in votes.items():
        confidence = labels.count(1) / len(labels)
        label = 1 if confidence > 0.5 else 0
        out.append(ProbabilisticLabel(doc_id, label, confidence))
    return out


def hybrid_labeling(doc_id: str, label: int, confidence: float, bandit_action: BanditAction) -> ProbabilisticLabel:
    """Applies the bandit algorithm to the labeling process and scales the labeling confidence."""
    scaled_confidence = confidence * bandit_action.propensity * bandit_action.expected_reward
    scaled_confidence = max(0.0, min(scaled_confidence, 1.0))  # clamp confidence to [0, 1]
    return ProbabilisticLabel(doc_id, label, scaled_confidence)


def hybrid_bandit_update(bandit_action: BanditAction, labeling_result: ProbabilisticLabel) -> BanditAction:
    """Updates the bandit policy based on the labeling results."""
    updated_propensity = bandit_action.propensity * labeling_result.confidence
    updated_expected_reward = bandit_action.expected_reward * labeling_result.confidence
    updated_confidence_bound = bandit_action.confidence_bound * (1 - labeling_result.confidence)
    return BanditAction(bandit_action.action_id, updated_propensity, updated_expected_reward, updated_confidence_bound, bandit_action.algorithm)


def hybrid_error_detection(bandit_action: BanditAction, labeling_result: ProbabilisticLabel) -> float:
    """Relaxes the error-detection threshold based on the bandit algorithm's confidence bound."""
    error_threshold = bandit_action.confidence_bound * (1 - labeling_result.confidence)
    return error_threshold


def liquid_time_constant(x_t: np.ndarray, w_t: np.ndarray, m_t: float, tau: float, dt: float) -> np.ndarray:
    """LTC recurrent dynamics with the TTT weight matrix."""
    w_t_eye = np.eye(w_t.shape[0])
    x_t_plus_1 = x_t + dt * (-(w_t_eye / tau + m_t * w_t) @ x_t)
    return x_t_plus_1


def ttt_weight_matrix_update(w_t: np.ndarray, x_t: np.ndarray, m_t: float, eta: float, lambda_: float, r_t: float) -> np.ndarray:
    """Updates the TTT weight matrix using the LTC effective time-constant."""
    gradient = np.outer(w_t @ x_t, x_t)
    w_t_plus_1 = w_t - eta * (gradient + lambda_ * r_t * w_t)
    return w_t_plus_1


if __name__ == "__main__":
    # Test the hybrid labeling function
    doc_id = "test_doc"
    label = 1
    confidence = 0.8
    bandit_action = BanditAction("action_1", 0.5, 0.2, 0.1, "algorithm_1")
    labeling_result = hybrid_labeling(doc_id, label, confidence, bandit_action)
    print(labeling_result)

    # Test the hybrid bandit update function
    updated_bandit_action = hybrid_bandit_update(bandit_action, labeling_result)
    print(updated_bandit_action)

    # Test the hybrid error detection function
    error_threshold = hybrid_error_detection(bandit_action, labeling_result)
    print(error_threshold)

    # Test the LTC recurrent dynamics
    x_t = np.array([1.0, 2.0])
    w_t = np.array([[0.1, 0.2], [0.3, 0.4]])
    m_t = 0.5
    tau = 1.0
    dt = 0.1
    x_t_plus_1 = liquid_time_constant(x_t, w_t, m_t, tau, dt)
    print(x_t_plus_1)

    # Test the TTT weight matrix update
    eta = 0.1
    lambda_ = 0.5
    r_t = 0.2
    w_t_plus_1 = ttt_weight_matrix_update(w_t, x_t, m_t, eta, lambda_, r_t)
    print(w_t_plus_1)