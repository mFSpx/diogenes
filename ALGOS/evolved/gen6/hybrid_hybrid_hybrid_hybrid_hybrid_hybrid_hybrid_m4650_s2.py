# DARWIN HAMMER — match 4650, survivor 2
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
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco


def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
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
    scaled_confidence = confidence * bandit_action.propensity * bandit_action.expected_reward
    return ProbabilisticLabel(doc_id, label, scaled_confidence)


def hybrid_bandit_update(bandit_action: BanditAction, labeling_result: ProbabilisticLabel) -> BanditAction:
    updated_propensity = bandit_action.propensity * labeling_result.confidence
    updated_expected_reward = bandit_action.expected_reward * labeling_result.confidence
    updated_confidence_bound = bandit_action.confidence_bound * (1 - labeling_result.confidence)
    return BanditAction(bandit_action.action_id, updated_propensity, updated_expected_reward, updated_confidence_bound, bandit_action.algorithm)


def hybrid_error_detection(bandit_action: BanditAction, labeling_result: ProbabilisticLabel) -> float:
    error_threshold = bandit_action.confidence_bound * labeling_result.confidence
    return error_threshold


def liquid_time_constant(x_t: np.ndarray, w_t: np.ndarray, m_t: float, tau: float, dt: float) -> np.ndarray:
    x_t_plus_1 = x_t + dt * (-(1 / tau + m_t) * x_t + m_t * (w_t @ x_t))
    return x_t_plus_1


def ttt_weight_matrix_update(w_t: np.ndarray, x_t: np.ndarray, m_t: float, eta: float, lambda_: float, r_t: float) -> np.ndarray:
    w_t_plus_1 = w_t - eta * (1 + lambda_ * r_t) * np.linalg.norm(w_t @ x_t)
    return w_t_plus_1


def hybrid_liquid_time_constant(bandit_action: BanditAction, x_t: np.ndarray, w_t: np.ndarray, tau: float, dt: float) -> np.ndarray:
    m_t = bandit_action.confidence_bound
    x_t_plus_1 = liquid_time_constant(x_t, w_t, m_t, tau, dt)
    return x_t_plus_1


def hybrid_ttt_weight_matrix_update(bandit_action: BanditAction, w_t: np.ndarray, x_t: np.ndarray, eta: float, lambda_: float, r_t: float) -> np.ndarray:
    m_t = bandit_action.confidence_bound
    w_t_plus_1 = ttt_weight_matrix_update(w_t, x_t, m_t, eta, lambda_, r_t)
    return w_t_plus_1


if __name__ == "__main__":
    doc_id = "test_doc"
    label = 1
    confidence = 0.8
    bandit_action = BanditAction("action_1", 0.5, 0.2, 0.1, "algorithm_1")
    labeling_result = hybrid_labeling(doc_id, label, confidence, bandit_action)
    print(labeling_result)

    updated_bandit_action = hybrid_bandit_update(bandit_action, labeling_result)
    print(updated_bandit_action)

    error_threshold = hybrid_error_detection(bandit_action, labeling_result)
    print(error_threshold)

    x_t = np.array([1.0, 2.0])
    w_t = np.array([[0.1, 0.2], [0.3, 0.4]])
    tau = 1.0
    dt = 0.1
    x_t_plus_1 = hybrid_liquid_time_constant(bandit_action, x_t, w_t, tau, dt)
    print(x_t_plus_1)

    eta = 0.1
    lambda_ = 0.5
    r_t = 0.2
    w_t_plus_1 = hybrid_ttt_weight_matrix_update(bandit_action, w_t, x_t, eta, lambda_, r_t)
    print(w_t_plus_1)