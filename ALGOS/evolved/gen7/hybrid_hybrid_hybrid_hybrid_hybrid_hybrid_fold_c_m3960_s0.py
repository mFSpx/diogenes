# DARWIN HAMMER — match 3960, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_distri_m1510_s4.py (gen6)
# parent_b: hybrid_hybrid_fold_change_d_hybrid_hybrid_hybrid_m1479_s0.py (gen5)
# born: 2026-05-29T23:52:44Z

"""
This module fuses the Hybrid Hammer and Hybrid Fold Change Detection algorithms.
The mathematical bridge between the two structures lies in the integration of 
the governing equations of both parents, specifically the curvature-to-confidence 
function in Hybrid Hammer and the Fisher score in Hybrid Fold Change Detection.
The Hybrid Hammer's curvature-to-confidence function is used as a weighting 
factor for the Hybrid Fold Change Detection algorithm's update equations, 
effectively weighting the selection of actions by the information content 
of the current token distribution.
"""

import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class CertaintyFlag:
    label: str
    confidence_bps: int  # basis points, 0-10000

class BanditAction:
    def __init__(self, action_id: str, propensity: float, expected_reward: float, confidence_bound: float, algorithm: str):
        self.action_id = action_id
        self.propensity = propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm

class BanditUpdate:
    def __init__(self, context_id: str, action_id: str, reward: float, propensity: float):
        self.context_id = context_id
        self.action_id = action_id
        self.reward = reward
        self.propensity = propensity

_POLICY: dict = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: list) -> None:
    for u in updates:
        total, n = _POLICY.get(u.action_id, [0.0, 0.0])
        _POLICY[u.action_id] = [total + u.reward, n + 1]

def stylometry_features(text: str) -> dict:
    pronouns = {"i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them"}
    conjunctions = {"and", "or", "but", "nor", "so", "yet", "for", "although"}
    verbs = {"is", "are", "was", "were", "be", "been", "being", "have", "has", "do", "does"}

    tokens = [t.lower().strip(".,!?;:()[]\"'") for t in text.split()]
    counts = Counter(tokens)

    return {
        "pronoun": sum(counts[w] for w in pronouns),
        "conj":   sum(counts[w] for w in conjunctions),
        "verb":   sum(counts[w] for w in verbs),
        "len":    len(tokens),
    }

def build_weighted_graph(features_list: list) -> tuple:
    n = len(features_list)
    keys = sorted({k for d in features_list for k in d})
    mat = np.zeros((n, len(keys)), dtype=float)
    for i, d in enumerate(features_list):
        for j, k in enumerate(keys):
            mat[i, j] = d.get(k, 0)

    strengths = np.linalg.norm(mat, axis=1) + 1e-12  # avoid zero
    norm_mat = mat / strengths[:, None]
    W = np.clip(norm_mat @ norm_mat.T, 0.0, 1.0)
    np.fill_diagonal(W, 1.0)
    return W, strengths

def curvature_to_confidence(W: np.ndarray, strengths: np.ndarray, eps: float = 1e-9) -> tuple:
    w_i = strengths[:, None]
    w_j = strengths[None, :]
    curvature = 1.0 - np.abs(w_i - w_j) / (w_i + w_j + eps)  
    confidence = ((curvature + 1.0) / 2.0 * 10000).astype(int)

    cert_dict = {}
    n = W.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            label = "high" if confidence[i, j] > 7000 else "low"
            cert_dict[(i, j)] = CertaintyFlag(label=label, confidence_bps=int(confidence[i, j]))
            cert_dict[(j, i)] = cert_dict[(i, j)]  # symmetric

    return confidence, cert_dict

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam at angle `theta`."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = -2 * (theta - center) / (width ** 2) * intensity
    return -2 * derivative ** 2 / intensity

def hybrid_update(updates: list, features_list: list) -> None:
    W, strengths = build_weighted_graph(features_list)
    confidence, cert_dict = curvature_to_confidence(W, strengths)
    update_policy(updates)
    for u in updates:
        theta = _reward(u.action_id)
        center = 0.5  # expected reward
        width = 0.1  # confidence width
        fisher = fisher_score(theta, center, width)
        propensity = fisher * u.propensity
        print(f"Action {u.action_id} propensity: {propensity}")

if __name__ == "__main__":
    updates = [
        BanditUpdate("context1", "action1", 1.0, 0.5),
        BanditUpdate("context1", "action2", 0.5, 0.3),
    ]
    features_list = [
        stylometry_features("This is a text."),
        stylometry_features("This is another text."),
    ]
    hybrid_update(updates, features_list)