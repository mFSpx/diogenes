# DARWIN HAMMER — match 1510, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m1019_s1.py (gen5)
# parent_b: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s3.py (gen2)
# born: 2026-05-29T23:37:06Z

import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Hashable, Mapping
import numpy as np

@dataclass
class CertaintyFlag:
    label: str
    confidence_bps: int  # basis points, 0-10000

def stylometry_features(text: str) -> Dict[str, int]:
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

def build_weighted_graph(features_list: List[Dict[str, int]]) -> Tuple[np.ndarray, np.ndarray]:
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

def curvature_to_confidence(W: np.ndarray, strengths: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, Dict[Tuple[int, int], CertaintyFlag]]:
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

def max_plus_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.max(A[:, :, None] + B[None, :, :], axis=1)

def tropical_broadcast(confidence: np.ndarray, iterations: int = 5) -> np.ndarray:
    C = confidence.astype(float) / 10000.0
    b = np.max(C, axis=1)
    for _ in range(iterations):
        b = np.max(C + b[None, :], axis=1)
    return np.clip(b, 0.0, 1.0)

def hoeffding_candidate_selection(broadcast: np.ndarray, n_samples: int = 30, delta: float = 0.05) -> List[int]:
    epsilon = math.sqrt(math.log(2.0 / delta) / (2.0 * n_samples))
    threshold = 0.5 + epsilon
    candidates = [i for i, val in enumerate(broadcast) if val > threshold]
    return candidates

def acceptance_probability(delta_e: float, temperature: float) -> float:
    if delta_e < 0:
        return 1.0
    try:
        return math.exp(-delta_e / temperature)
    except OverflowError:
        return 0.0

def simulated_annealing_update(current_leaders: List[int], candidates: List[int], temperature: float) -> List[int]:
    delta_e = abs(len(candidates) - len(current_leaders))
    prob = acceptance_probability(delta_e, temperature)
    if random.random() < prob:
        return candidates.copy()
    else:
        return current_leaders.copy()

def hybrid_stylometry_tropical_election(texts: List[str], temperature: float = 1.0, n_samples: int = 30, delta: float = 0.05) -> List[int]:
    features_list = [stylometry_features(text) for text in texts]
    W, strengths = build_weighted_graph(features_list)
    confidence, _ = curvature_to_confidence(W, strengths)
    broadcast = tropical_broadcast(confidence)
    candidates = hoeffding_candidate_selection(broadcast, n_samples, delta)
    current_leaders = []
    return simulated_annealing_update(current_leaders, candidates, temperature)