# DARWIN HAMMER — match 3921, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m530_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_rectified_flo_m1340_s1.py (gen5)
# born: 2026-05-29T23:52:37Z

import math
import numpy as np
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        hashlib.blake2b(data, digest_size=8).digest(), "big"
    )

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def ternary_vector_similarity(vector_a: List[int], vector_b: List[int]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have equal length")
    return sum(1 for a, b in zip(vector_a, vector_b) if a == b) / len(vector_a)

def length(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def bayesian_posterior(p_prior: float, L: float, FP: float) -> float:
    denominator = L * p_prior + FP * (1.0 - p_prior)
    if denominator == 0.0:
        return 0.0
    return (p_prior * L) / denominator

def encode_actions(actions: List[MathAction], k: int = 128) -> np.ndarray:
    signatures = []
    for act in actions:
        tokens = [act.id, f"{act.expected_value:.6f}"]
        sig = signature(tokens, k=k)
        signatures.append(sig)
    arr = np.array(signatures, dtype=np.uint64).astype(np.float64)
    arr /= float(1 << 64)
    return arr

def build_posterior_tree(
    points: np.ndarray,
    prior_fn=similarity,
    fp: float = 0.05,
    distance_scale: float = 10.0,
) -> Dict[Tuple[int, int], float]:
    n = points.shape[0]
    tree: Dict[Tuple[int, int], float] = {}
    int_sigs = (points * (1 << 64)).astype(np.uint64).tolist()
    for i in range(n):
        for j in range(i + 1, n):
            p_prior = prior_fn(int_sigs[i], int_sigs[j])
            dist = length(points[i], points[j])
            likelihood = sigmoid(-distance_scale * dist)
            post = bayesian_posterior(p_prior, likelihood, fp)
            tree[(i, j)] = post
    return tree

def rectified_flow_interpolate(
    src: np.ndarray,
    tgt: np.ndarray,
    steps: int = 10,
    posterior_weight: float = 1.0,
) -> List[np.ndarray]:
    if steps < 2:
        raise ValueError("steps must be >= 2")
    flow: List[np.ndarray] = []
    for t in np.linspace(0.0, 1.0, steps):
        interp = (1 - t) * src + t * tgt
        rectified = np.maximum(interp, 0.0)
        transformed = sigmoid(rectified)
        flow.append(posterior_weight * transformed)
    return flow

def hybrid_score(
    actions: List[MathAction],
    k: int = 128,
    distance_scale: float = 10.0,
    fp: float = 0.05,
) -> np.ndarray:
    points = encode_actions(actions, k=k)
    tree = build_posterior_tree(points, distance_scale=distance_scale, fp=fp)
    scores = np.zeros(len(actions))
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            score_ij = tree[(i, j)]
            scores[i] += score_ij
            scores[j] += score_ij
    return scores / (len(actions) - 1)

import hashlib