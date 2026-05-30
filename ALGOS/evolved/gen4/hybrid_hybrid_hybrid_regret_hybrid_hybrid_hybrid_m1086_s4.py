# DARWIN HAMMER — match 1086, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hoeffding_tre_m44_s3.py (gen3)
# born: 2026-05-29T23:32:53Z

import hashlib
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

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
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [0] * k
    sig = []
    for i in range(k):
        min_h = min(_hash(i, t) for t in toks)
        sig.append(min_h)
    return sig


def jaccard_minhash(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class StoreState:
    dance: float = 1.0


def compute_regret_bandit_scores(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    store: StoreState,
    reference_action_ids: List[str],
    minhash_k: int = 128,
) -> Dict[str, float]:
    cf_lookup = {cf.action_id: cf for cf in counterfactuals}

    signatures = {}
    for act in actions:
        tokens = list(act.id)
        signatures[act.id] = minhash_signature(tokens, k=minhash_k)

    ref_sigs = [signatures[aid] for aid in reference_action_ids if aid in signatures]
    if not ref_sigs:
        ref_sig = [0] * minhash_k
    else:
        ref_sig = [int(np.median([s[i] for s in ref_sigs])) for i in range(minhash_k)]

    scores = {}
    for act in actions:
        cf = cf_lookup.get(act.id)
        counterfactual = cf.outcome_value if cf else 0.0
        R = act.expected_value - act.cost - act.risk + counterfactual

        g = sigmoid(R)

        sim = jaccard_minhash(signatures[act.id], ref_sig)

        S = g * (1.0 + sim) * store.dance
        scores[act.id] = S
    return scores


@dataclass
class Endpoint:
    failures: int
    failure_threshold: int
    righting_time_index: float

    @property
    def failure_rate(self) -> float:
        if self.failure_threshold == 0:
            return 0.0
        return min(1.0, self.failures / self.failure_threshold)


def build_ssm_matrices(endpoints: List[Endpoint]) -> Tuple[np.ndarray, float]:
    if not endpoints:
        raise ValueError("At least one endpoint required")
    A = np.array([ep.righting_time_index for ep in endpoints], dtype=float)
    mean_failure = np.mean([ep.failure_rate for ep in endpoints])
    b = -mean_failure
    return A, b


def compute_health_scores(
    bandit_scores: Dict[str, float],
    endpoints: List[Endpoint],
) -> np.ndarray:
    A, b = build_ssm_matrices(endpoints)

    scores_vec = np.array(list(bandit_scores.values()), dtype=float)
    if len(A) < len(scores_vec):
        repeats = int(np.ceil(len(scores_vec) / len(A)))
        A_ext = np.tile(A, repeats)[: len(scores_vec)]
    else:
        A_ext = A[: len(scores_vec)]

    y = A_ext * scores_vec + b
    return y


@dataclass
class TreeNode:
    gain: float = 0.0
    count: int = 0


def compute_tropical_network(
    health_scores: np.ndarray,
    weights1: np.ndarray,
    bias1: np.ndarray,
    weights2: np.ndarray,
    bias2: float,
) -> float:
    z = np.max(health_scores[:, np.newaxis] + weights1, axis=0) + bias1
    g = np.max(z + weights2) + bias2
    return g


def hoeffding_bound(
    gain: float,
    count: int,
    confidence: float,
) -> float:
    if count == 0:
        return float("inf")
    return math.sqrt(math.log(1 / (1 - confidence)) / (2 * count))


def update_tree_node(
    node: TreeNode,
    gain: float,
    confidence: float,
) -> bool:
    node.gain += gain
    node.count += 1
    bound = hoeffding_bound(node.gain, node.count, confidence)
    return node.gain > bound


def main():
    # Example usage
    actions = [MathAction("id1", 10.0), MathAction("id2", 20.0)]
    counterfactuals = [MathCounterfactual("id1", 5.0)]
    store = StoreState()
    reference_action_ids = ["id1"]
    bandit_scores = compute_regret_bandit_scores(
        actions, counterfactuals, store, reference_action_ids
    )

    endpoints = [Endpoint(1, 10, 0.5), Endpoint(2, 20, 0.8)]
    health_scores = compute_health_scores(bandit_scores, endpoints)

    weights1 = np.array([0.1, 0.2])
    bias1 = np.array([0.01, 0.02])
    weights2 = np.array([0.3, 0.4])
    bias2 = 0.1
    gain = compute_tropical_network(health_scores, weights1, bias1, weights2, bias2)

    node = TreeNode()
    confidence = 0.95
    split = update_tree_node(node, gain, confidence)
    print(split)


if __name__ == "__main__":
    main()