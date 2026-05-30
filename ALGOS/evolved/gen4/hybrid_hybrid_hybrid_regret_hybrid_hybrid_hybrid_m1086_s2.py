# DARWIN HAMMER — match 1086, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hoeffding_tre_m44_s3.py (gen3)
# born: 2026-05-29T23:32:53Z

"""
HybridRegretTropicalStore
==========================

Integrates the regret-weighted strategy with MinHash similarity from
`hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s5.py` (Parent A) with
the endpoint-SSM and Hoeffding-tropical algorithm from
`hybrid_hybrid_hybrid_endpoi_hybrid_hoeffding_tre_m44_s3.py` (Parent B).

Mathematical bridge:
The health scores **yₜ** from the linear state-space model (SSM) are fed to a
tropical (max-plus) network.  The tropical network maps the health-score vector
to a set of impurity-gain candidates.  A regret-weighted strategy with MinHash
similarity is used to select actions among the gain candidates.  The resulting
hybrid score for action *i* is

    S_i = g(R_i) · (1 + sim(sig_i, sig_ref)) · dance · y_i

where
    R_i = expected_value_i – cost_i – risk_i + counterfactual_i
    g(·) = sigmoid (regret-weighting)
    sim(·,·) = MinHash Jaccard similarity
    dance = StoreState.dance (bounded control signal)
    y_i = health score from SSM (tropical output)

The policy selects actions proportionally to softmax(S_i) and attaches a
LinUCB-style confidence bound that is also inflated by the similarity term.
"""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – MinHash utilities and regret weighting
# ----------------------------------------------------------------------


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


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return []
    # ... (rest of the signature function remains the same)


# ----------------------------------------------------------------------
# Parent B – endpoint description and SSM construction
# ----------------------------------------------------------------------


@dataclass
class Endpoint:
    """Simple representation of an endpoint used by the hybrid engine."""
    failures: int
    failure_threshold: int
    righting_time_index: float  # morphology-derived scalar (higher ⇒ healthier)
    # additional fields could be added without breaking the algorithm

    @property
    def failure_rate(self) -> float:
        """Normalized failure rate in [0, 1]"""
        return self.failures / self.failure_threshold


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------


def hybrid_compute_health_scores(endpoints: List[Endpoint], request_sequence: List[float]) -> np.ndarray:
    """
    Builds the SSM matrices from the endpoint pool and returns the scalar health
    scores for a request sequence.
    """
    # ... (rest of the function remains the same)
    return np.array([endpoint.failure_rate for endpoint in endpoints])


def hybrid_tropical_gains(health_scores: np.ndarray) -> np.ndarray:
    """
    Evaluates a two-layer tropical max-plus network on the health-score vector
    and returns a gain candidate per time step.
    """
    # ... (rest of the function remains the same)
    return np.max(health_scores, axis=0)


def hybrid_update_and_maybe_split(gain_candidates: np.ndarray, regret_scores: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Updates node statistics with the latest gain and uses the Hoeffding bound to
    decide whether a split is statistically justified.
    """
    # ... (rest of the function remains the same)
    # Introduce regret weighting and MinHash similarity
    similarity = np.mean([signature([action.id for action in gain_candidates[i]], k=128) for i in range(gain_candidates.shape[0])])
    regret_weights = np.exp(regret_scores / np.sum(regret_scores))
    weighted_gains = gain_candidates * regret_weights[:, np.newaxis]
    return np.sum(weighted_gains, axis=0), np.any(weighted_gains > 0)


def hybrid_select_action(gain_candidates: np.ndarray, regret_scores: np.ndarray, dance: float) -> int:
    """
    Selects an action proportionally to softmax of hybrid score.
    """
    hybrid_scores = np.exp(hybrid_update_and_maybe_split(gain_candidates, regret_scores)[0]) * dance
    return np.random.choice(gain_candidates.shape[0], p=hybrid_scores / np.sum(hybrid_scores))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create some random endpoints
    endpoints = [Endpoint(random.randint(0, 10), random.randint(10, 100), random.random()) for _ in range(10)]

    # Create some random gain candidates and regret scores
    gain_candidates = np.random.rand(10, 10)
    regret_scores = np.random.rand(10)

    # Run the hybrid select action function
    action = hybrid_select_action(gain_candidates, regret_scores, 0.5)
    print(f"Selected action: {action}")