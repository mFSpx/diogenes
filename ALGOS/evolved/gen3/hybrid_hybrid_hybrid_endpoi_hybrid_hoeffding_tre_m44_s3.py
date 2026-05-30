# DARWIN HAMMER — match 44, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_endpoint_circ_state_space_duality_m1_s2.py (gen2)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s2.py (gen1)
# born: 2026-05-29T23:25:28Z

"""Hybrid Endpoint‑SSM & Hoeffding‑Tropical Algorithm
===================================================

Parent A – ``hybrid_hybrid_endpoint_circ_state_space_duality_m1_s2.py``  
Parent B – ``hybrid_hoeffding_tree_tropical_maxplus_m18_s2.py``

Mathematical bridge
-------------------
* Each *endpoint* is interpreted as a state variable of a linear state‑space
  model (SSM).  The SSM produces a scalar health score **yₜ** for every
  request step *t* (see Parent A).

* The sequence of health scores is fed to a *tropical* (max‑plus) network.
  In the max‑plus semiring the linear‑algebraic operations of a ReLU network
  become **t_add = max** and **t_mul = +**.  The tropical network maps the
  health‑score vector to a set of *impurity‑gain* candidates (see Parent B).

* A Hoeffding bound is applied to the stream of gain candidates to decide,
  with probability `1‑δ`, when a decision‑tree node may be split.

The three core functions below implement this pipeline:

1. ``hybrid_compute_health_scores`` – builds the SSM matrices from the
   endpoint pool and returns the scalar health scores for a request sequence.

2. ``hybrid_tropical_gains`` – evaluates a two‑layer tropical max‑plus network
   on the health‑score vector and returns a gain candidate per time step.

3. ``hybrid_update_and_maybe_split`` – updates node statistics with the latest
   gain and uses the Hoeffding bound to decide whether a split is statistically
   justified.

All operations rely only on ``numpy`` and the Python standard library.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – endpoint description and SSM construction
# ----------------------------------------------------------------------


@dataclass
class Endpoint:
    """Simple representation of an endpoint used by the hybrid engine."""
    failures: int
    failure_threshold: int
    righting_time_index: float  # morphology‑derived scalar (higher ⇒ healthier)
    # additional fields could be added without breaking the algorithm

    @property
    def failure_rate(self) -> float:
        """Normalized failure rate in [0, 1]."""
        if self.failure_threshold == 0:
            return 0.0
        return min(1.0, max(0.0, self.failures / self.failure_threshold))

    @property
    def recovery_priority(self) -> float:
        """Normalized recovery priority in [0, 1] derived from morphology."""
        # we map the righting_time_index (assumed ≥0) to a priority in [0,1]
        # using a simple sigmoid for boundedness.
        return 1.0 / (1.0 + math.exp(-self.righting_time_index))

    @property
    def health_score(self) -> float:
        """Combined health score used as the output projection Cₜ."""
        fr = self.failure_rate
        rp = self.recovery_priority
        return (1.0 - fr) * (1.0 - rp)


def _build_ssm_matrices(endpoints: List[Endpoint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the per‑step SSM matrices A, B, C from a list of endpoints.

    Returns
    -------
    A : (N, N) diagonal matrix – same for every time step.
    B : (N, 1) column vector – same for every time step.
    C : (1, N) row vector – same for every time step.
    """
    N = len(endpoints)
    failure_rates = np.array([ep.failure_rate for ep in endpoints], dtype=np.float64)
    A = np.diag(failure_rates)                     # decay due to failures
    B = np.array([[ep.righting_time_index] for ep in endpoints], dtype=np.float64)  # morphology input
    C = np.array([[ep.health_score for ep in endpoints]], dtype=np.float64)        # output projection
    return A, B, C


def hybrid_compute_health_scores(endpoints: List[Endpoint], inputs: np.ndarray) -> np.ndarray:
    """
    Evaluate the SSM (state‑space duality) for a sequence of scalar inputs.

    Parameters
    ----------
    endpoints : list[Endpoint]
        The pool of endpoints; each contributes one state dimension.
    inputs : (T,) array_like
        Scalar request values (e.g. request size, latency, etc.).

    Returns
    -------
    scores : (T,) ndarray
        The scalar health score yₜ produced at each time step.
    """
    if inputs.ndim != 1:
        raise ValueError("inputs must be a 1‑D array of scalars")
    T = inputs.shape[0]
    A, B, C = _build_ssm_matrices(endpoints)
    N = A.shape[0]

    # Initialise hidden state h₀ = 0
    h = np.zeros((N, 1), dtype=np.float64)
    scores = np.empty(T, dtype=np.float64)

    for t in range(T):
        x_t = inputs[t]
        # hₜ = A·hₜ₋₁ + B·xₜ   (B is (N,1), xₜ scalar)
        h = A @ h + B * x_t
        # yₜ = C·hₜ
        scores[t] = (C @ h).item()
    return scores


# ----------------------------------------------------------------------
# Parent B – tropical max‑plus algebra
# ----------------------------------------------------------------------


def t_add(a: float, b: float) -> float:
    """Tropical addition (max)."""
    return max(a, b)


def t_mul(a: float, b: float) -> float:
    """Tropical multiplication (plus)."""
    return a + b


def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical matrix multiplication: (A ⊗ B)_{ij} = max_k (A_{ik} + B_{kj})

    Both inputs are ordinary NumPy arrays; the operation is performed in the
    max‑plus semiring.
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("inner dimensions must match for tropical matmul")
    result = np.full((A.shape[0], B.shape[1]), -np.inf, dtype=np.float64)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            # compute max_k (A[i,k] + B[k,j])
            max_val = -np.inf
            for k in range(A.shape[1]):
                val = t_mul(A[i, k], B[k, j])
                if val > max_val:
                    max_val = val
            result[i, j] = max_val
    return result


def tropical_network_eval(x: np.ndarray, weights: List[np.ndarray]) -> np.ndarray:
    """
    Evaluate a feed‑forward tropical network (max‑plus) with given weight matrices.

    Parameters
    ----------
    x : (d,) ndarray
        Input vector.
    weights : list of 2‑D ndarrays
        Weight matrices for each layer.  No bias terms are used for simplicity.

    Returns
    -------
    y : (output_dim,) ndarray
        Network output in the tropical semiring.
    """
    # Convert input to column vector
    h = x.reshape(-1, 1)
    for W in weights:
        h = t_matmul(W, h)
    return h.ravel()


def hybrid_tropical_gains(scores: np.ndarray,
                          weight1: np.ndarray,
                          weight2: np.ndarray) -> np.ndarray:
    """
    Map health scores to impurity‑gain candidates via a two‑layer tropical network.

    Parameters
    ----------
    scores : (T,) ndarray
        Health scores produced by the SSM.
    weight1 : (H, T) ndarray
        First‑layer tropical weight matrix (H hidden units).
    weight2 : (1, H) ndarray
        Second‑layer tropical weight matrix (single output unit).

    Returns
    -------
    gains : (T,) ndarray
        Gain candidate for each time step (scalar per step).
    """
    T = scores.shape[0]
    gains = np.empty(T, dtype=np.float64)

    # The network is evaluated independently for each time step because the
    # tropical layers are linear in the max‑plus sense.
    for t in range(T):
        x_t = np.array([scores[t]])  # treat scalar as 1‑D vector
        # expand to match expected input dimension of weight1
        # weight1 expects dimension T; we broadcast the single value.
        x_vec = np.full(weight1.shape[1], x_t.item(), dtype=np.float64)
        out = tropical_network_eval(x_vec, [weight1, weight2])
        gains[t] = out.item()
    return gains


# ----------------------------------------------------------------------
# Hoeffding bound + split decision (Parent B)
# ----------------------------------------------------------------------


@dataclass
class NodeStats:
    """
    Statistics kept for a streaming decision‑tree node.
    """
    n: int = 0                                 # number of observed examples
    best_gain: float = -np.inf                 # highest gain seen so far
    second_best_gain: float = -np.inf          # second highest gain
    sum_gain: float = 0.0                      # optional: cumulative gain

    def update(self, gain: float) -> None:
        """Incorporate a new gain observation."""
        self.n += 1
        self.sum_gain += gain
        if gain > self.best_gain:
            self.second_best_gain = self.best_gain
            self.best_gain = gain
        elif gain > self.second_best_gain:
            self.second_best_gain = gain


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a variable bounded in [0, r]."""
    if n <= 0:
        return float('inf')
    return math.sqrt((r ** 2 * math.log(1.0 / delta)) / (2.0 * n))


def hybrid_update_and_maybe_split(node: NodeStats,
                                  gain: float,
                                  delta: float = 0.05,
                                  r: float = 1.0) -> bool:
    """
    Update node statistics with a new gain and decide whether to split.

    The decision follows the classic Hoeffding criterion:
        n >= (r^2 * ln(1/δ)) / (Δg^2)
    where Δg = best_gain – second_best_gain.

    Returns
    -------
    should_split : bool
        True if the Hoeffding bound guarantees that the current best split
        is statistically better than the second best with probability 1‑δ.
    """
    node.update(gain)

    # If we have fewer than 2 distinct candidates we cannot split yet.
    if node.n < 2 or node.best_gain == -np.inf or node.second_best_gain == -np.inf:
        return False

    delta_gain = node.best_gain - node.second_best_gain
    if delta_gain <= 0.0:
        # No observable advantage
        return False

    epsilon = hoeffding_bound(r, delta, node.n)
    return epsilon < delta_gain


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


def _random_endpoint() -> Endpoint:
    """Utility to generate a random endpoint for the demo."""
    failures = random.randint(0, 20)
    failure_threshold = random.randint(10, 30) or 1
    righting_time_index = random.uniform(-2.0, 2.0)
    return Endpoint(failures, failure_threshold, righting_time_index)


def _demo():
    random.seed(42)
    np.random.seed(0)

    # 1. Create a small pool of endpoints
    endpoints = [_random_endpoint() for _ in range(4)]

    # 2. Simulate a request sequence (e.g. request sizes)
    T = 12
    inputs = np.random.rand(T) * 5.0  # arbitrary positive scalars

    # 3. Compute health scores via the SSM
    scores = hybrid_compute_health_scores(endpoints, inputs)

    # 4. Define a tiny tropical network (H hidden units = 3)
    H = 3
    weight1 = np.random.randn(H, T)  # shape (H, T)
    weight2 = np.random.randn(1, H)  # shape (1, H)

    # 5. Obtain gain candidates from the tropical network
    gains = hybrid_tropical_gains(scores, weight1, weight2)

    # 6. Stream the gains through a Hoeffding‑based split decision
    node = NodeStats()
    split_occurred = False
    for t, g in enumerate(gains):
        if hybrid_update_and_maybe_split(node, g, delta=0.05, r=1.0):
            print(f"Split decided at time step {t} (n={node.n}, best_gain={node.best_gain:.4f}, "
                  f"second_best={node.second_best_gain:.4f})")
            split_occurred = True
            break

    if not split_occurred:
        print("No split was statistically justified after processing all gains.")
    # Print a few intermediate values for sanity checking
    print("\n--- Debug Info ---")
    print("Endpoints (failure_rate, recovery_priority, health_score):")
    for i, ep in enumerate(endpoints):
        print(f"  E{i}: fr={ep.failure_rate:.3f}, rp={ep.recovery_priority:.3f}, hs={ep.health_score:.3f}")
    print("\nHealth scores:", scores)
    print("\nTropical gains:", gains)
    print("\nFinal node stats:", node)


if __name__ == "__main__":
    _demo()