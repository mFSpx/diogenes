# DARWIN HAMMER — match 5337, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hdc_se_hybrid_ssim_hybrid_h_m1852_s1.py (gen4)
# born: 2026-05-30T00:01:26Z

"""Hybrid RBF‑HDC Bandit Algorithm

Parents:
- **PARENT A**: Gaussian radial‑basis‑function surrogate model driving a
  contextual bandit (reward prediction, social interaction).
- **PARENT B**: Hyperdimensional (HDC) encoding with sparse winner‑take‑all
  (WTA) pooling, Structural Similarity Index (SSIM) and a Hoeffding‑type
  uncertainty term.

Mathematical bridge:
Both parents operate on high‑dimensional vectors.  In this fusion the
centers of the RBF surrogate are *bipolar hypervectors* (the same objects
used by the HDC side).  The surrogate predicts a scalar reward for any
hypervector `x`.  For a given query hypervector `Q` and a candidate action
hypervector `A` we compute three quantities:

1. **Kernel reward** `r̂ = surrogate.predict(A)` – Gaussian RBF evaluation.
2. **Sparse dot‑product similarity** `σ = ⟨Q, A_mask⟩` where `A_mask` is
   the result of a winner‑take‑all mask applied to `A`.
3. **Structural similarity** `s = SSIM(Q, A)` which is injected as the
   uncertainty factor `ε = sqrt((1/(2·N))·ln(2/δ))` of a Hoeffding bound.

The fused priority used by the bandit decision rule is

    priority = r̂ · σ · (1 - ε) · s

Thus the RBF surrogate guides exploitation, the sparse HDC dot product
encodes contextual relevance, and the SSIM‑scaled Hoeffding term provides
a principled exploration‑exploitation balance.

The module below implements the combined mathematics and provides three
core functions demonstrating the hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Primitive utilities shared by both parents
# ----------------------------------------------------------------------


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def random_vector(dim: int = 10000, seed: str | int | None = None) -> List[int]:
    """Generate a bipolar hypervector (values -1 or +1)."""
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> List[int]:
    """Deterministic bipolar vector derived from a string."""
    seed = int.from_bytes(
        __import__("hashlib").sha256(symbol.encode("utf-8")).digest()[:8], "big"
    )
    return random_vector(dim, seed)


def bind_bipolar(a: List[int], b: List[int]) -> List[int]:
    """Element‑wise multiplication (XOR‑like bind for bipolar vectors)."""
    return [x * y for x, y in zip(a, b)]


def top_k_mask(vec: np.ndarray, k: int) -> np.ndarray:
    """Return a mask that keeps the `k` largest‑absolute‑value entries."""
    if k <= 0:
        raise ValueError("k must be positive")
    if k > vec.size:
        k = vec.size
    # indices of the k largest absolute values
    idx = np.argpartition(np.abs(vec), -k)[-k:]
    mask = np.zeros_like(vec, dtype=bool)
    mask[idx] = True
    return mask


def ssim(x: np.ndarray, y: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """Simplified SSIM for bipolar vectors (values -1/1)."""
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator if denominator != 0 else 0.0


def hoeffding_epsilon(N: int, delta: float = 0.05) -> float:
    """Hoeffding bound uncertainty term ε."""
    if N <= 0:
        raise ValueError("N must be positive")
    return math.sqrt((1.0 / (2 * N)) * math.log(2.0 / delta))


# ----------------------------------------------------------------------
# RBF Surrogate (Parent A)
# ----------------------------------------------------------------------


@dataclass
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Sequence[float]) -> float:
        """Gaussian RBF prediction for vector `x`."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


def build_surrogate(
    data: Iterable[Tuple[Sequence[float], float]],
    epsilon: float = 1.0,
) -> RBFSurrogate:
    """Create an RBFSurrogate from (center, reward) pairs."""
    centers = []
    weights = []
    for vec, reward in data:
        centers.append(tuple(vec))
        weights.append(reward)
    return RBFSurrogate(centers=centers, weights=weights, epsilon=epsilon)


def update_surrogate(
    surrogate: RBFSurrogate,
    new_center: Sequence[float],
    new_reward: float,
) -> RBFSurrogate:
    """Return a new surrogate with an additional training sample."""
    new_centers = surrogate.centers + [tuple(new_center)]
    new_weights = surrogate.weights + [new_reward]
    return RBFSurrogate(
        centers=new_centers, weights=new_weights, epsilon=surrogate.epsilon
    )


# ----------------------------------------------------------------------
# Hybrid priority computation (fusion of both parents)
# ----------------------------------------------------------------------


def compute_priority(
    query_hv: List[int],
    action_hv: List[int],
    surrogate: RBFSurrogate,
    N: int,
    delta: float = 0.05,
    wta_k: int = 100,
) -> float:
    """
    Fuse RBF reward, sparse HDC similarity and SSIM‑scaled Hoeffding bound.

    Parameters
    ----------
    query_hv : List[int]
        Bipolar query hypervector (context).
    action_hv : List[int]
        Bipolar action hypervector.
    surrogate : RBFSurrogate
        Trained RBF model that predicts a scalar reward from a hypervector.
    N : int
        Dimensionality of the hypervectors (used for ε).
    delta : float
        Confidence parameter for Hoeffding bound.
    wta_k : int
        Number of dimensions kept by the winner‑take‑all mask.

    Returns
    -------
    float
        The fused priority score.
    """
    # 1. Kernel reward prediction
    r_hat = surrogate.predict(action_hv)

    # 2. Sparse dot‑product similarity
    a_np = np.array(action_hv, dtype=np.int8)
    mask = top_k_mask(a_np, wta_k)
    a_masked = a_np * mask  # zero out non‑selected dimensions
    q_np = np.array(query_hv, dtype=np.int8)
    sigma = float(np.dot(q_np, a_masked))

    # 3. Structural similarity (SSIM) and Hoeffding ε
    s = ssim(q_np.astype(float), a_np.astype(float))
    epsilon = hoeffding_epsilon(N, delta)

    # 4. Fused priority
    priority = r_hat * sigma * (1.0 - epsilon) * s
    return priority


def select_best_action(
    query_hv: List[int],
    actions: List[Tuple[str, List[int]]],
    surrogate: RBFSurrogate,
    N: int,
    delta: float = 0.05,
    wta_k: int = 100,
) -> Tuple[str, float]:
    """
    Evaluate all candidate actions and return the identifier of the
    action with the highest fused priority together with that priority.
    """
    best_id = None
    best_score = -math.inf
    for act_id, act_hv in actions:
        score = compute_priority(
            query_hv, act_hv, surrogate, N, delta=delta, wta_k=wta_k
        )
        if score > best_score:
            best_score = score
            best_id = act_id
    if best_id is None:
        raise RuntimeError("No actions provided")
    return best_id, best_score


def social_interaction(
    x: List[int],
    g_best: List[int],
    k: int = 1,
    r1: float | None = None,
    seed: int | str | None = None,
) -> np.ndarray:
    """
    Simple social‑interaction operator borrowed from Parent A.
    It nudges vector `x` toward the global best `g_best` using a random
    coefficient `r`.  The result is a new hypervector that can be fed back
    into the surrogate or the HDC pipeline.
    """
    if len(x) != len(g_best):
        raise ValueError("x and g_best must share dimension")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    rng = random.Random(seed)
    r = rng.random() if r1 is None else r1
    # Linear interpolation in bipolar space, followed by sign binarisation
    new_vec = [
        1 if (1 - r) * xi + r * gi >= 0 else -1 for xi, gi in zip(x, g_best)
    ]
    return np.array(new_vec, dtype=np.int8)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    DIM = 4096  # dimensionality for the hypervectors
    NUM_ACTIONS = 5

    # Create a query hypervector (e.g., representing current context)
    query = random_vector(DIM, seed="query")

    # Generate candidate actions with deterministic symbols
    actions = [
        (f"act_{i}", symbol_vector(f"action_{i}", dim=DIM))
        for i in range(NUM_ACTIONS)
    ]

    # Simulated reward data for surrogate training:
    # reward = dot(query, action) + Gaussian noise
    training_data = []
    for _, act_hv in actions:
        noisy_reward = float(np.dot(query, act_hv)) + random.gauss(0, 10)
        training_data.append((act_hv, noisy_reward))

    # Build the RBF surrogate
    surrogate = build_surrogate(training_data, epsilon=0.001)

    # Choose the best action according to the hybrid priority
    best_id, best_score = select_best_action(
        query_hv=query,
        actions=actions,
        surrogate=surrogate,
        N=DIM,
        delta=0.05,
        wta_k=200,
    )
    print(f"Selected action: {best_id} with priority {best_score:.4f}")

    # Demonstrate social interaction update for the selected action
    selected_hv = dict(actions)[best_id]
    g_best = selected_hv  # pretend this is the global best
    updated_vec = social_interaction(selected_hv, g_best, k=1, seed=42)

    # Update surrogate with a new observation after interaction
    new_reward = float(np.dot(query, updated_vec)) + random.gauss(0, 5)
    surrogate = update_surrogate(surrogate, updated_vec, new_reward)

    # Re‑evaluate to ensure the pipeline still works
    _, new_score = select_best_action(
        query_hv=query,
        actions=actions,
        surrogate=surrogate,
        N=DIM,
        delta=0.05,
        wta_k=200,
    )
    print(f"New priority after update: {new_score:.4f}")

    sys.exit(0)