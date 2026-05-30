# DARWIN HAMMER — match 2649, survivor 4
# gen: 5
# parent_a: bandit_router.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s6.py (gen4)
# born: 2026-05-29T23:43:18Z

"""Hybrid Bandit‑RBF Router.

This module fuses the two parent algorithms:

* **Parent A** provides a classic contextual bandit router that estimates an
  empirical reward 𝑟̂ₐ = total_rewardₐ / countₐ and a simple confidence term.
* **Parent B** supplies an RBF surrogate model that, given a context vector
  **x**, predicts a reward ŷ(x)=∑ᵢ wᵢ·K(‖x−cᵢ‖) with a Gaussian kernel K.

The mathematical bridge is the *reward estimator*.
Both parents output a scalar estimate of the expected reward for an action.
We therefore construct a hybrid estimator

    𝑟̃ₐ(x) = α·𝑟̂ₐ   +   (1−α)·ŷₐ(x)

where α∈[0,1] balances empirical statistics and the surrogate prediction.
The hybrid estimator is then plugged into the same selection logic
(LinUCB‑style, Thompson, or ε‑greedy) used by Parent A.

The resulting system preserves the lightweight bandit bookkeeping while
leveraging a non‑parametric RBF model that can capture smooth context‑reward
relationships."""


from __future__ import annotations
import math, random, sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Sequence, Tuple, Callable, Any
import numpy as np


# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = Sequence[float]


# ----------------------------------------------------------------------
# Bandit core (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str          # identifier linking to a stored context vector
    action_id: str
    reward: float
    propensity: float


_POLICY: Dict[str, List[float]] = {}          # action_id → [total_reward, count]
_CONTEXT_STORE: Dict[str, Vector] = {}        # context_id → vector representation
_SURROGATE: Dict[str, "RBFSurrogate"] = {}    # action_id → surrogate model


def reset_hybrid() -> None:
    """Clear all learned statistics, stored contexts and surrogate models."""
    _POLICY.clear()
    _CONTEXT_STORE.clear()
    _SURROGATE.clear()


def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def _context_vector(context: Dict[str, float]) -> List[float]:
    """Deterministically turn a dict into a dense vector (sorted keys)."""
    return [float(context[k]) for k in sorted(context.keys())]


# ----------------------------------------------------------------------
# RBF surrogate (from Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel exp(-(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class RBFSurrogate:
    """Thin RBF interpolant for a single action."""
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        self.centers: List[Vector] = []   # stored context vectors
        self.weights: List[float] = []    # solved coefficients

    def _kernel_matrix(self, X: List[Vector]) -> np.ndarray:
        n = len(X)
        K = np.empty((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                K[i, j] = gaussian(euclidean(X[i], X[j]), self.epsilon)
        return K

    def fit(self, X: List[Vector], y: List[float]) -> None:
        """Fit the surrogate by solving K·w = y."""
        if not X:
            self.centers = []
            self.weights = []
            return
        K = self._kernel_matrix(X)
        try:
            w = np.linalg.solve(K, np.array(y, dtype=float))
        except np.linalg.LinAlgError:
            # fall back to least‑squares if K is singular
            w, *_ = np.linalg.lstsq(K, np.array(y, dtype=float), rcond=None)
        self.centers = X
        self.weights = w.tolist()

    def predict(self, x: Vector) -> float:
        """Predict using the current set of centers and weights."""
        if not self.centers:
            return 0.0
        contributions = [
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        ]
        return float(sum(contributions))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def update_hybrid(updates: List[BanditUpdate],
                  contexts: Dict[str, Dict[str, float]]) -> None:
    """
    Incorporate a batch of bandit feedback.

    * updates  – list of BanditUpdate objects.
    * contexts – mapping from context_id to raw context dicts.
    """
    # 1. Update empirical statistics
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

    # 2. Store raw contexts (if not already stored)
    for cid, ctx in contexts.items():
        if cid not in _CONTEXT_STORE:
            _CONTEXT_STORE[cid] = _context_vector(ctx)

    # 3. Re‑fit per‑action RBF surrogates
    #    Gather (context, reward) pairs per action.
    per_action: Dict[str, List[Tuple[Vector, float]]] = {}
    for u in updates:
        vec = _CONTEXT_STORE.get(u.context_id)
        if vec is None:
            continue
        per_action.setdefault(u.action_id, []).append((vec, float(u.reward)))

    for action, pairs in per_action.items():
        X, y = zip(*pairs)
        surrogate = _SURROGATE.get(action)
        if surrogate is None:
            surrogate = RBFSurrogate(epsilon=1.0)
            _SURROGATE[action] = surrogate
        surrogate.fit(list(X), list(y))


def _hybrid_estimate(action: str,
                     context_vec: Vector,
                     alpha: float) -> Tuple[float, float]:
    """
    Return (expected_reward, confidence_bound) for *action* given *context_vec*.

    expected_reward = α·empirical + (1‑α)·surrogate_prediction
    confidence_bound follows the LinUCB‑style term from Parent A.
    """
    empirical = _empirical_reward(action)
    surrogate = _SURROGATE.get(action)
    pred = surrogate.predict(context_vec) if surrogate else 0.0
    expected = alpha * empirical + (1.0 - alpha) * pred

    count = _POLICY.get(action, [0.0, 0.0])[1]
    confidence = 1.0 / math.sqrt(1.0 + count)   # same as Parent A
    return expected, confidence


def select_hybrid_action(context: Dict[str, float],
                         actions: List[str],
                         algorithm: str = 'linucb',
                         epsilon: float = 0.1,
                         alpha: float = 0.5,
                         seed: int | str | None = 7) -> BanditAction:
    """
    Choose an action using the hybrid reward estimator.

    * algorithm – 'linucb', 'epsilon_greedy', or 'thompson'.
    * epsilon   – exploration probability for ε‑greedy.
    * alpha     – weight of empirical reward (0 ≤ α ≤ 1).
    """
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)

    ctx_vec = _context_vector(context)

    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
        exp, conf = _hybrid_estimate(chosen, ctx_vec, alpha)
        return BanditAction(chosen,
                            1.0 / len(actions),
                            exp,
                            conf,
                            algorithm)

    if algorithm == 'thompson':
        # Sample from a Beta distribution whose parameters are nudged by the hybrid estimate.
        def sample(a: str) -> float:
            emp = _empirical_reward(a)
            surrogate = _SURROGATE.get(a)
            pred = surrogate.predict(ctx_vec) if surrogate else 0.0
            hybrid = alpha * emp + (1.0 - alpha) * pred
            # map hybrid∈[0,1] into Beta parameters; add 1 to avoid zeros.
            return rng.betavariate(1 + max(0.0, hybrid), 1 + max(0.0, 1.0 - hybrid))
        chosen = max(actions, key=sample)
        exp, conf = _hybrid_estimate(chosen, ctx_vec, alpha)
        return BanditAction(chosen,
                            1.0 / len(actions),
                            exp,
                            conf,
                            algorithm)

    # Default: LinUCB‑style scoring
    scale = math.sqrt(sum(v * v for v in context.values())) if context else 1.0
    def score(a: str) -> float:
        exp, _ = _hybrid_estimate(a, ctx_vec, alpha)
        count = _POLICY.get(a, [0.0, 0.0])[1]
        ucb = exp + 0.1 * scale / math.sqrt(1.0 + count)
        return ucb

    chosen = max(actions, key=score)
    exp, conf = _hybrid_estimate(chosen, ctx_vec, alpha)
    return BanditAction(chosen,
                        1.0 / len(actions),
                        exp,
                        conf,
                        algorithm)


def predict_reward(action: str,
                   context: Dict[str, float]) -> float:
    """
    Public helper: return the hybrid expected reward for *action*.
    """
    ctx_vec = _context_vector(context)
    exp, _ = _hybrid_estimate(action, ctx_vec, alpha=0.5)
    return exp


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny problem
    actions = ["a1", "a2", "a3"]
    # Simulated contexts (feature space 2‑D)
    contexts = {
        "c1": {"x1": 0.1, "x2": 0.3},
        "c2": {"x1": 0.4, "x2": 0.2},
        "c3": {"x1": 0.9, "x2": 0.7},
    }

    # Generate synthetic updates
    updates = [
        BanditUpdate(context_id="c1", action_id="a1", reward=1.0, propensity=1/3),
        BanditUpdate(context_id="c2", action_id="a2", reward=0.0, propensity=1/3),
        BanditUpdate(context_id="c3", action_id="a1", reward=1.0, propensity=1/3),
        BanditUpdate(context_id="c1", action_id="a3", reward=0.5, propensity=1/3),
    ]

    reset_hybrid()
    update_hybrid(updates, contexts)

    # Query selection for a new context
    new_context = {"x1": 0.5, "x2": 0.4}
    chosen = select_hybrid_action(new_context, actions,
                                  algorithm='linucb',
                                  epsilon=0.2,
                                  alpha=0.6,
                                  seed=42)
    print("Chosen action:", asdict(chosen))

    # Predict reward for each action
    for a in actions:
        print(f"Hybrid reward estimate for {a}:", predict_reward(a, new_context))

    sys.exit(0)