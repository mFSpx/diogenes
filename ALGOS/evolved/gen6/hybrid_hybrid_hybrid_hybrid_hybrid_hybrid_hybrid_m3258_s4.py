# DARWIN HAMMER — match 3258, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1136_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m1347_s0.py (gen5)
# born: 2026-05-29T23:48:55Z

"""
Hybrid Regret‑RBF Bandit
========================

This module fuses the two parent algorithms:

* **Parent A** – MinHash‑based similarity and regret‑matching with a
  temperature‑scaled soft‑max selector.
* **Parent B** – Radial‑basis‑function (RBF) surrogate model for expected
  rewards, pheromone‑biased selection and Count‑Min sketches for
  reward statistics.

**Mathematical bridge**

The bridge is the *similarity‑weighted RBF kernel*.  For a given
context we compute a MinHash signature and obtain a Jaccard‑like
similarity to the stored contexts.  That similarity is used as the
amplitude (latent variable) of the Gaussian RBF centred at each stored
context:


    φ_i(x) = sim(sig(x), sig(c_i)) * exp( -ε * ||x - c_i||² )


The surrogate prediction `ĥ(x) = Σ_i w_i φ_i(x)` supplies an
*expected reward estimate* that drives the regret‑matching update.
Thus the information‑density weighting of the Fisher‑style score in the
original parents is realised through the similarity‑weighted RBF
kernel.

The hybrid algorithm therefore proceeds as:

1. Encode a new context as a MinHash signature.
2. Evaluate the similarity‑weighted RBF surrogate to obtain an
   expected reward.
3. Update regrets using a regret‑matching rule.
4. Select the next action via a temperature‑scaled soft‑max over
   accumulated regrets.

All operations rely only on ``numpy`` and the Python standard library.
"""

import hashlib
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# MinHash utilities (Parent A)
# ----------------------------------------------------------------------


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        hashlib.blake2b(data, digest_size=8).digest(), "big"
    )


def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """
    Compute a MinHash signature of length ``k`` for a set of tokens.
    Empty tokens are ignored.
    """
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        # maximal hash value for all positions → zero similarity with anything
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """
    Jaccard‑like similarity between two MinHash signatures.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# RBF surrogate utilities (Parent B)
# ----------------------------------------------------------------------


Vector = Sequence[float]


def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF: exp( - (ε·r)² )."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def count_min_sketch(rewards: Iterable[int], width: int = 256, depth: int = 4) -> np.ndarray:
    """
    Simple Count‑Min sketch for integer rewards.
    """
    sketch = np.zeros((depth, width), dtype=int)
    for reward in rewards:
        for i in range(depth):
            h = int(
                hashlib.sha256(f"{reward}:{i}".encode()).hexdigest(),
                16,
            ) % width
            sketch[i, h] += 1
    return sketch


def estimate_mean_reward(sketch: np.ndarray) -> float:
    """Mean estimate from a Count‑Min sketch."""
    return float(np.mean(sketch))


def estimate_variance(sketch: np.ndarray) -> float:
    """Variance estimate from a Count‑Min sketch."""
    return float(np.var(sketch))


def fit_similarity_weighted_rbf(
    contexts: List[Vector],
    rewards: List[float],
    signatures: List[List[int]],
    epsilon: float = 1.0,
) -> Callable[[Vector, List[int]], float]:
    """
    Fit a surrogate that predicts reward for a new context ``x`` using
    similarity‑weighted RBFs.

    Returns a function ``predict(x, sig_x)``.
    """
    if not contexts:
        raise ValueError("at least one context required")
    if len(contexts) != len(rewards) or len(contexts) != len(signatures):
        raise ValueError("contexts, rewards and signatures must align")

    # Solve for linear weights w in least‑squares sense:
    #   Φ w = y ,  Φ_ij = sim(sig_i, sig_j) * exp(-ε*||c_i-c_j||²)
    n = len(contexts)
    phi = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            sim = jaccard_similarity(signatures[i], signatures[j])
            dist = euclidean(contexts[i], contexts[j])
            phi[i, j] = sim * gaussian_rbf(dist, epsilon)

    # Regularised least squares (ridge) to avoid singularity
    ridge = 1e-9
    w = np.linalg.solve(phi.T @ phi + ridge * np.eye(n), phi.T @ np.array(rewards))

    def predictor(x: Vector, sig_x: List[int]) -> float:
        """Predict expected reward for context ``x``."""
        phi_x = np.empty(n, dtype=float)
        for i in range(n):
            sim = jaccard_similarity(sig_x, signatures[i])
            dist = euclidean(x, contexts[i])
            phi_x[i] = sim * gaussian_rbf(dist, epsilon)
        return float(phi_x @ w)

    return predictor


# ----------------------------------------------------------------------
# Regret‑matching utilities (Parent A)
# ----------------------------------------------------------------------


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature‑scaled soft‑max."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = values / temperature
    max_val = np.max(scaled)
    exp_vals = np.exp(scaled - max_val)  # for numerical stability
    return exp_vals / np.sum(exp_vals)


def regret_update(
    regrets: defaultdict,
    action_id: str,
    reward_estimate: float,
    learning_rate: float = 0.1,
) -> None:
    """
    Regret‑matching update for a single action.

    ``regrets`` maps action identifiers to cumulative regret values.
    """
    # Positive part of regret drives probability mass
    pos_regret = max(regrets[action_id], 0.0)
    # Increment proportional to the difference between estimated reward
    # and the current average (here simplified as the current regret)
    delta = learning_rate * (reward_estimate - pos_regret)
    regrets[action_id] = regrets[action_id] + delta


def select_action(
    regrets: defaultdict,
    temperature: float = 1.0,
) -> str:
    """
    Choose an action using a soft‑max over the *positive* regrets.
    """
    actions = list(regrets.keys())
    pos_regrets = np.array([max(regrets[a], 0.0) for a in actions], dtype=float)
    probs = _softmax(pos_regrets, temperature)
    return random.choices(actions, weights=probs, k=1)[0]


# ----------------------------------------------------------------------
# Hybrid class tying everything together
# ----------------------------------------------------------------------


@dataclass
class HybridBanditRegret:
    """
    A hybrid bandit that:

    * stores contexts as vectors together with MinHash signatures,
    * builds a similarity‑weighted RBF surrogate,
    * maintains a Count‑Min sketch of observed rewards,
    * updates regrets based on surrogate predictions,
    * selects actions via soft‑max regret‑matching.
    """
    epsilon: float = 1.0
    temperature: float = 1.0
    learning_rate: float = 0.1
    sketch_width: int = 256
    sketch_depth: int = 4
    k_hash: int = 128  # MinHash length

    # internal state
    _contexts: List[Vector] = None
    _signatures: List[List[int]] = None
    _rewards: List[float] = None
    _regrets: defaultdict = None
    _sketch: np.ndarray = None
    _predictor: Callable[[Vector, List[int]], float] = None

    def __post_init__(self):
        self._contexts = []
        self._signatures = []
        self._rewards = []
        self._regrets = defaultdict(float)
        self._sketch = np.zeros((self.sketch_depth, self.sketch_width), dtype=int)
        self._predictor = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        action_id: str,
        context_tokens: Iterable[str],
        context_vector: Vector,
        reward: float,
    ) -> None:
        """
        Incorporate a new (action, context, reward) observation.
        """
        # 1. Update Count‑Min sketch
        self._sketch += count_min_sketch(
            [int(reward)], width=self.sketch_width, depth=self.sketch_depth
        )

        # 2. Store raw data
        sig = minhash_signature(context_tokens, k=self.k_hash)
        self._contexts.append(list(context_vector))
        self._signatures.append(sig)
        self._rewards.append(float(reward))
        self._regrets[action_id] = self._regrets.get(action_id, 0.0)

        # 3. Re‑fit surrogate (lightweight incremental approach)
        self._predictor = fit_similarity_weighted_rbf(
            self._contexts,
            self._rewards,
            self._signatures,
            epsilon=self.epsilon,
        )

        # 4. Estimate expected reward for the observed context
        reward_est = self._predictor(context_vector, sig)

        # 5. Regret update
        regret_update(
            self._regrets,
            action_id,
            reward_est,
            learning_rate=self.learning_rate,
        )

    def recommend(self) -> Tuple[str, float]:
        """
        Recommend the next action based on current regrets.
        Returns (action_id, selection_probability).
        """
        if not self._regrets:
            raise RuntimeError("no actions have been observed yet")
        actions = list(self._regrets.keys())
        pos_regrets = np.array(
            [max(self._regrets[a], 0.0) for a in actions], dtype=float
        )
        probs = _softmax(pos_regrets, self.temperature)
        chosen = random.choices(actions, weights=probs, k=1)[0]
        return chosen, float(probs[actions.index(chosen)])

    def reward_statistics(self) -> Tuple[float, float]:
        """
        Return (mean_estimate, variance_estimate) of observed rewards
        using the Count‑Min sketch.
        """
        return estimate_mean_reward(self._sketch), estimate_variance(self._sketch)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    random.seed(42)

    # Define a tiny synthetic problem
    actions = ["A", "B", "C"]
    tokens_per_action = {
        "A": ["apple", "red", "fruit"],
        "B": ["banana", "yellow", "fruit"],
        "C": ["carrot", "orange", "vegetable"],
    }
    vectors_per_action = {
        "A": (1.0, 0.0, 0.0),
        "B": (0.0, 1.0, 0.0),
        "C": (0.0, 0.0, 1.0),
    }

    # Simulated stochastic reward function
    def true_reward(action_id: str) -> float:
        base = {"A": 1.0, "B": 0.8, "C": 0.3}[action_id]
        return base + random.gauss(0, 0.05)

    bandit = HybridBanditRegret(epsilon=0.8, temperature=0.5, learning_rate=0.2)

    # Run a few interaction rounds
    for step in range(20):
        # Recommend an action based on current regrets
        try:
            act, prob = bandit.recommend()
        except RuntimeError:
            # First round – pick randomly
            act = random.choice(actions)
            prob = 1.0 / len(actions)

        # Observe reward
        rew = true_reward(act)

        # Feed observation back into the model
        bandit.observe(
            action_id=act,
            context_tokens=tokens_per_action[act],
            context_vector=vectors_per_action[act],
            reward=rew,
        )

        if step % 5 == 0:
            mean, var = bandit.reward_statistics()
            print(
                f"Step {step:02d} | Action {act} (p={prob:.2f}) | Reward {rew:.3f} | "
                f"Mean≈{mean:.3f} Var≈{var:.3f}"
            )

    # Final recommendation
    final_act, final_prob = bandit.recommend()
    print(f"\nFinal recommendation: {final_act} with probability {final_prob:.3f}")

    # Ensure the predictor is callable
    assert bandit._predictor is not None
    test_vec = (0.5, 0.5, 0.0)
    test_sig = minhash_signature(["apple", "banana"], k=bandit.k_hash)
    pred = bandit._predictor(test_vec, test_sig)
    print(f"Surrogate prediction for test context: {pred:.3f}")