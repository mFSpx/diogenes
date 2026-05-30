# DARWIN HAMMER — match 255, survivor 0
# gen: 4
# parent_a: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s6.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_workshare_allocator_m60_s1.py (gen2)
# born: 2026-05-29T23:27:54Z

"""Hybrid Perceptual‑Hash RBF + Store‑Modulated Bandit

Parents
-------
* **Parent A** – ``hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s6.py``  
  Provides a perceptual hash that clusters high‑dimensional vectors and,
  inside each cluster, fits an independent Radial‑Basis‑Function (RBF)
  surrogate model.

* **Parent B** – ``hybrid_hybrid_bandit_router_workshare_allocator_m60_s1.py``  
  Supplies a contextual multi‑armed bandit whose exploration/exploitation
  balance is driven by a ``StoreState`` (honey‑bee‑style store).  The store
  yields a scalar *dance* that can be used as a control signal.

Mathematical Bridge
-------------------
The bridge is the *store dance* that modulates the RBF kernel width
(``ε``) for the cluster that currently handles a datum.  Concretely, for a
vector ``x`` belonging to hash‑cluster ``c`` with centre matrix ``X_c`` we
define


ε_c = ε₀ · (1 + store.dance)
K_c(x, X_c) = exp(−ε_c · ‖x − X_c‖²)


The RBF surrogate output ``f̂_c(x) = w_cᵀ K_c(x, X_c)`` is then used as the
reward signal for a bandit whose context is the cluster identifier.  The
bandit selects an action, the store updates its level based on the received
reward, and the updated ``dance`` immediately influences the next kernel
width.  Thus the two topologies are fused into a single adaptive system.

The module below implements this hybrid algorithm, exposing three core
functions that demonstrate the combined behaviour.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Parent A – perceptual hash utilities
# ----------------------------------------------------------------------


def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of a numeric sequence.

    A bit is set to 1 when the corresponding value is greater‑or‑equal to the
    mean of the (first 64) values.
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def cluster_by_phash(hashes: Dict[str, int], max_distance: int = 4) -> List[List[str]]:
    """Group keys whose hashes are within ``max_distance`` Hamming distance.

    The algorithm is greedy: each key is placed into the first existing
    cluster whose representative (the first element) is close enough; otherwise
    a new cluster is started.
    """
    clusters: List[List[str]] = []
    for key, h in hashes.items():
        placed = False
        for cluster in clusters:
            rep_key = cluster[0]
            if hamming_distance(h, hashes[rep_key]) <= max_distance:
                cluster.append(key)
                placed = True
                break
        if not placed:
            clusters.append([key])
    return clusters


# ----------------------------------------------------------------------
# Parent A – RBF surrogate utilities
# ----------------------------------------------------------------------


def rbf_kernel(x: np.ndarray, centers: np.ndarray, epsilon: float) -> np.ndarray:
    """Compute RBF kernel vector K_i = exp(-ε·‖x‑μ_i‖²)."""
    diff = centers - x  # shape (n_centers, d)
    sq_norm = np.einsum("ij,ij->i", diff, diff)
    return np.exp(-epsilon * sq_norm)


def fit_rbf_weights(
    X: np.ndarray, y: np.ndarray, epsilon: float, reg: float = 1e-8
) -> np.ndarray:
    """Solve (K + reg·I) w = y for RBF weights."""
    K = np.exp(-epsilon * np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2) ** 2)
    A = K + reg * np.eye(K.shape[0])
    return np.linalg.solve(A, y)


def rbf_predict(x: np.ndarray, centers: np.ndarray, weights: np.ndarray, epsilon: float) -> float:
    """Predict using a fitted RBF model."""
    k = rbf_kernel(x, centers, epsilon)
    return float(k @ weights)


# ----------------------------------------------------------------------
# Parent B – Store dynamics
# ----------------------------------------------------------------------


@dataclass
class StoreState:
    """Honey‑bee style store that produces a bounded control signal."""

    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply the store equation and recompute the dance duration."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._store_last_delta(delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the most recent Δ."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))

    def _store_last_delta(self, delta: float) -> None:
        self._last_delta = delta


# ----------------------------------------------------------------------
# Parent B – Bandit utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""

    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridBandit"


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""

    context_id: str
    action_id: str
    reward: float
    propensity: float


class ContextualBandit:
    """Simple epsilon‑greedy bandit with per‑context statistics."""

    def __init__(self, actions: List[str], epsilon: float = 0.1):
        self.epsilon = epsilon
        self.actions = actions
        # stats: context → action → (sum_reward, count)
        self._stats: Dict[str, Dict[str, Tuple[float, int]]] = {}

    def _ensure_context(self, ctx: str) -> None:
        if ctx not in self._stats:
            self._stats[ctx] = {a: (0.0, 0) for a in self.actions}

    def select(self, context_id: str) -> BanditAction:
        self._ensure_context(context_id)
        if random.random() < self.epsilon:
            # explore uniformly
            a_id = random.choice(self.actions)
        else:
            # exploit: pick action with highest empirical mean
            means = {
                a: (s[0] / s[1] if s[1] > 0 else 0.0)
                for a, s in self._stats[context_id].items()
            }
            a_id = max(means, key=means.get)
        sum_r, cnt = self._stats[context_id][a_id]
        exp_reward = sum_r / cnt if cnt > 0 else 0.0
        confidence = (
            1.0 / math.sqrt(cnt + 1)  # simple confidence proxy
        )
        propensity = self.epsilon / len(self.actions) + (1 - self.epsilon) * (1.0 if cnt == 0 else 0.0)
        return BanditAction(
            action_id=a_id,
            propensity=propensity,
            expected_reward=exp_reward,
            confidence_bound=confidence,
        )

    def update(self, upd: BanditUpdate) -> None:
        self._ensure_context(upd.context_id)
        sum_r, cnt = self._stats[upd.context_id][upd.action_id]
        self._stats[upd.context_id][upd.action_id] = (sum_r + upd.reward, cnt + 1)


# ----------------------------------------------------------------------
# Hybrid data structure per hash‑cluster
# ----------------------------------------------------------------------


@dataclass
class HybridClusterModel:
    """Holds an RBF surrogate and a bandit policy for a specific hash cluster."""

    centers: np.ndarray                # shape (n_samples, d)
    weights: np.ndarray                # shape (n_samples,)
    epsilon0: float                    # base kernel width
    bandit: ContextualBandit
    store: StoreState

    def predict(self, x: np.ndarray) -> float:
        """RBF prediction using the current store‑modulated epsilon."""
        eps = self.epsilon0 * (1.0 + self.store.dance)
        return rbf_predict(x, self.centers, self.weights, eps)

    def update_rbf(self, x: np.ndarray, y: float) -> None:
        """Incrementally refit the RBF model with a new sample."""
        # Append the new sample
        self.centers = np.vstack([self.centers, x])
        self.weights = np.append(self.weights, y)  # temporary placeholder
        # Re‑fit weights with updated epsilon
        eps = self.epsilon0 * (1.0 + self.store.dance)
        self.weights = fit_rbf_weights(self.centers, np.append(self.weights, y), eps)

    def step(self, x: np.ndarray, y: float) -> Tuple[float, BanditAction]:
        """Full hybrid step: predict, select action, receive reward, update all."""
        pred = self.predict(x)
        action = self.bandit.select(context_id=str(id(self)))
        # Use the true target y as the reward (could be any function of pred)
        reward = -abs(pred - y)  # negative absolute error as reward
        # Update store with reward as inflow
        self.store.update(inflow=[reward], outflow=[])
        # Update bandit with observed reward
        self.bandit.update(
            BanditUpdate(
                context_id=str(id(self)),
                action_id=action.action_id,
                reward=reward,
                propensity=action.propensity,
            )
        )
        # Optionally refine RBF (here we simply add the point)
        self.update_rbf(x, y)
        return pred, action


# ----------------------------------------------------------------------
# Fusion functions
# ----------------------------------------------------------------------


def build_hybrid_models(
    data: Dict[str, Vector],
    targets: Dict[str, float],
    epsilon0: float = 0.01,
    max_phash_distance: int = 4,
) -> Dict[str, HybridClusterModel]:
    """Create a HybridClusterModel for each perceptual‑hash cluster.

    Returns a mapping from cluster identifier (string of joined keys) to the
    corresponding model.
    """
    # 1. Compute hashes
    hashes = {k: compute_phash(list(v)) for k, v in data.items()}
    # 2. Cluster by Hamming distance
    raw_clusters = cluster_by_phash(hashes, max_distance=max_phash_distance)
    # 3. Build models
    models: Dict[str, HybridClusterModel] = {}
    for idx, cluster_keys in enumerate(raw_clusters):
        # Gather vectors and targets
        X = np.array([data[k] for k in cluster_keys])
        y = np.array([targets[k] for k in cluster_keys])
        # Fit initial RBF weights
        eps = epsilon0
        w = fit_rbf_weights(X, y, eps)
        # Initialise bandit with three dummy actions
        bandit = ContextualBandit(actions=["A", "B", "C"], epsilon=0.2)
        store = StoreState()
        cluster_id = f"cluster_{idx}"
        models[cluster_id] = HybridClusterModel(
            centers=X,
            weights=w,
            epsilon0=epsilon0,
            bandit=bandit,
            store=store,
        )
    return models


def hybrid_predict(
    key: str,
    vector: Vector,
    models: Dict[str, HybridClusterModel],
    hashes: Dict[str, int],
    max_phash_distance: int = 4,
) -> Tuple[float, BanditAction]:
    """Predict for a single datum using the appropriate cluster model.

    The function finds the nearest hash cluster, runs the hybrid step and
    returns the surrogate prediction together with the selected bandit action.
    """
    # Compute hash for the query
    q_hash = compute_phash(list(vector))
    # Find the first cluster whose representative is within distance
    chosen_model = None
    for cid, model in models.items():
        # Representative hash: first key of the cluster (stored in hashes)
        # We retrieve it via the first key that maps to this model (reverse lookup)
        # For simplicity we store a mapping from model id to representative hash
        # during construction; here we recompute by scanning.
        rep_key = next(k for k, v in hashes.items() if v == hashes.get(cid.split("_")[0], q_hash))
        if hamming_distance(q_hash, hashes[rep_key]) <= max_phash_distance:
            chosen_model = model
            break
    if chosen_model is None:
        # fallback: use any model (e.g., first)
        chosen_model = next(iter(models.values()))
    x = np.array(vector)
    pred, action = chosen_model.step(x, targets.get(key, 0.0))
    return pred, action


def hybrid_update(
    key: str,
    vector: Vector,
    true_value: float,
    models: Dict[str, HybridClusterModel],
    hashes: Dict[str, int],
    max_phash_distance: int = 4,
) -> None:
    """Explicitly update the hybrid system with a new observation."""
    # The update logic mirrors ``hybrid_predict`` but discards the return values.
    q_hash = compute_phash(list(vector))
    chosen_model = None
    for cid, model in models.items():
        rep_key = next(k for k, v in hashes.items() if v == hashes.get(cid.split("_")[0], q_hash))
        if hamming_distance(q_hash, hashes[rep_key]) <= max_phash_distance:
            chosen_model = model
            break
    if chosen_model is None:
        chosen_model = next(iter(models.values()))
    x = np.array(vector)
    # Run a hybrid step – the internal update mechanisms already adjust everything.
    chosen_model.step(x, true_value)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Synthetic dataset
    random.seed(42)
    np.random.seed(42)
    data: Dict[str, Vector] = {}
    targets: Dict[str, float] = {}
    for i in range(30):
        vec = np.random.randn(10).tolist()
        key = f"sample_{i}"
        data[key] = vec
        targets[key]