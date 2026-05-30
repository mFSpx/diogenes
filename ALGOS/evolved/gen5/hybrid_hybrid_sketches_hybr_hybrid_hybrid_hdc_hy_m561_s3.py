# DARWIN HAMMER — match 561, survivor 3
# gen: 5
# parent_a: hybrid_sketches_hybrid_bandit_router_m31_s1.py (gen2)
# parent_b: hybrid_hybrid_hdc_hybrid_hy_hybrid_hybrid_rbf_su_m182_s3.py (gen4)
# born: 2026-05-29T23:29:49Z

import numpy as np
import hashlib
import random
import math
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Tuple, Any

# ----------------------------------------------------------------------
# Core data structures
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
    context_id: str
    action_id: str
    reward: float
    propensity: float

# ----------------------------------------------------------------------
# Simple in‑memory policy tracker
# ----------------------------------------------------------------------
_POLICY: Dict[str, Tuple[float, int]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        total, cnt = _POLICY.get(u.action_id, (0.0, 0))
        _POLICY[u.action_id] = (total + float(u.reward), cnt + 1)

def _reward(a: str) -> float:
    total, cnt = _POLICY.get(a, (0.0, 0))
    return total / cnt if cnt else 0.0

# ----------------------------------------------------------------------
# Count‑Min Sketch with numpy backing
# ----------------------------------------------------------------------
class CountMinSketch:
    """A lightweight Count‑Min Sketch using SHA‑256 as hash functions."""
    def __init__(self, width: int = 64, depth: int = 4, seed: int = 0):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.seed = seed

    def _hash(self, item: str, d: int) -> int:
        h = hashlib.sha256(f'{self.seed}:{d}:{item}'.encode()).hexdigest()
        return int(h, 16) % self.width

    def add(self, item: str, count: int = 1) -> None:
        for d in range(self.depth):
            idx = self._hash(item, d)
            self.table[d, idx] += count

    def estimate(self, item: str) -> int:
        """Return the minimum count across hash tables (standard CMS query)."""
        return min(self.table[d, self._hash(item, d)] for d in range(self.depth))

    def bulk_add(self, items: List[str]) -> None:
        for it in items:
            self.add(it)

# ----------------------------------------------------------------------
# Random projection utilities
# ----------------------------------------------------------------------
def _projection_matrix(dim: int, width: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Gaussian random projection (mean 0, var 1)
    return rng.standard_normal(size=(width, dim))

_PROJ_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}

def get_projection(dim: int, width: int, seed: int = 42) -> np.ndarray:
    key = (dim, width, seed)
    if key not in _PROJ_CACHE:
        _PROJ_CACHE[key] = _projection_matrix(dim, width, seed)
    return _PROJ_CACHE[key]

# ----------------------------------------------------------------------
# Symbolic hyper‑vector generation (binary ±1)
# ----------------------------------------------------------------------
def symbol_vector(symbol: str, dim: int = 10000) -> np.ndarray:
    h = hashlib.sha256(symbol.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    return np.where(rng.integers(0, 2, size=dim) == 0, -1, 1).astype(np.int8)

# ----------------------------------------------------------------------
# Binding and bundling for binary hyper‑vectors
# ----------------------------------------------------------------------
def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b

def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("at least one vector required")
    stacked = np.stack(vectors, axis=0)
    sums = stacked.sum(axis=0)
    return np.where(sums >= 0, 1, -1).astype(np.int8)

# ----------------------------------------------------------------------
# RBF surrogate
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

@dataclass
class RBFSurrogate:
    centers: List[np.ndarray]          # shape: (n_centers, dim)
    weights: np.ndarray                # shape: (n_centers,)
    epsilon: float = 1.0

    def predict(self, x: np.ndarray) -> float:
        return float(
            sum(
                w * gaussian(euclidean(x, c), self.epsilon)
                for w, c in zip(self.weights, self.centers)
            )
        )

# ----------------------------------------------------------------------
# Modulation of surrogate using CMS‑derived hyper‑vector
# ----------------------------------------------------------------------
def modulate_surrogate(surrogate: RBFSurrogate, modulation_vec: np.ndarray) -> RBFSurrogate:
    """
    Apply element‑wise binding between each centre and the modulation vector.
    The weights are left unchanged – modulation influences the geometry of the
    centres rather than their scaling.
    """
    if modulation_vec.ndim != 1:
        raise ValueError("modulation vector must be one‑dimensional")
    new_centers = [bind(c, modulation_vec.astype(c.dtype)) for c in surrogate.centers]
    return RBFSurrogate(new_centers, surrogate.weights.copy(), surrogate.epsilon)

# ----------------------------------------------------------------------
# Hybrid modulation vector construction
# ----------------------------------------------------------------------
def hybrid_modulation_vector(
    items: List[str],
    width: int = 64,
    depth: int = 4,
    dim: int = 10000,
    seed: int = 7
) -> np.ndarray:
    """
    1. Build a Count‑Min Sketch from the supplied items.
    2. Query the sketch for each distinct item to obtain a frequency estimate.
    3. Form a dense frequency vector of length `width` by aggregating estimates
       per hash bucket (the “sketch fingerprint”).
    4. Project this fingerprint into a high‑dimensional binary space via a
       random Gaussian matrix followed by a sign operation.
    """
    cms = CountMinSketch(width, depth, seed)
    cms.bulk_add(items)

    # Fingerprint: sum of estimated counts per column across all depth rows
    fingerprint = cms.table.min(axis=0).astype(np.float32)   # shape (width,)

    # Random projection → real‑valued vector
    proj = get_projection(dim, width, seed)
    projected = fingerprint @ proj                               # shape (dim,)

    # Binarise to obtain a hyper‑vector (±1)
    modulation = np.where(projected >= 0, 1, -1).astype(np.int8)
    return modulation

# ----------------------------------------------------------------------
# High‑level hybrid surrogate construction
# ----------------------------------------------------------------------
def hybrid_surrogate(
    items: List[str],
    centers: List[List[float]],
    weights: List[float],
    epsilon: float = 1.0,
    width: int = 64,
    depth: int = 4,
    dim: int = 10000,
    seed: int = 7
) -> RBFSurrogate:
    """
    Build an RBF surrogate whose centres are bound with a modulation vector
    derived from a Count‑Min Sketch of `items`. This creates a deeper
    mathematical coupling between the frequency‑based sketch and the kernel
    approximation.
    """
    modulation_vec = hybrid_modulation_vector(items, width, depth, dim, seed)

    # Convert centres to numpy arrays of matching dtype
    np_centers = [np.array(c, dtype=np.float32) for c in centers]
    # If centre dimension differs from `dim`, broadcast by truncation/padding
    target_dim = dim
    processed = []
    for c in np_centers:
        if c.size > target_dim:
            processed.append(c[:target_dim])
        elif c.size < target_dim:
            pad = np.zeros(target_dim - c.size, dtype=c.dtype)
            processed.append(np.concatenate([c, pad]))
        else:
            processed.append(c)

    surrogate = RBFSurrogate(processed, np.array(weights, dtype=np.float32), epsilon)
    return modulate_surrogate(surrogate, modulation_vec)

# ----------------------------------------------------------------------
# Bandit action selection with deeper surrogate integration
# ----------------------------------------------------------------------
def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    items: List[str],
    centers: List[List[float]],
    weights: List[float],
    algorithm: str = 'linucb',
    epsilon: float = 0.1,
    seed: Any = 7,
    width: int = 64,
    depth: int = 4,
    dim: int = 10000
) -> BanditAction:
    """
    Select an action using a hybrid of contextual bandit logic and the
    surrogate's predictions. The surrogate is recomputed on‑the‑fly from the
    current `items` (e.g., recent clicks) to reflect up‑to‑date modulation.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # Build surrogate once per decision
    surrogate = hybrid_surrogate(items, centers, weights, width=width, depth=depth, dim=dim, seed=seed)

    # Context vector for LinUCB‑style scoring (fallback to zero vector)
    ctx_vec = np.array([context.get(k, 0.0) for k in sorted(context.keys())], dtype=np.float32)
    if ctx_vec.size == 0:
        ctx_vec = np.zeros(1, dtype=np.float32)

    # Compute a score for each action: surrogate prediction + exploration bonus
    scores = {}
    for a in actions:
        # Simple encoding of action name into a numeric vector via hash
        h = int(hashlib.sha256(a.encode()).hexdigest(), 16)
        action_feat = np.full_like(ctx_vec, h % 101 / 100.0)  # deterministic pseudo‑feature
        pred = surrogate.predict(action_feat)
        exploration = epsilon * rng.random()
        scores[a] = _reward(a) + pred + exploration

    chosen = max(scores, key=scores.get)
    return BanditAction(
        action_id=chosen,
        propensity=1.0,
        expected_reward=_reward(chosen),
        confidence_bound=epsilon,
        algorithm=algorithm
    )

# ----------------------------------------------------------------------
# Example usage (executed only when run as script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample data
    items = ["item1", "item2", "item3", "item2", "item1"]
    centers = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    weights = [0.5, 0.5]

    # Build surrogate and make a prediction
    sur = hybrid_surrogate(items, centers, weights, dim=128)
    test_vec = np.array([0.2, 0.1, 0.3], dtype=np.float32)
    print("Surrogate prediction:", sur.predict(test_vec))

    # Bandit selection
    actions = ["actionA", "actionB"]
    ctx = {"feature1": 1.0, "feature2": 0.5}
    chosen = hybrid_select_action(ctx, actions, items, centers, weights, algorithm='epsilon_greedy')
    print("Chosen action:", chosen)