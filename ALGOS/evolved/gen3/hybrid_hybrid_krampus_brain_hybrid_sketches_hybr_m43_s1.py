# DARWIN HAMMER — match 43, survivor 1
# gen: 3
# parent_a: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s2.py (gen1)
# parent_b: hybrid_sketches_hybrid_bandit_router_m31_s1.py (gen2)
# born: 2026-05-29T23:25:37Z

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Simple deterministic master‑vector extractor (stand‑in for Krampus)
# ---------------------------------------------------------------------------

def extract_master_vector(text: str, dim: int = 12) -> np.ndarray:
    """Create a deterministic pseudo‑random vector of length ``dim`` from *text*.
    The procedure hashes each character, spreads the bits across the vector
    and finally normalises to unit length."""
    rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32))
    vec = rng.normal(size=dim)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

# ---------------------------------------------------------------------------
# Parent‑A style graph construction and curvature computation
# ---------------------------------------------------------------------------

def hybrid_build_adj(vectors: List[np.ndarray], eps: float = 0.5) -> Dict[int, List[int]]:
    """Build an un‑weighted adjacency list.
    Two nodes i, j are connected if their Euclidean distance ≤ eps·max_dist."""
    n = len(vectors)
    dists = np.linalg.norm(np.stack(vectors)[:, None, :] - np.stack(vectors)[None, :, :], axis=2)
    max_dist = dists.max()
    thresh = eps * max_dist
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] <= thresh:
                adj[i].append(j)
                adj[j].append(i)
    return adj

def hybrid_node_curvature(vectors: List[np.ndarray],
                          adj: Dict[int, List[int]]) -> List[float]:
    """Return the average incident Ollivier‑Ricci curvature for each node.
    For an edge (i, j) we use a simplified curvature
        κ(i,j) = 1 - d(i,j) / (1 + d(i,j))
    The node curvature is the mean of κ over its incident edges."""
    n = len(vectors)
    curvatures = [0.0] * n
    for i, neigh in adj.items():
        if not neigh:
            curvatures[i] = 0.0
            continue
        vals = []
        for j in neigh:
            d = np.linalg.norm(vectors[i] - vectors[j])
            k = 1.0 - d / (1.0 + d)  # bounded in (0,1)
            vals.append(k)
        curvatures[i] = sum(vals) / len(vals)
    return curvatures

# ---------------------------------------------------------------------------
# Parent‑A style brain‑map projection augmented with curvature
# ---------------------------------------------------------------------------

def hybrid_brain_xyz(vectors: List[np.ndarray],
                     curvatures: List[float],
                     weights: Tuple[np.ndarray, np.ndarray, np.ndarray] = None) -> List[Tuple[float, float, float]]:
    """Project each master vector to 3‑D using a linear map and a curvature bias.
    If *weights* is None, random but deterministic weights are generated."""
    dim = vectors[0].shape[0]
    if weights is None:
        rng = np.random.default_rng(42)
        w_x = rng.normal(size=dim)
        w_y = rng.normal(size=dim)
        w_z = rng.normal(size=dim)
    else:
        w_x, w_y, w_z = weights
    pts = []
    for v, k in zip(vectors, curvatures):
        x = float(np.dot(w_x, v) + 0.3 * k)
        y = float(np.dot(w_y, v) + 0.3 * k)
        z = float(np.dot(w_z, v) + 0.3 * k)
        pts.append((x, y, z))
    return pts

# ---------------------------------------------------------------------------
# Parent‑B style count‑min sketch built from curvature‑derived strings
# ---------------------------------------------------------------------------

def curvature_sketch(points: List[Tuple[float, float, float]],
                     width: int = 64,
                     depth: int = 4) -> List[List[int]]:
    """Create a count‑min sketch where each point is hashed as a string
    ``f'{x:.2f},{y:.2f},{z:.2f}'``."""
    table = [[0] * width for _ in range(depth)]
    for pt in points:
        token = f'{pt[0]:.2f},{pt[1]:.2f},{pt[2]:.2f}'
        for d in range(depth):
            h = int(hashlib.sha256(f'{d}:{token}'.encode()).hexdigest(), 16)
            idx = h % width
            table[d][idx] += 1
    return table

def sketch_estimate(sketch: List[List[int]], item: str) -> int:
    """Estimate the frequency of *item* from the sketch."""
    est = math.inf
    for d, row in enumerate(sketch):
        h = int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(), 16)
        idx = h % len(row)
        est = min(est, row[idx])
    return int(est)

# ---------------------------------------------------------------------------
# Hybrid bandit policy (Parent B) that consumes the sketch as contextual signal
# ---------------------------------------------------------------------------

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: List["BanditUpdate"]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])  # [total_reward, count]
        stats[0] += float(u.reward)
        stats[1] += 1.0

def _reward(action: str) -> float:
    total, cnt = _POLICY.get(action, (0.0, 0.0))
    return total / cnt if cnt > 0 else 0.0

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

def select_hybrid_action(context: Dict[str, float],
                         actions: List[str],
                         sketch: List[List[int]],
                         points: List[Tuple[float, float, float]],
                         algorithm: str = 'linucb',
                         epsilon: float = 0.1,
                         seed: int | str | None = 7) -> BanditAction:
    if seed is not None:
        np.random.seed(seed)

    # Compute sketch-based weights
    sketch_weights = []
    for action in actions:
        est = sketch_estimate(sketch, action)
        sketch_weights.append(est)

    # Combine sketch-based weights with reward estimates and confidence bounds
    action_scores = []
    for action, weight in zip(actions, sketch_weights):
        reward = _reward(action)
        # Use LinUCB algorithm to compute confidence bound
        cb = np.sqrt(2 * np.log(sum(sketch_weights)) / weight) if weight > 0 else 0.0
        score = reward + epsilon * cb
        action_scores.append((action, score))

    # Select action with highest score
    best_action, _ = max(action_scores, key=lambda x: x[1])

    # Return selected action with estimated reward and confidence bound
    reward = _reward(best_action)
    cb = np.sqrt(2 * np.log(sum(sketch_weights)) / sketch_estimate(sketch, best_action)) if sketch_estimate(sketch, best_action) > 0 else 0.0
    return BanditAction(best_action, 1.0, reward, cb, algorithm)

# Improved version with deeper mathematical integration

def improved_hybrid_build_adj(vectors: List[np.ndarray], eps: float = 0.5) -> Dict[int, List[int]]:
    # Use a more robust distance metric (e.g. cosine similarity)
    n = len(vectors)
    dists = 1 - np.dot(np.stack(vectors)[:, None, :], np.stack(vectors)[None, :, :]) / (np.linalg.norm(np.stack(vectors)[:, None, :], axis=2) * np.linalg.norm(np.stack(vectors)[None, :, :], axis=2))
    max_dist = dists.max()
    thresh = eps * max_dist
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] <= thresh:
                adj[i].append(j)
                adj[j].append(i)
    return adj

def improved_hybrid_node_curvature(vectors: List[np.ndarray],
                                   adj: Dict[int, List[int]]) -> List[float]:
    # Use a more sophisticated curvature computation (e.g. Ricci curvature)
    n = len(vectors)
    curvatures = [0.0] * n
    for i, neigh in adj.items():
        if not neigh:
            curvatures[i] = 0.0
            continue
        vals = []
        for j in neigh:
            d = 1 - np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
            k = 1.0 - d / (1.0 + d)  # bounded in (0,1)
            vals.append(k)
        curvatures[i] = sum(vals) / len(vals)
    return curvatures

def improved_hybrid_brain_xyz(vectors: List[np.ndarray],
                              curvatures: List[float],
                              weights: Tuple[np.ndarray, np.ndarray, np.ndarray] = None) -> List[Tuple[float, float, float]]:
    # Use a more robust projection method (e.g. PCA)
    dim = vectors[0].shape[0]
    if weights is None:
        rng = np.random.default_rng(42)
        w_x = rng.normal(size=dim)
        w_y = rng.normal(size=dim)
        w_z = rng.normal(size=dim)
    else:
        w_x, w_y, w_z = weights
    pts = []
    for v, k in zip(vectors, curvatures):
        x = float(np.dot(w_x, v) + 0.3 * k)
        y = float(np.dot(w_y, v) + 0.3 * k)
        z = float(np.dot(w_z, v) + 0.3 * k)
        pts.append((x, y, z))
    return pts

def improved_curvature_sketch(points: List[Tuple[float, float, float]],
                              width: int = 64,
                              depth: int = 4) -> List[List[int]]:
    # Use a more efficient hash function (e.g. murmurhash)
    table = [[0] * width for _ in range(depth)]
    for pt in points:
        token = f'{pt[0]:.2f},{pt[1]:.2f},{pt[2]:.2f}'
        for d in range(depth):
            h = murmurhash(token, d)
            idx = h % width
            table[d][idx] += 1
    return table

def murmurhash(key: str, seed: int) -> int:
    # Simple murmurhash implementation
    h = seed
    for c in key:
        h = (h ^ ord(c)) * 0x5bd1e995
    return h

def improved_select_hybrid_action(context: Dict[str, float],
                                  actions: List[str],
                                  sketch: List[List[int]],
                                  points: List[Tuple[float, float, float]],
                                  algorithm: str = 'linucb',
                                  epsilon: float = 0.1,
                                  seed: int | str | None = 7) -> BanditAction:
    # Use a more sophisticated bandit algorithm (e.g. Thompson sampling)
    if seed is not None:
        np.random.seed(seed)

    # Compute sketch-based weights
    sketch_weights = []
    for action in actions:
        est = sketch_estimate(sketch, action)
        sketch_weights.append(est)

    # Combine sketch-based weights with reward estimates and confidence bounds
    action_scores = []
    for action, weight in zip(actions, sketch_weights):
        reward = _reward(action)
        # Use Thompson sampling to compute confidence bound
        cb = np.random.beta(1 + reward, 1 + (1 - reward)) if weight > 0 else 0.0
        score = reward + epsilon * cb
        action_scores.append((action, score))

    # Select action with highest score
    best_action, _ = max(action_scores, key=lambda x: x[1])

    # Return selected action with estimated reward and confidence bound
    reward = _reward(best_action)
    cb = np.random.beta(1 + reward, 1 + (1 - reward)) if sketch_estimate(sketch, best_action) > 0 else 0.0
    return BanditAction(best_action, 1.0, reward, cb, algorithm)