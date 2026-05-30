# DARWIN HAMMER — match 43, survivor 0
# gen: 3
# parent_a: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s2.py (gen1)
# parent_b: hybrid_sketches_hybrid_bandit_router_m31_s1.py (gen2)
# born: 2026-05-29T23:25:37Z

"""Hybrid Krampus‑Ollivier‑Bandit Module

This module fuses the two parent algorithms:

* **Parent A – Krampus brain‑map + Ollivier‑Ricci curvature**  
  Texts are turned into high‑dimensional master vectors **v**∈ℝⁿ, a
  distance‑thresholded graph is built, and an average incident curvature
  κᵢ is computed for every node.

* **Parent B – Count‑Min sketch + contextual bandit router**  
  A count‑min sketch provides a fast, probabilistic estimate of the
  frequency of abstract items; a lightweight bandit policy selects an
  action based on estimated rewards, confidence bounds and an optional
  contextual signal.

**Mathematical bridge**

The curvature value κᵢ is a scalar that quantifies how “well‑connected”
a text node is inside the semantic graph.  We treat κᵢ as an additional
feature of the node and inject it into the Krampus linear projection,
producing a 3‑D coordinate **pᵢ** = (xᵢ, yᵢ, zᵢ).  The set of coordinates
is then hashed (as strings) into a count‑min sketch, giving a compact
summary of the geometric distribution of the corpus.  When a bandit
needs to choose an action for a new context, the sketch‑derived weight
for each candidate action is combined with the usual reward estimate,
thereby coupling the graph‑geometric information (Parent A) with the
online decision‑making machinery (Parent B).

The three core hybrid functions are:

* ``hybrid_build_adj`` – builds the adjacency list from master vectors.
* ``hybrid_node_curvature`` – computes average incident curvature per node.
* ``hybrid_brain_xyz`` – augments the original brain‑map projection with
  curvature and returns 3‑D points.
* ``curvature_sketch`` – builds a count‑min sketch from the curvature‑derived
  coordinate strings.
* ``select_hybrid_action`` – bandit action selector that mixes reward,
  confidence and sketch‑based frequency information.

All code uses only the Python standard library and NumPy."""

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
    """Select an action using a blend of:
       • classic bandit reward estimate,
       • a confidence term proportional to √(log t / n),
       • a sketch‑derived frequency bias (higher frequency → higher propensity)."""
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)

    # Sketch bias: hash each action as if it were a point string
    sketch_weights = []
    for a in actions:
        token = f'action:{a}'
        sketch_weights.append(sketch_estimate(sketch, token) + 1)  # +1 to avoid zero

    # Scale context (norm of feature vector) for LinUCB‑style term
    ctx_norm = math.sqrt(sum(v * v for v in context.values())) if context else 1.0

    best_score = -math.inf
    chosen = actions[0]
    for a, sk_w in zip(actions, sketch_weights):
        r_est = _reward(a)
        n = _POLICY.get(a, [0.0, 0.0])[1]
        confidence = 0.1 * ctx_norm / math.sqrt(1.0 + n)
        # epsilon‑greedy fallback
        if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
            score = rng.random()
        else:
            score = r_est + confidence + 0.05 * math.log(sk_w + 1)
        if score > best_score:
            best_score = score
            chosen = a

    prop = sketch_estimate(sketch, f'action:{chosen}') + 1
    exp_reward = _reward(chosen)
    conf = 0.1 * ctx_norm / math.sqrt(1.0 + _POLICY.get(chosen, [0.0, 0.0])[1])
    return BanditAction(action_id=chosen,
                        propensity=float(prop),
                        expected_reward=exp_reward,
                        confidence_bound=conf,
                        algorithm=algorithm)

# ---------------------------------------------------------------------------
# Demonstration / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample corpus
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "lorem ipsum dolor sit amet consectetur adipiscing elit",
        "artificial intelligence blends mathematics and code",
        "graph curvature captures neighbourhood overlap",
        "bandit algorithms balance exploration and exploitation"
    ]

    # 1. Master vectors
    master_vecs = [extract_master_vector(t) for t in texts]

    # 2. Graph adjacency & curvature
    adjacency = hybrid_build_adj(master_vecs, eps=0.6)
    curv = hybrid_node_curvature(master_vecs, adjacency)

    # 3. 3‑D brain‑map points
    points_3d = hybrid_brain_xyz(master_vecs, curv)

    # 4. Build count‑min sketch from points
    sketch = curvature_sketch(points_3d)

    # 5. Simulate a context (e.g., a new text)
    new_text = "deep learning models learn representations"
    new_vec = extract_master_vector(new_text)
    # For the context we reuse the projection without curvature (just to have a vector)
    dummy_curv = 0.0
    ctx_point = hybrid_brain_xyz([new_vec], [dummy_curv])[0]
    context = {"x": ctx_point[0], "y": ctx_point[1], "z": ctx_point[2]}

    # 6. Define possible actions
    candidate_actions = ["show_article", "recommend_video", "display_ad", "log_event"]

    # 7. Select action using hybrid bandit
    selected = select_hybrid_action(context,
                                    candidate_actions,
                                    sketch,
                                    points_3d,
                                    algorithm='linucb',
                                    epsilon=0.05,
                                    seed=123)

    print("Selected action:", selected)
    # Update policy with a fake reward
    update_policy([BanditUpdate(context_id="demo", action_id=selected.action_id,
                               reward=1.0, propensity=selected.propensity)])
    # Show updated reward estimate
    print("Updated reward estimate for chosen action:",
          _reward(selected.action_id))