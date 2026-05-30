# DARWIN HAMMER — match 8, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# parent_b: model_vram_scheduler.py (gen0)
# born: 2026-05-29T23:25:05Z

"""Hybrid VRAM Scheduler with Bayesian Tree Cost Integration.

This module fuses two parent algorithms:
- **hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py** – provides
  tree construction, Euclidean edge lengths, and Bayesian marginal/posterior
  calculations.
- **model_vram_scheduler.py** – defines VRAM budgeting dataclasses and the
  environment‑driven configuration for a 4 GB GPU.

The mathematical bridge is the *expected VRAM consumption* of a model
component tree.  Each node (e.g., base model, embedding, LoRA cartridge) has
a size (MiB) and a prior probability of being required.  A likelihood vector
represents evidence (e.g., prompt characteristics).  Using the Bayesian
marginal and posterior from the first parent we obtain a posterior probability
distribution over the nodes.  The *expected VRAM* is the dot product of this
distribution with the node‑size vector.  To assess decision hygiene we also
compute the Shannon entropy of the posterior; high entropy signals uncertainty
and may trigger a more conservative scheduling policy.

The scheduler therefore integrates:
1. Tree metrics (distances, edge lengths) – structural cost.
2. Bayesian update – probabilistic weighting of each component.
3. Expected VRAM and entropy – decision‑hygiene metrics.
4. VRAM budget check – the original scheduling logic.

The result is a single unified system that can advise whether the current
model configuration fits within the available GPU memory.
"""

import math
import random
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities from Parent A
# ----------------------------------------------------------------------
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)

    return adj, edge_len, dist

def bayes_marginal(prior: np.ndarray, likelihood: np.ndarray, false_positive: float) -> np.ndarray:
    """
    Vectorised marginal:  P(E) = L·p + FP·(1‑p)
    """
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: np.ndarray, likelihood: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """
    Vectorised posterior:  P(H|E) = (L·p) / P(E)
    Handles zero marginals safely.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        posterior = np.where(marginal > 0,
                             (likelihood * prior) / marginal,
                             0.0)
    # Normalise to guard against numerical drift
    total = posterior.sum()
    if total > 0:
        posterior = posterior / total
    return posterior

def shannon_entropy(probs: np.ndarray) -> float:
    """Binary Shannon entropy (base‑2) of a probability vector."""
    eps = np.finfo(float).eps
    probs = np.clip(probs, eps, 1 - eps)
    return -np.sum(probs * np.log2(probs))

# ----------------------------------------------------------------------
# VRAM‑related dataclasses from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

# ----------------------------------------------------------------------
# Environment helpers (mirroring Parent B)
# ----------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

# Default budget parameters (can be overridden via env)
DEFAULT_BUDGET_MB = _env_int("LUCIDOTA_VRAM_BUDGET_MB", 4096)
DEFAULT_RESERVE_MB = _env_int("LUCIDOTA_VRAM_RESERVE_MB", 768)

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def expected_vram_usage(sizes: np.ndarray, posterior: np.ndarray) -> float:
    """
    Expected VRAM (MiB) = Σ_i  size_i * P_i
    """
    return float(np.dot(sizes, posterior))

def hybrid_vram_decision(
    root: str,
    node_coords: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    sizes_mb: Dict[str, int],
    prior: np.ndarray,
    likelihood: np.ndarray,
    false_positive: float = 0.01,
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
) -> VramSlotPlan:
    """
    Compute a unified VRAM plan:
    1. Build the tree and obtain root‑to‑node distances (structural cost).
    2. Perform Bayesian update to get posterior probabilities of each node.
    3. Compute expected VRAM usage.
    4. Compute Shannon entropy of the posterior as a decision‑hygiene metric.
    5. Return a VramSlotPlan indicating fit / exceed status.
    """
    # --- 1. Tree metrics (structural cost) ---
    _, edge_len, dist = tree_metrics(node_coords, edges, root)

    # Convert dicts to ordered arrays matching the order of prior/likelihood
    node_order = list(sizes_mb.keys())
    sizes_arr = np.array([sizes_mb[n] for n in node_order], dtype=float)

    # --- 2. Bayesian update ---
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)

    # --- 3. Expected VRAM ---
    exp_vram = expected_vram_usage(sizes_arr, posterior)

    # --- 4. Decision hygiene (entropy) ---
    entropy = shannon_entropy(posterior)

    # Combine structural distance cost as a simple additive factor
    structural_cost = sum(edge_len.values())
    adjusted_vram = exp_vram + 0.01 * structural_cost  # small scaling factor

    # --- 5. Budget check ---
    available = budget_mb - reserve_mb
    fits = adjusted_vram <= available

    reason = (
        f"expected={exp_vram:.1f}MiB, structural={structural_cost:.2f}eucl, "
        f"adjusted={adjusted_vram:.1f}MiB, budget={available}MiB, entropy={entropy:.3f}"
    )
    action = "accept" if fits else "reject"

    plan = VramSlotPlan(
        artifact_id=root,
        artifact_kind="model_tree",
        action=action,
        estimated_mb=int(math.ceil(adjusted_vram)),
        reason=reason,
        detail={
            "expected_vram_mb": exp_vram,
            "structural_cost": structural_cost,
            "adjusted_vram_mb": adjusted_vram,
            "entropy_bits": entropy,
            "posterior": posterior.tolist(),
            "node_order": node_order,
        },
    )
    return plan

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny dependency tree:
    #   root (base model) -> embedding -> lora
    node_coords = {
        "base": (0.0, 0.0),
        "embed": (1.0, 0.0),
        "lora": (2.0, 0.0),
    }
    edges = [("base", "embed"), ("embed", "lora")]
    sizes_mb = {
        "base": 1800,      # typical base model size
        "embed": 384,      # embedding lane
        "lora": 128,       # LoRA cartridge
    }

    # Prior belief that each component will be needed (e.g., from usage stats)
    prior = np.array([0.9, 0.5, 0.2])
    # Likelihood derived from current prompt/context (simulated)
    likelihood = np.array([0.95, 0.6, 0.4])

    plan = hybrid_vram_decision(
        root="base",
        node_coords=node_coords,
        edges=edges,
        sizes_mb=sizes_mb,
        prior=prior,
        likelihood=likelihood,
        false_positive=0.02,
        budget_mb=DEFAULT_BUDGET_MB,
        reserve_mb=DEFAULT_RESERVE_MB,
    )

    print("Hybrid VRAM Plan:")
    for k, v in plan.as_dict().items():
        print(f"{k}: {v}")