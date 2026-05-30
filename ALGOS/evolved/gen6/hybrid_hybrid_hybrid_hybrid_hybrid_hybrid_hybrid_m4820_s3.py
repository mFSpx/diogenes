# DARWIN HAMMER — match 4820, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_sketch_m135_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m2355_s1.py (gen5)
# born: 2026-05-29T23:58:13Z

"""Hybrid Allocation‑Sheaf‑Bandit Fusion

Parents:
- `hybrid_hybrid_hybrid_worksh_hybrid_hybrid_sketch_m135_s0.py` (allocation + sheaf cohomology)
- `hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m2355_s1.py` (contextual bandit + Fisher information)

Mathematical Bridge
------------------
1. The allocation routine yields a scalar per group.  Interpreting this vector **s**
   as a 0‑cochain, the coboundary operator δ maps **s** to edge‑differences
   **δs**.  The L2 norm ‖δs‖₂ quantifies inconsistency across the group graph.
2. Treat the allocation vector **s** as a set of *latent parameters* θ of a
   temperature‑dependent Gaussian model `p(x|θ)`.  For a univariate Gaussian
   with fixed variance σ² the Fisher information w.r.t. θ is `I(θ)=1/σ²`.
   We therefore obtain a Fisher scalar for each group directly from the
   allocation values.
3. The bandit context is constructed by projecting the allocation vector onto
   its principal components (a tiny PCA).  The UCB for an action a in context c
   is augmented with the Fisher information of the underlying allocation:
   
   UCB(a) = μ_a + β * sqrt( I_a / n_a )
   
   where μ_a is the estimated reward, n_a the pull count, and I_a the Fisher
   information derived from the allocation of the group that action a belongs
   to.

The module provides three public functions that demonstrate the fused pipeline:
`allocate_and_residual` – allocation + sheaf residual,
`compute_fisher_from_allocation` – Fisher information per group,
`select_action_with_fisher` – contextual bandit decision using the above
information.  A minimal smoke test exercises the full flow."""

import datetime as dt
import hashlib
import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – allocation utilities
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1


def _pct(value: float) -> float:
    """Round a float to six decimal places (consistent with Parent A)."""
    return round(float(value), 6)


def weekday_weight_vector(ref_date: dt.date | None = None) -> np.ndarray:
    """Return a 4‑element weight vector that depends on the weekday.

    The original algorithm uses a deterministic hash of the ISO week day.
    Here we keep the same spirit but simplify the hash to a reproducible
    pseudo‑random number seeded by the weekday.
    """
    if ref_date is None:
        ref_date = dt.date.today()
    weekday = ref_date.isoweekday()  # 1 = Monday … 7 = Sunday
    rng = random.Random(weekday)  # deterministic per weekday
    raw = np.array([rng.random() for _ in GROUPS], dtype=float)
    # Normalize to sum to 1
    w = raw / raw.sum()
    return w


def allocate_hybrid(seed: int | None = None) -> Dict[str, float]:
    """Deterministic+LLM split allocation per group.

    Returns a dict mapping each group name to an allocated proportion that
    sums to 1.0.  The deterministic part comes from the weekday weight vector,
    the stochastic part is a small random perturbation.
    """
    if seed is not None:
        random.seed(seed)
    base = weekday_weight_vector()
    # Small random noise (max ±2 %)
    noise = np.array([random.uniform(-0.02, 0.02) for _ in GROUPS])
    alloc = base + noise
    # Clip negative values and renormalize
    alloc = np.clip(alloc, 0.0, None)
    alloc /= alloc.sum()
    return dict(zip(GROUPS, map(_pct, alloc)))


# ----------------------------------------------------------------------
# Parent A – sheaf (coboundary) utilities
# ----------------------------------------------------------------------
def _edge_index(i: int, j: int) -> Tuple[int, int]:
    """Return a sorted edge tuple for undirected graph."""
    return (min(i, j), max(i, j))


def build_coboundary_matrix(num_vertices: int) -> np.ndarray:
    """Construct the (num_edges × num_vertices) coboundary matrix δ.

    For a complete graph on `num_vertices` vertices each unordered pair (i,j)
    defines an edge.  The row corresponding to edge (i,j) has +1 at column i,
    -1 at column j, and 0 elsewhere.
    """
    edges = [_edge_index(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices)]
    m = np.zeros((len(edges), num_vertices), dtype=float)
    for row, (i, j) in enumerate(edges):
        m[row, i] = 1.0
        m[row, j] = -1.0
    return m


def sheaf_residual_from_allocation(allocation: Dict[str, float]) -> float:
    """Compute ‖δ s‖₂ where s is the allocation 0‑cochain.

    Returns the L2 norm of the edge‑difference vector.
    """
    vec = np.array([allocation[g] for g in GROUPS], dtype=float)
    δ = build_coboundary_matrix(len(GROUPS))
    edge_diffs = δ @ vec
    norm = np.linalg.norm(edge_diffs, ord=2)
    return float(_pct(norm))


# ----------------------------------------------------------------------
# Parent B – bandit & Fisher utilities (simplified)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    group: str                     # group this action belongs to
    propensity: float
    expected_reward: float
    confidence_bound: float = 0.0
    algorithm: str = "HybridBanditAI"


@dataclass
class BanditPolicy:
    """Tracks reward estimates and pull counts per action."""
    rewards: Dict[str, float]          # cumulative reward
    pulls: Dict[str, int]              # number of times action selected
    fisher: Dict[str, float]           # accumulated Fisher information


def init_policy(actions: List[BanditAction]) -> BanditPolicy:
    return BanditPolicy(
        rewards={a.action_id: 0.0 for a in actions},
        pulls={a.action_id: 0 for a in actions},
        fisher={a.action_id: 0.0 for a in actions},
    )


def compute_fisher_information(sigma_sq: float = 1.0) -> float:
    """Fisher information for a univariate Gaussian N(θ, σ²) w.r.t. θ."""
    if sigma_sq <= 0.0:
        raise ValueError("Variance must be positive")
    return 1.0 / sigma_sq


def _pca_projection(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Very small PCA: project rows onto the top `n_components` eigenvectors."""
    # Center
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    # Covariance
    cov = centered.T @ centered / (centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1][:n_components]
    components = eigvecs[:, idx]
    return centered @ components


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def allocate_and_residual(seed: int | None = None) -> Tuple[Dict[str, float], float]:
    """Run allocation (Parent A) and return both the allocation dict and the sheaf residual."""
    alloc = allocate_hybrid(seed)
    residual = sheaf_residual_from_allocation(alloc)
    return alloc, residual


def compute_fisher_from_allocation(allocation: Dict[str, float], base_variance: float = 0.05) -> Dict[str, float]:
    """Map each group's allocation to a Fisher information scalar.

    We treat the allocation proportion as the mean θ of a Gaussian with a
    fixed variance that is inversely proportional to the allocation magnitude.
    Larger allocations imply more confident (lower‑variance) estimates,
    yielding higher Fisher information.
    """
    fisher_per_group = {}
    for g, prop in allocation.items():
        # Prevent division by zero; clamp variance to a small positive number
        var = max(base_variance * (1.0 - prop), 1e-6)
        fisher = compute_fisher_information(var)
        fisher_per_group[g] = _pct(fisher)
    return fisher_per_group


def select_action_with_fisher(
    policy: BanditPolicy,
    actions: List[BanditAction],
    allocation: Dict[str, float],
    beta: float = 1.0,
) -> BanditAction:
    """Select the action with the highest Fisher‑augmented Upper‑Confidence‑Bound.

    The context for the bandit is the PCA projection of the allocation vector.
    For each action we compute:
        UCB = μ̂ + β * sqrt( I_g / (n_a + 1) )
    where μ̂ is the empirical mean reward, I_g is the Fisher information of the
    group the action belongs to, and n_a is the number of times the action has
    been selected.
    """
    # Build context (not used directly in this toy UCB, but kept for extensibility)
    alloc_vec = np.array([allocation[g] for g in GROUPS], dtype=float).reshape(1, -1)
    _ = _pca_projection(alloc_vec, n_components=2)  # placeholder

    # Pre‑compute Fisher per group
    fisher_group = compute_fisher_from_allocation(allocation)

    best_ucb = -math.inf
    best_action = None

    for act in actions:
        pulls = policy.pulls[act.action_id]
        reward_sum = policy.rewards[act.action_id]
        mu_hat = reward_sum / pulls if pulls > 0 else 0.0
        I_g = fisher_group.get(act.group, 0.0)
        # Upper‑Confidence‑Bound with Fisher scaling
        ucb = mu_hat + beta * math.sqrt(I_g / (pulls + 1))
        if ucb > best_ucb:
            best_ucb = ucb
            best_action = act

    # Return a copy with the computed confidence bound filled in
    if best_action is None:
        raise RuntimeError("No actions available for selection")
    return BanditAction(
        action_id=best_action.action_id,
        group=best_action.group,
        propensity=best_action.propensity,
        expected_reward=best_action.expected_reward,
        confidence_bound=_pct(best_ucb),
        algorithm=best_action.algorithm,
    )


def update_policy_with_observation(
    policy: BanditPolicy,
    action: BanditAction,
    reward: float,
    fisher: float,
) -> None:
    """Update reward sums, pull counts and accumulate Fisher information."""
    aid = action.action_id
    policy.rewards[aid] += reward
    policy.pulls[aid] += 1
    policy.fisher[aid] += fisher


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: allocation + sheaf residual
    allocation, residual = allocate_and_residual(seed=42)
    print("Allocation:", allocation)
    print("Sheaf residual (‖δs‖₂):", residual)

    # Step 2: define a tiny action set (two actions per group)
    actions = []
    for g in GROUPS:
        for i in range(2):
            act_id = f"{g}_act{i}"
            actions.append(
                BanditAction(
                    action_id=act_id,
                    group=g,
                    propensity=random.random(),
                    expected_reward=random.uniform(0, 1),
                )
            )

    # Initialise policy
    policy = init_policy(actions)

    # Simulate a few decision cycles
    for step in range(5):
        chosen = select_action_with_fisher(policy, actions, allocation, beta=0.5)
        # Fake reward: higher if expected_reward is high + some noise
        reward = chosen.expected_reward + random.gauss(0, 0.1)
        # Fisher for the chosen group's allocation
        fisher_dict = compute_fisher_from_allocation(allocation)
        fisher_val = fisher_dict.get(chosen.group, 0.0)
        update_policy_with_observation(policy, chosen, reward, fisher_val)
        print(f"Step {step+1}: chose {chosen.action_id} (group={chosen.group}) "
              f"UCB={chosen.confidence_bound}, reward={reward:.3f}, fisher={fisher_val:.3f}")

    print("Final policy state (sample):")
    sample_aid = actions[0].action_id
    print(f"  Action {sample_aid}: pulls={policy.pulls[sample_aid]}, "
          f"cum_reward={policy.rewards[sample_aid]:.3f}, fisher={policy.fisher[sample_aid]:.3f}")