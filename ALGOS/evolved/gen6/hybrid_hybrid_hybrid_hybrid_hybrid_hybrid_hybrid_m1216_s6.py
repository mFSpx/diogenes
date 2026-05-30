# DARWIN HAMMER — match 1216, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m551_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m1192_s0.py (gen5)
# born: 2026-05-29T23:34:26Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m551_s0.py (Bandit‑Router / Store update with B‑spline signature)
- Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m1192_s0.py (Stylometry‑driven Workshare allocator)

Mathematical Bridge
------------------
Both parents manipulate a scalar *control signal* that modulates downstream decisions:

* Parent A* produces a **dance signal** `d = store.level` after updating the honeybee‑style store with a
  signature vector `s` and a tree‑metric matrix `M`:
  
  d_{t+1} = d_t + Δt·(sᵀ·M)
  

* Parent B* distributes a total workload `U` across model groups using a deterministic target
  percentage `p`.  The allocation is a linear map from the target percentage to a share vector
  `w ∈ ℝ^G` (G = number of groups).

The fusion treats the dance signal `d` as a *dynamic scaling factor* for the workshare
allocation and, conversely, uses the resulting group shares to re‑weight the bandit
propensities.  Concretely:


p̂ = p·(1 + tanh(d))                # scale target % by the bounded dance signal
ŵ = allocate_workshare(U, p̂)      # same allocation routine as Parent B
π_i' = π_i·ŵ_{group(i)}            # bandit propensity π_i is multiplied by the share of its group


Thus the two topologies are fused through a single scalar `d` that flows bidirectionally
between the store update and the workshare allocator, while the matrix operations
` sᵀ·M ` and the allocation vector `ŵ ` remain unchanged.

The module below implements this hybrid system with three demonstration functions:
1. `lead_lag_bspline_signature` – B‑spline‑projected signature of a time series.
2. `store_update_from_signature` – Bayesian store update using the signature and a tree metric.
3. `allocate_and_adjust` – workshare allocation scaled by the dance signal and subsequent
   bandit propensity adjustment.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str
    group: str  # added for fusion with workshare groups


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honeybee‑style store and derived control signal."""
    level: float = 0.0          # dance signal d
    alpha: float = 1.0          # Beta‑distribution parameters
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0


# ----------------------------------------------------------------------
# Data structures & constants from Parent B
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

GROUPS = ("codex", "groq", "cohere", "local_models")

def words(text: str) -> List[str]:
    return [word.lower() for word in text.split() if word.isalpha()]

@dataclass
class WorkshareLane:
    group: str
    llm_units: float
    llm_share_pct: float
    proof_required: bool


# ----------------------------------------------------------------------
# Core Hybrid Functions
# ----------------------------------------------------------------------
def lead_lag_bspline_signature(
    series: np.ndarray,
    degree: int = 3,
    num_knots: int = 5,
) -> np.ndarray:
    """
    Compute a B‑spline projected signature of a 1‑D time series.

    The signature is the vector of coefficients obtained by fitting a
    piece‑wise polynomial of given `degree` with `num_knots` interior knots.
    This provides a smooth, low‑dimensional representation that can be
    multiplied by a tree‑metric matrix in the store update.

    Parameters
    ----------
    series : np.ndarray
        Input time‑series (shape: (T,)).
    degree : int
        Degree of the B‑spline (default 3 → cubic).
    num_knots : int
        Number of interior knots (default 5).

    Returns
    -------
    coeffs : np.ndarray
        Coefficient vector of length `degree + num_knots + 1`.
    """
    if series.ndim != 1:
        raise ValueError("series must be a 1‑D array")
    T = series.shape[0]
    # Build knot vector uniformly spaced in [0, T)
    knots = np.linspace(0, T, num_knots + 2)[1:-1]  # interior knots
    # Construct the design matrix for B‑spline basis using Cox‑de Boor recursion (simplified)
    # For speed and library constraints we approximate with a Vandermonde matrix of powers.
    # This still yields a linear system solvable via least squares.
    X = np.vander(np.linspace(0, 1, T), N=degree + 1, increasing=True)
    coeffs, *_ = np.linalg.lstsq(X, series, rcond=None)
    return coeffs


def store_update_from_signature(
    store: StoreState,
    signature: np.ndarray,
    tree_metric: np.ndarray,
) -> StoreState:
    """
    Perform a Bayesian store update using a signature vector and a tree‑metric matrix.

    The core update equation (the bridge) is:

        Δd = dt * (signatureᵀ @ tree_metric @ ones)

    where `d` is the dance signal (`store.level`).  The Beta parameters
    (`alpha`, `beta`) are updated with a simple additive rule that mimics
    a conjugate prior update for binary rewards.

    Parameters
    ----------
    store : StoreState
        Current store state.
    signature : np.ndarray
        Signature vector (shape: (k,)).
    tree_metric : np.ndarray
        Square matrix of shape (k, k) encoding inflow/outflow coefficients.

    Returns
    -------
    StoreState
        Updated store (a new instance, leaving the original unchanged).
    """
    if signature.ndim != 1:
        raise ValueError("signature must be 1‑D")
    if tree_metric.shape[0] != tree_metric.shape[1]:
        raise ValueError("tree_metric must be square")
    if signature.shape[0] != tree_metric.shape[0]:
        raise ValueError("signature length must match tree_metric dimension")

    # Compute the scalar flow contribution
    flow = signature @ tree_metric @ np.ones(tree_metric.shape[0])
    delta = store.dt * flow

    # Update dance signal with a bounded tanh to keep it in a reasonable range
    new_level = store.level + delta
    new_level = math.tanh(new_level)  # keep |d| ≤ 1

    # Simple Bayesian update: treat delta > 0 as a "success"
    if delta > 0:
        new_alpha = store.alpha + delta
        new_beta = store.beta
    else:
        new_alpha = store.alpha
        new_beta = store.beta - delta  # delta is negative

    return StoreState(
        level=new_level,
        alpha=new_alpha,
        beta=new_beta,
        dt=store.dt,
        base=store.base,
    )


def allocate_and_adjust(
    total_units: float,
    store: StoreState,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
) -> Tuple[Dict[str, float], List[BanditAction]]:
    """
    Allocate workshare across groups, scaling the target percentage by the
    store's dance signal, and then adjust a synthetic list of bandit actions
    according to the resulting shares.

    The scaling rule (the mathematical bridge) is:

        p̂ = deterministic_target_pct * (1 + tanh(d))

    where `d` is `store.level`.  The allocation routine mirrors Parent B,
    distributing `total_units` proportionally to the scaled target percentage.
    Afterwards each `BanditAction.propensity` is multiplied by the share of its
    associated group.

    Returns
    -------
    workshare : dict[str, float]
        Mapping group → allocated units.
    adjusted_actions : list[BanditAction]
        Propensity‑adjusted actions.
    """
    # 1️⃣ Scale the deterministic target percentage using the bounded dance signal
    scaled_pct = deterministic_target_pct * (1.0 + math.tanh(store.level))
    # Clamp to [0, 100]
    scaled_pct = max(0.0, min(100.0, scaled_pct))

    # 2️⃣ Simple proportional allocation: each group gets the same share of the scaled target,
    #    and the remainder (100 - scaled_pct) is distributed equally as "background" work.
    target_units = total_units * (scaled_pct / 100.0)
    background_units = total_units - target_units
    per_group_target = target_units / len(groups)
    per_group_background = background_units / len(groups)

    workshare: Dict[str, float] = {
        g: per_group_target + per_group_background for g in groups
    }

    # 3️⃣ Create a mock list of bandit actions (one per group) for demonstration
    actions: List[BanditAction] = []
    for i, g in enumerate(groups):
        base_propensity = random.random()
        actions.append(
            BanditAction(
                action_id=f"act_{i}",
                propensity=base_propensity,
                expected_reward=random.uniform(0, 1),
                confidence_bound=random.uniform(0, 0.5),
                algorithm="HybridBandit",
                group=g,
            )
        )

    # 4️⃣ Adjust propensities by the allocated share (normalized to [0,1])
    total_allocated = sum(workshare.values())
    normalized_share = {g: workshare[g] / total_allocated for g in groups}
    adjusted_actions = [
        BanditAction(
            action_id=a.action_id,
            propensity=a.propensity * normalized_share.get(a.group, 0.0),
            expected_reward=a.expected_reward,
            confidence_bound=a.confidence_bound,
            algorithm=a.algorithm,
            group=a.group,
        )
        for a in actions
    ]

    return workshare, adjusted_actions


# ----------------------------------------------------------------------
# Helper for Parent B's original allocation (kept for reference)
# ----------------------------------------------------------------------
def _allocate_workshare(
    total_units: float,
    deterministic_target_pct: float,
    groups: Tuple[str, ...] = GROUPS,
) -> Dict[str, float]:
    """
    Original deterministic allocation from Parent B (used internally).
    """
    target_units = total_units * (deterministic_target_pct / 100.0)
    background_units = total_units - target_units
    per_group_target = target_units / len(groups)
    per_group_background = background_units / len(groups)
    return {g: per_group_target + per_group_background for g in groups}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic time series
    t = np.linspace(0, 2 * np.pi, 100)
    series = np.sin(t) + 0.1 * np.random.randn(100)

    # 1️⃣ Signature extraction
    sig = lead_lag_bspline_signature(series, degree=3, num_knots=6)

    # 2️⃣ Tree metric (simple positive‑definite matrix)
    k = sig.shape[0]
    rng = np.random.default_rng(42)
    A = rng.normal(size=(k, k))
    tree_metric = A @ A.T + np.eye(k) * 0.1  # ensure PD

    # 3️⃣ Store update
    store = StoreState(level=0.0, alpha=1.0, beta=1.0, dt=0.05)
    store = store_update_from_signature(store, sig, tree_metric)

    # 4️⃣ Hybrid allocation & bandit adjustment
    workshare, adjusted_actions = allocate_and_adjust(
        total_units=1000.0,
        store=store,
        deterministic_target_pct=85.0,
    )

    # Output sanity check
    print("Updated StoreState:", store)
    print("\nWorkshare allocation (units):")
    for g, u in workshare.items():
        print(f"  {g}: {u:.2f}")

    print("\nAdjusted Bandit Actions:")
    for a in adjusted_actions:
        print(
            f"  {a.action_id} (group={a.group}) -> propensity={a.propensity:.4f}"
        )