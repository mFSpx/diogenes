# DARWIN HAMMER — match 3272, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2508_s2.py (gen6)
# born: 2026-05-29T23:48:54Z

"""Hybrid Decision‑Bandit‑Regret Fusion Module
================================================

This module fuses the two parent algorithms:

* **Parent A** – *hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s2.py*  
  Provides a 3‑dimensional resource vector for each entity  
  `e_i = [d_i, p_i, s_i]` (distance, privacy‑load, hygiene‑score) and a model
  vector `m_j = [RAM_j, α·τ_j·μ]`.  A combined resource matrix `A` is built and
  a binary selection vector `x` must satisfy `A.T @ x ≤ budgets`.

* **Parent B** – *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2508_s2.py*  
  Defines immutable `MathAction` objects with a regret‑weighted scalar
  `w = expected_value - cost - risk`.  It also supplies a min‑hash signature
  routine and a Jaccard‑like similarity measure.

**Mathematical Bridge**

The bridge is the *regret weight* `w` from Parent B, which is treated as an
additional resource dimension that participates in the same linear budget
constraints as the spatial, privacy and decision‑score dimensions from Parent A.
Furthermore, the min‑hash similarity between an entity’s signature and an
action’s signature is added to the entity’s privacy‑load `p_i`, thereby letting
the signature‑based topology of Parent B influence the privacy budgeting of
Parent A.

The resulting hybrid resource vector for an entity becomes  


h_i = [ d_i,
        p_i + Σ_k similarity(sig_i, sig_{a_k}),
        s_i,
        0.0 ]                # no regret contribution for pure entities


and for an action (treated as a “model”)  


h_j = [ RAM_j,
        α·τ_j·μ,
        0.0,
        w_j ]                # regret weight occupies the fourth dimension


All rows are stacked into a matrix `H`.  A greedy selector works on `H` and a
budget vector `B = [spatial_budget, privacy_budget, decision_budget,
regret_budget]` to produce a feasible binary indicator `x`.

The implementation below follows this design and provides three public
functions that demonstrate the hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent B – immutable actions and min‑hash utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Immutable description of a decision option."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

    @property
    def regret_weight(self) -> float:
        """Regret‑weighted scalar used throughout the hybrid model."""
        return self.expected_value - self.cost - self.risk


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on a seed and a token string."""
    h = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def signature(tokens: Iterable[str], k: int = 128) -> np.ndarray:
    """Min‑hash signature of a token set (length k)."""
    toks = [t for t in tokens if t]
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not toks:
        return np.full(k, (1 << 64) - 1, dtype=np.uint64)

    sig = np.empty(k, dtype=np.uint64)
    for i in range(k):
        hashes = [_hash(i, t) for t in toks]
        sig[i] = min(hashes)
    return sig


def similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Normalized Jaccard‑like similarity for two min‑hash signatures."""
    if sig_a.shape != sig_b.shape:
        raise ValueError("Signatures must have the same length")
    return np.mean(sig_a == sig_b)


# ----------------------------------------------------------------------
# Parent A – spatial / privacy / decision hygiene utilities
# ----------------------------------------------------------------------
def haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Return great‑circle distance in metres between two (lat, lon) pairs."""
    R = 6_371_000.0  # Earth radius in metres
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_entity_vector(
    entity: Dict[str, Any],
    reference_location: Tuple[float, float],
    beta: float,
    alpha: float,
    all_signatures: List[np.ndarray],
    action_signatures: List[np.ndarray],
) -> np.ndarray:
    """
    Compute the hybrid resource vector for a single entity.

    Returns a length‑4 numpy array:
    [ distance, privacy_load (+ similarity), hygiene_score, 0.0 ].
    """
    # 1️⃣ distance component
    d_i = haversine(entity["location"], reference_location)

    # 2️⃣ base privacy load from signature collision (decision hygiene)
    sig_i = signature(entity.get("tokens", []))
    collision = int(any(np.array_equal(sig_i, other) for other in all_signatures if not np.array_equal(sig_i, other)))
    p_i = beta * collision

    # 3️⃣ similarity contribution from all actions (Parent B bridge)
    sim_sum = sum(similarity(sig_i, a_sig) for a_sig in action_signatures)
    p_i += sim_sum

    # 4️⃣ decision hygiene score (placeholder – in real code this would be
    #    computed by the original decision‑hygiene algorithm)
    s_i = random.uniform(0.0, 1.0) * alpha

    return np.array([d_i, p_i, s_i, 0.0], dtype=np.float64)


def calculate_action_vector(
    action: MathAction,
    ram_consumption: float,
    tau: int,
    mu_privacy: float,
    alpha: float,
) -> np.ndarray:
    """
    Compute the hybrid resource vector for an action (treated as a model).

    Returns a length‑4 numpy array:
    [ RAM, α·τ·μ, 0.0, regret_weight ].
    """
    spatial = ram_consumption
    privacy = alpha * tau * mu_privacy
    decision = 0.0
    regret = action.regret_weight
    return np.array([spatial, privacy, decision, regret], dtype=np.float64)


# ----------------------------------------------------------------------
# Core hybrid machinery
# ----------------------------------------------------------------------
def build_hybrid_matrix(
    entities: List[Dict[str, Any]],
    actions: List[MathAction],
    reference_location: Tuple[float, float],
    beta: float = 1.0,
    alpha: float = 1.0,
    tau_factors: Sequence[int] = (1, 2, 3),
    privacy_records: Sequence[float] = (),
) -> Tuple[np.ndarray, List[str]]:
    """
    Assemble the hybrid resource matrix H (rows = entities ∪ actions,
    columns = [spatial, privacy, decision, regret]) and a list of row identifiers.
    """
    # Pre‑compute signatures for entities and actions
    entity_sigs = [signature(ent.get("tokens", [])) for ent in entities]
    action_sigs = [signature([action.id]) for action in actions]

    # Mean privacy risk μ (Parent A)
    mu = float(np.mean(privacy_records)) if privacy_records else 0.0

    # Build entity rows
    entity_rows = [
        calculate_entity_vector(
            ent,
            reference_location,
            beta,
            alpha,
            entity_sigs,
            action_sigs,
        )
        for ent in entities
    ]

    # Build action rows – assign a τ factor cyclically from the provided list
    action_rows = []
    for idx, act in enumerate(actions):
        tau = tau_factors[idx % len(tau_factors)]
        # Placeholder RAM consumption; in a real system this would be queried.
        ram = random.uniform(0.5, 4.0) * 1024  # MB → approximate MB value
        row = calculate_action_vector(act, ram, tau, mu, alpha)
        action_rows.append(row)

    # Stack and return
    H = np.vstack(entity_rows + action_rows)
    identifiers = [f"entity_{i}" for i in range(len(entities))] + [
        f"action_{a.id}" for a in actions
    ]
    return H, identifiers


def greedy_hybrid_selector(
    H: np.ndarray,
    budgets: np.ndarray,
    values: np.ndarray = None,
) -> np.ndarray:
    """
    Simple greedy selector that respects linear budgets.
    - `H` is the (n_rows × n_dims) resource matrix.
    - `budgets` is a 1‑D array of length n_dims.
    - `values` (optional) is a profit vector; if omitted, rows are ranked by
      the inverse of their total resource consumption.

    Returns a binary indicator vector `x` (length n_rows).
    """
    n_rows, n_dims = H.shape
    if values is None:
        # Inverse total consumption as a naive utility
        total = H.sum(axis=1) + 1e-9
        values = 1.0 / total

    # Compute efficiency = value / (weighted resource usage)
    # Use equal weighting for all dimensions.
    efficiency = values / (H.sum(axis=1) + 1e-9)

    order = np.argsort(-efficiency)  # descending
    remaining = budgets.copy()
    x = np.zeros(n_rows, dtype=int)

    for idx in order:
        row = H[idx]
        if np.all(row <= remaining + 1e-9):
            x[idx] = 1
            remaining -= row
    return x


def evaluate_selection(
    H: np.ndarray,
    x: np.ndarray,
    budgets: np.ndarray,
) -> Tuple[bool, np.ndarray]:
    """
    Verify that a selection vector `x` respects the budgets.
    Returns (is_feasible, used_resources).
    """
    used = H.T @ x
    feasible = np.all(used <= budgets + 1e-9)
    return feasible, used


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy entities
    entities = [
        {
            "location": (37.7749, -122.4194),  # San Francisco
            "tokens": ["sensor", "temperature", "outdoor"],
        },
        {
            "location": (34.0522, -118.2437),  # Los Angeles
            "tokens": ["sensor", "humidity", "indoor"],
        },
        {
            "location": (40.7128, -74.0060),   # New York
            "tokens": ["camera", "security", "outdoor"],
        },
    ]

    # Dummy actions
    actions = [
        MathAction(id="A1", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="A2", expected_value=5.0, cost=1.0, risk=0.5),
        MathAction(id="A3", expected_value=7.5, cost=3.0, risk=2.0),
    ]

    # Reference location (center of USA)
    ref_loc = (39.8283, -98.5795)

    # Privacy risk records (simulated)
    privacy_risks = [0.2, 0.5, 0.1, 0.3]

    # Build hybrid matrix
    H, ids = build_hybrid_matrix(
        entities,
        actions,
        reference_location=ref_loc,
        beta=1.5,
        alpha=0.8,
        tau_factors=(1, 2, 3),
        privacy_records=privacy_risks,
    )

    # Define budgets: spatial (metres), privacy, decision score, regret
    budgets = np.array(
        [
            5_000_000.0,   # ~5,000 km total distance allowance
            15.0,          # privacy budget
            2.0,           # decision‑score budget
            8.0,           # regret budget
        ]
    )

    # Greedy selection
    x = greedy_hybrid_selector(H, budgets)

    # Verify feasibility
    feasible, used = evaluate_selection(H, x, budgets)

    # Print results
    print("Selected rows:")
    for sel, identifier in zip(x, ids):
        if sel:
            print(f"  - {identifier}")

    print("\nResource usage vs budgets:")
    for dim, (used_val, bud) in enumerate(zip(used, budgets), start=1):
        print(f"  Dim {dim}: used {used_val:.3f} / budget {bud}")

    print("\nFeasibility:", feasible)

    sys.exit(0)