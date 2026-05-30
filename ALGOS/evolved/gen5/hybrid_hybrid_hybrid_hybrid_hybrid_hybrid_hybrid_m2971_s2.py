# DARWIN HAMMER — match 2971, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1830_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_percyphon_hyb_m1057_s1.py (gen4)
# born: 2026-05-29T23:47:00Z

"""Hybrid Regret‑Bandit‑Koopman‑Honeybee‑Morphology Engine
=====================================================

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – ``hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1830_s3.py``  
  Provides a regret‑weighted probability distribution `p_t`, its Gini coefficient
  `G_t`, a honeybee‑style store `S_t`, and a Koopman operator `K` that forecasts
  the mean‑reward vector `μ̂_t`.

* **Parent B** – ``hybrid_hybrid_hybrid_hybrid_hybrid_percyphon_hyb_m1057_s1.py``  
  Supplies health scores `h_i` that are normalised into a work‑share vector
  **W**, and morphology descriptors (`σ`, `φ`) that yield a global scaling factor
  `μ_shape`.

**Mathematical Bridge**

The work‑share vector **W** (derived from health scores) is used as the
regret‑weighted distribution `p_t`.  Its Gini coefficient modulates the
honeybee store, producing a confidence scalar `C_t = S_t·G_t`.  The Koopman
operator forecasts the next reward vector `μ̂ = K·p_t`.  A context vector
`c_t` is built from the morphology descriptors of the current entity; its
structural similarity to the forecast (`ssim`) is measured by a cosine‑like
metric.  The unified decision index for each action *a* is


U_a = μ̂_a + η·‖c_t‖·C_t·ssim(c_t, μ̂)


Finally the indices are turned into procedural‑slot allocations using the
shape‑derived scaling factor `μ_shape`:


n_a = round( (U_a / ΣU) * B * μ_shape )


The implementation below integrates these equations into a single coherent
system."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Shared Data Structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """Action descriptor used by the regret engine."""
    id: str
    expected_value: float
    cost: float = 0.0


@dataclass(frozen=True)
class ModelTier:
    """Simple descriptor for an ML model tier."""
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


@dataclass(frozen=True)
class Morphology:
    """Geometric description of an entity."""
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class ProceduralSlot:
    """One slot generated for a procedural entity."""
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Honeybee‑style Store
# ----------------------------------------------------------------------


@dataclass
class StoreState:
    """Scalar store that evolves with inflow/outflow streams."""
    level: float = 1.0
    decay: float = 0.95  # exponential decay per step

    def update(self, inflow: float, outflow: float) -> None:
        """Apply inflow/outflow and decay."""
        self.level = self.decay * self.level + inflow - outflow
        if self.level < 0.0:
            self.level = 0.0


# ----------------------------------------------------------------------
# Core Mathematical Primitives
# ----------------------------------------------------------------------


def gini_coefficient(p: np.ndarray) -> float:
    """Gini coefficient of a non‑negative vector `p` (must sum to 1)."""
    if p.ndim != 1:
        raise ValueError("p must be a 1‑D array")
    sorted_p = np.sort(p)  # ascending
    n = p.size
    cumulative = np.cumsum(sorted_p)
    gini = 1.0 - 2.0 * np.sum(cumulative) / (n * np.sum(sorted_p)) + (1.0 / n)
    return max(0.0, min(1.0, gini))


def koopman_forecast(p: np.ndarray, K: np.ndarray, steps: int = 1) -> np.ndarray:
    """Linear Koopman forecast `μ̂ = K^steps · p`."""
    if K.shape[0] != K.shape[1] or K.shape[0] != p.size:
        raise ValueError("K must be a square matrix compatible with p")
    K_power = np.linalg.matrix_power(K, steps)
    return K_power @ p


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0,1] (0 for orthogonal, 1 for identical)."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Both inputs must be 1‑D vectors")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0.0, 1.0))


# ----------------------------------------------------------------------
# Hybrid Engine Functions
# ----------------------------------------------------------------------


def compute_hybrid_decision_indices(
    p: np.ndarray,
    K: np.ndarray,
    store: StoreState,
    c: np.ndarray,
    eta: float = 0.1,
) -> np.ndarray:
    """
    Compute the unified decision index vector `U` for each action.

    Parameters
    ----------
    p : np.ndarray
        Regret‑weighted probability distribution (also the work‑share vector).
    K : np.ndarray
        Koopman operator matrix.
    store : StoreState
        Honeybee store providing the scalar level `S_t`.
    c : np.ndarray
        Context vector derived from morphology.
    eta : float
        Exploration coefficient.

    Returns
    -------
    np.ndarray
        Decision indices `U_a` for every action `a`.
    """
    # Gini of the distribution
    G = gini_coefficient(p)

    # Confidence scalar
    C = store.level * G

    # Forecasted reward vector
    mu_hat = koopman_forecast(p, K, steps=1)

    # Similarity between context and forecast
    sim = cosine_similarity(c, mu_hat)

    # Norm of context
    norm_c = np.linalg.norm(c)

    # Decision index per action
    U = mu_hat + eta * norm_c * C * sim
    return U


def allocate_procedural_slots(
    decision_indices: np.ndarray,
    base_budget: int,
    morph_scale: float,
) -> List[int]:
    """
    Convert decision indices into integer slot allocations.

    Parameters
    ----------
    decision_indices : np.ndarray
        Vector `U` from `compute_hybrid_decision_indices`.
    base_budget : int
        Global slot budget `B`.
    morph_scale : float
        Scaling factor `μ_shape` derived from morphology.

    Returns
    -------
    List[int]
        Slot counts `n_i` for each action.
    """
    if decision_indices.ndim != 1:
        raise ValueError("decision_indices must be 1‑D")
    total = decision_indices.sum()
    if total == 0.0:
        # avoid division by zero – allocate equally
        proportions = np.full_like(decision_indices, 1.0 / decision_indices.size)
    else:
        proportions = decision_indices / total
    raw = proportions * base_budget * morph_scale
    # Round and ensure at least one slot for non‑zero proportions
    slots = [max(1, int(round(v))) if p > 0 else 0 for v, p in zip(raw, proportions)]
    return slots


def generate_procedural_slots(
    actions: List[MathAction],
    slot_counts: List[int],
) -> List[ProceduralSlot]:
    """
    Produce concrete `ProceduralSlot` objects from slot counts.

    Parameters
    ----------
    actions : List[MathAction]
        List of actions (used for naming).
    slot_counts : List[int]
        Number of slots allocated to each action.

    Returns
    -------
    List[ProceduralSlot]
        Generated procedural slots.
    """
    slots: List[ProceduralSlot] = []
    idx = 0
    for action, count in zip(actions, slot_counts):
        for i in range(count):
            uuid = f"{action.id}-{idx:04d}"
            slot = ProceduralSlot(
                slot_index=idx,
                name=f"{action.id}_slot_{i}",
                alias=action.id,
                persona="hybrid_agent",
                uuid=uuid,
                ternary_offset=random.choice([-1, 0, 1]),
            )
            slots.append(slot)
            idx += 1
    return slots


# ----------------------------------------------------------------------
# Helper Functions for Parent‑Specific Data
# ----------------------------------------------------------------------


def health_scores_from_risk(risk_values: List[float]) -> np.ndarray:
    """
    Convert reconstruction‑risk scores into health scores `h_i`,
    then normalise to a work‑share vector `p_t`.
    Higher risk → lower health.
    """
    risk_arr = np.array(risk_values, dtype=float)
    # Simple inversion with epsilon to avoid division by zero
    eps = 1e-6
    health = 1.0 / (risk_arr + eps)
    # Normalise to sum to 1 (work‑share vector)
    p = health / health.sum()
    return p


def morphology_context_vector(morph: Morphology) -> np.ndarray:
    """
    Build a context vector `c_t` from morphology.
    Normalise to unit length.
    """
    vec = np.array([morph.length, morph.width, morph.height, morph.mass], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def shape_scaling_factor(sphericity: float, flatness: float) -> float:
    """Global scaling factor μ = (σ + φ) / 2."""
    return (sphericity + flatness) / 2.0


def random_koopman_operator(dim: int) -> np.ndarray:
    """
    Generate a stochastic Koopman matrix (rows sum to 1) to act as a linear
    forecast operator.
    """
    mat = np.random.rand(dim, dim)
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # ---- Define a small set of actions ----
    actions = [
        MathAction(id="A", expected_value=1.2),
        MathAction(id="B", expected_value=0.8),
        MathAction(id="C", expected_value=1.5),
        MathAction(id="D", expected_value=0.5),
    ]

    # ---- Simulate reconstruction risk scores (Parent B) ----
    risk_scores = [random.uniform(0.1, 2.0) for _ in actions]
    p_vector = health_scores_from_risk(risk_scores)  # work‑share / p_t

    # ---- Initialise honeybee store (Parent A) ----
    store = StoreState(level=1.0, decay=0.97)
    store.update(inflow=0.05, outflow=0.02)  # arbitrary dynamics

    # ---- Morphology (Parent B) ----
    morph = Morphology(length=1.8, width=0.9, height=0.4, mass=2.3)
    c_vec = morphology_context_vector(morph)

    # ---- Shape descriptors for scaling (Parent B) ----
    sphericity = 0.78
    flatness = 0.42
    mu_shape = shape_scaling_factor(sphericity, flatness)

    # ---- Koopman operator (Parent A) ----
    K = random_koopman_operator(dim=len(actions))

    # ---- Compute hybrid decision indices ----
    U = compute_hybrid_decision_indices(p=p_vector, K=K, store=store, c=c_vec, eta=0.15)

    # ---- Allocate procedural slots ----
    base_budget = 12  # total slots to distribute
    slot_counts = allocate_procedural_slots(decision_indices=U, base_budget=base_budget, morph_scale=mu_shape)

    # ---- Generate concrete slots ----
    slots = generate_procedural_slots(actions, slot_counts)

    # ---- Simple verification output ----
    print("Risk scores :", ["{:.3f}".format(r) for r in risk_scores])
    print("Work‑share p :", ["{:.3f}".format(v) for v in p_vector])
    print("Gini coeff  :", "{:.3f}".format(gini_coefficient(p_vector)))
    print("Store level :", "{:.3f}".format(store.level))
    print("Koopman forecast μ̂ :", ["{:.3f}".format(v) for v in koopman_forecast(p_vector, K)])
    print("Context vector c :", ["{:.3f}".format(v) for v in c_vec])
    print("Decision indices U :", ["{:.3f}".format(v) for v in U])
    print("Slot allocation per action :", slot_counts)
    print("Generated", len(slots), "procedural slots. Sample:")
    for s in slots[:5]:
        print(s.as_dict())
    sys.exit(0)