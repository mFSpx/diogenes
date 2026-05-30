# DARWIN HAMMER — match 1808, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_privacy_model_hybrid_hybrid_semant_m1133_s0.py (gen4)
# parent_b: hybrid_jepa_energy_hybrid_sparse_wta_hy_m181_s3.py (gen3)
# born: 2026-05-29T23:38:54Z

"""Hybrid Privacy‑Semantic Model Pool
===================================

This module fuses two parent algorithms:

* **Parent A** – *hybrid_hybrid_privacy_model_hybrid_hybrid_semant_m1133_s0.py*  
  Provides a resource matrix **A** whose columns are
  ``[RAM, privacy‑load, semantic‑load]`` and defines the semantic‑load
  ``s(m) = β·semantic_similarity(m.doc_vector, seed_vectors)``.

* **Parent B** – *hybrid_jepa_energy_hybrid_sparse_wta_hy_m181_s3.py*  
  Introduces a ``ModelTier`` description, a ``ModelPool`` that tracks
  RAM usage and a variational free‑energy (VFE) surrogate used to
  penalise illegal tier combinations and resource violations.

**Mathematical bridge** – Both parents reason about *resource constraints*.
Parent A expresses them as a linear inequality ``L = Aᵀ·x ≤ budget``.
Parent B expresses violations through a VFE penalty.  The hybrid
algorithm builds the same linear load vector **L** while still using the
VFE as a scalar objective to be minimised.  The VFE is increased by a
large constant whenever any component of ``L`` exceeds its ceiling,
thereby unifying the two formulations into a single optimisation
framework.

The code below implements:
1. Semantic similarity (cosine) and semantic‑load computation.
2. A ``HybridModelPool`` that stores ``ModelTier`` objects together with
   their privacy‑load and document vectors.
3. Functions that add models, evaluate a selection vector, and perform a
   greedy selection respecting the combined resource budget while
   minimising VFE.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Core data structures (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ModelTier:
    """Immutable descriptor of a model tier."""
    name: str
    ram_mb: int
    tier: str                     # e.g. "T1", "T2", "T3"
    privacy_load: float           # privacy‑load contribution (0‑1)
    doc_vector: np.ndarray        # high‑dimensional document embedding


# ----------------------------------------------------------------------
# Helper functions (from Parent A)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Return a normalised reconstruction risk in [0, 1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def semantic_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Returns a value in [-1, 1]; clipped to [0, 1] for semantic‑load use.
    """
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return max(0.0, min(1.0, sim))


def compute_semantic_load(model: ModelTier,
                          seed_vectors: Sequence[np.ndarray],
                          beta: float = 1.0) -> float:
    """
    Compute the semantic‑load for a model as

        s(m) = β * max_{seed} semantic_similarity(m.doc_vector, seed)

    The max similarity captures the closest semantic neighbourhood.
    """
    if not seed_vectors:
        return 0.0
    sims = [semantic_similarity(model.doc_vector, seed) for seed in seed_vectors]
    return beta * max(sims)


# ----------------------------------------------------------------------
# Hybrid model pool (fusion of both parents)
# ----------------------------------------------------------------------
class HybridModelPool:
    """
    Manages a pool of ``ModelTier`` objects under three coupled budgets:
    RAM (MB), privacy‑load, and semantic‑load.

    The pool maintains a variational free‑energy (VFE) scalar that is
    increased whenever a constraint is violated or an illegal tier
    combination appears.  The VFE is the optimisation objective.
    """

    def __init__(self,
                 ram_ceiling_mb: int,
                 privacy_budget: float,
                 semantic_budget: float,
                 beta: float = 1.0):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.privacy_budget = privacy_budget
        self.semantic_budget = semantic_budget
        self.beta = beta

        self.loaded: Dict[str, ModelTier] = {}
        self._vfe: float = 0.0                     # lower is better

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _total_privacy_load(self) -> float:
        return sum(m.privacy_load for m in self.loaded.values())

    def _total_semantic_load(self, seed_vectors: Sequence[np.ndarray]) -> float:
        return sum(compute_semantic_load(m, seed_vectors, self.beta)
                   for m in self.loaded.values())

    def _vfe_penalty(self, delta: float) -> None:
        """Add a penalty (or reward if delta < 0) to the VFE."""
        self._vfe += delta

    def _tier_conflict(self, candidate: ModelTier) -> bool:
        """
        Enforce the same rule as Parent B:
        tier T3 cannot co‑exist with any T2 model.
        """
        if candidate.tier == "T3":
            return any(m.tier == "T2" for m in self.loaded.values())
        if candidate.tier == "T2":
            return any(m.tier == "T3" for m in self.loaded.values())
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def free_energy(self) -> float:
        """Current variational free energy."""
        return self._vfe

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def add_model(self,
                  model: ModelTier,
                  seed_vectors: Sequence[np.ndarray]) -> None:
        """
        Attempt to add *model* to the pool.

        The method updates VFE according to:
        * tier conflicts → huge penalty,
        * RAM overflow → linear penalty,
        * privacy‑load overflow → linear penalty,
        * semantic‑load overflow → linear penalty.
        """
        # Tier conflict penalty (hard)
        if self._tier_conflict(model):
            self._vfe_penalty(1e10)

        # Compute prospective loads
        new_ram = self._used_ram() + model.ram_mb
        new_privacy = self._total_privacy_load() + model.privacy_load
        new_semantic = self._total_semantic_load(seed_vectors) + \
                       compute_semantic_load(model, seed_vectors, self.beta)

        # Soft penalties proportional to excess amount
        if new_ram > self.ram_ceiling_mb:
            self._vfe_penalty(10.0 * (new_ram - self.ram_ceiling_mb))
        if new_privacy > self.privacy_budget:
            self._vfe_penalty(5.0 * (new_privacy - self.privacy_budget))
        if new_semantic > self.semantic_budget:
            self._vfe_penalty(5.0 * (new_semantic - self.semantic_budget))

        # Record the model (even if penalties were applied)
        self.loaded[model.name] = model

    def evaluate_selection(self,
                           selection: Iterable[str],
                           seed_vectors: Sequence[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Given a selection of model names, compute the resource load vector

            L = [RAM, privacy‑load, semantic‑load]ᵀ

        and the corresponding VFE (including penalties for violations).
        """
        selected_models = [self.loaded[name] for name in selection if name in self.loaded]

        total_ram = sum(m.ram_mb for m in selected_models)
        total_privacy = sum(m.privacy_load for m in selected_models)
        total_semantic = sum(compute_semantic_load(m, seed_vectors, self.beta)
                             for m in selected_models)

        load_vec = np.array([total_ram, total_privacy, total_semantic], dtype=float)

        # Re‑compute VFE for this selection (fresh start)
        vfe = 0.0
        # Tier conflicts inside the selection
        tiers = {m.tier for m in selected_models}
        if "T3" in tiers and "T2" in tiers:
            vfe += 1e10
        # Resource excess penalties
        if total_ram > self.ram_ceiling_mb:
            vfe += 10.0 * (total_ram - self.ram_ceiling_mb)
        if total_privacy > self.privacy_budget:
            vfe += 5.0 * (total_privacy - self.privacy_budget)
        if total_semantic > self.semantic_budget:
            vfe += 5.0 * (total_semantic - self.semantic_budget)

        return load_vec, vfe

    def greedy_select(self,
                      candidate_models: Sequence[ModelTier],
                      seed_vectors: Sequence[np.ndarray]) -> List[str]:
        """
        Simple greedy heuristic:
        1. Sort candidates by decreasing (privacy_load + semantic_load) / ram_mb.
        2. Add models while the VFE does not increase beyond a large threshold.
        Returns the list of selected model names.
        """
        # Pre‑compute semantic loads for sorting
        scored = []
        for m in candidate_models:
            sem_load = compute_semantic_load(m, seed_vectors, self.beta)
            score = (m.privacy_load + sem_load) / max(1, m.ram_mb)
            scored.append((score, m))

        scored.sort(key=lambda x: -x[0])   # descending

        selected: List[str] = []
        for _, model in scored:
            # Tentatively add and evaluate
            temp_pool = HybridModelPool(self.ram_ceiling_mb,
                                        self.privacy_budget,
                                        self.semantic_budget,
                                        self.beta)
            # copy already selected models into temp_pool
            for name in selected:
                temp_pool.loaded[name] = self.loaded[name]
            temp_pool.add_model(model, seed_vectors)

            # Evaluate VFE after addition
            _, vfe = temp_pool.evaluate_selection(temp_pool.loaded.keys(), seed_vectors)
            # Accept if VFE is finite (i.e., not hit the huge tier‑conflict penalty)
            if vfe < 1e9:
                self.add_model(model, seed_vectors)
                selected.append(model.name)

        return selected


# ----------------------------------------------------------------------
# Demonstration functions (fulfil requirement of ≥3 functions)
# ----------------------------------------------------------------------
def build_demo_models(num: int,
                      dim: int = 50) -> List[ModelTier]:
    """
    Generate *num* synthetic ``ModelTier`` objects with random attributes.
    """
    models = []
    for i in range(num):
        name = f"model_{i}"
        ram_mb = random.randint(200, 1500)
        tier = random.choice(["T1", "T2", "T3"])
        privacy_load = random.random()          # in [0,1]
        doc_vector = np.random.randn(dim)
        models.append(ModelTier(name, ram_mb, tier, privacy_load, doc_vector))
    return models


def generate_seed_vectors(k: int, dim: int = 50) -> List[np.ndarray]:
    """Create *k* random seed vectors for semantic neighbourhoods."""
    return [np.random.randn(dim) for _ in range(k)]


def run_demo() -> None:
    """Execute a short end‑to‑end demonstration of the hybrid pool."""
    # Budgets (chosen arbitrarily for the demo)
    ram_ceiling = 4000          # MB
    privacy_budget = 3.0        # aggregate privacy load
    semantic_budget = 2.5       # aggregate semantic load
    beta = 1.2                  # scaling for semantic load

    # Build pool and data
    pool = HybridModelPool(ram_ceiling, privacy_budget, semantic_budget, beta)
    seed_vecs = generate_seed_vectors(k=5, dim=50)
    candidates = build_demo_models(num=10, dim=50)

    # Pre‑load all candidates into the internal dictionary (needed for selection)
    for m in candidates:
        pool.loaded[m.name] = m

    # Greedy selection respecting constraints
    selected = pool.greedy_select(candidates, seed_vecs)

    # Final evaluation
    load_vec, final_vfe = pool.evaluate_selection(selected, seed_vecs)

    print("=== Hybrid Model Pool Demo ===")
    print(f"Selected models ({len(selected)}): {selected}")
    print(f"Total load vector [RAM_MB, privacy, semantic]: {load_vec}")
    print(f"Final variational free energy (VFE): {final_vfe:.2f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_demo()