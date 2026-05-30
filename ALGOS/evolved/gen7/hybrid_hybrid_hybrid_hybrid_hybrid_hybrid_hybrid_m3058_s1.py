# DARWIN HAMMER — match 3058, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m2581_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_jepa_e_hybrid_hybrid_hybrid_m2161_s3.py (gen6)
# born: 2026-05-29T23:47:35Z

"""Hybrid Memory‑Similarity‑Pool Scheduler

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m2581_s2.py (weekday sinusoidal
  weighting, Bayesian marginal, SSIM similarity)
- hybrid_hybrid_hybrid_jepa_e_hybrid_hybrid_hybrid_m2161_s3.py (model pool with
  RAM ceiling, LRU eviction, energy bookkeeping)

Mathematical bridge:
Both parents treat resource distribution as a weighted probability problem.
Parent A produces a weekday‑dependent stochastic vector **w** and a Bayesian
marginal **m** for each task; it also yields a similarity factor **s** (SSIM‑like)
that refines the allocation.  Parent B manages a finite memory pool **C**
with an energy term **E** that rewards successful loads and penalises overflow.

The hybrid formulation multiplies the three probabilistic ingredients

    a_i = w_i · m_i · (s_i + ε)

to obtain a raw allocation score **a**.  After normalising **a** we map it to the
available memory **C** and let the **ModelPool** admit the highest‑scoring
models, evicting the least‑recently used ones when the ceiling would be breached.
The pool’s energy is updated with a simple reward/penalty law, thus unifying the
core topologies of both ancestors into a single system."""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core structures from Parent B
# ----------------------------------------------------------------------


class ModelTier:
    """Immutable description of a model that can be loaded into the pool."""
    __slots__ = ("name", "ram_mb", "tier")

    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

    def __repr__(self) -> str:
        return f"ModelTier(name={self.name!r}, ram_mb={self.ram_mb}, tier={self.tier!r})"


class ModelPool:
    """
    Manages a set of loaded models under a RAM ceiling.
    Energy:
        • positive → penalties (overflow, tier conflict)
        • negative → rewards (successful load, useful eviction)
    """

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb: int = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._energy: float = 0.0
        self._access_order: List[str] = []          # LRU list

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _used(self) -> int:
        """Current RAM consumption in MB."""
        return sum(m.ram_mb for m in self.loaded.values())

    def _record_access(self, name: str) -> None:
        """Refresh LRU order for *name*."""
        if name in self._access_order:
            self._access_order.remove(name)
        self._access_order.append(name)

    def _evict_lru(self) -> ModelTier | None:
        """Evict the least‑recently used model; return it or None."""
        if not self._access_order:
            return None
        lru_name = self._access_order.pop(0)
        evicted = self.loaded.pop(lru_name)
        # reward for freeing memory
        self._energy -= evicted.ram_mb * 0.01
        return evicted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, model: ModelTier) -> bool:
        """
        Attempt to load *model* respecting the RAM ceiling.
        Returns True on success, False on failure (after possible eviction).
        """
        # tier‑conflict penalty (simple example: disallow two T3 together)
        if model.tier == "T3" and any(m.tier == "T3" for m in self.loaded.values()):
            self._energy += 5.0          # penalty
            return False

        # Try direct load
        if self._used() + model.ram_mb <= self.ram_ceiling_mb:
            self.loaded[model.name] = model
            self._record_access(model.name)
            self._energy -= model.ram_mb * 0.01   # small cost
            return True

        # Evict until we fit or nothing left
        while self._used() + model.ram_mb > self.ram_ceiling_mb:
            evicted = self._evict_lru()
            if evicted is None:
                break
        if self._used() + model.ram_mb <= self.ram_ceiling_mb:
            self.loaded[model.name] = model
            self._record_access(model.name)
            self._energy -= model.ram_mb * 0.01
            return True

        # Still cannot fit
        self._energy += 10.0               # heavy penalty
        return False

    def access(self, name: str) -> ModelTier | None:
        """Mark *name* as recently used and return the model if present."""
        model = self.loaded.get(name)
        if model:
            self._record_access(name)
        return model

    @property
    def energy(self) -> float:
        return self._energy

    def status(self) -> Tuple[int, int, float]:
        """Return (used_mb, free_mb, energy)."""
        used = self._used()
        return used, self.ram_ceiling_mb - used, self._energy


# ----------------------------------------------------------------------
# Functions from Parent A
# ----------------------------------------------------------------------


def weekday_weight_vector(groups: List[str], dow: int) -> np.ndarray:
    """
    Normalised sinusoidal weight vector for *groups* based on weekday ``dow``.
    The vector is row‑stochastic (sums to 1).

    Parameters
    ----------
    groups : List[str]
        Identifiers for the groups/tasks.
    dow : int
        Day of week where 0 = Sunday … 6 = Saturday.

    Returns
    -------
    np.ndarray
        Weight vector of shape (len(groups),).
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight = raw / raw.sum()
    return weight


def compute_bayesian_marginal(feature_vec: np.ndarray, prior: float = 0.5) -> float:
    """
    Simple Bayesian marginal for a task given its feature vector.
    The likelihood is approximated by the mean of the (clipped) feature vector.

    Parameters
    ----------
    feature_vec : np.ndarray
        1‑D array of non‑negative features (already normalised to [0,1]).
    prior : float, optional
        Prior probability of the task being important.

    Returns
    -------
    float
        Posterior marginal probability in [0,1].
    """
    if feature_vec.ndim != 1:
        raise ValueError("feature_vec must be 1‑D")
    likelihood = np.clip(feature_vec.mean(), 0.0, 1.0)
    numerator = prior * likelihood
    denominator = numerator + (1.0 - prior) * (1.0 - likelihood)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def ssim_similarity(profile: np.ndarray, reference: np.ndarray, C1: float = 0.01, C2: float = 0.03) -> float:
    """
    SSIM‑like similarity between *profile* and *reference*.
    The implementation follows the classic SSIM formula but operates on
    1‑D vectors.

    Returns a value in [0,1].
    """
    mu_x = profile.mean()
    mu_y = reference.mean()
    sigma_x = profile.var()
    sigma_y = reference.var()
    sigma_xy = ((profile - mu_x) * (reference - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


# ----------------------------------------------------------------------
# Hybrid orchestration
# ----------------------------------------------------------------------


def hybrid_score_vector(
    models: List[ModelTier],
    dow: int,
    feature_map: Dict[str, np.ndarray],
    profile_map: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute the raw hybrid scores ``a_i = w_i * m_i * (s_i + ε)`` for each model.

    Parameters
    ----------
    models : List[ModelTier]
        Candidate models.
    dow : int
        Day‑of‑week (0‑6) for the weekday weighting.
    feature_map : Dict[str, np.ndarray]
        Mapping from model name to its feature vector (for Bayesian marginal).
    profile_map : Dict[str, np.ndarray]
        Mapping from model name to its performance profile (for SSIM).

    Returns
    -------
    np.ndarray
        Normalised score vector (sums to 1) of length ``len(models)``.
    """
    epsilon = 1e-6
    names = [m.name for m in models]

    # 1️⃣ Weekday sinusoidal weights
    w = weekday_weight_vector(names, dow)                     # shape (n,)

    # 2️⃣ Bayesian marginal per model
    m = np.array([compute_bayesian_marginal(feature_map.get(n, np.array([0.0]))) for n in names])

    # 3️⃣ SSIM similarity to the mean profile
    # Build a reference profile as the mean of all provided profiles
    all_profiles = np.stack([profile_map.get(n, np.zeros(1)) for n in names])
    if all_profiles.shape[1] == 0:
        # degenerate case – no profile information
        s = np.ones_like(w)
    else:
        reference = all_profiles.mean(axis=0)
        s = np.array([ssim_similarity(profile_map.get(n, np.zeros_like(reference)), reference) for n in names])

    # Hybrid raw scores
    raw = w * m * (s + epsilon)

    # Normalise to a probability‑like vector
    if raw.sum() == 0.0:
        # fallback to uniform distribution
        return np.full_like(raw, 1.0 / raw.size)
    return raw / raw.sum()


def allocate_models_hybrid(
    pool: ModelPool,
    dow: int,
    feature_map: Dict[str, np.ndarray],
    profile_map: Dict[str, np.ndarray],
    candidate_models: List[ModelTier],
) -> List[str]:
    """
    Allocate models into *pool* based on hybrid scores.
    Models are considered in descending order of their hybrid score; each
    successful load updates the pool's energy.  The function returns the list
    of model names that are finally resident in the pool.

    Parameters
    ----------
    pool : ModelPool
        The memory pool to be populated.
    dow : int
        Current day‑of‑week.
    feature_map : Dict[str, np.ndarray]
        Feature vectors for Bayesian marginal.
    profile_map : Dict[str, np.ndarray]
        Performance profiles for SSIM.
    candidate_models : List[ModelTier]
        All models that could be loaded.

    Returns
    -------
    List[str]
        Names of models that are loaded after the allocation pass.
    """
    scores = hybrid_score_vector(candidate_models, dow, feature_map, profile_map)
    # Pair each model with its score and sort descending
    scored_models = sorted(zip(candidate_models, scores), key=lambda x: x[1], reverse=True)

    for model, score in scored_models:
        success = pool.load(model)
        # Reward successful load proportional to its hybrid confidence
        if success:
            pool._energy -= score * 0.5          # small negative energy (reward)
        else:
            pool._energy += (1.0 - score) * 0.5   # penalty for rejection

    return list(pool.loaded.keys())


def hybrid_status_report(pool: ModelPool) -> str:
    """Pretty‑print the current status of the ModelPool."""
    used, free, energy = pool.status()
    lines = [
        f"Memory used: {used} MB / {pool.ram_ceiling_mb} MB",
        f"Free memory: {free} MB",
        f"Pool energy: {energy:.2f}",
        f"Loaded models ({len(pool.loaded)}):",
    ]
    for name, model in pool.loaded.items():
        lines.append(f"  - {name} ({model.ram_mb} MB, tier={model.tier})")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(0)

    # Create a pool with a modest ceiling
    pool = ModelPool(ram_ceiling_mb=2000)

    # Generate dummy candidate models
    candidates = [
        ModelTier(name=f"model_{i}", ram_mb=random.randint(100, 500), tier=random.choice(["T1", "T2", "T3"]))
        for i in range(10)
    ]

    # Dummy feature vectors (10‑dim, values in [0,1])
    feature_map = {m.name: np.random.rand(10) for m in candidates}

    # Dummy performance profiles (15‑dim)
    profile_map = {m.name: np.random.rand(15) for m in candidates}

    # Run allocation for a Wednesday (dow=3)
    allocated = allocate_models_hybrid(pool, dow=3, feature_map=feature_map,
                                       profile_map=profile_map, candidate_models=candidates)

    # Print results
    print("=== Hybrid Allocation Result ===")
    print(hybrid_status_report(pool))
    print("\nAllocated models:", allocated)