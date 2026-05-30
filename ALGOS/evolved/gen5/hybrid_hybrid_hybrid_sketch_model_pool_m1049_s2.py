# DARWIN HAMMER — match 1049, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_sketches_rlct_hybrid_hybrid_bayes__m98_s2.py (gen4)
# parent_b: model_pool.py (gen0)
# born: 2026-05-29T23:32:34Z

"""Hybrid Sketch‑Bayesian‑RLCT + Model‑Pool Resource Manager.

Parents:
- hybrid_hybrid_sketches_rlct_hybrid_hybrid_bayes__m98_s2.py (sketch‑based
  log‑likelihood, Gaussian conjugate update, RLCT asymptotics, curvature‑aware
  bandit).
- model_pool.py (RAM‑constrained loading/eviction of tiered models).

Mathematical bridge:
Both parents manipulate a *resource budget* that can be expressed as a scalar.
In the sketch side the posterior covariance Σ determines an effective
dimension *m = dim(θ)* and enters the RLCT estimate  

    RLCT = λ·log n − (m−1)·log log n ,   λ = ½·log |Σ| .

The RLCT value quantifies the “information pressure” of the current data.
In the model‑pool side the RAM ceiling is a hard budget.  We map the RLCT
pressure to a *budget factor* β = sigmoid(RLCT) ∈ (0,1) that scales the
available RAM for loading new models.  Moreover, the MinHash‑based Ollivier‑Ricci
curvature matrix C_{ij} (Jaccard similarity of sketch signatures) is used as a
bandit‑style weighting when selecting which models to keep under the scaled
budget.  The resulting system jointly updates the Bayesian posterior,
estimates RLCT, and allocates model resources in a mathematically coupled way.

The module provides three high‑level hybrid operations:
1. ``build_hybrid_sketch`` – builds Count‑Min, HyperLogLog, and MinHash
   structures from a stream of string items.
2. ``bayesian_sketch_update`` – performs a Gaussian conjugate update using
   sketch‑derived pseudo‑observations and returns posterior parameters.
3. ``allocate_models_via_rlct`` – given a ``ModelPool`` and a list of candidate
   ``ModelTier`` objects, loads a subset respecting the RLCT‑scaled RAM ceiling
   and curvature‑aware bandit preferences.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Sketch primitives (adapted from Parent A)
# ----------------------------------------------------------------------


def _hash(item: str, seed: int) -> int:
    """Deterministic integer hash for a given seed."""
    h = hashlib.blake2b(digest_size=8)
    h.update(item.encode("utf-8"))
    h.update(seed.to_bytes(2, "little"))
    return int.from_bytes(h.digest(), "little")


def count_min_sketch(
    items: Iterable[str], width: int = 128, depth: int = 5
) -> List[List[int]]:
    """Very lightweight Count‑Min sketch returning a depth×width matrix."""
    sketch = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            col = _hash(item, d) % width
            sketch[d][col] += 1
    return sketch


def hyperloglog_estimate(items: Iterable[str]) -> int:
    """Simplified distinct‑count estimator (exact for modest inputs)."""
    # Real HLL would use registers; here we just count distinct items.
    return len(set(items))


def minhash_signature(items: Iterable[str], num_perm: int = 64) -> List[int]:
    """Return a MinHash signature (list of minima of hashed permutations)."""
    sig = [sys.maxsize] * num_perm
    for item in items:
        for i in range(num_perm):
            hv = _hash(item, i)
            if hv < sig[i]:
                sig[i] = hv
    return sig


def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    assert len(sig1) == len(sig2)
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def build_hybrid_sketch(items: Iterable[str]) -> Dict[str, object]:
    """Construct Count‑Min, HyperLogLog, and MinHash structures."""
    items = list(items)  # materialise once
    cm = count_min_sketch(items)
    hll = hyperloglog_estimate(items)
    mh = minhash_signature(items)
    return {"count_min": cm, "hyperloglog": hll, "minhash": mh, "raw_items": items}


# ----------------------------------------------------------------------
# Bayesian update using sketch‑derived pseudo‑observations
# ----------------------------------------------------------------------


def _sketch_pseudo_observations(
    sketch: Dict[str, object], dim: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert sketch statistics into a synthetic observation vector ``y`` and
    observation precision matrix ``Λ`` (diagonal) for a Gaussian likelihood.
    The total count from Count‑Min is used as the observation magnitude;
    the HyperLogLog estimate provides a variance proxy.
    """
    cm = sketch["count_min"]
    total_counts = sum(sum(row) for row in cm)
    hll = sketch["hyperloglog"]
    # Scale pseudo‑observation to the dimension
    y = np.full((dim,), total_counts / dim, dtype=float)
    # Use distinct count to set variance; larger distinct count → lower variance
    variance = max(1.0, hll)  # avoid zero
    precision = np.diag(np.full(dim, 1.0 / variance))
    return y, precision


def bayesian_sketch_update(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    sketch: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a conjugate Gaussian update where the likelihood is defined by
    sketch‑derived pseudo‑observations.  Returns posterior mean and covariance.
    """
    dim = prior_mean.shape[0]
    y, Λ = _sketch_pseudo_observations(sketch, dim)

    # Compute posterior precision = Σ₀⁻¹ + Λ
    Σ0_inv = np.linalg.inv(prior_cov)
    Σ_post_inv = Σ0_inv + Λ
    Σ_post = np.linalg.inv(Σ_post_inv)

    # Posterior mean μ_post = Σ_post ( Σ₀⁻¹ μ₀ + Λ y )
    μ_post = Σ_post @ (Σ0_inv @ prior_mean + Λ @ y)

    return μ_post, Σ_post


# ----------------------------------------------------------------------
# RLCT estimate (Parent A) and curvature‑aware bandit weight (Parent A)
# ----------------------------------------------------------------------


def hybrid_rlct_estimate(posterior_cov: np.ndarray, sketch: Dict[str, object]) -> float:
    """
    Compute the Real Log‑Canonical Threshold (RLCT) estimate using the posterior
    covariance and the total observation count extracted from the sketch.
    """
    dim = posterior_cov.shape[0]
    # λ = ½·log|Σ|
    sign, logdet = np.linalg.slogdet(posterior_cov)
    if sign <= 0:
        # Numerical safeguard: fallback to trace‑based proxy
        λ = 0.5 * np.log(np.trace(posterior_cov) + 1e-12)
    else:
        λ = 0.5 * logdet

    # Effective sample size n from Count‑Min total count
    total_counts = sum(sum(row) for row in sketch["count_min"])
    n = max(total_counts, 2)  # avoid log(0) and loglog(≤1)

    rlct = λ * math.log(n) - (dim - 1) * math.log(math.log(n))
    return rlct


def curvature_matrix(models: List["ModelTier"], minhash_sig: List[int]) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature between models using Jaccard similarity
    of their name‑derived MinHash signatures.  The result is a symmetric matrix
    with entries in [0,1].
    """
    num = len(models)
    C = np.zeros((num, num))
    for i, mi in enumerate(models):
        sig_i = minhash_signature([mi.name], len(minhash_sig))
        for j, mj in enumerate(models):
            if i <= j:
                sig_j = minhash_signature([mj.name], len(minhash_sig))
                sim = jaccard_similarity(sig_i, sig_j)
                C[i, j] = C[j, i] = sim
    return C


# ----------------------------------------------------------------------
# Model pool (adapted from Parent B) with RLCT‑scaled budget
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded under current constraints."""


class ModelPool:
    """
    Manages a set of loaded models under a RAM ceiling.  The ceiling can be
    dynamically scaled by an RLCT‑derived factor β ∈ (0,1) to reflect data
    pressure.
    """

    def __init__(self, ram_ceiling_mb: int = 6000):
        self._ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    # ------------------------------------------------------------------
    # Basic pool operations (unchanged from Parent B)
    # ------------------------------------------------------------------
    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model if constraints allow; otherwise raise."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise ModelLoadError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self._ram_ceiling_mb:
            raise ModelLoadError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """Evict oldest loaded models until the new model fits."""
        while self.loaded and model.ram_mb + self._used() > self._ram_ceiling_mb:
            evicted_name = next(iter(self.loaded))
            self.loaded.pop(evicted_name)
        self.load(model)

    # ------------------------------------------------------------------
    # RLCT‑aware allocation
    # ------------------------------------------------------------------
    def _scaled_ceiling(self, rlct: float) -> int:
        """
        Scale the static RAM ceiling by a sigmoid of the RLCT value.
        Larger RLCT → tighter budget (β close to 1), smaller RLCT → relaxed budget.
        """
        beta = 1.0 / (1.0 + math.exp(-rlct / 10.0))  # smooth sigmoid
        return int(self._ram_ceiling_mb * beta)

    def allocate_models_via_rlct(
        self,
        candidates: List[ModelTier],
        rlct: float,
        curvature: np.ndarray,
    ) -> List[ModelTier]:
        """
        Select a subset of ``candidates`` to load, respecting the RLCT‑scaled RAM
        ceiling and using curvature as a bandit‑style weight.  The algorithm:

        1. Compute a utility score u_i = (ram_mb_i)⁻¹ * (1 + avg_curvature_i).
        2. Sort candidates by decreasing utility.
        3. Greedily load while the scaled budget permits, evicting if necessary.
        Returns the list of successfully loaded models.
        """
        scaled_ceiling = self._scaled_ceiling(rlct)
        used = self._used()

        # Pre‑compute average curvature per model (including self‑curvature = 1)
        avg_curv = curvature.mean(axis=1) if curvature.size else np.ones(len(candidates))

        utilities = [
            (i, (1.0 / cand.ram_mb) * (1.0 + avg_curv[i]))
            for i, cand in enumerate(candidates)
        ]
        utilities.sort(key=lambda x: x[1], reverse=True)

        loaded_models: List[ModelTier] = []
        for idx, _util in utilities:
            model = candidates[idx]
            if model.name in self.loaded:
                continue  # already present
            if model.ram_mb + used <= scaled_ceiling:
                self.loaded[model.name] = model
                used += model.ram_mb
                loaded_models.append(model)
            else:
                # Try eviction of lowest‑utility loaded model(s)
                while self.loaded and model.ram_mb + used > scaled_ceiling:
                    # evict the currently loaded model with smallest utility
                    evict_idx = min(
                        self.loaded.values(),
                        key=lambda m: (1.0 / m.ram_mb),
                    )
                    used -= evict_idx.ram_mb
                    self.loaded.pop(evict_idx.name)
                if model.ram_mb + used <= scaled_ceiling:
                    self.loaded[model.name] = model
                    used += model.ram_mb
                    loaded_models.append(model)
                else:
                    # Cannot fit even after eviction
                    continue
        return loaded_models


# ----------------------------------------------------------------------
# High‑level hybrid workflow
# ----------------------------------------------------------------------


def hybrid_workflow(
    items: Iterable[str],
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    model_candidates: List[ModelTier],
    pool: ModelPool,
) -> Tuple[np.ndarray, np.ndarray, float, List[ModelTier]]:
    """
    End‑to‑end hybrid routine:

    1. Build sketch from ``items``.
    2. Perform Bayesian update → posterior.
    3. Estimate RLCT.
    4. Compute curvature matrix from model names using the sketch's MinHash.
    5. Allocate models in ``pool`` according to RLCT‑scaled budget.

    Returns (posterior_mean, posterior_cov, rlct, loaded_models).
    """
    sketch = build_hybrid_sketch(items)
    post_mean, post_cov = bayesian_sketch_update(prior_mean, prior_cov, sketch)
    rlct = hybrid_rlct_estimate(post_cov, sketch)

    # Curvature based on model names and the global MinHash signature
    curv = curvature_matrix(model_candidates, sketch["minhash"])

    loaded = pool.allocate_models_via_rlct(model_candidates, rlct, curv)
    return post_mean, post_cov, rlct, loaded


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data stream
    data_stream = [f"sample_{i%50}" for i in range(1000)]

    # Prior for a 5‑dimensional parameter vector
    d = 5
    mu0 = np.zeros(d)
    sigma0 = np.eye(d) * 10.0

    # Model candidates (mirroring Parent B definitions)
    TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
    TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
    TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
    TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")
    candidates = [
        TIER_T1_QWEN_0_5B,
        TIER_T2_REASONING,
        TIER_T2_TOOL,
        TIER_T3_QWEN_7B,
    ]

    # Initialise pool with default ceiling (6000 MB)
    pool = ModelPool(ram_ceiling_mb=6000)

    # Run the hybrid pipeline
    posterior_mu,