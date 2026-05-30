# DARWIN HAMMER — match 1556, survivor 5
# gen: 4
# parent_a: variational_free_energy.py (gen0)
# parent_b: hybrid_hybrid_label_foundry_hybrid_hybrid_decisi_m172_s1.py (gen3)
# born: 2026-05-29T23:37:33Z

"""
Hybrid Variational Free Energy & Adaptive Labeling Fusion

Parent A: ``variational_free_energy.py`` – provides Gaussian KL divergence,
free‑energy computation and precision‑weighted belief updates (active inference).

Parent B: ``hybrid_hybrid_label_foundry_hybrid_hybrid_decisi_m172_s1.py`` – defines
a labeling pipeline with morphology‑derived *recovery priority* and
text‑feature‑derived *entropy modulation* that steer pruning probabilities.

Mathematical Bridge
-------------------
We treat the *recovery priority* as an additional precision factor that scales
the inverse variance of the variational distribution **q**.  The *entropy
modulation* rescales the surprise term ``-ln p(o)`` (the data‑fit component of
free energy).  The hybrid free‑energy therefore becomes


F_hybrid = w_r * KL[ N(μ_q, σ_q²) || N(μ_p, σ_p²) ]  -  w_e * ln p(o)


where ``w_r = 1 + recovery_priority`` and ``w_e = 1 + entropy_factor``.
Both weights are ≥ 1, ensuring that higher priority or richer text increases
the influence of the corresponding term.  Belief updates use the combined
precision ``τ = (1/σ_q²) * w_r * w_e`` to perform a precision‑weighted
gradient step, unifying perception (belief update) and action (label
pruning) into a single computation.

The module implements three core hybrid functions:
    * ``hybrid_free_energy`` – evaluates the weighted free‑energy.
    * ``hybrid_belief_update`` – performs a precision‑weighted Gaussian update.
    * ``hybrid_aggregate_labels`` – aggregates labeling‑function votes while
      computing confidences from the hybrid free‑energy.
"""

from __future__ import annotations

import sys
import random
import math
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data structures (from Parent B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # 0 or 1


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # in [0, 1]


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class TextFeatures:
    evidence_count: int
    planning_count: int
    delay_count: int


# ---------------------------------------------------------------------------
# Core Gaussian utilities (from Parent A)
# ---------------------------------------------------------------------------


def kl_gaussian(
    mu_q: np.ndarray | float,
    sigma_q: np.ndarray | float,
    mu_p: np.ndarray | float,
    sigma_p: np.ndarray | float,
) -> float:
    """KL divergence KL[N(μ_q,σ_q²) || N(μ_p,σ_p²)] for (possibly) vector Gaussians.

    The formula is summed over all dimensions when arrays are supplied.
    """
    mu_q = np.asarray(mu_q, dtype=float)
    sigma_q = np.asarray(sigma_q, dtype=float)
    mu_p = np.asarray(mu_p, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float)

    if np.any(sigma_q <= 0) or np.any(sigma_p <= 0):
        raise ValueError("Standard deviations must be strictly positive.")

    term1 = np.log(sigma_p / sigma_q)
    term2 = (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2)
    kl = term1 + term2 - 0.5
    return float(np.sum(kl))


def free_energy_gaussian(
    mu_q: np.ndarray | float,
    sigma_q: np.ndarray | float,
    mu_p: np.ndarray | float,
    sigma_p: np.ndarray | float,
    log_likelihood: float,
) -> float:
    """Standard variational free energy for Gaussian q and Gaussian prior p.

    F = KL[q||p] - log p(o)   (log_likelihood = log p(o))
    """
    return kl_gaussian(mu_q, sigma_q, mu_p, sigma_p) - log_likelihood


# ---------------------------------------------------------------------------
# Bridge utilities (Recovery priority & Entropy modulation)
# ---------------------------------------------------------------------------


def recovery_priority(morph: Morphology) -> float:
    """
    Compute a dimensionless priority weight from morphology.
    Larger volume and mass increase priority.
    Normalised to roughly [0, 1] for typical ranges.
    """
    volume = morph.length * morph.width * morph.height
    # Simple linear scaling and clipping
    priority = (0.5 * volume + 0.5 * morph.mass) / (1.0 + 0.5 * volume + 0.5 * morph.mass)
    return float(priority)


def entropy_modulation(txt: TextFeatures) -> float:
    """
    Compute an entropy‑modulation factor from text feature counts.
    The richer (more evidence/planning) the text, the higher the factor.
    """
    total = txt.evidence_count + txt.planning_count + txt.delay_count
    if total == 0:
        return 0.0
    # Weight evidence and planning positively, delay negatively
    richness = (txt.evidence_count + 0.8 * txt.planning_count - 0.5 * txt.delay_count) / total
    # Map to [0, 1] via sigmoid for stability
    factor = 1.0 / (1.0 + math.exp(-5 * (richness - 0.5)))
    return float(factor)


# ---------------------------------------------------------------------------
# Hybrid core functions
# ---------------------------------------------------------------------------


def hybrid_free_energy(
    mu_q: np.ndarray | float,
    sigma_q: np.ndarray | float,
    mu_p: np.ndarray | float,
    sigma_p: np.ndarray | float,
    log_likelihood: float,
    morph: Morphology,
    txt: TextFeatures,
) -> float:
    """
    Weighted free energy where:
        w_r = 1 + recovery_priority
        w_e = 1 + entropy_modulation
    """
    w_r = 1.0 + recovery_priority(morph)
    w_e = 1.0 + entropy_modulation(txt)

    base_kl = kl_gaussian(mu_q, sigma_q, mu_p, sigma_p)
    weighted_kl = w_r * base_kl
    weighted_surprise = -w_e * log_likelihood

    return weighted_kl + weighted_surprise


def hybrid_belief_update(
    prior_mu: float,
    prior_sigma: float,
    obs_mu: float,
    obs_sigma: float,
    morph: Morphology,
    txt: TextFeatures,
    lr: float = 0.5,
) -> Tuple[float, float]:
    """
    Precision‑weighted Gaussian update using hybrid weights.
    The effective precision of q is scaled by w_r * w_e.
    """
    w_r = 1.0 + recovery_priority(morph)
    w_e = 1.0 + entropy_modulation(txt)

    # Precisions (inverse variances)
    tau_q = w_r * w_e / (prior_sigma ** 2)
    tau_o = 1.0 / (obs_sigma ** 2)

    # Posterior precision and mean (standard Gaussian conjugacy)
    tau_post = tau_q + tau_o
    mu_post = (tau_q * prior_mu + tau_o * obs_mu) / tau_post
    sigma_post = math.sqrt(1.0 / tau_post)

    # Optional learning‑rate blend towards the posterior
    new_mu = (1 - lr) * prior_mu + lr * mu_post
    new_sigma = (1 - lr) * prior_sigma + lr * sigma_post

    return new_mu, new_sigma


def hybrid_aggregate_labels(
    batches: List[List[LabelingFunctionResult]],
    morph_map: Dict[str, Morphology],
    txt_map: Dict[str, TextFeatures],
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
) -> List[ProbabilisticLabel]:
    """
    Aggregate labeling‑function votes for each document, then compute a
    confidence using the hybrid free‑energy between the aggregated belief
    (as a Gaussian) and a simple Bernoulli‑like likelihood.

    For each document:
        * Count votes for label 1 (positive) and 0 (negative).
        * Treat the proportion as an observation mean with variance inversely
          proportional to the number of votes.
        * Perform a hybrid belief update.
        * Convert the posterior mean into a confidence (sigmoid) and emit a
          ``ProbabilisticLabel``.
    """
    # Collect votes per document
    vote_dict: Dict[str, List[int]] = {}
    for batch in batches:
        for res in batch:
            if res.label not in (0, 1):
                continue
            vote_dict.setdefault(res.doc_id, []).append(res.label)

    out: List[ProbabilisticLabel] = []
    for doc_id, votes in vote_dict.items():
        n = len(votes)
        if n == 0:
            continue
        pos = sum(votes)
        obs_mu = pos / n  # proportion of positive votes
        # Variance of a binomial proportion ≈ p(1-p)/n ; add epsilon for stability
        eps = 1e-6
        obs_sigma = math.sqrt(max(obs_mu * (1 - obs_mu) / n, eps))

        morph = morph_map.get(doc_id, Morphology(1.0, 1.0, 1.0, 1.0))
        txt = txt_map.get(doc_id, TextFeatures(0, 0, 0))

        # Hybrid belief update
        post_mu, post_sigma = hybrid_belief_update(
            prior_mu, prior_sigma, obs_mu, obs_sigma, morph, txt, lr=0.7
        )

        # Compute a synthetic log‑likelihood for free‑energy (Bernoulli)
        # log p(o) = log Bernoulli(pos|n, theta) with theta = post_mu (clipped)
        theta = np.clip(post_mu, eps, 1 - eps)
        log_likelihood = pos * math.log(theta) + (n - pos) * math.log(1 - theta)

        # Hybrid free energy (mainly for diagnostics; not used further here)
        _ = hybrid_free_energy(
            post_mu,
            post_sigma,
            prior_mu,
            prior_sigma,
            log_likelihood,
            morph,
            txt,
        )

        # Confidence as sigmoid of posterior mean
        confidence = 1.0 / (1.0 + math.exp(-10 * (post_mu - 0.5)))

        label = int(post_mu >= 0.5)
        out.append(ProbabilisticLabel(doc_id=doc_id, label=label, confidence=confidence))

    return out


# ---------------------------------------------------------------------------
# Simple decorator mirroring Parent B (kept for API compatibility)
# ---------------------------------------------------------------------------


def labeling_function(name: str | None = None):
    """Decorator that tags a function as a labeling function."""
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Construct toy morphology and text feature dictionaries for two docs
    morphologies = {
        "docA": Morphology(length=2.0, width=1.5, height=0.5, mass=0.8),
        "docB": Morphology(length=1.0, width=1.0, height=1.0, mass=1.2),
    }

    texts = {
        "docA": TextFeatures(evidence_count=5, planning_count=3, delay_count=1),
        "docB": TextFeatures(evidence_count=2, planning_count=1, delay_count=4),
    }

    # Simulated labeling‑function results
    batch1 = [
        LabelingFunctionResult(lf_name="lf1", doc_id="docA", label=1),
        LabelingFunctionResult(lf_name="lf2", doc_id="docA", label=0),
        LabelingFunctionResult(lf_name="lf1", doc_id="docB", label=0),
    ]

    batch2 = [
        LabelingFunctionResult(lf_name="lf3", doc_id="docA", label=1),
        LabelingFunctionResult(lf_name="lf2", doc_id="docB", label=1),
        LabelingFunctionResult(lf_name="lf3", doc_id="docB", label=0),
    ]

    result = hybrid_aggregate_labels(
        batches=[batch1, batch2],
        morph_map=morphologies,
        txt_map=texts,
        prior_mu=0.5,
        prior_sigma=0.3,
    )

    for pl in result:
        print(
            f"Document {pl.doc_id}: label={pl.label}, confidence={pl.confidence:.3f}"
        )

    # Demonstrate direct hybrid belief update on a synthetic observation
    mu, sigma = hybrid_belief_update(
        prior_mu=0.0,
        prior_sigma=1.0,
        obs_mu=0.8,
        obs_sigma=0.2,
        morph=morphologies["docA"],
        txt=texts["docA"],
    )
    print(f"\nHybrid belief after update: mu={mu:.4f}, sigma={sigma:.4f}")

    # Compute hybrid free energy for diagnostic purposes
    log_like = -0.2  # dummy log‑likelihood
    fe = hybrid_free_energy(
        mu_q=mu,
        sigma_q=sigma,
        mu_p=0.0,
        sigma_p=1.0,
        log_likelihood=log_like,
        morph=morphologies["docA"],
        txt=texts["docA"],
    )
    print(f"Hybrid free energy (diagnostic): {fe:.4f}")

    sys.exit(0)