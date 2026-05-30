# DARWIN HAMMER — match 1556, survivor 6
# gen: 4
# parent_a: variational_free_energy.py (gen0)
# parent_b: hybrid_hybrid_label_foundry_hybrid_hybrid_decisi_m172_s1.py (gen3)
# born: 2026-05-29T23:37:33Z

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
# Data structures
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
# Core Gaussian utilities
# ---------------------------------------------------------------------------


def kl_gaussian(
    mu_q: np.ndarray | float,
    sigma_q: np.ndarray | float,
    mu_p: np.ndarray | float,
    sigma_p: np.ndarray | float,
) -> float:
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
    return kl_gaussian(mu_q, sigma_q, mu_p, sigma_p) - log_likelihood


# ---------------------------------------------------------------------------
# Bridge utilities
# ---------------------------------------------------------------------------


def recovery_priority(morph: Morphology) -> float:
    volume = morph.length * morph.width * morph.height
    priority = (0.5 * volume + 0.5 * morph.mass) / (1.0 + 0.5 * volume + 0.5 * morph.mass)
    return float(priority)


def entropy_modulation(txt: TextFeatures) -> float:
    total = txt.evidence_count + txt.planning_count + txt.delay_count
    if total == 0:
        return 0.0
    richness = (txt.evidence_count + 0.8 * txt.planning_count - 0.5 * txt.delay_count) / total
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
    w_r = 1.0 + recovery_priority(morph)
    w_e = 1.0 + entropy_modulation(txt)

    tau_q = w_r * w_e / (prior_sigma ** 2)
    tau_o = 1.0 / (obs_sigma ** 2)

    tau_post = tau_q + tau_o
    mu_post = (tau_q * prior_mu + tau_o * obs_mu) / tau_post
    sigma_post = math.sqrt(1.0 / tau_post)

    new_mu = (1 - lr) * prior_mu + lr * mu_post
    new_sigma = math.sqrt((1 - lr) * (prior_sigma ** 2) + lr * (sigma_post ** 2))

    return new_mu, new_sigma


def hybrid_aggregate_labels(
    batches: List[List[LabelingFunctionResult]],
    morph_map: Dict[str, Morphology],
    txt_map: Dict[str, TextFeatures],
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
) -> List[ProbabilisticLabel]:
    vote_dict: Dict[str, List[int]] = {}
    for batch in batches:
        for res in batch:
            if res.label not in (0, 1):
                continue
            vote_dict.setdefault(res.doc_id, []).append(res.label)

    out: List[ProbabilisticLabel] = []
    for doc_id, votes in vote_dict.items():
        n_pos = votes.count(1)
        n_neg = len(votes) - n_pos
        if n_pos + n_neg == 0:
            continue

        obs_mu = n_pos / (n_pos + n_neg)
        obs_sigma = math.sqrt(obs_mu * (1 - obs_mu) / (n_pos + n_neg))

        morph = morph_map.get(doc_id)
        txt = txt_map.get(doc_id)
        if morph is None or txt is None:
            continue

        posterior_mu, posterior_sigma = hybrid_belief_update(
            prior_mu, prior_sigma, obs_mu, obs_sigma, morph, txt
        )
        confidence = 1.0 / (1.0 + math.exp(-posterior_mu))

        out.append(ProbabilisticLabel(doc_id, 1 if confidence > 0.5 else 0, confidence))

    return out