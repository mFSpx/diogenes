# DARWIN HAMMER — match 562, survivor 1
# gen: 5
# parent_a: model_vram_scheduler.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_decisi_m176_s2.py (gen4)
# born: 2026-05-29T23:29:50Z

import os
import json
import random
import math
import re
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Re‑use dataclasses from the original VRAM scheduler (Parent A)
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
# Evidence extraction (Parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)


def extract_evidence_features(text: str) -> Dict[str, int]:
    """Count occurrences of evidence‑related tokens in *text*."""
    matches = EVIDENCE_RE.findall(text)
    return {"evidence_count": len(matches)}


# ----------------------------------------------------------------------
# Prior construction using curvature (mathematical bridge)
# ----------------------------------------------------------------------
def curvature_weight(i: int, j: int, scale: float = 0.1) -> float:
    """Simple surrogate for Ollivier‑Ricci curvature between two artifacts."""
    distance = abs(i - j)
    return math.exp(-scale * distance)


def build_prior(artifact_ids: List[str], base_memories: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a Gaussian prior (mean vector, covariance matrix) for VRAM usage.

    *Mean*  – the known static memory footprints (in MB).  
    *Covariance* – pairwise curvature‑derived couplings, modelling that
    loading one artifact influences the memory pressure of another.
    """
    mean = np.array(base_memories, dtype=float)

    n = len(artifact_ids)
    cov = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                cov[i, j] = mean[i] * 0.05  # 5 % self‑uncertainty
            else:
                curv = curvature_weight(i, j)
                cov[i, j] = curv * min(mean[i], mean[j]) * 0.02
    # Ensure positive‑definiteness by adding a tiny jitter and taking the symmetric part
    cov = (cov + cov.T) / 2 + np.eye(n) * 1e-3
    return mean, cov


# ----------------------------------------------------------------------
# Bayesian update (Kalman‑like) using evidence as observation
# ----------------------------------------------------------------------
def bayesian_vram_update(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    observed_vec: np.ndarray,
    observation_cov: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a Gaussian Bayesian update:

        Σ_post = (Σ_prior⁻¹ + Σ_obs⁻¹)⁻¹
        μ_post = Σ_post (Σ_prior⁻¹ μ_prior + Σ_obs⁻¹ y)

    Returns posterior mean and covariance.
    """
    inv_prior = np.linalg.inv(prior_cov)
    inv_obs = np.linalg.inv(observation_cov)

    post_cov = np.linalg.inv(inv_prior + inv_obs)
    post_mean = post_cov @ (inv_prior @ prior_mean + inv_obs @ observed_vec)

    return post_mean, post_cov


def observation_from_evidence(
    artifact_ids: List[str],
    base_memories: List[int],
    evidence_feat: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert evidence counts into a noisy observation vector.

    The observation is the base memory vector perturbed by a factor that grows
    with the number of evidence tokens (more evidence ⇒ lower observation noise).
    """
    count = evidence_feat.get("evidence_count", 0)
    factor = max(0.5, 1.0 - 0.05 * count)  # clamp to ≥0.5
    obs_vec = np.array(base_memories, dtype=float) * factor

    # Observation variance proportional to (1‑factor)
    var = (1.0 - factor) * np.array(base_memories, dtype=float) * 0.2 + 1.0
    obs_cov = np.diag(var)

    return obs_vec, obs_cov


# ----------------------------------------------------------------------
# Hybrid planning function (demonstrates the fused operation)
# ----------------------------------------------------------------------
def hybrid_vram_plan(
    artifact_info: List[Tuple[str, str, int]],
    budget_mb: int,
    evidence_text: str,
) -> List[VramSlotPlan]:
    """
    Generate an advisory VRAM plan using Bayesian‑fused evidence.

    *artifact_info*: List of (artifact_id, kind, static_memory_mb).
    *budget_mb*: Total VRAM budget.
    *evidence_text*: Free‑form log/text to extract evidence features.
    """
    ids, kinds, mems = zip(*artifact_info)
    ids = list(ids)
    kinds = list(kinds)
    mems = list(mems)

    # 1️⃣ Prior from static footprints + curvature coupling
    prior_mean, prior_cov = build_prior(ids, mems)

    # 2️⃣ Observation derived from textual evidence
    evidence_feat = extract_evidence_features(evidence_text)
    obs_vec, obs_cov = observation_from_evidence(ids, mems, evidence_feat)

    # 3️⃣ Bayesian update → posterior estimate of needed VRAM per artifact
    post_mean, post_cov = bayesian_vram_update(prior_mean, prior_cov, obs_vec, obs_cov)

    # Ensure estimates are non-negative
    post_mean = np.maximum(post_mean, 0)

    # 4️⃣ Greedy allocation respecting budget (simple heuristic)
    plan: List[VramSlotPlan] = []
    remaining = budget_mb
    for idx, (aid, kind, est_mb) in enumerate(zip(ids, kinds, post_mean.astype(int))):
        action = "keep" if est_mb <= remaining else "offload"
        reason = (
            "posterior_estimate_fits"
            if action == "keep"
            else "budget_exceeded_after_posterior"
        )
        detail = {
            "posterior_estimate_mb": int(est_mb),
            "evidence_count": evidence_feat.get("evidence_count", 0),
            "prior_estimate_mb": mems[idx],
        }
        plan.append(
            VramSlotPlan(
                artifact_id=aid,
                artifact_kind=kind,
                action=action,
                estimated_mb=int(est_mb),
                reason=reason,
                detail=detail,
            )
        )
        if action == "keep":
            remaining -= int(est_mb)

    # Append a summary slot for remaining budget
    plan.append(
        VramSlotPlan(
            artifact_id="__budget_summary__",
            artifact_kind="budget",
            action="summary",
            estimated_mb=remaining,
            reason="remaining_after_allocation",
            detail={"original_budget_mb": budget_mb},
        )
    )
    return plan


# ----------------------------------------------------------------------
# Additional helper demonstrating matrix‑level hybrid operation
# ----------------------------------------------------------------------
def curvature_covariance_matrix(n: int, scale: float = 0.08) -> np.ndarray:
    """
    Produce a curvature-derived covariance matrix.
    """
    cov = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            cov[i, j] = curvature_weight(i, j, scale)
    return (cov + cov.T) / 2