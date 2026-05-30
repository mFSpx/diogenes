# DARWIN HAMMER — match 562, survivor 2
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

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)

def extract_evidence_features(text: str) -> Dict[str, int]:
    matches = EVIDENCE_RE.findall(text)
    return {"evidence_count": len(matches)}

def curvature_weight(i: int, j: int, scale: float = 0.1) -> float:
    distance = abs(i - j)
    return math.exp(-scale * distance)

def build_prior(artifact_ids: List[str], base_memories: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.array(base_memories, dtype=float)

    n = len(artifact_ids)
    cov = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                cov[i, j] = mean[i] * 0.05
            else:
                curv = curvature_weight(i, j)
                cov[i, j] = curv * min(mean[i], mean[j]) * 0.02
    cov += np.eye(n) * 1e-3
    return mean, cov

def bayesian_vram_update(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    observed_vec: np.ndarray,
    observation_cov: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
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
    count = evidence_feat.get("evidence_count", 0)
    factor = max(0.5, 1.0 - 0.05 * count)
    obs_vec = np.array(base_memories, dtype=float) * factor

    var = (1.0 - factor) * np.array(base_memories, dtype=float) * 0.2 + 1.0
    obs_cov = np.diag(var)

    return obs_vec, obs_cov

def hybrid_vram_plan(
    artifact_info: List[Tuple[str, str, int]],
    budget_mb: int,
    evidence_text: str,
) -> List[VramSlotPlan]:
    ids, kinds, mems = zip(*artifact_info)
    ids = list(ids)
    kinds = list(kinds)
    mems = list(mems)

    prior_mean, prior_cov = build_prior(ids, mems)
    evidence_feat = extract_evidence_features(evidence_text)
    obs_vec, obs_cov = observation_from_evidence(ids, mems, evidence_feat)
    post_mean, _ = bayesian_vram_update(prior_mean, prior_cov, obs_vec, obs_cov)

    plan = []
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

def curvature_covariance_matrix(n: int, scale: float = 0.08) -> np.ndarray:
    cov = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                cov[i, j] = 1.0
            else:
                curv = math.exp(-scale * abs(i - j))
                cov[i, j] = curv
    return cov

def improved_hybrid_vram_plan(
    artifact_info: List[Tuple[str, str, int]],
    budget_mb: int,
    evidence_text: str,
) -> List[VramSlotPlan]:
    ids, kinds, mems = zip(*artifact_info)
    ids = list(ids)
    kinds = list(kinds)
    mems = list(mems)

    prior_mean, prior_cov = build_prior(ids, mems)
    evidence_feat = extract_evidence_features(evidence_text)
    obs_vec, obs_cov = observation_from_evidence(ids, mems, evidence_feat)
    post_mean, _ = bayesian_vram_update(prior_mean, prior_cov, obs_vec, obs_cov)

    # Improved greedy allocation with sorting
    sorted_indices = sorted(range(len(post_mean)), key=lambda i: post_mean[i])
    plan = []
    remaining = budget_mb
    for idx in sorted_indices:
        aid, kind, est_mb = ids[idx], kinds[idx], post_mean[idx]
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