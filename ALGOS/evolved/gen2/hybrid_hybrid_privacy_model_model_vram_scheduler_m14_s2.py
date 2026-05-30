# DARWIN HAMMER — match 14, survivor 2
# gen: 2
# parent_a: hybrid_privacy_model_pool_m7_s1.py (gen1)
# parent_b: model_vram_scheduler.py (gen0)
# born: 2026-05-29T23:22:41Z

"""Hybrid VRAM‑Privacy Scheduler

Parents:
- PARENT ALGORITHM A: ``hybrid_privacy_model_pool_m7_s1.py`` – provides
  ``reconstruction_risk_score`` (unique quasi‑identifiers / total records) and
  a simple differential‑privacy aggregate.
- PARENT ALGORITHM B: ``model_vram_scheduler.py`` – offers VRAM inspection,
  file‑receipt utilities and a budget‑oriented residency planner.

Mathematical bridge:
Both systems reason about limited resources.  A is a *probabilistic* risk
estimate 𝑟∈[0,1]; B is a *deterministic* memory consumption 𝑚 (MiB).  By treating
the risk as a probability that a model will be accessed, we can compute the
*expected* VRAM load:

    E[VRAM] = Σ_i ( r_i · m_i )

where r_i = reconstruction_risk_score for model i and m_i = model.ram_mb.
The hybrid planner uses this expectation together with the DP‑aggregate of
risks to decide which models to admit, evict or pre‑empt under a hard VRAM
budget.  The core equations are therefore a dot‑product (matrix multiplication)
and a summed (DP) aggregation, unifying the two topologies into a single
decision engine.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (adapted from both parents)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str


# Example tiers (mirroring parent A)
TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Parent A – probability that a record can be re‑identified."""
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """
    Parent A – deterministic aggregation (noise added later at runtime).
    Here we simply sum the values; the signature is kept for compatibility.
    """
    return sum(values)


def anonymize_for_indexing(record: Mapping[str, Any], redact_keys: set[str] | None = None) -> dict[str, Any]:
    """Parent A – redaction helper."""
    redact = redact_keys or {"email", "phone", "ssn", "secret", "token", "password"}
    return {k: ("<redacted>" if k.lower() in redact else v) for k, v in record.items()}


@dataclass(frozen=True)
class VramSlotPlan:
    """Parent B – a single allocation decision."""
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# GPU inspection (parent B, but safe fallback)
# ----------------------------------------------------------------------


def gpu_memory() -> dict[str, Any]:
    """Return a minimal GPU memory snapshot.  If ``nvidia-smi`` is unavailable,
    fall back to a synthetic 4 GB device."""
    if not shutil.which("nvidia-smi"):
        # Synthetic fallback – matches the default budget of parent B.
        return {
            "status": "fallback",
            "selected_index": 0,
            "total_mb": 4096,
            "used_mb": 0,
            "free_mb": 4096,
            "gpus": [{"index": 0, "name": "fallback-gpu", "total_mb": 4096, "used_mb": 0, "free_mb": 4096}],
        }

    cp = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,pstate",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    if cp.returncode != 0 or not cp.stdout.strip():
        return {"status": "error", "stderr": cp.stderr[-500:]}
    gpus: List[dict[str, Any]] = []
    for line in cp.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, total, used, free, driver, pstate = parts[:7]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "total_mb": int(total),
                "used_mb": int(used),
                "free_mb": int(free),
                "driver_version": driver,
                "pstate": pstate,
            }
        )
    if not gpus:
        return {"status": "error", "stdout": cp.stdout[-500:], "stderr": cp.stderr[-500:]}
    first = gpus[0]
    return {"status": "ok", "selected_index": first["index"], **first, "gpus": gpus}


# ----------------------------------------------------------------------
# Hybrid core mathematics
# ----------------------------------------------------------------------


def compute_model_risks(models: List[ModelTier], total_records: int) -> np.ndarray:
    """
    Compute a risk vector **r** for the supplied models.
    The number of unique quasi‑identifiers for a model is approximated by the
    count of already‑loaded models plus one (mirroring parent A's heuristic).
    """
    risks = []
    for i, _ in enumerate(models):
        unique_qi = i + 1  # simplistic proxy
        risk = reconstruction_risk_score(unique_qi, total_records)
        risks.append(risk)
    return np.array(risks, dtype=float)


def expected_vram_usage(models: List[ModelTier], risk_vector: np.ndarray) -> float:
    """
    Expected VRAM load  E = r·m  where:
        r – risk vector (probability of access)
        m – memory vector (MiB)
    """
    memory_vec = np.array([m.ram_mb for m in models], dtype=float)
    return float(risk_vector @ memory_vec)  # dot product


def hybrid_admission_decision(
    candidate_models: List[ModelTier],
    total_vram_budget_mb: int,
    total_records: int,
    epsilon: float = 1.0,
) -> List[ModelTier]:
    """
    Greedy admission based on descending risk‑to‑memory ratio.
    The DP‑aggregate of risks is used as a scaling factor to bias the decision.
    Returns the subset of models that can fit under the budget.
    """
    # Compute risks and the DP aggregate (sum) for scaling.
    risks = compute_model_risks(candidate_models, total_records)
    risk_sum = dp_aggregate(risks, epsilon=epsilon)

    # Ratio = (risk * scaling) / memory  – higher means more “valuable”.
    scaling = 1.0 + exp(-risk_sum)  # smooth function of total risk
    ratios = (risks * scaling) / np.array([m.ram_mb for m in candidate_models], dtype=float)

    # Sort indices by descending ratio.
    order = np.argsort(-ratios)
    admitted: List[ModelTier] = []
    used = 0
    for idx in order:
        model = candidate_models[idx]
        if used + model.ram_mb <= total_vram_budget_mb:
            admitted.append(model)
            used += model.ram_mb
    return admitted


def generate_hybrid_plan(models: List[ModelTier], total_vram_budget_mb: int, total_records: int) -> List[VramSlotPlan]:
    """
    Produce a list of :class:`VramSlotPlan` objects describing the hybrid
    allocation.  Each plan entry records the model name, estimated memory,
    and a reason derived from the risk‑to‑memory ratio.
    """
    admitted = hybrid_admission_decision(models, total_vram_budget_mb, total_records)
    risk_vec = compute_model_risks(models, total_records)

    plans: List[VramSlotPlan] = []
    for model in models:
        idx = models.index(model)
        risk = float(risk_vec[idx])
        action = "load" if model in admitted else "evict"
        reason = (
            f"risk={risk:.3f} {'high' if risk > 0.5 else 'low'}; "
            f"{'fits' if action == 'load' else 'exceeds'} budget"
        )
        plans.append(
            VramSlotPlan(
                artifact_id=model.name,
                artifact_kind="model",
                action=action,
                estimated_mb=model.ram_mb,
                reason=reason,
                detail={"risk": risk, "tier": model.tier},
            )
        )
    return plans


# ----------------------------------------------------------------------
# Demonstration helpers (three required functions)
# ----------------------------------------------------------------------


def demo_compute_expectation():
    """Show expected VRAM usage for a fixed set of models."""
    models = [TIER_T1_QWEN_0_5B, TIER_T2_REASONING, TIER_T2_TOOL]
    total_records = 10_000
    risks = compute_model_risks(models, total_records)
    exp_usage = expected_vram_usage(models, risks)
    print("Risk vector:", risks)
    print(f"Expected VRAM usage: {exp_usage:.2f} MiB")


def demo_hybrid_admission():
    """Demonstrate which models survive a 6 GB budget."""
    models = [TIER_T1_QWEN_0_5B, TIER_T2_REASONING, TIER_T2_TOOL, TIER_T3_QWEN_7B]
    budget_mb = 6_000
    total_records = 5_000
    admitted = hybrid_admission_decision(models, budget_mb, total_records)
    print(f"Budget: {budget_mb} MiB")
    print("Admitted models:", [m.name for m in admitted])


def demo_generate_plan():
    """Create a full hybrid plan and pretty‑print it."""
    models = [TIER_T1_QWEN_0_5B, TIER_T2_REASONING, TIER_T2_TOOL, TIER_T3_QWEN_7B]
    budget_mb = 8_000
    total_records = 12_000
    plans = generate_hybrid_plan(models, budget_mb, total_records)
    for p in plans:
        print(json.dumps(p.as_dict(), indent=2))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    print("=== Demo: Expected VRAM usage ===")
    demo_compute_expectation()
    print("\n=== Demo: Hybrid admission ===")
    demo_hybrid_admission()
    print("\n=== Demo: Full hybrid plan ===")
    demo_generate_plan()