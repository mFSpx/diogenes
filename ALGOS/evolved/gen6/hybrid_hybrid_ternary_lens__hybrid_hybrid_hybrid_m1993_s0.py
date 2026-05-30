# DARWIN HAMMER — match 1993, survivor 0
# gen: 6
# parent_a: hybrid_ternary_lens_audit_hybrid_hybrid_privac_m154_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_fisher_locali_m711_s0.py (gen5)
# born: 2026-05-29T23:40:24Z

"""Hybrid Ternary Lens Audit with Fisher‑Weighted Risk & Temporal Scheduling.

Parents:
- hybrid_ternary_lens_audit_hybrid_hybrid_privac_m154_s0.py (A)
- hybrid_hybrid_hybrid_hybrid_fisher_locali_m711_s0.py (B)

Mathematical bridge:
Both parents define a *reconstruction risk score*  r = unique_qi / total_records .
Parent B introduces a Fisher‑information based weighting w(θ) = exp(‑½·((θ‑μ)/σ)²) (Gaussian beam),
while Parent A uses that risk to predict RAM/VRAM exhaustion for model loading/eviction.
The hybrid therefore:
1. Computes r for each model tier.
2. Treats the tier’s VRAM usage as the “θ” argument of the Gaussian beam,
   with μ set to a target VRAM budget and σ controlling scheduling aggressiveness.
3. Forms a Fisher‑weighted risk  f = r·w(θ) .
4. Uses f to drive a temporal scheduler that respects a global VRAM budget and
   analyses how f evolves over time (chronological extraction).

The module provides three core hybrid functions:
- `weighted_reconstruction_risk`
- `vram_scheduler`
- `temporal_risk_analysis`
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Dict
import json
import random
import sys
import math
import numpy as np
import statistics

# ----------------------------------------------------------------------
# Shared data structures (from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1", 1024)
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2", 2048)
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2", 2048)
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3", 4096)

# ----------------------------------------------------------------------
# Parent A utilities (trimmed)
# ----------------------------------------------------------------------
def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def load_manifest(path: Path) -> dict[str, Any]:
    """Load a JSON manifest and validate classifications."""
    data = json.loads(path.read_text(encoding="utf-8"))
    # Simplified validation – real code would check against CLASSIFICATIONS
    return data

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Base risk: proportion of unique quasi‑identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

# ----------------------------------------------------------------------
# Parent B utilities (trimmed)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian kernel used as a Fisher‑information weighting."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, mu: float, sigma: float) -> float:
    """Convenient wrapper around gaussian_beam for Fisher weighting."""
    return gaussian_beam(theta, mu, sigma)

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def weighted_reconstruction_risk(
    model_tiers: Iterable[ModelTier],
    unique_qi: int,
    total_records: int,
    vram_target: int,
    sigma: float = 500.0,
) -> Dict[str, float]:
    """
    Compute Fisher‑weighted reconstruction risk for each model tier.

    For a tier with VRAM usage v, the weight is w = exp(-½·((v‑vram_target)/σ)²).
    The final score is f = r·w where r is the base reconstruction risk.

    Returns a mapping ``tier_name -> weighted risk``.
    """
    base_risk = reconstruction_risk_score(unique_qi, total_records)
    weighted: Dict[str, float] = {}
    for tier in model_tiers:
        weight = fisher_score(tier.vram_mb, vram_target, sigma)
        weighted[tier.name] = base_risk * weight
    return weighted


def vram_scheduler(
    weighted_risks: Dict[str, float],
    tiers_lookup: Dict[str, ModelTier],
    global_vram_budget_mb: int,
) -> Tuple[List[ModelTier], List[ModelTier]]:
    """
    Decide which models to *load* and which to *evict* based on weighted risk.

    - Models with higher weighted risk are prioritized for loading.
    - The function greedily adds models until the VRAM budget would be exceeded,
      then marks the remaining models for eviction.

    Returns ``(loaded, evicted)`` lists of ``ModelTier`` objects.
    """
    # Sort tiers by descending weighted risk
    sorted_names = sorted(weighted_risks, key=weighted_risks.get, reverse=True)

    loaded: List[ModelTier] = []
    evicted: List[ModelTier] = []
    used_vram = 0

    for name in sorted_names:
        tier = tiers_lookup[name]
        if used_vram + tier.vram_mb <= global_vram_budget_mb:
            loaded.append(tier)
            used_vram += tier.vram_mb
        else:
            evicted.append(tier)

    return loaded, evicted


def temporal_risk_analysis(
    timestamps: Iterable[datetime],
    weighted_risks: Iterable[float],
) -> Tuple[float, float]:
    """
    Perform a simple linear regression (least‑squares) of weighted risk over time.

    Returns ``(slope, intercept)`` where time is expressed as seconds since epoch.
    The slope indicates how risk evolves; positive slope → increasing risk.
    """
    times = np.array([ts.timestamp() for ts in timestamps], dtype=float)
    risks = np.array(list(weighted_risks), dtype=float)

    if len(times) < 2:
        raise ValueError("At least two data points are required for temporal analysis")

    # Linear regression via numpy polyfit (degree 1)
    slope, intercept = np.polyfit(times, risks, 1)
    return float(slope), float(intercept)


# ----------------------------------------------------------------------
# Helper to build a lookup dictionary for tiers
# ----------------------------------------------------------------------
def _build_tier_lookup(tiers: Iterable[ModelTier]) -> Dict[str, ModelTier]:
    return {t.name: t for t in tiers}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample data
    tiers = [TIER_T1_QWEN_0_5B, TIER_T2_REASONING, TIER_T2_TOOL, TIER_T3_QWEN_7B]
    lookup = _build_tier_lookup(tiers)

    # Simulated dataset characteristics
    unique_qi = 1234
    total_records = 10000

    # Desired VRAM budget (e.g., the amount we would like each model to stay near)
    vram_target = 2000  # MB
    sigma = 600.0

    # 1. Compute weighted risks
    w_risks = weighted_reconstruction_risk(
        model_tiers=tiers,
        unique_qi=unique_qi,
        total_records=total_records,
        vram_target=vram_target,
        sigma=sigma,
    )
    print("Weighted risks per tier:", w_risks)

    # 2. Schedule models under a global VRAM budget
    global_budget = 6000  # MB
    loaded, evicted = vram_scheduler(w_risks, lookup, global_budget)
    print("Loaded models:", [m.name for m in loaded])
    print("Evicted models:", [m.name for m in evicted])

    # 3. Temporal analysis – pretend we have three snapshots
    now = datetime.now(timezone.utc)
    timestamps = [
        now.replace(hour=0, minute=0, second=0, microsecond=0),
        now.replace(hour=6, minute=0, second=0, microsecond=0),
        now.replace(hour=12, minute=0, second=0, microsecond=0),
    ]
    # Vary the weighted risk slightly to simulate change
    risk_series = [w_risks["qwen-0.5b"], w_risks["reasoning-t2"], w_risks["tool-t2"]]
    slope, intercept = temporal_risk_analysis(timestamps, risk_series)
    print(f"Temporal regression: slope={slope:.6f}, intercept={intercept:.6f}")

    # Final sanity check – ensure no exception was raised
    sys.exit(0)