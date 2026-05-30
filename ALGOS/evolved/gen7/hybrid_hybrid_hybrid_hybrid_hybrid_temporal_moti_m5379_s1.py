# DARWIN HAMMER — match 5379, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m2390_s1.py (gen6)
# parent_b: hybrid_temporal_motifs_possum_filter_m87_s1.py (gen1)
# born: 2026-05-30T00:01:36Z

"""
Hybrid module unifying the Hybrid Bandit-RBF-HDC Model with the 
temporal_spatial_fusion algorithm.

The mathematical bridge between the two parents lies in the combination of 
the bandit's expected reward and the temporal motif's support count. 
The expected reward is replaced by the RBF surrogate's prediction for the 
vector [context, action_one_hot], and then modulated by the temporal motif's 
support count. The trust factor from the cockpit metrics is used to scale 
the velocity of the linguistic vector transport, and the z-score of the 
support distribution across patterns is used to drive the hybrid recovery 
priority and decision-making of the ModelPool.

This module provides three hybrid functions:
1. `hybrid_style_target` – compute the trust-weighted style target.
2. `hybrid_bandit_priority` – fuses the bandit's expected reward and the 
   temporal motif's support count into a single priority value.
3. `hybrid_euler_step`   – Euler integration toward the target style, 
   with a step size modulated by the trust factor and the z-score.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = Sequence[float]
BipolarVector = List[int]

# ----------------------------------------------------------------------
# Bandit core (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

# ----------------------------------------------------------------------
# Temporal motif utilities (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int

def sessionize_events(events: List[dict], gap_seconds: float = 1800.0) -> List[List[dict]]:
    """Group events into sessions separated by a temporal gap."""
    sessions: List[List[dict]] = []
    cur: List[dict] = []
    last: float | None = None
    for e in sorted(events, key=lambda x: float(x.get('t', 0))):
        t = float(e.get('t', 0))
        if cur and last is not None and t - last > gap_seconds:
            sessions.append(cur)
            cur = []
        cur.append(e)
        last = t
    if cur:
        sessions.append(cur)
    return sessions

def compute_support_count(temporal_motifs: List[TemporalMotif]) -> List[float]:
    """Compute the support count for each temporal motif."""
    support_counts = [motif.support for motif in temporal_motifs]
    return support_counts

def compute_z_score(support_counts: List[float]) -> float:
    """Compute the z-score of the support distribution."""
    mean = np.mean(support_counts)
    std_dev = np.std(support_counts)
    if std_dev == 0:
        return 0
    z_score = (np.mean(support_counts) - np.mean(support_counts)) / std_dev
    return z_score

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_style_target(bandit_action: BanditAction, temporal_motif: TemporalMotif, trust_factor: float) -> float:
    """Compute the trust-weighted style target."""
    style_target = bandit_action.expected_reward * temporal_motif.support * trust_factor
    return style_target

def hybrid_bandit_priority(bandit_action: BanditAction, temporal_motif: TemporalMotif) -> float:
    """Fuse the bandit's expected reward and the temporal motif's support count into a single priority value."""
    priority = bandit_action.expected_reward * temporal_motif.support
    return priority

def hybrid_euler_step(bandit_action: BanditAction, temporal_motif: TemporalMotif, trust_factor: float, z_score: float) -> float:
    """Euler integration toward the target style, with a step size modulated by the trust factor and the z-score."""
    step_size = trust_factor * z_score
    euler_step = bandit_action.expected_reward * temporal_motif.support * step_size
    return euler_step

if __name__ == "__main__":
    # Smoke test
    bandit_action = BanditAction("action_id", 0.5, 10.0, 0.1, "algorithm")
    temporal_motif = TemporalMotif(("pattern1", "pattern2"), 5)
    trust_factor = 0.8
    z_score = compute_z_score([5.0, 3.0, 4.0, 2.0])
    print(hybrid_style_target(bandit_action, temporal_motif, trust_factor))
    print(hybrid_bandit_priority(bandit_action, temporal_motif))
    print(hybrid_euler_step(bandit_action, temporal_motif, trust_factor, z_score))