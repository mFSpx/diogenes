# DARWIN HAMMER — match 1490, survivor 6
# gen: 4
# parent_a: workshare_allocator.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s2.py (gen3)
# born: 2026-05-29T23:37:01Z

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
K: int = len(GROUPS)                     # number of arms
GAMMA: float = 0.07                      # exploration parameter for EXP3
DECAY_OUTFLOW: float = 0.1               # base outflow decay factor
MAX_EVASION_DELTA: float = 1.0
EVASION_ALPHA: float = 3.0
HOEFFDING_DELTA: float = 0.05            # confidence level for Hoeffding bound
SN_GAP_MIN: float = 0.01                 # lower bound for signal‑to‑noise gap
SN_GAP_MAX: float = 10.0                 # optional upper bound for stability


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def evasion_delta(t: int, t_max: int = 100) -> float:
    """Exponential decay schedule for eviction magnitude."""
    if t < 0 or t_max <= 0:
        raise ValueError("invalid evasion schedule")
    return MAX_EVASION_DELTA * math.exp(-EVASION_ALPHA * min(t, t_max) / t_max)


def clamp(x: List[float], lower: float, upper: float) -> List[float]:
    """Clamp each component of a vector to [lower, upper]."""
    return [min(upper, max(lower, xi)) for xi in x]


def hoeffding_epsilon(num_samples: int, delta: float = HOEFFDING_DELTA) -> float:
    """Hoeffding bound ε = sqrt(ln(2/δ) / (2·n))."""
    if num_samples <= 0:
        return 1.0
    return math.sqrt(math.log(2.0 / delta) / (2.0 * num_samples))


def signal_to_noise_gap(deterministic_units: float, llm_units: float) -> float:
    """Ratio of deterministic to LLM units, bounded for numerical stability."""
    if llm_units <= 0:
        return SN_GAP_MAX
    gap = deterministic_units / llm_units
    return max(SN_GAP_MIN, min(gap, SN_GAP_MAX))


def _pct(value: float) -> float:
    """Round to six decimal places for reproducibility."""
    return round(float(value), 6)


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass
class ContextState:
    """All mutable state that belongs to a particular context."""
    weights: List[float] = field(default_factory=lambda: [1.0] * K)   # EXP3 weights
    store: List[float] = field(default_factory=lambda: [0.0] * K)    # virtual store per arm
    counts: List[int] = field(default_factory=lambda: [0] * K)      # number of observations per arm
    total_observations: int = 0                                      # sum of counts

    def propensities(self) -> List[float]:
        """Return the current probability simplex derived from EXP3 weights."""
        w_sum = sum(self.weights)
        if w_sum == 0:
            return [1.0 / K] * K
        base = [(1.0 - GAMMA) * w / w_sum for w in self.weights]
        uniform = [GAMMA / K] * K
        return [b + u for b, u in zip(base, uniform)]

    def update_weights(self, arm_idx: int, reward: float) -> None:
        """EXP3 weight update with Hoeffding‑scaled reward estimate."""
        p = self.propensities()[arm_idx]
        if p <= 0:
            return
        eps = hoeffding_epsilon(self.counts[arm_idx])
        # Clip reward to [0,1] to keep EXP3 guarantees; scale by Hoeffding bound
        clipped = max(0.0, min(1.0, reward))
        estimated = clipped / (p * eps)
        self.weights[arm_idx] *= math.exp(GAMMA * estimated / K)


# ----------------------------------------------------------------------
# Global registry of contexts
# ----------------------------------------------------------------------
_CONTEXTS: Dict[str, ContextState] = {}


def _get_context(state_id: str) -> ContextState:
    """Retrieve or lazily create a ContextState."""
    if state_id not in _CONTEXTS:
        _CONTEXTS[state_id] = ContextState()
    return _CONTEXTS[state_id]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def allocate_hybrid_workshare(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
    context_id: str = "global",
) -> Dict[str, Any]:
    """
    Allocate ``total_units`` according to the hybrid scheme.

    Returns a dictionary compatible with the original Parent A output,
    enriched with a deeper bandit‑driven LLM split and an updated virtual store.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units

    # ------------------------------------------------------------------
    # Bandit step: compute propensities and allocate LLM residual
    # ------------------------------------------------------------------
    ctx = _get_context(context_id)
    propensities = ctx.propensities()
    prop_array = np.array(propensities, dtype=float)
    prop_array /= prop_array.sum()                     # safeguard against drift
    propensities = prop_array.tolist()

    shares = [p * llm_units for p in propensities]

    # ------------------------------------------------------------------
    # Store dynamics: inflow from allocation, outflow via evasion and S/N gap
    # ------------------------------------------------------------------
    sn_gap = signal_to_noise_gap(deterministic_units, llm_units)

    for idx, group in enumerate(groups):
        inflow = shares[idx]
        outflow = evasion_delta(ctx.counts[idx]) * ctx.store[idx]
        eps = hoeffding_epsilon(ctx.counts[idx])
        delta_store = inflow - outflow * sn_gap * eps
        ctx.store[idx] = max(0.0, ctx.store[idx] + delta_store)

    # ------------------------------------------------------------------
    # Assemble output compatible with Parent A
    # ------------------------------------------------------------------
    lanes = []
    for i, group in enumerate(groups):
        lane = {
            "group": group,
            "llm_units": _pct(shares[i]),
            "llm_share_pct": (
                _pct(100.0 * shares[i] / llm_units) if llm_units > 0 else 0.0
            ),
            "proof_required": True,
            "store": _pct(ctx.store[i]),
        }
        lanes.append(lane)

    jzloads: List[Dict[str, Any]] = [
        {
            "kind": "OBJECT",
            "id": "project2501_hybrid_workshare_policy",
            "type": "workshare_policy",
            "deterministic_target_pct": _pct(deterministic_target_pct),
            "llm_residual_pct": _pct(100.0 - deterministic_target_pct),
        },
        {
            "kind": "EVENT",
            "id": "project2501_hybrid_allocation",
            "type": "allocation_computed",
            "total_units": _pct(total_units),
            "deterministic_units": _pct(deterministic_units),
            "llm_units": _pct(llm_units),
        },
    ]

    for lane in lanes:
        jzloads.append(
            {
                "kind": "EDGE",
                "from": "project2501_hybrid_workshare_policy",
                "to": f"model_group:{lane['group']}",
                "type": "ASSIGNS_LLM_RESIDUAL_DYNAMIC",
                "llm_units": lane["llm_units"],
                "llm_share_pct": lane["llm_share_pct"],
                "store": lane["store"],
            }
        )

    return {
        "schema": "lucidota.project2501.hybrid_workshare_allocation.v2",
        "total_units": _pct(total_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
        "jzloads": jzloads,
    }


def record_reward(
    *,
    context_id: str,
    group: str,
    reward: float,
) -> None:
    """
    Incorporate an observed ``reward`` for ``group`` into the EXP3 policy
    of ``context_id``.  The reward is assumed to lie in the interval [0, 1];
    values outside this range are clipped to preserve theoretical guarantees.
    """
    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r}")

    ctx = _get_context(context_id)
    arm_idx = GROUPS.index(group)

    # Update statistics
    ctx.counts[arm_idx] += 1
    ctx.total_observations += 1

    # EXP3 weight update with Hoeffding‑scaled reward
    ctx.update_weights(arm_idx, reward)


def simulate_step(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    context_id: str = "global",
    reward_sampler: Any = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper that performs a full allocation‑reward cycle.
    ``reward_sampler`` is a callable ``(group: str) -> float`` that returns a
    synthetic reward for the given group.  If omitted, a uniform random reward
    in [0,1] is used.
    """
    allocation = allocate_hybrid_workshare(
        total_units=total_units,
        deterministic_target_pct=deterministic_target_pct,
        context_id=context_id,
    )

    sampler = reward_sampler or (lambda g: random.random())
    for lane in allocation["lanes"]:
        group = lane["group"]
        reward = sampler(group)
        record_reward(context_id=context_id, group=group, reward=reward)

    # Return allocation together with the sampled rewards for inspection
    allocation["sampled_rewards"] = {
        lane["group"]: _pct(sampler(lane["group"])) for lane in allocation["lanes"]
    }
    return allocation


def dump_contexts(path: Path) -> None:
    """Serialise the internal context registry to JSON for offline analysis."""
    import json

    serialisable = {
        cid: {
            "weights": ctx.weights,
            "store": ctx.store,
            "counts": ctx.counts,
            "total_observations": ctx.total_observations,
        }
        for cid, ctx in _CONTEXTS.items()
    }
    path.write_text(json.dumps(serialisable, indent=2))


def load_contexts(path: Path) -> None:
    """Load a previously dumped context registry."""
    import json

    data = json.loads(path.read_text())
    _CONTEXTS.clear()
    for cid, payload in data.items():
        _CONTEXTS[cid] = ContextState(
            weights=payload["weights"],
            store=payload["store"],
            counts=payload["counts"],
            total_observations=payload["total_observations"],
        )