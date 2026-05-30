# DARWIN HAMMER — match 5127, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_rete_bandit_g_hybrid_hybrid_hybrid_m1966_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1078_s0.py (gen5)
# born: 2026-05-30T00:00:07Z

"""Hybrid Workshare Allocation Module
Parent A: hybrid_hybrid_rete_bandit_g_hybrid_hybrid_hybrid_m1966_s1.py
Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1078_s0.py

Mathematical Bridge
-------------------
The fusion is built on the observation that both parents manipulate a *resource
allocation* problem using probabilistic updates:

* Parent A distributes deterministic vs. LLM units and adapts the LLM share
  with a Bayesian‑style update that depends on the day of the week.
* Parent B models a honey‑bee store whose “dance” (pheromone signal) evolves
  according to a linear differential equation and whose multivector
  representation can be projected onto a scalar weight for each work‑share
  group.

The hybrid algorithm therefore:

1. Treats the store’s dance value as the *likelihood* for a Beta‑distribution
   update of the LLM‑share probability (Bayesian update).
2. Encodes the current store state in a multivector; the magnitude of its
   grade‑1 (vector) part yields a *pheromone weight* for each group.
3. Combines the posterior LLM‑share probability with the pheromone weights
   to allocate LLM units across groups while preserving the deterministic
   portion.

The three core functions below implement this mathematically unified system.
"""

import sys
import math
import random
from datetime import date, datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Data structures from Parent B (slightly trimmed for the hybrid)
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    """Dynamic store representing honey‑bee “dance” pheromone signal."""
    level: float = 0.0
    alpha: float = 1.0   # inflow coefficient
    beta: float = 1.0    # outflow coefficient
    dt: float = 1.0
    base: float = 1.0   # baseline dance intensity
    gain: float = 1.0   # gain applied to delta
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Linear store update → returns new level and delta."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._store_last_delta(delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Pheromone intensity after applying gain and clipping."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

    def _store_last_delta(self, delta: float) -> None:
        self._store_last_delta = lambda _: None  # placeholder to silence mypy
        self._last_delta = delta


class Multivector:
    """Very small multivector implementation sufficient for grade‑1 projection."""
    def __init__(self, components: Dict[str, float], n: int):
        # components keyed by blade string, e.g. 'e1', 'e2', 'e12', ...
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k: int) -> 'Multivector':
        """Return a new multivector containing only blades of grade k."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def vector_magnitude(self) -> float:
        """L2 norm of the grade‑1 (vector) part."""
        vec = self.grade(1).components
        return math.sqrt(sum(v * v for v in vec.values()))

    def __repr__(self) -> str:
        return f"Multivector({self.components}, n={self.n})"


# ----------------------------------------------------------------------
# Helper functions from Parent A (Bayesian / day‑of‑week logic)
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round to six decimal places, mimicking Parent A's helper."""
    return round(float(value), 6)


def day_of_week_likelihood(day: int) -> float:
    """Map day of week (0=Mon … 6=Sun) to a likelihood factor in [0.5, 1.5].

    The mapping is sinusoidal so that mid‑week days are favoured.
    """
    # Shift so that Wednesday (day=2) is maximal
    angle = (day - 2) * (2 * math.pi / 7)
    return 1.0 + 0.5 * math.cos(angle)


def beta_posterior(alpha: float, beta: float, successes: int, failures: int) -> Tuple[float, float]:
    """Standard Beta‑distribution posterior update."""
    return alpha + successes, beta + failures


# ----------------------------------------------------------------------
# Hybrid core functions (the required three)
# ----------------------------------------------------------------------
def compute_pheromone_weights(store: StoreState, groups: Tuple[str, ...]) -> Dict[str, float]:
    """
    Derive a weight for each group from the store's dance signal using a
    multivector representation.

    The multivector is built with a scalar part equal to the current level
    and a vector part where each component is proportional to the group's
    index. The resulting vector magnitude is normalised across groups.
    """
    # Build a simple multivector: scalar = level, vector blades = e{i}
    components = {"s": store.level}
    for i, grp in enumerate(groups, start=1):
        components[f"e{i}"] = store.dance * (i / len(groups))

    mv = Multivector(components, n=len(groups) + 1)
    magnitude = mv.vector_magnitude()
    if magnitude == 0:
        # fallback: uniform weights
        return {g: 1.0 / len(groups) for g in groups}

    # Weight for group i is proportional to its blade coefficient
    raw_weights = {}
    for i, grp in enumerate(groups, start=1):
        coeff = mv.components.get(f"e{i}", 0.0)
        raw_weights[grp] = max(0.0, coeff)  # ensure non‑negative

    total = sum(raw_weights.values())
    if total == 0:
        return {g: 1.0 / len(groups) for g in groups}
    return {g: _pct(w / total) for g, w in raw_weights.items()}


def bayesian_llm_share(total_units: float,
                       deterministic_target_pct: float,
                       day: int,
                       prior_alpha: float = 2.0,
                       prior_beta: float = 2.0) -> Tuple[float, float]:
    """
    Compute the posterior probability that a unit should be allocated to LLM
    workshare, using a Beta prior updated with a likelihood derived from the
    day‑of‑week factor.

    Returns (deterministic_units, llm_units) after applying the posterior mean.
    """
    # Prior predictive mean
    prior_mean = prior_alpha / (prior_alpha + prior_beta)

    # Day‑of‑week likelihood acts as pseudo‑observations
    likelihood = day_of_week_likelihood(day)  # in [0.5, 1.5]
    pseudo_success = likelihood * 10   # scale to integer‑like counts
    pseudo_failure = (1.0 - (likelihood - 0.5) / 1.0) * 10

    post_alpha, post_beta = beta_posterior(prior_alpha, prior_beta,
                                            int(pseudo_success),
                                            int(pseudo_failure))
    posterior_mean = post_alpha / (post_alpha + post_beta)

    # Blend the posterior mean with the deterministic target percentage
    llm_share_pct = (100.0 - deterministic_target_pct) * posterior_mean / prior_mean
    llm_share_pct = max(0.0, min(100.0, llm_share_pct))

    deterministic_units = _pct(total_units * deterministic_target_pct / 100.0)
    llm_units = _pct(total_units - deterministic_units)
    # Adjust llm_units according to the posterior‑scaled share
    llm_units = _pct(total_units * llm_share_pct / 100.0)
    deterministic_units = _pct(total_units - llm_units)

    return deterministic_units, llm_units


def hybrid_allocate_workshare(total_units: float,
                              deterministic_target_pct: float = 90.0,
                              groups: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models"),
                              day: int = None,
                              store: StoreState = None) -> Dict[str, float]:
    """
    Unified allocation routine.

    1. Compute deterministic vs. LLM units via a Bayesian update that depends
       on the day of the week.
    2. Update the StoreState with synthetic inflow/outflow derived from the
       deterministic share (as “resource consumption”) and obtain a pheromone
       weight for each group.
    3. Distribute the LLM units across groups according to the pheromone weights.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("At least one group required")

    # ------------------------------------------------------------------
    # Step 1 – Bayesian LLM share
    # ------------------------------------------------------------------
    today = day if day is not None else date.today().weekday()
    det_units, llm_units = bayesian_llm_share(total_units,
                                              deterministic_target_pct,
                                              today)

    # ------------------------------------------------------------------
    # Step 2 – Store dynamics and pheromone weighting
    # ------------------------------------------------------------------
    store = store or StoreState()
    # Treat deterministic units as “inflow” to the store, LLM units as “outflow”
    inflow = [det_units * 0.01]   # small fraction influences the store
    outflow = [llm_units * 0.01]
    store.update(inflow, outflow)

    pheromone_weights = compute_pheromone_weights(store, groups)

    # ------------------------------------------------------------------
    # Step 3 – Final per‑group allocation
    # ------------------------------------------------------------------
    per_group = {}
    for grp in groups:
        grp_llm = _pct(llm_units * pheromone_weights[grp])
        per_group[grp] = {
            "deterministic_units": _pct(det_units / len(groups)),
            "llm_units": grp_llm,
            "total_units": _pct(det_units / len(groups) + grp_llm),
            "pheromone_weight": pheromone_weights[grp],
            "dance_intensity": _pct(store.dance),
        }

    # Assemble a flat summary dict for convenience
    summary = {
        "total_units": _pct(total_units),
        "deterministic_units": det_units,
        "llm_units": llm_units,
        "store_level": _pct(store.level),
        "store_dance": _pct(store.dance),
        "allocation_per_group": per_group,
    }
    return summary


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic run
    result = hybrid_allocate_workshare(
        total_units=1_000.0,
        deterministic_target_pct=85.0,
        groups=("codex", "groq", "cohere", "local_models"),
        day=2,  # Wednesday – maximal likelihood
    )
    print("Hybrid Allocation Result:")
    for k, v in result.items():
        if k != "allocation_per_group":
            print(f"{k}: {v}")
    print("\nPer‑group breakdown:")
    for grp, data in result["allocation_per_group"].items():
        print(f"{grp}: {data}")