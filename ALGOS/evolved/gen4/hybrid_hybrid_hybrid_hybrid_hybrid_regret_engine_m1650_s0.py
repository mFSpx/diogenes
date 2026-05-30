# DARWIN HAMMER — match 1650, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_korpus_text_m128_s1.py (gen3)
# parent_b: hybrid_regret_engine_hybrid_doomsday_cale_m19_s3.py (gen2)
# born: 2026-05-29T23:37:57Z

"""Hybrid module combining epistemic certainty flags with regret-engine and doomsday calendar math.

Parent A (hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s3) provides a
`CertaintyFlag` data class whose `confidence_bps` (basis points, 0‑10000) encodes
a scalar weight w = confidence_bps / 10000.

Parent B (hybrid_regret_engine_hybrid_doomsday_calendar_m19_s3) supplies two
regret-weighted structures:
* Actions with expected values, costs, and risks
* Counterfactuals with outcome values and probabilities

The mathematical bridge is the **weight-scaled Gini coefficient**:
for a given set of regret-weighted actions, we compute the Gini coefficient of
the epistemic-weighted action values, which quantifies the unevenness of the
action distribution.  The resulting Gini coefficient is then multiplied by the
epistemic weight w, attenuated by the entropy E (0‑log₂|Σ|) of the action
distribution.  The functions below implement this pipeline and also offer a
Bayesian-style update of a `CertaintyFlag` using the hybrid score.

Only the Python standard library and NumPy are used.
"""

import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Tuple, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Minimal re-implementation of the epistemic certainty helpers (Parent A)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # 0 .. 10000
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat())

# ----------------------------------------------------------------------
# Minimal re-implementation of the regret-engine helpers (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

def compute_regret_weighted_strategy(actions: list[MathAction], counterfactuals: list[MathCounterfactual]) -> Dict[str, float]:
    if not actions:
        return {}
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions}
    best = max(vals.values())
    w = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}

def rank_actions_by_ev(actions: list[MathAction]) -> list[MathAction]:
    return sorted(actions, key=lambda a: (-(a.expected_value - a.cost - a.risk), a.id))

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def doomsday(year: int, month: int, day: int) -> int:
    return (dt.date(year, month, day).weekday() + 1) % 7

def weekday_distribution(year: int, month: int, num_days: int) -> np.ndarray:
    weekdays = [doomsday(year, month, day) for day in range(1, num_days + 1)]
    weekday_counts = np.zeros(7)
    for weekday in weekdays:
        weekday_counts[weekday - 1] += 1
    return weekday_counts

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------

def hybrid_gini_coefficient(actions: list[MathAction], counterfactuals: list[MathCounterfactual], certainty_flag: CertaintyFlag) -> float:
    """Compute the weight-scaled Gini coefficient."""
    regressed_strategy = compute_regret_weighted_strategy(actions, counterfactuals)
    epistemic_weight = certainty_flag.confidence_bps / 10000
    weighted_gini = gini_coefficient([regressed_strategy[a] * epistemic_weight for a in regressed_strategy])
    entropy = -sum(regressed_strategy.values() * np.log2(regressed_strategy.values()))
    return weighted_gini * np.exp(-entropy)

def hybrid_certainty_flag(actions: list[MathAction], counterfactuals: list[MathCounterfactual], certainty_flag: CertaintyFlag) -> CertaintyFlag:
    """Update the certainty flag using the hybrid Gini coefficient."""
    hybrid_score = hybrid_gini_coefficient(actions, counterfactuals, certainty_flag)
    return CertaintyFlag(
        label=certainty_flag.label,
        confidence_bps=hybrid_score * 10000,
        authority_class=certainty_flag.authority_class,
        rationale=certainty_flag.rationale,
        evidence_refs=certainty_flag.evidence_refs,
        generated_at=certainty_flag.generated_at
    )

def hybrid_weekday_distribution(year: int, month: int, num_days: int, actions: list[MathAction], certainty_flag: CertaintyFlag) -> np.ndarray:
    """Compute the weekday distribution with regret-weighted actions."""
    regressed_strategy = compute_regret_weighted_strategy(actions, [])
    return weekday_distribution(year, month, num_days) * regressed_strategy

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    actions = [
        MathAction(id="A1", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="A2", expected_value=20.0, cost=1.0, risk=0.5),
        MathAction(id="A3", expected_value=30.0, cost=3.0, risk=1.5)
    ]
    counterfactuals = [
        MathCounterfactual(action_id="A1", outcome_value=15.0, probability=0.8),
        MathCounterfactual(action_id="A2", outcome_value=25.0, probability=0.9),
        MathCounterfactual(action_id="A3", outcome_value=40.0, probability=0.7)
    ]
    certainty_flag = CertaintyFlag(label="FACT", confidence_bps=8000, authority_class="Expert", rationale="Strong evidence")
    print(hybrid_gini_coefficient(actions, counterfactuals, certainty_flag))
    print(hybrid_certainty_flag(actions, counterfactuals, certainty_flag))
    print(hybrid_weekday_distribution(2024, 3, 31, actions, certainty_flag))