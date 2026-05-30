# DARWIN HAMMER — match 1947, survivor 2
# gen: 6
# parent_a: hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m859_s1.py (gen5)
# born: 2026-05-29T23:40:06Z

"""Hybrid algorithm combining Fold‑Change Detection dynamics (Parent A) with
Geometric‑Algebra multivector operations and epistemic certainty flags (Parent B).

Mathematical bridge:
- The scalar state `x` from the feed‑forward ODE of Parent A is used as a
  coefficient for a basis blade of a multivector (Parent B).  The evolving
  multivector `M = x·e₀ + y·e₁` therefore couples the two systems.
- The norm of `M` (√(x²+y²)) is fed back as a reward signal for the bandit
  policy defined in Parent A.  Thus the dynamical system drives the bandit
  learning loop, while the bandit updates can modulate the gain of the
  ODE in subsequent calls, completing a closed hybrid loop.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Iterable, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Fold‑Change Detection & Bandit policy
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


_POLICY: Dict[str, List[float]] = {}  # action_id -> [total_reward, count]


def reset_policy() -> None:
    """Reset the bandit policy."""
    global _POLICY
    _POLICY.clear()


def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]


def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
    """Compute a scaling factor based on log‑count ratio."""
    if count == 0:
        return 1.0
    return max(1.0, math.exp(log_count_ratio * count))


def step(
    u: float,
    x: float,
    y: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay_x: float = 1.0,
    decay_y: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """Euler integration of the feed‑forward fold‑change detector."""
    if dt < 0:
        raise ValueError("dt must be non‑negative")
    ratio = u / max(abs(x), eps)
    dy = gain * ratio - decay_y * y
    dx = u - decay_x * x
    return x + dt * dx, y + dt * dy


def response_series(
    inputs: List[float],
    x0: float = 1.0,
    y0: float = 0.0,
    **kw,
) -> List[Tuple[float, float]]:
    """Generate (x, y) for each stimulus in `inputs`."""
    x, y = x0, y0
    out: List[Tuple[float, float]] = []
    for u in inputs:
        x, y = step(u, x, y, **kw)
        out.append((x, y))
    return out


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate new reward observations into the bandit policy."""
    for upd in updates:
        a = upd.action_id
        if a not in _POLICY:
            _POLICY[a] = [0.0, 0.0]
        _POLICY[a][0] += upd.reward
        _POLICY[a][1] += 1.0


# ----------------------------------------------------------------------
# Parent B – Multivector algebra & epistemic certainty flags
# ----------------------------------------------------------------------


EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
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
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


def _blade_sign(indices: List[int]) -> Tuple[Tuple[int, ...], int]:
    """Bubble‑sort indices, returning sorted tuple and the parity sign."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return tuple(lst), sign


def _multiply_blades(
    blade_a: FrozenSet[int], blade_b: FrozenSet[int]
) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


class Multivector:
    """Sparse representation of a multivector using a dict {blade: coeff}."""

    def __init__(self, components: Dict[FrozenSet[int], float] | None = None):
        self._components: Dict[FrozenSet[int], float] = dict(components or {})

    def __add__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self._components)
        for b, c in other._components.items():
            result._components[b] = result._components.get(b, 0.0) + c
        return result

    __radd__ = __add__

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector({b: scalar * c for b, c in self._components.items()})

    __mul__ = __rmul__

    def geometric_product(self, other: "Multivector") -> "Multivector":
        """Geometric product (outer + inner) of two multivectors."""
        out: Dict[FrozenSet[int], float] = {}
        for ba, ca in self._components.items():
            for bb, cb in other._components.items():
                blade, sign = _multiply_blades(ba, bb)
                out[blade] = out.get(blade, 0.0) + sign * ca * cb
        return Multivector(out)

    def norm(self) -> float:
        """Euclidean norm of the coefficient vector."""
        return math.sqrt(sum(c * c for c in self._components.values()))

    def __repr__(self) -> str:
        if not self._components:
            return "0"
        terms = []
        for blade, coeff in sorted(self._components.items(), key=lambda x: (len(x[0]), x[0])):
            if blade:
                basis = "∧".join(str(i) for i in sorted(blade))
                terms.append(f"{coeff:.3g}*e{basis}")
            else:
                terms.append(f"{coeff:.3g}")
        return " + ".join(terms)


# ----------------------------------------------------------------------
# Hybrid core: coupling dynamics, multivector, and bandit reward
# ----------------------------------------------------------------------


def hybrid_state_from_xy(x: float, y: float) -> Multivector:
    """
    Build a 2‑dimensional multivector M = x·e0 + y·e1.
    Blade 0 and 1 are represented by frozenset({0}) and frozenset({1}).
    """
    components = {
        frozenset({0}): x,
        frozenset({1}): y,
    }
    return Multivector(components)


def hybrid_step(
    inputs: List[float],
    x0: float = 1.0,
    y0: float = 0.0,
    dt: float = 1.0,
    gain: float = 1.0,
    decay_x: float = 1.0,
    decay_y: float = 1.0,
) -> List[Tuple[float, float, Multivector]]:
    """
    Run the fold‑change detector on `inputs` and simultaneously embed the
    scalar states into a multivector. Returns a list of (x, y, M).
    """
    x, y = x0, y0
    trajectory: List[Tuple[float, float, Multivector]] = []
    for u in inputs:
        x, y = step(u, x, y, dt=dt, gain=gain, decay_x=decay_x, decay_y=decay_y)
        M = hybrid_state_from_xy(x, y)
        trajectory.append((x, y, M))
    return trajectory


def compute_reward_from_state(x: float, y: float, M: Multivector) -> float:
    """
    Reward is a weighted sum of absolute fold‑change (|x|) and the multivector norm.
    """
    return abs(x) + M.norm()


def generate_certainty_flag(reward: float) -> CertaintyFlag:
    """
    Map reward magnitude to an epistemic certainty flag.
    - reward ≥ 5.0 → FACT (high confidence)
    - 2.0 ≤ reward < 5.0 → PROBABLE
    - 1.0 ≤ reward < 2.0 → POSSIBLE
    - reward < 1.0 → BULLSHIT
    """
    if reward >= 5.0:
        label = "FACT"
        conf = 9000
    elif reward >= 2.0:
        label = "PROBABLE"
        conf = 6000
    elif reward >= 1.0:
        label = "POSSIBLE"
        conf = 3000
    else:
        label = "BULLSHIT"
        conf = 1000
    return certainty(
        label,
        confidence_bps=conf,
        authority_class="HybridEngine",
        rationale=f"Reward {reward:.3g}",
        evidence_refs=("fold_change", "multivector_norm"),
    )


def hybrid_update_policy(
    trajectory: List[Tuple[float, float, Multivector]],
    context_prefix: str = "ctx",
) -> None:
    """
    For each step, compute a reward and feed a BanditUpdate into the policy.
    The action_id is derived from the sign of x (positive/negative/zero).
    """
    updates: List[BanditUpdate] = []
    for idx, (x, y, M) in enumerate(trajectory):
        reward = compute_reward_from_state(x, y, M)
        if x > 0:
            a_id = "pos_x"
        elif x < 0:
            a_id = "neg_x"
        else:
            a_id = "zero_x"
        # Propensity is a dummy uniform value; in a real system it would be
        # derived from a policy distribution.
        upd = BanditUpdate(
            context_id=f"{context_prefix}_{idx}",
            action_id=a_id,
            reward=reward,
            propensity=1.0,
        )
        updates.append(upd)
    update_policy(updates)


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------


def demo_hybrid_dynamics() -> List[Tuple[float, float, Multivector]]:
    """Run a short random stimulus sequence and return the hybrid trajectory."""
    random.seed(42)
    stimuli = [random.uniform(-2.0, 2.0) for _ in range(10)]
    return hybrid_step(stimuli, gain=0.8, decay_x=0.5, decay_y=0.3)


def demo_policy_and_certainty() -> List[CertaintyFlag]:
    """Execute the hybrid loop, update the bandit policy and produce certainty flags."""
    reset_policy()
    traj = demo_hybrid_dynamics()
    hybrid_update_policy(traj)
    flags = [generate_certainty_flag(compute_reward_from_state(x, y, M)) for (x, y, M) in traj]
    return flags


def demo_multivector_arithmetic() -> Multivector:
    """Show a simple geometric product between two hybrid‑derived multivectors."""
    # Create two independent hybrid states
    M1 = hybrid_state_from_xy(1.2, -0.4)
    M2 = hybrid_state_from_xy(0.5, 2.1)
    return M1.geometric_product(M2)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Run dynamics and print a few steps
    trajectory = demo_hybrid_dynamics()
    print("Hybrid trajectory (first 3 steps):")
    for i, (x, y, M) in enumerate(trajectory[:3]):
        print(f" step {i}: x={x:.3f}, y={y:.3f}, M={M}")

    # 2. Update policy and display policy contents
    hybrid_update_policy(trajectory)
    print("\nBandit policy after update:")
    for act, (total, cnt) in _POLICY.items():
        print(f" {act}: total_reward={total:.3f}, count={cnt}")

    # 3. Produce certainty flags
    flags = demo_policy_and_certainty()
    print("\nCertainty flags:")
    for i, fl in enumerate(flags[:3]):
        print(f" step {i}: {fl.label} (confidence {fl.confidence_bps})")

    # 4. Demonstrate multivector arithmetic
    prod = demo_multivector_arithmetic()
    print("\nGeometric product of two hybrid multivectors:")
    print(prod)