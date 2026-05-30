# DARWIN HAMMER — match 1947, survivor 3
# gen: 6
# parent_a: hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m859_s1.py (gen5)
# born: 2026-05-29T23:40:06Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Iterable, FrozenSet
import numpy as np

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
    global _POLICY
    _POLICY.clear()


def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]


def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
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
    if dt < 0:
        raise ValueError("dt must be non-negative")
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
    x, y = x0, y0
    out: List[Tuple[float, float]] = []
    for u in inputs:
        x, y = step(u, x, y, **kw)
        out.append((x, y))
    return out


def update_policy(updates: List[BanditUpdate]) -> None:
    for upd in updates:
        a = upd.action_id
        if a not in _POLICY:
            _POLICY[a] = [0.0, 0.0]
        _POLICY[a][0] += upd.reward
        _POLICY[a][1] += 1.0


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
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


class Multivector:
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
        out: Dict[FrozenSet[int], float] = {}
        for ba, ca in self._components.items():
            for bb, cb in other._components.items():
                blade, sign = _multiply_blades(ba, bb)
                out[blade] = out.get(blade, 0.0) + sign * ca * cb
        return Multivector(out)

    def norm(self) -> float:
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


def hybrid_state_from_xy(x: float, y: float) -> Multivector:
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
    x, y = x0, y0
    out: List[Tuple[float, float, Multivector]] = []
    for u in inputs:
        x, y = step(u, x, y, dt=dt, gain=gain, decay_x=decay_x, decay_y=decay_y)
        multivector = hybrid_state_from_xy(x, y)
        reward = multivector.norm()
        out.append((x, y, multivector))
        gain = gain * (1 + 0.1 * reward)
    return out


def main():
    inputs = [1.0, 2.0, 3.0, 4.0, 5.0]
    x0 = 1.0
    y0 = 0.0
    dt = 1.0
    gain = 1.0
    decay_x = 1.0
    decay_y = 1.0
    result = hybrid_step(inputs, x0=x0, y0=y0, dt=dt, gain=gain, decay_x=decay_x, decay_y=decay_y)
    for x, y, multivector in result:
        print(f"x: {x}, y: {y}, multivector: {multivector}")


if __name__ == "__main__":
    main()