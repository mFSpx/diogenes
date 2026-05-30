# DARWIN HAMMER — match 859, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_korpus_text_m128_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_pherom_m96_s1.py (gen4)
# born: 2026-05-29T23:31:21Z

import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Tuple, Dict, List
import numpy as np

EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


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
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

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


def _blade_sign(indices: list) -> tuple:
    """Return (sorted_blade, sign) after bubble-sorting index list."""
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
                sign *= 1
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> tuple:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: dict, n: int):
        # components: {frozenset(indices): coefficient}
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coeff for blade, coeff in self.components.items() if len(blade) == k}, self.n
        )

    def __add__(self, other: "Multivector") -> "Multivector":
        if self.n != other.n:
            raise ValueError("Multivectors must have the same dimension")
        components = self.components.copy()
        for blade, coeff in other.components.items():
            components[blade] = components.get(blade, 0) + coeff
        return Multivector(components, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        components = {blade: coeff * scalar for blade, coeff in self.components.items()}
        return Multivector(components, self.n)


def integrate_epistemic_certainty(
    certainty_flags: List[CertaintyFlag], multivector: Multivector
) -> Multivector:
    """Integrate epistemic certainty into the multivector coefficients."""
    weighted_components = {}
    for flag in certainty_flags:
        weight = flag.confidence_bps / 10000
        for blade, coeff in multivector.components.items():
            weighted_components[blade] = weighted_components.get(blade, 0) + weight * coeff
    return Multivector(weighted_components, multivector.n)


def calculate_pheromone_strengths(
    multivector: Multivector, certainty_flags: List[CertaintyFlag]
) -> List[float]:
    """Calculate pheromone strengths based on the multivector coefficients and certainty flags."""
    pheromone_strengths = []
    for flag in certainty_flags:
        weight = flag.confidence_bps / 10000
        pheromone_strength = 0
        for blade, coeff in multivector.components.items():
            pheromone_strength += weight * coeff
        pheromone_strengths.append(pheromone_strength)
    return pheromone_strengths


def calculate_shannon_entropy(text: str) -> float:
    """Calculate the Shannon entropy of a text."""
    frequency = {}
    for symbol in text:
        if symbol not in frequency:
            frequency[symbol] = 0
        frequency[symbol] += 1
    entropy = 0.0
    for count in frequency.values():
        p_x = count / len(text)
        entropy += -p_x * math.log(p_x, 2)
    return entropy


def calculate_entropy_weighted_pheromone_strengths(
    multivector: Multivector, certainty_flags: List[CertaintyFlag], text: str
) -> List[float]:
    """Calculate pheromone strengths based on the multivector coefficients, certainty flags and Shannon entropy."""
    pheromone_strengths = []
    entropy = calculate_shannon_entropy(text)
    for flag in certainty_flags:
        weight = flag.confidence_bps / 10000
        pheromone_strength = 0
        for blade, coeff in multivector.components.items():
            pheromone_strength += weight * coeff * (1 + entropy / (1 + entropy))
        pheromone_strengths.append(pheromone_strength)
    return pheromone_strengths


if __name__ == "__main__":
    # Create a multivector
    multivector = Multivector({frozenset([1, 2]): 0.5, frozenset([3, 4]): 0.3}, 5)

    # Create certainty flags
    certainty_flags = [
        certainty("FACT", confidence_bps=8000, authority_class="primary", rationale="reason 1"),
        certainty("PROBABLE", confidence_bps=6000, authority_class="secondary", rationale="reason 2"),
    ]

    # Integrate epistemic certainty into the multivector coefficients
    integrated_multivector = integrate_epistemic_certainty(certainty_flags, multivector)

    # Calculate entropy weighted pheromone strengths
    text = "example text"
    entropy_weighted_pheromone_strengths = calculate_entropy_weighted_pheromone_strengths(
        integrated_multivector, certainty_flags, text
    )

    # Print results
    print("Integrated multivector coefficients:")
    for blade, coeff in integrated_multivector.components.items():
        print(f"{blade}: {coeff}")
    print("Entropy weighted pheromone strengths:")
    for strength in entropy_weighted_pheromone_strengths:
        print(strength)