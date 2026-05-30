# DARWIN HAMMER — match 3028, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_gliner_zero_s_hybrid_doomsday_cale_m49_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m1412_s2.py (gen6)
# born: 2026-05-29T23:47:19Z

"""
Hybrid algorithm that fuses the geometric embedding and Gini coefficient calculation 
of `hybrid_hybrid_gliner_zero_s_hybrid_doomsday_cale_m49_s1.py` 
with the Multivector operations and Fisher scoring of 
`hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m1412_s2.py`.

The mathematical bridge is established by representing the extracted spans 
as Multivector components, where each span's start and length are used 
to compute a weighted Fisher score. The Gini coefficient is then 
applied to the Fisher scores to evaluate the inequality of the 
span lengths, providing a measure of the diversity of the extracted 
information.

The hybrid algorithm calculates a unified metric that combines 
the spatial coherence of the extracted spans with the Fisher scores 
and Gini coefficient of their lengths, yielding a measure that 
rewards both high-confidence spans and diverse, well-connected layouts.
"""

import numpy as np
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"

class Multivector:
    def __init__(self, components: dict[frozenset, float] = None):
        self.components: dict[frozenset, float] = dict(components or {})

    def __add__(self, other: "Multivector") -> "Multivector":
        res = self.components.copy()
        for k, v in other.components.items():
            res[k] = res.get(k, 0.0) + v
            if abs(res[k]) < 1e-15:
                del res[k]
        return Multivector(res)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({k: -v for k, v in self.components.items()})

    def __mul__(self, other: "Multivector") -> "Multivector":
        result: dict[frozenset, float] = {}
        for ba, ca in self.components.items():
            for bb, cb in other.components.items():
                combined = list(ba) + list(bb)
                result[frozenset(combined)] = result.get(frozenset(combined), 0.0) + ca * cb
        return Multivector(result)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def gini_coefficient(values: List[float]) -> float:
    values = np.array(values)
    values = values.flatten()
    if len(values) == 0:
        return 0.0
    index = np.argsort(values)
    n = len(values)
    index = index[::-1]
    values = values[index]
    A = np.sum(values * np.arange(n))
    B = np.sum(values) * (n - 1) / 2.0
    if B == 0:
        return 0.0
    return (A - B) / B

def hybrid_metric(spans: List[Span], center: float, width: float) -> float:
    multivector = Multivector()
    fisher_scores = []
    for span in spans:
        fisher_score_val = fisher_score(span.start, center, width)
        multivector.components[frozenset([span.start, span.end])] = fisher_score_val
        fisher_scores.append(fisher_score_val)
    gini_val = gini_coefficient(fisher_scores)
    return gini_val * multivector.components[frozenset([span.start for span in spans])]

def main():
    spans = [Span(1, 5, "text1", "label1", 0.5), Span(10, 15, "text2", "label2", 0.7)]
    center = 5.0
    width = 2.0
    metric = hybrid_metric(spans, center, width)
    print(metric)

if __name__ == "__main__":
    main()