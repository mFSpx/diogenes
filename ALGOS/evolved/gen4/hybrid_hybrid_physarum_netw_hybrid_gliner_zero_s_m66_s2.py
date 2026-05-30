# DARWIN HAMMER — match 66, survivor 2
# gen: 4
# parent_a: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s1.py (gen3)
# parent_b: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s3.py (gen1)
# born: 2026-05-29T23:26:34Z

"""
Unified Algorithm: Flux-Based Gliner Hybrid
Fuses the principles of Flux-Based Conductance Update (Parent Algorithm A: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s1.py)
and a Zero-Shot Extractor with Minimum Cost Tree (Parent Algorithm B: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s3.py).

The mathematical bridge between the two parents lies in the integration of the 
store differential equation in the UnifiedBanditTTT class (Parent A) with the 
label extraction and scoring mechanism in the Span class (Parent B). Specifically, 
the update_conductance function from Parent A can be seen as a time-stepping scheme 
for integrating the store differential equation, which can be used to influence the 
learning rate and propensity of the contextual bandit in Parent B.

By fusing these two components, we develop a unified algorithm that leverages the 
strengths of both parents to extract labels and compute scores based on a flux-based 
conductance update mechanism.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


# ----------------------------------------------------------------------
# Unified Flux-Based Gliner Hybrid
# ----------------------------------------------------------------------
class UnifiedFluxGliner:
    def __init__(self, text: str, labels: List[str]):
        self.text = text
        self.labels = labels
        self.spans = []

    def extract_labels(self):
        for label in self.labels:
            pattern = re.compile(r"(?<!\w)" + re.escape(label) + r"(?!\w)")
            for m in pattern.finditer(self.text):
                span = Span(m.start(), m.end(), self.text, label, 0.0)
                self.spans.append(span)

    def compute_scores(self, conductance: float, edge_length: float, pressure_a: float, pressure_b: float):
        for span in self.spans:
            q = flux(conductance, edge_length, pressure_a, pressure_b)
            span.score = update_conductance(span.score, q)

    def get_spans(self):
        return self.spans


import re

def main():
    text = "This is a test sentence with labels Operator and Rainmaker."
    labels = ["Operator", "Rainmaker"]
    unified_gliner = UnifiedFluxGliner(text, labels)
    unified_gliner.extract_labels()
    conductance = 1.0
    edge_length = 1.0
    pressure_a = 1.0
    pressure_b = 0.0
    unified_gliner.compute_scores(conductance, edge_length, pressure_a, pressure_b)
    for span in unified_gliner.get_spans():
        print(span)


if __name__ == "__main__":
    main()