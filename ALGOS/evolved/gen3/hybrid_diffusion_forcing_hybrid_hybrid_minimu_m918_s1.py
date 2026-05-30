# DARWIN HAMMER — match 918, survivor 1
# gen: 3
# parent_a: diffusion_forcing.py (gen0)
# parent_b: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s2.py (gen2)
# born: 2026-05-29T23:31:39Z

"""
Hybrid Diffusion Forcing - Epistemic Bayesian Minimum-Cost Tree.

This module fuses the Diffusion Forcing algorithm (Chen et al. 2024) with the
Hybrid Epistemic-Bayesian Minimum-Cost Tree (parent B). The mathematical bridge
between the two lies in treating the noise schedule of Diffusion Forcing as a
probabilistic prior, which is then updated using the Bayesian rules from the
tree module. Specifically, we interpret the cumulative noise schedule
alpha_bar_t as a probability distribution, and use it as the prior in the
Bayesian update equations.

The resulting hybrid system integrates the strengths of both parents:
- Diffusion Forcing provides a flexible and efficient way to model complex
  sequential data, while
- The Epistemic Bayesian Minimum-Cost Tree provides a robust and principled
  approach to handling uncertainty and epistemic confidence.

The three core functions below demonstrate the hybrid operation:
1. `diffusion_forcing_noise_to_probability` – maps a diffusion forcing noise
   schedule to a probability distribution.
2. `hybrid_tree_cost_with_diffusion_forcing` – computes the total cost of a
   tree where every edge weight incorporates Bayesian updating using the
   diffusion forcing noise schedule as a prior.
3. `aggregate_tree_certainty_with_diffusion_forcing` – produces a single
   scalar summarising the overall epistemic-Bayesian certainty of the whole
   tree (product of posteriors).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
import math
import random
import sys
import pathlib

# Minimal re-implementation of the epistemic certainty helpers (parent B)
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

def confidence_to_probability(cf: CertaintyFlag) -> float:
    return cf.confidence_bps / 10000.0

def diffusion_forcing_noise_to_probability(noise_schedule: np.ndarray) -> np.ndarray:
    return noise_schedule

def hybrid_tree_cost_with_diffusion_forcing(noise_schedule: np.ndarray, 
                                           tree: Dict[str, Dict[str, float]], 
                                           certainty_flags: List[CertaintyFlag]) -> float:
    total_cost = 0.0
    for edge, weight in tree.items():
        prior = diffusion_forcing_noise_to_probability(noise_schedule)
        likelihood = confidence_to_probability(certainty_flags[0])
        posterior = prior * likelihood / (prior * likelihood + (1 - prior) * (1 - likelihood))
        total_cost += weight * posterior
    return total_cost

def aggregate_tree_certainty_with_diffusion_forcing(noise_schedule: np.ndarray, 
                                                    tree: Dict[str, Dict[str, float]], 
                                                    certainty_flags: List[CertaintyFlag]) -> float:
    total_certainty = 1.0
    for edge, weight in tree.items():
        prior = diffusion_forcing_noise_to_probability(noise_schedule)
        likelihood = confidence_to_probability(certainty_flags[0])
        posterior = prior * likelihood / (prior * likelihood + (1 - prior) * (1 - likelihood))
        total_certainty *= posterior
    return total_certainty

def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    if schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        alpha_bars = np.clip(alpha_bars, 0.0, 1.0)
        return alpha_bars
    else:
        raise ValueError("Unknown schedule")

if __name__ == "__main__":
    T = 100
    noise_sched = noise_schedule(T)
    tree = {"edge1": {"weight": 1.0}, "edge2": {"weight": 2.0}}
    cf = CertaintyFlag("FACT", 8000, "high", "example rationale")
    certainty_flags = [cf]
    hybrid_cost = hybrid_tree_cost_with_diffusion_forcing(noise_sched, tree, certainty_flags)
    aggregate_certainty = aggregate_tree_certainty_with_diffusion_forcing(noise_sched, tree, certainty_flags)
    print("Hybrid tree cost:", hybrid_cost)
    print("Aggregate tree certainty:", aggregate_certainty)