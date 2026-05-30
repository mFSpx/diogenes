# DARWIN HAMMER — match 918, survivor 0
# gen: 3
# parent_a: diffusion_forcing.py (gen0)
# parent_b: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s2.py (gen2)
# born: 2026-05-29T23:31:39Z

"""
Hybrid Diffusion Forcing with Epistemic Certainty.

This module integrates the Diffusion Forcing algorithm with the Epistemic Certainty framework.
The mathematical bridge is established by treating the cumulative noise schedule alpha_bar
as a prior probability distribution for the epistemic certainty model.
Specifically, we use the noise schedule to update the confidence of the CertaintyFlag objects,
which in turn affects the edge weights in the hybrid tree cost computation.

Parents:
- `diffusion_forcing.py` – assigns an independent noise level to each token in the sequence,
  enabling causal planning and flexible time horizons.
- `hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s2.py` – defines a hybrid epistemic-Bayesian
  minimum-cost tree, where each edge weight incorporates Bayesian updating and epistemic confidence.
"""

import math
import numpy as np
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Tuple, Dict, List

EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""


def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """Return the cumulative noise schedule alpha_bar, shape (T+1,)."""
    if schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        return np.clip(alpha_bars, 0.001, 0.999)  # Clip to ensure numerical stability


def confidence_to_probability(cf: CertaintyFlag) -> float:
    """Map a CertaintyFlag to a probability."""
    return cf.confidence_bps / 10000


def hybrid_tree_cost_with_certainty(alpha_bars: np.ndarray, cf: CertaintyFlag) -> float:
    """Compute the total cost of a tree where every edge weight incorporates Bayesian updating and epistemic confidence."""
    p = confidence_to_probability(cf)
    updated_p = p * alpha_bars[0] + (1 - p) * alpha_bars[-1]
    return updated_p


def aggregate_tree_certainty(alpha_bars: np.ndarray, cfs: List[CertaintyFlag]) -> float:
    """Produce a single scalar summarising the overall epistemic-Bayesian certainty of the whole tree."""
    posteriors = [hybrid_tree_cost_with_certainty(alpha_bars, cf) for cf in cfs]
    return np.prod(posteriors)


def diffusion_forcing_loss(alpha_bars: np.ndarray, cfs: List[CertaintyFlag], epsilon: np.ndarray) -> float:
    """compute the diffusion forcing loss with epistemic certainty."""
    posteriors = [hybrid_tree_cost_with_certainty(alpha_bars, cf) for cf in cfs]
    return np.sum([np.abs(epsilon[i] - posteriors[i]) ** 2 for i in range(len(posteriors))])


if __name__ == "__main__":
    T = 10
    alpha_bars = noise_schedule(T)
    cf = CertaintyFlag("FACT", 8000, "authority", "rationale")
    cost = hybrid_tree_cost_with_certainty(alpha_bars, cf)
    print(cost)
    epsilon = np.random.rand(T)
    loss = diffusion_forcing_loss(alpha_bars, [cf] * T, epsilon)
    print(loss)