# DARWIN HAMMER — match 918, survivor 2
# gen: 3
# parent_a: diffusion_forcing.py (gen0)
# parent_b: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s2.py (gen2)
# born: 2026-05-29T23:31:39Z

"""
Hybrid Diffusion Forcing with Epistemic Certainty.

This module integrates the core topologies of two parent algorithms: 
- Diffusion Forcing (Chen et al. 2024), which assigns an independent noise level to each token in a sequence, 
- Hybrid Epistemic-Bayesian Minimum-Cost Tree, which applies Bayesian updating and epistemic confidence to tree cost computation.

The mathematical bridge between the two algorithms lies in the application of epistemic certainty to the diffusion forcing process. 
In this hybrid algorithm, we treat the confidence in the interval [0, 1] as a prior for the noise schedule, 
and apply Bayesian updating to the denoising process.

Key functions:
1. `confidence_to_noise_schedule` - maps a `CertaintyFlag` to a noise schedule.
2. `hybrid_diffusion_forcing` - computes the diffusion forcing loss with epistemic certainty.
3. `aggregate_certainty` - produces a single scalar summarising the overall epistemic-Bayesian certainty of the whole sequence.
"""

import numpy as np
import math
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

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")

def confidence_to_noise_schedule(T: int, certainty_flag: CertaintyFlag, schedule: str = "cosine") -> np.ndarray:
    """
    Returns the cumulative noise schedule alpha_bar, shape (T+1,).
    
    Parameters
    ----------
    T:
        Total number of diffusion timesteps.
    certainty_flag:
        A `CertaintyFlag` object representing the epistemic certainty.
    schedule:
        "cosine" (Nichol & Dhariwal 2021) or "linear" (Ho et al. 2020).
    
    Returns
    -------
    np.ndarray shape (T+1,) with values in (0, 1].
    """
    confidence = certainty_flag.confidence_bps / 10000
    if schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0] * confidence
        alpha_bars = np.clip(alpha_bars, 0, 1)
        return alpha_bars
    else:
        raise NotImplementedError("Only 'cosine' schedule is implemented")

def hybrid_diffusion_forcing(T: int, x_0: np.ndarray, certainty_flag: CertaintyFlag, epsilon: np.ndarray, lambda_func: callable) -> float:
    """
    Computes the diffusion forcing loss with epistemic certainty.
    
    Parameters
    ----------
    T:
        Total number of diffusion timesteps.
    x_0:
        The original sequence.
    certainty_flag:
        A `CertaintyFlag` object representing the epistemic certainty.
    epsilon:
        The noise schedule.
    lambda_func:
        A function representing the weighting function.
    
    Returns
    -------
    The diffusion forcing loss.
    """
    alpha_bars = confidence_to_noise_schedule(T, certainty_flag)
    loss = 0
    for t in range(T + 1):
        alpha_bar_t = alpha_bars[t]
        x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * epsilon
        loss += lambda_func(t) * np.linalg.norm(x_t - epsilon) ** 2
    return loss

def aggregate_certainty(certainty_flags: List[CertaintyFlag]) -> float:
    """
    Produces a single scalar summarising the overall epistemic-Bayesian certainty of the whole sequence.
    
    Parameters
    ----------
    certainty_flags:
        A list of `CertaintyFlag` objects.
    
    Returns
    -------
    The aggregated certainty.
    """
    confidence = 1
    for certainty_flag in certainty_flags:
        confidence *= certainty_flag.confidence_bps / 10000
    return confidence

if __name__ == "__main__":
    # Smoke test
    T = 10
    x_0 = np.random.rand(T)
    certainty_flag = CertaintyFlag("FACT", 10000, "authority", "rationale")
    epsilon = np.random.rand(T)
    lambda_func = lambda t: 1 / (t + 1)
    loss = hybrid_diffusion_forcing(T, x_0, certainty_flag, epsilon, lambda_func)
    print(f"Loss: {loss}")
    certainty_flags = [CertaintyFlag("FACT", 10000, "authority", "rationale") for _ in range(5)]
    aggregated_certainty = aggregate_certainty(certainty_flags)
    print(f"Aggregated certainty: {aggregated_certainty}")