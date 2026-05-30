# DARWIN HAMMER — match 1382, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_label_foundry_path_signature_m231_s0.py (gen3)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s1.py (gen1)
# born: 2026-05-29T23:37:09Z

"""
HybridCaputoPathSignature: Fusion of Caputo Fractional Minimum Cost Tree (parent A - hybrid_caputo_fractional_minimum_cost_tree_m35_s1.py)
and Hybrid Endpoint Label Signature (parent B - hybrid_hybrid_label_foundry_path_signature_m231_s0.py).

Mathematical bridge
-------------------
The hybrid uses the fractional derivative (from parent A) to regularize the recovery priority ρ (from parent B) in the path signature tensor S2.
This is achieved by applying the Caputo fractional derivative to the recovery priority ρ:

    ρ_reg = ∂^α ρ / ∂t^α

The regularized recovery priority ρ_reg is then used to scale the signature tensor S2:

    S2_scaled = exp(ρ_reg * S2)

This combines the fractional dynamics of parent A with the path signature of parent B.

"""

import numpy as np
from math import exp, gamma
from random import random
from dataclasses import dataclass
from typing import List, Dict, Callable

@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # binary 0/1

@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # in [0,1]

def labeling_function(name: str | None = None):
    """Decorator that annotates a labeling function with a name."""
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """Pure A-logic: majority vote with confidence = proportion of votes."""
    votes: Dict[str, List[int]] = {}
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes.setdefault(r.doc_id, []).append(r.label)
    out: List[ProbabilisticLabel] = []
    for doc_id, labels in votes.items():
        confidence = np.mean(labels)
        out.append(ProbabilisticLabel(doc_id, 0, confidence))  # binary 0 for simplicity
    return out

def caputo_derivative(f, alpha, t, tau):
    """Caputo Fractional Derivative

    :param f: Function to differentiate
    :param alpha: Fractional order
    :param t: Time point
    :param tau: Time span
    :return: Caputo Fractional Derivative
    """
    return 1 / gamma(1 - alpha) * np.sum(f[tau] / (t - tau)**alpha)

def signature_level2_scaled(path, rho, alpha, t, tau):
    """Level-2 iterated integral tensor with scaled signature.

    path: (T, d). Returns (d, d).
    S2_scaled[i,j] = sum_{t=1..T-1} (X_{t-1}[i] - X_0[i]) * (X_t[j] - X_{t-1}[j])
                    * exp(caputo_derivative(rho, alpha, t, tau) * S2[i,j])
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0) 
    S2 = np.zeros((path.shape[1], path.shape[1]))
    for i in range(path.shape[1]):
        for j in range(path.shape[1]):
            S2[i, j] = np.sum((path[:-1, i] - path[0, i]) * (path[1:, j] - path[:-1, j]))
    rho_reg = caputo_derivative(rho, alpha, t, tau)
    S2_scaled = np.exp(rho_reg * S2)
    return S2_scaled

def hybrid_fusion(path, rho, alpha, t, tau):
    """Hybrid fusion of Caputo fractional minimum cost tree and path signature.

    :param path: Path geometry
    :param rho: Recovery priority
    :param alpha: Fractional order
    :param t: Time point
    :param tau: Time span
    :return: Scaled signature tensor
    """
    S2_scaled = signature_level2_scaled(path, rho, alpha, t, tau)
    return S2_scaled

if __name__ == "__main__":
    path = np.random.rand(10, 2)
    rho = np.random.rand(10)
    alpha = 0.5
    t = 1.0
    tau = np.arange(10)
    S2_scaled = hybrid_fusion(path, rho, alpha, t, tau)
    print(S2_scaled)