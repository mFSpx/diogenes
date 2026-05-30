# DARWIN HAMMER — match 166, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s2.py (gen2)
# parent_b: hybrid_bayes_claim_kernel_hybrid_ternary_route_m84_s1.py (gen3)
# born: 2026-05-29T23:27:24Z

"""Hybrid Algorithm: Distributed Leader Election ↔ Hoeffding Tree ↔ Bayesian Edge Reliability ↔ Tropical Max‑Plus Cost

Parents
-------
* **Parent A** – `hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s2.py`  
  Provides a probabilistic leader‑election framework (broadcast and acceptance
  probabilities, simulated annealing temperature) together with Hoeffding‑bound
  statistical guarantees and a Tropical max‑plus algebra (t_add, t_mul,
  t_matmul).

* **Parent B** – `hybrid_bayes_claim_kernel_hybrid_ternary_route_m84_s1.py`  
  Supplies a Bayesian hypothesis kernel that updates edge reliabilities
  (`MathHypothesis`) using evidence (`MathEvidence`).  The posterior
  probability is used as a multiplicative confidence factor on edge lengths.

Mathematical Bridge
-------------------
The bridge is the **probabilistic weight** that appears in both families:

* In Parent A the acceptance probability `p_accept = exp(-ΔE / T)` is a
  temperature‑scaled confidence that a proposed state (e.g. a split in a
  decision tree) should be taken.
* In Parent B the posterior `P(H|E)` is a confidence that an edge is reliable.

Both are numbers in **[0, 1]** and can therefore be multiplied with a physical
quantity (split gain, edge length).  The hybrid algorithm therefore:

1. Uses `acceptance_probability` together with a Hoeffding bound to decide
   whether a node should become a *leader* (or a tree split).
2. Updates edge reliabilities with Bayesian evidence.
3. Forms a **weighted cost matrix** `Ĉ = C ⊙ P` where `C` holds raw edge lengths
   and `P` holds posterior reliabilities.
4. Evaluates the global cost of a routing tree with **Tropical max‑plus algebra**:
   `cost = max_path_sum(Ĉ) = max_{paths} Σ (length·posterior)`.  The max‑plus
   operations `t_add` (max) and `t_mul` (addition) are reused from Parent A.

The three core functions below demonstrate this fusion.  They can be
combined in a distributed leader‑election loop, a streaming Hoeffding‑tree
learner, or a Bayesian‑aware routing optimizer."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Tuple, Mapping, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Types shared between the two parent topologies
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]
Edge = Tuple[Node, Node]

@dataclass(frozen=True)
class MathEvidence:
    """Observation used to update an edge hypothesis."""
    id: str
    measurement: float          # observed length or signal strength
    noise_std: float            # standard deviation of measurement noise

@dataclass(frozen=True)
class MathHypothesis:
    """Bayesian hypothesis attached to a tree edge."""
    id: str
    prior: float                # prior reliability (0‑1)
    posterior: float            # current posterior (0‑1)
    evidence_ids: Tuple[str, ...] = ()

# ----------------------------------------------------------------------
# Parent‑A utilities (probabilistic leader election & tropical algebra)
# ----------------------------------------------------------------------
def broadcast_probability(phase: int, step: int) -> float:
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

def acceptance_probability(delta_e: float, temperature: float) -> float:
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def t_add(x, y):
    """Tropical addition = maximum."""
    return np.maximum(x, y)

def t_mul(x, y):
    """Tropical multiplication = ordinary addition."""
    return np.add(x, y)

def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical matrix multiplication (max‑plus):
    (A ⊗ B)[i, j] = max_k (A[i, k] + B[k, j])
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible dimensions for tropical matmul")
    # Use broadcasting: (n, m, 1) + (1, m, p) -> (n, m, p) then max over axis=1
    A_exp = A[:, :, np.newaxis]          # shape (n, m, 1)
    B_exp = B[np.newaxis, :, :]          # shape (1, m, p)
    C = np.max(A_exp + B_exp, axis=1)    # shape (n, p)
    return C

# ----------------------------------------------------------------------
# Parent‑B utility (Bayesian hypothesis update)
# ----------------------------------------------------------------------
def update_hypothesis(
    hypothesis: MathHypothesis,
    evidence: MathEvidence,
    likelihood_ratio: float,
) -> MathHypothesis:
    """Update posterior using odds form to retain numerical stability."""
    if likelihood_ratio < 0:
        raise ValueError("likelihood_ratio must be non‑negative")
    # Convert posterior to odds, apply likelihood ratio, convert back.
    odds = hypothesis.posterior / (1.0 - hypothesis.posterior + 1e-12)
    new_odds = odds * likelihood_ratio
    posterior = new_odds / (1.0 + new_odds)
    posterior = max(0.0, min(1.0, posterior))
    new_eids = hypothesis.evidence_ids + (evidence.id,)
    return replace(hypothesis, posterior=posterior, evidence_ids=new_eids)

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def hybrid_split_decision(
    delta_e: float,
    step: int,
    phase: int,
    r: float,
    delta: float,
    n_samples: int,
    temperature_schedule: List[float],
) -> bool:
    """
    Combine acceptance probability (simulated annealing) with a Hoeffding bound
    to decide whether a node becomes a leader / a split point.

    Returns True if the probabilistic test passes.
    """
    # 1️⃣ Acceptance based on temperature schedule
    temperature = temperature_schedule[step] if step < len(temperature_schedule) else cooling_temperature(step)
    p_accept = acceptance_probability(delta_e, temperature)

    # 2️⃣ Hoeffding bound on the observed gain (ΔE) treated as a random variable
    bound = hoeffding_bound(r, delta, n_samples)

    # 3️⃣ Broadcast probability modulates the final decision
    p_broadcast = broadcast_probability(phase, step)

    # Final probability is the product of the three independent confidences
    final_p = p_accept * (1.0 - bound) * p_broadcast
    return random.random() < final_p

def bayesian_update_edges(
    hypotheses: Dict[Edge, MathHypothesis],
    evidences: List[MathEvidence],
    likelihood_func: Any,
) -> Dict[Edge, MathHypothesis]:
    """
    Update all edge hypotheses with the supplied evidences.
    `likelihood_func` receives (hypothesis, evidence) and must return a
    non‑negative likelihood ratio.
    """
    updated = {}
    for edge, hyp in hypotheses.items():
        post = hyp
        for ev in evidences:
            lr = likelihood_func(post, ev)
            post = update_hypothesis(post, ev, lr)
        updated[edge] = post
    return updated

def tropical_bayesian_cost(
    cost_matrix: np.ndarray,
    hypotheses: Dict[Edge, MathHypothesis],
    node_index: Dict[Node, int],
) -> float:
    """
    Compute the global routing cost of a tree using Bayesian posteriors as
    multiplicative confidence factors and Tropical max‑plus algebra to aggregate
    path costs.

    The procedure:
    1. Build a weighted matrix Ĉ where Ĉ[i, j] = length(i, j) * posterior(i, j)
       (0 if no edge exists).
    2. Compute the tropical closure Ĉ* = I ⊕ Ĉ ⊕ Ĉ⊗Ĉ ⊕ … (here we iterate a
       fixed number of times sufficient for small graphs).
    3. The overall cost is the maximum entry of Ĉ* (the longest confident path).
    """
    n = cost_matrix.shape[0]
    weighted = np.full((n, n), -np.inf)          # -inf is tropical zero (identity for max)

    # Populate weighted matrix with length * posterior
    for (u, v), hyp in hypotheses.items():
        i, j = node_index[u], node_index[v]
        weighted[i, j] = cost_matrix[i, j] * hyp.posterior
        # Assume undirected graph for simplicity
        weighted[j, i] = cost_matrix[j, i] * hyp.posterior

    # Tropical identity matrix (0 on diagonal, -inf elsewhere)
    I = np.full((n, n), -np.inf)
    np.fill_diagonal(I, 0.0)

    # Compute closure by repeated tropical multiplication (n‑1 times is enough for
    # a graph without positive cycles).
    closure = I.copy()
    power = weighted.copy()
    for _ in range(n - 1):
        closure = t_add(closure, power)            # closure = closure ⊕ power
        power = t_matmul(power, weighted)          # power = power ⊗ weighted

    # The overall cost is the maximum finite entry in the closure
    finite_vals = closure[np.isfinite(closure)]
    return float(np.max(finite_vals)) if finite_vals.size > 0 else 0.0

# ----------------------------------------------------------------------
# Simple likelihood function for demonstration
# ----------------------------------------------------------------------
def gaussian_likelihood_ratio(hypothesis: MathHypothesis, evidence: MathEvidence) -> float:
    """
    Likelihood ratio for a Gaussian measurement model:
        LR = exp( - (m - μ)^2 / (2σ^2) )
    where μ is the expected length (here we use hypothesis.prior as proxy)
    and σ is the noise standard deviation.
    """
    mu = hypothesis.prior * evidence.measurement   # crude proxy
    sigma = evidence.noise_std + 1e-9
    exponent = - ((evidence.measurement - mu) ** 2) / (2 * sigma ** 2)
    return math.exp(exponent)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ----- Build a tiny graph ------------------------------------------------
    nodes = ['A', 'B', 'C']
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Raw length matrix (symmetric)
    lengths = np.array([
        [0.0, 5.0, 3.0],
        [5.0, 0.0, 2.0],
        [3.0, 2.0, 0.0],
    ])

    # Initialise hypotheses with uniform priors
    edges = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]
    hyps = {
        e: MathHypothesis(id=f"H_{e[0]}{e[1]}", prior=0.8, posterior=0.8)
        for e in edges
    }

    # ----- Simulate evidence -------------------------------------------------
    evidences = [
        MathEvidence(id="e1", measurement=5.2, noise_std=0.3),
        MathEvidence(id="e2", measurement=2.1, noise_std=0.2),
    ]

    # Update hypotheses with Bayesian evidence
    hyps = bayesian_update_edges(hyps, evidences, gaussian_likelihood_ratio)

    # ----- Hybrid split decision (example) -----------------------------------
    delta_e = 0.7                      # observed gain
    step = 2
    phase = 1
    r = 1.0
    delta = 0.05
    n_samples = 30
    temperature_schedule = [1.0, 0.8, 0.6, 0.5, 0.4]

    split = hybrid_split_decision(
        delta_e, step, phase, r, delta, n_samples, temperature_schedule
    )
    print(f"Hybrid split decision: {'accept' if split else 'reject'}")

    # ----- Compute tropical Bayesian cost ------------------------------------
    total_cost = tropical_bayesian_cost(lengths, hyps, node_idx)
    print(f"Tropical Bayesian routing cost: {total_cost:.4f}")

    # Simple sanity check: cost should be finite and non‑negative
    assert total_cost >= 0.0
    sys.exit(0)