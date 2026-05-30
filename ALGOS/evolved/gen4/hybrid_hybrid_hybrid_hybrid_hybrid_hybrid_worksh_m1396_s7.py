# DARWIN HAMMER — match 1396, survivor 7
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s0.py (gen3)
# parent_b: hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py (gen2)
# born: 2026-05-29T23:35:58Z

"""Hybrid Bayesian‑LTC Allocation Module
=====================================

Parents
-------
* **Parent A** – `hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s0.py`
  Provides Bayesian update utilities (`bayes_marginal`, `bayes_update`),
  geometric helpers (`length`) and a label‑scoring routine (`label_score`).
* **Parent B** – `hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py`
  Implements a Liquid‑Time‑Constant (LTC) cell whose effective time‑constant
  τₛᵧₛ(t) modulates a resource‑allocation schedule over calendar days.

Mathematical Bridge
-------------------
For each calendar day *t* we treat the **group‑wise allocation percentages**
as *priors* *P(g)*.  The *semantic‑neighbor distance* between the day’s
document and a group‑specific label yields a *likelihood* *L(g|d)*.
Using the Bayesian marginal

M(g) = L(g|d)·P(g) + fp·(1‑P(g))

with a small false‑positive rate *fp*, we obtain the *posterior*

P⁺(g) = P(g)·L(g|d) / M(g)

The posterior vector **p⁺** (summing to ≈1) is fed as the *base LLM share*
into the LTC cell.  The day‑of‑week (scaled to [0, 1]) is the external input
*I(t)*.  The LTC computes an input‑dependent effective time‑constant

τ_eff(t) = τ / (1 + τ·σ(w·x(t) + b·I(t)))          (σ = sigmoid)

which in turn scales the LLM portion:

llm_units(t) = llm_base · (τ_eff(t) / τ_max)

Thus the Bayesian posterior informs *what* is allocated, while the LTC
dynamics determine *how much* is allocated on that day, achieving a
tight mathematical coupling of the two parent systems.

The module below implements this hybrid pipeline with three public
functions:
1. `init_hybrid_ltc` – create an LTC state object.
2. `hybrid_allocate_by_dates` – compute per‑day, per‑group allocations.
3. `summarize_hybrid_savings` – compare baseline vs. hybrid allocations.
"""

import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Parent A utilities (Bayesian & label scoring)
# ---------------------------------------------------------------------------

Point = tuple[float, float]
Edge = tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) = L·P + fp·(1‑P)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P·L / P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

def parse_labels(label: str) -> list[str]:
    """Simple splitter – in real code this would be more sophisticated."""
    return [lbl.strip() for lbl in label.split(",") if lbl.strip()]

class Span:
    """Placeholder for a scored text span."""
    def __init__(self, score: float):
        self.score = score

def literal_fallback(text: str, labels: list[str], case_sensitive: bool = False) -> list[Span]:
    """Very naive scoring: count occurrences of each label."""
    if not case_sensitive:
        text = text.lower()
        labels = [l.lower() for l in labels]
    scores = []
    for lbl in labels:
        count = text.split().count(lbl)
        scores.append(Span(score=float(count)))
    return scores

def label_score(text: str, label: str) -> float:
    """Score a label against text via literal fallback."""
    labels = parse_labels(label)
    spans = literal_fallback(text, labels, case_sensitive=False)
    return sum(span.score for span in spans)

def semantic_neighbor_distance(doc: str, label: str) -> float:
    """
    Mocked semantic‑neighbor distance.
    In a real system this would be a cosine distance or similar.
    Here we map the label score (higher = more similar) to a distance
    in [0,1] by an exponential decay.
    """
    score = label_score(doc, label)
    # Convert to a pseudo‑probability (capped at 1)
    prob = min(1.0, score / (len(doc.split()) + 1e-9))
    # Distance = 1 - probability, then squash to [0,1]
    return 1.0 - prob

# ---------------------------------------------------------------------------
# Parent B utilities (Liquid Time‑Constant)
# ---------------------------------------------------------------------------

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def sigmoid(x: float) -> float:
    """Standard logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))

class LTCState:
    """Scalar Liquid‑Time‑Constant cell."""
    def __init__(self, tau: float = 1.0, weight: float = 1.0, bias: float = 0.0):
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        self.tau = tau
        self.w = weight
        self.b = bias
        self.x = 0.0          # hidden state
        self.tau_history: list[float] = []

    def step(self, input_val: float) -> float:
        """
        Perform one discrete update.
        input_val ∈ [0,1] (day‑of‑week scaled).
        Returns the effective time‑constant τ_eff for this step.
        """
        gate = sigmoid(self.w * self.x + self.b * input_val)
        tau_eff = self.tau / (1.0 + self.tau * gate)
        # Simple Euler integration for hidden state
        dt = 1.0  # one day
        dx = (-self.x + input_val) * dt / tau_eff
        self.x += dx
        self.tau_history.append(tau_eff)
        return tau_eff

# ---------------------------------------------------------------------------
# Hybrid Functions
# ---------------------------------------------------------------------------

GROUPS = ("codex", "groq", "cohere", "local_models")
DETERMINISTIC_PCT = 0.30   # 30 % of total units are deterministic

def init_hybrid_ltc(tau: float = 1.0, weight: float = 1.0, bias: float = 0.0) -> LTCState:
    """
    Initialise the LTC component of the hybrid system.
    Returns an LTCState instance ready for stepping through days.
    """
    return LTCState(tau=tau, weight=weight, bias=bias)

def compute_posterior_allocation(prior_vec: np.ndarray,
                                 doc: str,
                                 false_positive: float = 0.01) -> np.ndarray:
    """
    Given a prior probability vector over GROUPS and a document string,
    compute the Bayesian posterior using semantic‑neighbor distances as
    likelihoods.

    Parameters
    ----------
    prior_vec : np.ndarray
        Shape (G,) where G = len(GROUPS). Must sum to 1.
    doc : str
        Text associated with the current day.
    false_positive : float
        Small probability for the complement event.

    Returns
    -------
    np.ndarray
        Posterior probability vector (renormalised to sum to 1).
    """
    if prior_vec.shape != (len(GROUPS),):
        raise ValueError("prior_vec must have shape (len(GROUPS),)")
    likelihoods = np.empty_like(prior_vec)
    for i, grp in enumerate(GROUPS):
        # Use the group name as a proxy label
        dist = semantic_neighbor_distance(doc, grp)
        # Convert distance to likelihood: closer => higher likelihood
        likelihoods[i] = 1.0 - dist  # in [0,1]
    marginals = np.array([
        bayes_marginal(p, l, false_positive)
        for p, l in zip(prior_vec, likelihoods)
    ])
    post = np.array([
        bayes_update(p, l, m)
        for p, l, m in zip(prior_vec, likelihoods, marginals)
    ])
    # Renormalise to avoid numerical drift
    total = post.sum()
    if total == 0.0:
        # fallback to uniform if everything collapsed
        return np.full_like(post, 1.0 / len(post))
    return post / total

def hybrid_allocate_by_dates(dates: list[date],
                             total_units_per_day: int,
                             docs_by_date: dict[date, str],
                             ltc: LTCState) -> dict[date, dict[str, float]]:
    """
    Compute per‑day, per‑group allocations.

    Steps per day:
    1. Scale day‑of‑week (Mon=0 … Sun=6) to [0,1] and feed to LTC.
    2. Obtain τ_eff(t) and keep track of the maximal τ observed.
    3. Form a deterministic share (fixed proportion of total_units).
    4. Derive a prior vector for the LLM share (equal split among groups).
    5. Update the prior with the day’s document via Bayesian posterior.
    6. Scale the posterior LLM portion by τ_eff(t) / τ_max.

    Returns
    -------
    dict[date, dict[group, allocated_units]]
    """
    allocations: dict[date, dict[str, float]] = {}
    # Step 4 – deterministic share is split equally among groups
    det_share_per_group = (total_units_per_day * DETERMINISTIC_PCT) / len(GROUPS)
    # Prepare a uniform prior for the LLM part
    llm_prior = np.full(len(GROUPS), 1.0 / len(GROUPS))

    # First pass: collect τ_eff values to compute τ_max
    tau_eff_list: list[float] = []
    for cur_date in dates:
        dow_scaled = (cur_date.weekday() / 6.0)  # Monday=0 … Sunday=6 → [0,1]
        tau_eff = ltc.step(dow_scaled)
        tau_eff_list.append(tau_eff)
    tau_max = max(tau_eff_list) if tau_eff_list else 1.0

    # Reset LTC state for a second deterministic pass (so allocations are reproducible)
    ltc.x = 0.0
    ltc.tau_history.clear()

    for idx, cur_date in enumerate(dates):
        dow_scaled = (cur_date.weekday() / 6.0)
        tau_eff = ltc.step(dow_scaled)
        doc = docs_by_date.get(cur_date, "")
        posterior = compute_posterior_allocation(llm_prior, doc)

        llm_base = total_units_per_day * (1.0 - DETERMINISTIC_PCT)
        llm_scaled = llm_base * (tau_eff / tau_max)

        # Allocate per group
        group_alloc = {}
        for i, grp in enumerate(GROUPS):
            group_alloc[grp] = det_share_per_group + posterior[i] * llm_scaled
        allocations[cur_date] = group_alloc

    return allocations

def summarize_hybrid_savings(allocations: dict[date, dict[str, float]],
                             total_units_per_day: int) -> dict[str, float]:
    """
    Compare the hybrid allocation against a naïve baseline where the LLM
    portion is split uniformly (no Bayesian or LTC modulation).

    Returns a dictionary with:
    - 'baseline_total': total units allocated across all days (should equal
      total_units_per_day * number_of_days).
    - 'hybrid_total'  : total units actually allocated (should be equal).
    - 'llm_savings_pct': percentage reduction in variance of LLM share
      compared to the baseline (higher = more focused allocation).
    """
    days = list(allocations.keys())
    baseline_total = total_units_per_day * len(days)

    # Compute baseline LLM variance (uniform split)
    baseline_llm_per_day = total_units_per_day * (1.0 - DETERMINISTIC_PCT) / len(GROUPS)
    baseline_variance = np.var([baseline_llm_per_day] * len(days))

    # Hybrid LLM variance
    hybrid_llm_vals = []
    for cur_date in days:
        day_total = sum(allocations[cur_date].values())
        llm_total = day_total - (total_units_per_day * DETERMINISTIC_PCT)
        hybrid_llm_vals.append(llm_total)
    hybrid_variance = np.var(hybrid_llm_vals)

    savings_pct = 0.0
    if baseline_variance > 0:
        savings_pct = (1.0 - hybrid_variance / baseline_variance) * 100.0

    return {
        "baseline_total": _pct(baseline_total),
        "hybrid_total": _pct(sum(hybrid_llm_vals) + DETERMINISTIC_PCT * total_units_per_day * len(days)),
        "llm_savings_pct": _pct(savings_pct)
    }

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a week of dates starting today
    start = date.today()
    dates = [start + timedelta(days=i) for i in range(7)]

    # Mock documents: each day gets a short sentence mentioning some groups
    sample_texts = [
        "The codex model performed well today.",
        "groq and cohere were compared.",
        "local_models are useful for edge devices.",
        "codex and groq both excel.",
        "cohere shows improvement.",
        "local_models and codex collaborate.",
        "All models performed equally."
    ]
    docs_by_date = {d: txt for d, txt in zip(dates, sample_texts)}