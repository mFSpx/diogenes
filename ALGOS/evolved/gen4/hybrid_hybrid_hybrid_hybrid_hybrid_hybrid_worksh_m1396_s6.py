# DARWIN HAMMER — match 1396, survivor 6
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s0.py (gen3)
# parent_b: hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py (gen2)
# born: 2026-05-29T23:35:58Z

"""Hybrid Semantic‑Bayesian + Liquid‑Time‑Constant Allocation.

Parents:
- **Parent A** – semantic_neighbors + Bayesian evidence update (semantic
  distances → likelihoods, label scoring → prior).
- **Parent B** – work‑share allocation modulated by a Liquid‑Time‑Constant
  (LTC) network whose effective time‑constant τₛᵧₛ(t) depends on an external
  input I(t).

Mathematical Bridge
-------------------
For each time step *t* (a calendar day) we first compute a Bayesian
posterior *pₜ* for a given document using the semantic‑neighbor distances as
likelihoods and a label‑derived prior.  This posterior *pₜ∈[0,1]* becomes the
external input *I(t)* of the LTC.  The LTC gating function *f* is a sigmoid
parameterised by a learned weight *w* and bias *b*:

    f(I) = 1 / (1 + exp(‑w·I‑b))

The effective time‑constant is then

    τₛᵧₛ(t) = τ / (1 + τ·f(I(t)))

where *τ* is a base constant.  The maximal observed τₛᵧₛ over the processed
date sequence, τₘₐₓ, normalises the LLM‑portion of the resource allocation:

    llm_units(t) = llm_base(t) · (τₛᵧₛ(t) / τₘₐₓ)

The deterministic portion of the total units is split uniformly across the
pre‑defined groups, while the LTC‑scaled LLM units are distributed proportionally
to the same groups.  Thus the Bayesian evidence from the semantic graph drives
the temporal dynamics of the LTC, which in turn reshapes the allocation schedule.

The module provides three public functions that illustrate the hybrid
operation and a small smoke‑test when executed as a script.
"""

import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Point = tuple[float, float]
Edge = tuple[str, str]

# ---------------------------------------------------------------------------
# Parent A – Bayesian / Semantic utilities
# ---------------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) = L·P + FP·(1‑P)."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P·L / P(E)."""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def label_score(text: str, label: str) -> float:
    """
    Dummy label scoring: count case‑insensitive occurrences of the label
    in the text, normalised by length.
    """
    txt = text.lower()
    lbl = label.lower()
    count = txt.count(lbl)
    return count / max(1, len(txt.split()))

def semantic_neighbors(doc_id: str, k: int = 5) -> list[tuple[str, float]]:
    """
    Placeholder semantic neighbor generator.
    Returns *k* synthetic neighbor IDs with random distances in (0, 1].
    """
    random.seed(hash(doc_id) & 0xffffffff)
    neighbors = []
    for i in range(k):
        nid = f"{doc_id}_nbr_{i}"
        dist = random.random() * 0.9 + 0.1  # avoid zero
        neighbors.append((nid, dist))
    return neighbors

# ---------------------------------------------------------------------------
# Parent B – Liquid‑Time‑Constant (LTC) utilities
# ---------------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

# ---------------------------------------------------------------------------
# Hybrid System
# ---------------------------------------------------------------------------
class HybridLTC:
    """
    Encapsulates the parameters of the hybrid system.
    """
    def __init__(self,
                 base_tau: float = 1.0,
                 weight: float = 5.0,
                 bias: float = -2.5,
                 deterministic_pct: float = 0.6):
        """
        Parameters
        ----------
        base_tau : float
            Base time constant τ.
        weight, bias : float
            Parameters of the gating sigmoid f(I) = sigmoid(weight·I + bias).
        deterministic_pct : float
            Fraction of total units allocated deterministically (0‑1).
        """
        if not 0.0 <= deterministic_pct <= 1.0:
            raise ValueError("deterministic_pct must be in [0,1]")
        self.base_tau = float(base_tau)
        self.w = float(weight)
        self.b = float(bias)
        self.deterministic_pct = deterministic_pct

    # -----------------------------------------------------------------------
    # 1️⃣ Bayesian posterior from semantic neighbourhood
    # -----------------------------------------------------------------------
    def posterior_for_doc(self,
                          text: str,
                          label: str,
                          doc_id: str,
                          false_positive: float = 0.05) -> float:
        """
        Compute a Bayesian posterior using:
        - prior  = label_score(text, label)
        - likelihood = average similarity derived from semantic neighbor distances
        - false_positive = configurable noise term
        """
        prior = label_score(text, label)
        neighbors = semantic_neighbors(doc_id, k=5)
        # Convert distances to similarities (higher = more similar)
        similarities = [1.0 / d for _, d in neighbors]
        likelihood = sum(similarities) / max(1, len(similarities))
        # Clamp to [0,1]
        likelihood = max(0.0, min(1.0, likelihood))
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)
        return posterior

    # -----------------------------------------------------------------------
    # 2️⃣ LTC effective time‑constant given an external input I (posterior)
    # -----------------------------------------------------------------------
    def effective_tau(self, input_signal: float) -> float:
        """
        τₛᵧₛ = τ / (1 + τ·f(I)), where f(I) = sigmoid(w·I + b).
        """
        gating = sigmoid(self.w * input_signal + self.b)
        return self.base_tau / (1.0 + self.base_tau * gating)

    # -----------------------------------------------------------------------
    # 3️⃣ Allocation schedule over a date range
    # -----------------------------------------------------------------------
    def allocate_by_dates(self,
                          start_date: date,
                          end_date: date,
                          total_units_per_day: int,
                          texts: dict[str, str],
                          labels: dict[str, str],
                          doc_ids: list[str]) -> dict[date, dict[str, float]]:
        """
        For each day in the inclusive range [start_date, end_date]:
        - Pick a document ID cyclically from *doc_ids*.
        - Compute posterior pₜ using its text and label.
        - Feed pₜ as I(t) to the LTC and obtain τₛᵧₛ(t).
        - After processing the whole window, normalise τₛᵧₛ(t) by τₘₐₓ
          and allocate resources.

        Returns
        -------
        allocations : dict
            {date: {group: allocated_units, ...}, ...}
        """
        # -------------------------------------------------------------------
        # Phase 1 – collect τₛᵧₛ(t) for each day
        # -------------------------------------------------------------------
        day_count = (end_date - start_date).days + 1
        tau_series = []
        posterior_series = []
        doc_cycle = (doc_ids * ((day_count // len(doc_ids)) + 1))[:day_count]

        for idx, cur_date in enumerate(
                (start_date + timedelta(days=i) for i in range(day_count))):
            doc_id = doc_cycle[idx]
            text = texts.get(doc_id, "")
            label = labels.get(doc_id, "")
            posterior = self.posterior_for_doc(text, label, doc_id)
            tau = self.effective_tau(posterior)
            tau_series.append(tau)
            posterior_series.append(posterior)

        tau_max = max(tau_series) if tau_series else 1.0

        # -------------------------------------------------------------------
        # Phase 2 – allocate per day
        # -------------------------------------------------------------------
        allocations: dict[date, dict[str, float]] = {}
        for idx, cur_date in enumerate(
                (start_date + timedelta(days=i) for i in range(day_count))):
            tau = tau_series[idx]
            posterior = posterior_series[idx]

            # deterministic share
            det_units = total_units_per_day * self.deterministic_pct
            det_per_group = det_units / len(GROUPS)

            # LLM share modulated by LTC
            llm_base = total_units_per_day * (1.0 - self.deterministic_pct)
            llm_units = llm_base * (tau / tau_max)
            llm_per_group = llm_units / len(GROUPS)

            # Assemble allocation dict
            day_alloc = {g: _pct(det_per_group + llm_per_group) for g in GROUPS}
            allocations[cur_date] = day_alloc

        return allocations

    # -----------------------------------------------------------------------
    # 4️⃣ Summarisation utility
    # -----------------------------------------------------------------------
    def summarize(self,
                  allocations: dict[date, dict[str, float]],
                  total_units_per_day: int) -> str:
        """
        Produce a one‑line summary comparing baseline (no LTC modulation)
        to the hybrid allocation.
        """
        baseline_llm = total_units_per_day * (1.0 - self.deterministic_pct)
        baseline_total = total_units_per_day * len(GROUPS)  # just for scale

        hybrid_total = sum(
            sum(groups.values()) for groups in allocations.values()
        )
        savings = (baseline_total * len(allocations) - hybrid_total) / (baseline_total * len(allocations))
        return (f"Baseline total units (per day × days): {baseline_total * len(allocations):.2f} | "
                f"Hybrid total units: {hybrid_total:.2f} | "
                f"Savings: {_pct(savings * 100)} %")

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic dataset
    start = date.today()
    end = start + timedelta(days=6)  # one week

    docs = [f"doc_{i}" for i in range(3)]
    texts = {
        "doc_0": "The quick brown fox jumps over the lazy dog.",
        "doc_1": "Artificial intelligence and machine learning are evolving rapidly.",
        "doc_2": "Quantum computing promises exponential speed‑up for certain problems."
    }
    labels = {
        "doc_0": "animal",
        "doc_1": "technology",
        "doc_2": "physics"
    }

    hybrid = HybridLTC(base_tau=1.2, weight=7.0, bias=-3.0, deterministic_pct=0.55)
    alloc = hybrid.allocate_by_dates(start, end, total_units_per_day=1000,
                                     texts=texts, labels=labels, doc_ids=docs)

    # Print a concise table
    print("Date".ljust(12), *[g.ljust(12) for g in GROUPS])
    for d in sorted(alloc):
        row = [d.isoformat()]
        row.extend(str(alloc[d][g]).ljust(12) for g in GROUPS)
        print(" ".join(row))

    print("\n", hybrid.summarize(alloc, total_units_per_day=1000))
    sys.exit(0)