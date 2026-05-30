# DARWIN HAMMER — match 5637, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2046_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_vorono_m1379_s1.py (gen5)
# born: 2026-05-30T00:03:41Z

"""Hybrid Decision‑Regret‑Temporal Analyzer (HDR‑T)

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2046_s0.py (HDR‑A)
  Provides a 6‑dimensional feature count vector **f**, a regret‑weighted probability
  vector **p**, and an edge‑expectation matrix **E**. The core metric is the bilinear
  form **C = fᵀ·E·p**.
- hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_vorono_m1379_s1.py (Doomsday‑Voronoi)
  Supplies a vectorised date‑to‑weekday function `weekday_sakamoto` and a Gini
  coefficient calculator.

Mathematical Bridge:
The weekday index of each action’s timestamp is turned into a *temporal weight*
vector **w** ∈ ℝⁿ (e.g. w_i = 1/(weekday_i+1)). By scaling the columns of **E**
with **w**, we obtain a temporally‑aware expectation matrix **Ẽ = E·diag(w)**.
The unified decision metric becomes

    C = fᵀ · Ẽ · p = fᵀ · (E·diag(w)) · p

which simultaneously accounts for stylometric evidence, regret‑weighted action
importance, and temporal relevance. The per‑action contributions are
c_i = fᵀ·Ẽ[:,i]·p_i, whose inequality is quantified by the Gini coefficient.

The module implements this fused pipeline with three public functions and a
smoke‑test.
"""

import re
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex feature extraction (6 features)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don\'t|stop|limit|restriction|rule|policy)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:outcome|result|consequence|effect|impact|resolution|settle|final|end)\b",
    re.I,
)


def extract_features(text: str) -> np.ndarray:
    """Return a 6‑dimensional feature count vector **f** for *text*."""
    counts = np.array(
        [
            len(EVIDENCE_RE.findall(text)),
            len(PLANNING_RE.findall(text)),
            len(DELAY_RE.findall(text)),
            len(SUPPORT_RE.findall(text)),
            len(BOUNDARY_RE.findall(text)),
            len(OUTCOME_RE.findall(text)),
        ],
        dtype=np.float64,
    )
    return counts


# ----------------------------------------------------------------------
# Parent B – vectorised Doomsday (weekday) implementation
# ----------------------------------------------------------------------
def weekday_sakamoto(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """
    Compute weekday indices for vectorised (year, month, day) arrays using
    Tomohiko Sakamoto's algorithm.  Result: 0 = Sunday … 6 = Saturday.
    """
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


def temporal_weights_from_dates(dates: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Convert a list of (year, month, day) tuples into a weight vector **w**.
    Weekday 0 (Sunday) gets the highest weight, weekday 6 (Saturday) the lowest:
        w_i = 1 / (weekday_i + 1)
    """
    if not dates:
        return np.array([], dtype=np.float64)

    years, months, days = zip(*dates)
    wdays = weekday_sakamoto(
        np.array(years, dtype=np.int64),
        np.array(months, dtype=np.int64),
        np.array(days, dtype=np.int64),
    )
    weights = 1.0 / (wdays.astype(np.float64) + 1.0)
    return weights


# ----------------------------------------------------------------------
# Helper utilities (regret probabilities, edge matrix, Gini)
# ----------------------------------------------------------------------
def regret_weighted_probabilities(n_actions: int, seed: int = 42) -> np.ndarray:
    """
    Generate a regret‑weighted probability vector **p** of length *n_actions*.
    Higher index → higher assumed regret, thus larger weight before normalisation.
    """
    rng = random.Random(seed)
    raw = np.array([rng.random() + i * 0.1 for i in range(n_actions)], dtype=np.float64)
    p = raw / raw.sum()
    return p


def edge_expectation_matrix(n_actions: int, seed: int = 123) -> np.ndarray:
    """
    Produce a deterministic 6×n_actions matrix **E**.
    Each column corresponds to an action; rows correspond to the six feature
    categories. Values are drawn from a uniform distribution in [0,1).
    """
    rng = np.random.default_rng(seed)
    return rng.random((6, n_actions), dtype=np.float64)


def gini_coefficient(x: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1‑D array *x*.
    Returns 0 for perfect equality and 1 for maximal inequality.
    """
    if x.ndim != 1:
        raise ValueError("gini_coefficient expects a 1‑D array")
    if x.size == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = x.size
    cumulative = np.cumsum(sorted_x, dtype=np.float64)
    sum_x = cumulative[-1]
    if sum_x == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_x) / n
    return float(gini)


# ----------------------------------------------------------------------
# Core hybrid operation
# ----------------------------------------------------------------------
def hybrid_cost(
    f: np.ndarray,
    E: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute the temporally‑aware hybrid cost.

    Parameters
    ----------
    f : (6,) feature count vector
    E : (6, n) edge‑expectation matrix
    p : (n,) regret‑weighted probability vector
    w : (n,) temporal weight vector (derived from dates)

    Returns
    -------
    total_cost : float
        The scalar C = fᵀ·(E·diag(w))·p
    contributions : (n,) ndarray
        Per‑action contributions c_i = fᵀ·E[:,i]·w_i·p_i
    """
    if f.shape != (6,):
        raise ValueError("Feature vector f must have shape (6,)")
    if E.shape[0] != 6:
        raise ValueError("Edge matrix E must have 6 rows")
    if not (p.shape == w.shape == (E.shape[1],)):
        raise ValueError("Vectors p, w must match the number of actions (columns of E)")

    # Scale each column of E by the corresponding temporal weight
    E_tilde = E * w  # broadcasting over rows

    # Per‑action scalar: s_i = fᵀ·E_tilde[:,i]
    s = f @ E_tilde  # shape (n,)

    contributions = s * p
    total_cost = float(contributions.sum())
    return total_cost, contributions


# ----------------------------------------------------------------------
# Public API exposing the three required functions
# ----------------------------------------------------------------------
def analyze_text_with_actions(
    text: str,
    action_dates: List[Tuple[int, int, int]],
    seed: int = 0,
) -> Tuple[float, float]:
    """
    End‑to‑end hybrid analysis:
    1. Extract feature vector **f** from *text*.
    2. Build **E**, **p**, and temporal weights **w** from *action_dates*.
    3. Compute total hybrid cost and Gini of per‑action contributions.

    Returns
    -------
    total_cost : float
    gini : float
    """
    n_actions = len(action_dates)
    if n_actions == 0:
        raise ValueError("At least one action/date must be provided")

    f = extract_features(text)
    E = edge_expectation_matrix(n_actions, seed=seed + 1)
    p = regret_weighted_probabilities(n_actions, seed=seed + 2)
    w = temporal_weights_from_dates(action_dates)

    total_cost, contributions = hybrid_cost(f, E, p, w)
    gini = gini_coefficient(contributions)
    return total_cost, gini


def batch_hybrid_analysis(
    texts: List[str],
    action_dates_list: List[List[Tuple[int, int, int]]],
) -> List[Tuple[float, float]]:
    """
    Apply `analyze_text_with_actions` to a batch of inputs.
    The i‑th entry of *texts* is paired with the i‑th entry of
    *action_dates_list*.
    """
    if len(texts) != len(action_dates_list):
        raise ValueError("Mismatched batch lengths")
    results = []
    for txt, dates in zip(texts, action_dates_list):
        results.append(analyze_text_with_actions(txt, dates))
    return results


def summarize_results(results: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Aggregate a list of (cost, gini) tuples:
    - average cost
    - average Gini
    """
    if not results:
        return 0.0, 0.0
    costs, ginis = zip(*results)
    avg_cost = float(np.mean(costs))
    avg_gini = float(np.mean(ginis))
    return avg_cost, avg_gini


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The evidence was verified and the plan was set. "
        "We will wait until tomorrow, then delegate to the team. "
        "Boundary conditions are clear, and the outcome will be recorded."
    )
    # Three hypothetical actions with timestamps
    action_dates = [
        (2026, 5, 30),  # Monday
        (2026, 6, 1),   # Wednesday
        (2026, 6, 4),   # Saturday
    ]

    cost, gini = analyze_text_with_actions(sample_text, action_dates)
    print(f"Hybrid cost: {cost:.6f}")
    print(f"Gini of contributions: {gini:.4f}")

    # Batch example
    batch_texts = [sample_text, sample_text.upper()]
    batch_dates = [action_dates, [(2026, 7, 10), (2026, 7, 12), (2026, 7, 14), (2026, 7, 15)]]
    batch_res = batch_hybrid_analysis(batch_texts, batch_dates)
    avg_cost, avg_gini = summarize_results(batch_res)
    print(f"Batch average cost: {avg_cost:.6f}")
    print(f"Batch average Gini: {avg_gini:.4f}")