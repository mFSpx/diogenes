# DARWIN HAMMER — match 1730, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_decision_hygi_rete_bandit_gate_m28_s1.py (gen2)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_hybrid_cockpi_m1134_s1.py (gen4)
# born: 2026-05-29T23:40:01Z

"""
Hybrid Module: fusion of hybrid_decision_hygiene_shannon_entropy (Parent A) and
hybrid_physarum_network_hybrid_hybrid_bandit (Parent B).

Mathematical Bridge
-------------------
* Parent A provides a Shannon‑entropy estimator for a textual context and a set of
  regex‑based evidence counters (evidence, planning, delay, support, boundary,
  outcome).
* Parent B defines a Physarum‑inspired conductance dynamics
  `flux = g / L * (p_a - p_b)` and a bandit‑style update
  `g_{t+1} = max(0, g_t + dt·(gain·|q| - decay·g_t))`.

The fusion treats the normalized entropy `Ĥ ∈ [0,1]` (low entropy ⇒ high confidence)
as a *trust* scalar that multiplicatively modulates both the conductance update
and the bandit gain.  The evidence counters are reduced to a single quality score
`Ê ∈ [0,1]` via `anti_slop_ratio`.  The combined weight

    w = (1 - Ĥ) * Ê

is applied to the gain term of the Physarum update and to the reward scaling of
the bandit step, yielding a unified dynamics that reacts to textual uncertainty
and evidential support.

The module therefore offers three core hybrid functions:
    1. `shannon_entropy(text)`
    2. `extract_evidence_features(text)`
    3. `hybrid_conductance_update(...)` – Physarum update modulated by entropy &
       evidence.
    4. `hybrid_bandit_step(...)` – Bandit reward update modulated by the same
       weight.

All functions are pure NumPy / std‑lib and can be exercised via the smoke test
below.
"""

import math
import re
import sys
import random
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regexes and Shannon entropy
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
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)

def shannon_entropy(text: str) -> float:
    """Return the Shannon entropy of the character distribution in *text*."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    entropy = -np.sum(probs * np.log2(probs + np.finfo(float).eps))
    # Normalise to [0,1] by dividing by log2(|alphabet|) (max possible entropy)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return float(entropy / max_entropy)

def extract_evidence_features(text: str) -> Dict[str, int]:
    """Count matches for each evidence‑related regex in *text*."""
    return {
        "evidence": len(EVIDENCE_RE.findall(text)),
        "planning": len(PLANNING_RE.findall(text)),
        "delay": len(DELAY_RE.findall(text)),
        "support": len(SUPPORT_RE.findall(text)),
        "boundary": len(BOUNDARY_RE.findall(text)),
        "outcome": len(OUTCOME_RE.findall(text)),
    }

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0,1]."""
    if total_claims_emitted <= 0:
        return 1.0
    ratio = claims_with_evidence / total_claims_emitted
    return max(0.0, min(1.0, ratio))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known‑good, clamped to [0,1]."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    ratio = displayed_ok / total
    return max(0.0, min(1.0, ratio))

# ----------------------------------------------------------------------
# Parent B – Physarum conductance and bandit dynamics
# ----------------------------------------------------------------------

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    """Physarum flux on an edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(
    conductance: float,
    q: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
    weight: float = 1.0,
) -> float:
    """
    Conductance update with optional *weight* that scales the gain term.
    The weight stems from the hybrid trust factor.
    """
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    delta = dt * (gain * weight * abs(q) - decay * conductance)
    return max(0.0, conductance + delta)

def hybrid_conductance_update(
    conductance: float,
    edge_length: float,
    pressure_a: float,
    pressure_b: float,
    entropy: float,
    evidence_score: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
) -> float:
    """
    Unified Physarum step where the gain term is modulated by a trust weight:

        weight = (1 - entropy) * evidence_score

    *entropy* is normalised Shannon entropy ∈ [0,1];
    *evidence_score* is anti‑slop ratio ∈ [0,1].
    """
    weight = (1.0 - entropy) * evidence_score
    q = flux(conductance, edge_length, pressure_a, pressure_b)
    return update_conductance(conductance, q, dt=dt, gain=gain, decay=decay, weight=weight)

def hybrid_bandit_step(
    value_estimate: float,
    reward: float,
    honesty: float,
    entropy: float,
    dt: float = 1.0,
    gain: float = 0.5,
    decay: float = 0.1,
) -> float:
    """
    Bandit‑style value update with trust weighting.

        weight = (1 - entropy) * honesty

    The reward is scaled by *weight* before being added to the estimate.
    """
    weight = (1.0 - entropy) * honesty
    delta = dt * (gain * weight * reward - decay * value_estimate)
    return max(0.0, value_estimate + delta)

# ----------------------------------------------------------------------
# Helper to combine evidence counters into a single quality metric
# ----------------------------------------------------------------------

def compute_evidence_score(features: Dict[str, int]) -> float:
    """
    Convert raw feature counts into a normalised evidence quality score.

    For demonstration we treat the sum of all positive evidence categories
    (evidence, planning, support, outcome) as *supported claims* and the total
    number of matches across all categories as *total claims*.
    """
    supported = (
        features.get("evidence", 0)
        + features.get("planning", 0)
        + features.get("support", 0)
        + features.get("outcome", 0)
    )
    total = sum(features.values())
    return anti_slop_ratio(supported, total)

# ----------------------------------------------------------------------
# Example high‑level hybrid operation
# ----------------------------------------------------------------------

def hybrid_simulation_step(
    conductance: float,
    edge_length: float,
    pressure_a: float,
    pressure_b: float,
    bandit_value: float,
    reward: float,
    displayed_ok: int,
    unknown_displayed_as_ok: int,
    text_context: str,
) -> Tuple[float, float]:
    """
    Perform one hybrid iteration:
      1. Compute entropy of *text_context*.
      2. Extract evidence features → evidence_score.
      3. Update conductance (Physarum) using the hybrid weight.
      4. Compute cockpit honesty → weight for bandit update.
      5. Update bandit value.
    Returns (new_conductance, new_bandit_value).
    """
    entropy = shannon_entropy(text_context)
    features = extract_evidence_features(text_context)
    evidence_score = compute_evidence_score(features)
    new_conductance = hybrid_conductance_update(
        conductance,
        edge_length,
        pressure_a,
        pressure_b,
        entropy,
        evidence_score,
    )
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    new_bandit_value = hybrid_bandit_step(
        bandit_value,
        reward,
        honesty,
        entropy,
    )
    return new_conductance, new_bandit_value

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    sample_text = (
        "The evidence was verified and the plan was documented. "
        "We will wait tomorrow before proceeding, but the outcome looks good."
    )
    # Initial physarum state
    g = 0.5          # conductance
    L = 1.2          # edge length
    p_a = 2.0        # pressure node A
    p_b = 1.0        # pressure node B

    # Bandit state
    v = 0.3          # value estimate
    r = 1.0          # observed reward

    # Cockpit metrics
    displayed_ok = 8
    unknown_displayed_as_ok = 2

    new_g, new_v = hybrid_simulation_step(
        conductance=g,
        edge_length=L,
        pressure_a=p_a,
        pressure_b=p_b,
        bandit_value=v,
        reward=r,
        displayed_ok=displayed_ok,
        unknown_displayed_as_ok=unknown_displayed_as_ok,
        text_context=sample_text,
    )
    print(f"Old conductance: {g:.4f} → New conductance: {new_g:.4f}")
    print(f"Old bandit value: {v:.4f} → New bandit value: {new_v:.4f}")
    print(f"Entropy: {shannon_entropy(sample_text):.4f}")
    print(f"Evidence features: {extract_evidence_features(sample_text)}")
    sys.exit(0)