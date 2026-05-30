# DARWIN HAMMER — match 373, survivor 1
# gen: 4
# parent_a: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_decisi_fisher_localization_m153_s1.py (gen3)
# born: 2026-05-29T23:28:27Z

"""
Hybrid Algorithm: physarum_bandit_fisher_integration
Parents:
- hybrid_physarum_network_hybrid_hybrid_bandit_m11_s4.py (physarum flux + bandit routing)
- hybrid_hybrid_hybrid_decisi_fisher_localization_m153_s1.py (regex feature extraction + Fisher information scoring)

Mathematical Bridge:
The bandit action provides an inflow/outflow quantity q = propensity - confidence_bound.
The Fisher information derived from textual features supplies a data‑driven gain factor G ≥ 1.
Conductance is updated by scaling the bandit‑driven term with G, i.e.

    Δσ = dt * ( G * |q| - decay * σ )

where σ is the current conductance. The flux term from the physarum model
(σ / L * (p_a - p_b)) is then added, yielding a unified update that blends
exploration‑exploitation dynamics with information‑theoretic weighting of
contextual textual evidence.
"""

import math
import random
import sys
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (physarum + bandit)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""
    action_id: str
    propensity: float            # interpreted as inflow rate
    expected_reward: float
    confidence_bound: float      # interpreted as outflow rate
    algorithm: str


def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on an edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 1.0, gain: float = 1.0,
                       decay: float = 0.05) -> float:
    """Standard physarum conductance update."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


# ----------------------------------------------------------------------
# Parent B components (regex feature extraction + Fisher information)
# ----------------------------------------------------------------------


# Regex feature set – identical to parent B
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
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|anger|hasty|rash|reckless|spontaneou?s?)\b",
    re.I,
)

REGEX_FEATURES: List[Tuple[str, re.Pattern]] = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("support", SUPPORT_RE),
    ("boundary", BOUNDARY_RE),
    ("outcome", OUTCOME_RE),
    ("impulsive", IMPULSIVE_RE),
]


def extract_feature_counts(text: str) -> Dict[str, int]:
    """Count occurrences of each regex feature in the supplied text."""
    counts: Dict[str, int] = {}
    for name, pattern in REGEX_FEATURES:
        matches = pattern.findall(text)
        counts[name] = len(matches)
    return counts


def fisher_information_score(counts: Dict[str, int]) -> float:
    """
    Compute a scalar Fisher‑information‑like score from feature counts.

    For a multinomial model with probabilities p_i = n_i / N,
    the Fisher information for the parameter vector θ (log‑probabilities) is
    I = Σ (1 / p_i) (∂p_i/∂θ)^2 = Σ p_i (1 - p_i).

    We approximate this by Σ p_i (1 - p_i) and return the sum,
    which lies in [0, 0.25 * K] (K = number of features). The score is then
    normalized to [0, 1] by dividing by the theoretical maximum (K/4).
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    K = len(counts)
    max_info = K / 4.0  # when each p_i = 0.5 (theoretical max for binary case)
    info = 0.0
    for n in counts.values():
        p = n / total
        info += p * (1.0 - p)
    return min(1.0, info / max_info)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def hybrid_gain_from_text(text: str) -> float:
    """
    Derive a multiplicative gain factor G ≥ 1 from textual evidence.

    G = 1 + fisher_score, ensuring that the physarum update is never
    weakened by the Fisher component.
    """
    counts = extract_feature_counts(text)
    fisher_score = fisher_information_score(counts)
    return 1.0 + fisher_score


def hybrid_conductance_update(
    bandit_action: BanditAction,
    edge_length: float,
    pressure_a: float,
    pressure_b: float,
    conductance: float,
    context_text: str,
    eps: float = 1e-12,
) -> float:
    """
    Unified update of edge conductance.

    1. Compute bandit‑driven quantity q = propensity - confidence_bound.
    2. Obtain data‑driven gain G from the Fisher score of `context_text`.
    3. Update conductance using the physarum rule with gain G.
    4. Add the physarum flux term to the updated conductance.
    """
    q = bandit_action.propensity - bandit_action.confidence_bound
    G = hybrid_gain_from_text(context_text)
    updated_sigma = update_conductance(
        conductance, q, dt=1.0, gain=G, decay=0.05
    )
    flux_val = flux(updated_sigma, edge_length, pressure_a, pressure_b, eps)
    return updated_sigma + flux_val


def simulate_network_step(
    actions: List[BanditAction],
    edge_params: List[Tuple[float, float, float, float]],  # (length, p_a, p_b, sigma)
    texts: List[str],
) -> List[float]:
    """
    Perform a single simulation step over a list of edges.

    Parameters
    ----------
    actions : list of BanditAction
        One action per edge.
    edge_params : list of (edge_length, pressure_a, pressure_b, conductance)
    texts : list of str
        Contextual textual evidence for each edge.

    Returns
    -------
    new_conductances : list of float
        Updated conductance values after the hybrid operation.
    """
    if not (len(actions) == len(edge_params) == len(texts)):
        raise ValueError("All input lists must have the same length.")
    new_sigmas: List[float] = []
    for act, (L, pa, pb, sigma), txt in zip(actions, edge_params, texts):
        new_sigma = hybrid_conductance_update(
            act, L, pa, pb, sigma, txt
        )
        new_sigmas.append(new_sigma)
    return new_sigmas


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a deterministic random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Example bandit actions for three edges
    actions = [
        BanditAction(
            action_id="edge_0",
            propensity=0.8,
            expected_reward=0.5,
            confidence_bound=0.2,
            algorithm="UCB1",
        ),
        BanditAction(
            action_id="edge_1",
            propensity=0.3,
            expected_reward=0.4,
            confidence_bound=0.1,
            algorithm="Thompson",
        ),
        BanditAction(
            action_id="edge_2",
            propensity=0.6,
            expected_reward=0.7,
            confidence_bound=0.5,
            algorithm="EpsilonGreedy",
        ),
    ]

    # Edge parameters: (length, pressure_a, pressure_b, conductance)
    edge_params = [
        (1.5, 1.0, 0.0, 0.2),
        (2.0, 0.8, 0.3, 0.5),
        (1.0, 1.2, 0.4, 0.3),
    ]

    # Contextual texts containing various regex features
    texts = [
        "The evidence was verified and the source was documented. We plan the next steps.",
        "Please wait, we need to delay the release. The team will support the rollout later.",
        "All boundaries were respected, and the outcome is verified as green. No impulsive actions.",
    ]

    new_conductances = simulate_network_step(actions, edge_params, texts)

    for i, sigma in enumerate(new_conductances):
        print(f"Edge {i} new conductance: {sigma:.6f}")