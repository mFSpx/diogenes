# DARWIN HAMMER — match 1811, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hdc_hybrid_hy_m996_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_rbf_su_m971_s2.py (gen4)
# born: 2026-05-29T23:38:57Z

"""Hybrid Decision‑Bandit Hyperdimensional RBF Model
=================================================

This module fuses the core ideas of the two parent algorithms:

* **Parent A** – *Hybrid Decision Hygiene + Hyperdimensional Computing*  
  – extracts a set of textual hygiene features with regular expressions, maps them
    to a high‑dimensional bipolar vector (dimension = 10 000) and applies positive /
    negative feature weights.

* **Parent B** – *Bandit + RBF Surrogate Learning Vector*  
  – defines a temperature‑dependent developmental rate using the Schoolfield model
    and uses it as a modulation factor for RBF‑kernel similarity scores that drive a
    contextual bandit.

**Mathematical bridge**  
The bridge is the *hyperdimensional feature vector* produced by Parent A.  
It is fed to an RBF kernel (Gaussian similarity) that measures the distance between the
current context and a prototype vector stored for each bandit action (Parent B).  
The raw similarity (the “signal”) is multiplied by the *Schoolfield developmental
rate* evaluated at the current temperature, thereby modulating the bandit’s propensity
scores with a biologically‑inspired temperature factor.

The result is a unified system that (i) captures decision‑hygiene cues in a
high‑dimensional space, (ii) evaluates their similarity to learned action prototypes,
and (iii) selects actions with temperature‑aware confidence bounds.
"""

import sys
import math
import random
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Dict

import numpy as np
import re

# ----------------------------------------------------------------------
# Constants from Parent A
# ----------------------------------------------------------------------
DIM = 10_000  # hyperdimensional vector dimension

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

# Regex patterns for feature extraction (truncated list for brevity)
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(r"\b(?:support|help|assist|aid|back|backing|sponsor)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|limit|threshold|cap|wall|edge)\b", re.I)
OUTCOME_RE = re.compile(r"\b(?:outcome|result|conclusion|effect|impact)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:impulsive|rash|hasty|quick|snap)\b", re.I)
SCARCITY_RE = re.compile(r"\b(?:scarcity|rare|limited|shortage|few)\b", re.I)
RISK_RE = re.compile(r"\b(?:risk|danger|hazard|threat|exposure)\b", re.I)

_REGEX_MAP = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
    "risk": RISK_RE,
}

# ----------------------------------------------------------------------
# Data structures from Parent B
# ----------------------------------------------------------------------
Vector = Sequence[float]

@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a bandit arm."""
    action_id: str
    propensity: float = 0.0          # will be filled by the hybrid model
    expected_reward: float = 0.0     # placeholder for downstream learning
    confidence_bound: float = 0.0   # placeholder for UCB‑style exploration
    algorithm: str = "HybridHDCBandit"

@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the Schoolfield temperature‑dependence model."""
    rho_25: float = 1.0                # rate at 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0   # activation enthalpy (J mol⁻¹)
    t_low: float = 283.15              # low temperature bound (K)
    t_high: float = 307.15             # high temperature bound (K)
    delta_h_low: float = -45_000.0     # low‑temp deactivation enthalpy
    delta_h_high: float = 65_000.0     # high‑temp deactivation enthalpy
    r_cal: float = 1.987               # gas constant (cal mol⁻¹ K⁻¹)

# ----------------------------------------------------------------------
# Core mathematical building blocks
# ----------------------------------------------------------------------
def _bipolar_vector(seed: int = None) -> np.ndarray:
    """Generate a random bipolar (+1 / –1) vector of length DIM."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=DIM, replace=True).astype(np.int8)


def compute_hdc_feature_vector(text: str) -> np.ndarray:
    """
    Extract decision‑hygiene features from *text* and embed them into a
    hyperdimensional bipolar vector.

    Steps
    -----
    1. Initialise a zero vector of length DIM.
    2. For each feature in ``_FEATURE_ORDER`` test the associated regex.
    3. If the feature is present, add the corresponding positive weight
       (or subtract the negative weight) to a *slice* of the bipolar base
       vector.  The slice length for feature *i* is ``abs(weight_i) // 10``.
    4. Finally, the vector is binarised to ``+1`` / ``‑1`` (sign function).

    The function returns a ``np.ndarray`` of dtype ``int8`` containing only
    ``+1`` and ``‑1`` values.
    """
    # Start from a random bipolar seed – this gives each run a reproducible
    # high‑dimensional basis while still being deterministic for the same seed.
    base_vec = _bipolar_vector(seed=42)

    # Accumulator for weighted contributions (float to allow addition/subtraction)
    accumulator = np.zeros(DIM, dtype=np.float32)

    for idx, feature in enumerate(_FEATURE_ORDER):
        pattern = _REGEX_MAP[feature]
        match = bool(pattern.search(text))
        pos_w = _POSITIVE_WEIGHTS[idx]
        neg_w = _NEGATIVE_WEIGHTS[idx]

        # Determine slice length; ensure at least 1 element.
        slice_len = max(1, (pos_w + neg_w) // 10)

        # Choose a deterministic slice start based on feature index
        start = (idx * 1234) % (DIM - slice_len + 1)
        end = start + slice_len

        if match:
            # Positive contribution
            accumulator[start:end] += pos_w
        else:
            # Negative contribution (penalise missing evidence)
            accumulator[start:end] -= neg_w

    # Combine base bipolar signs with accumulated weighted bias
    combined = base_vec.astype(np.float32) + accumulator
    # Binarise to +1 / -1
    hdc_vec = np.where(combined >= 0, 1, -1).astype(np.int8)
    return hdc_vec


def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rbf_similarity(a: Vector, b: Vector, sigma: float = 1.0) -> float:
    """
    Radial‑basis‑function (Gaussian) similarity between two vectors.

    ``sigma`` is the bandwidth; larger values produce smoother similarity.
    """
    dist = euclidean(a, b)
    return math.exp(- (dist ** 2) / (2 * sigma ** 2))


def schoolfield_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Compute the temperature‑dependent developmental rate using the
    Schoolfield model.

    The formula (simplified) is::

        rate = rho_25 *
               exp(-ΔH_a / (R * (1/T - 1/298.15))) /
               (1 + exp(ΔH_l / R * (1/T_l - 1/T)) +
                exp(ΔH_h / R * (1/T_h - 1/T)))

    where:
        ΔH_a  – activation enthalpy
        ΔH_l  – low‑temperature deactivation enthalpy
        ΔH_h  – high‑temperature deactivation enthalpy
        R     – gas constant (converted to J mol⁻¹ K⁻¹)
    """
    R = params.r_cal * 4.184  # convert cal·mol⁻¹·K⁻¹ → J·mol⁻¹·K⁻¹
    T = temp_k
    T_ref = 298.15  # 25 °C in Kelvin

    # Activation term
    act = math.exp(-params.delta_h_activation / R * (1 / T - 1 / T_ref))

    # Low‑temperature deactivation term
    low = math.exp(params.delta_h_low / R * (1 / params.t_low - 1 / T))

    # High‑temperature deactivation term
    high = math.exp(params.delta_h_high / R * (1 / params.t_high - 1 / T))

    denominator = 1 + low + high
    rate = params.rho_25 * act / denominator
    return max(rate, 0.0)  # rates cannot be negative


# ----------------------------------------------------------------------
# Hybrid model that merges both parent behaviours
# ----------------------------------------------------------------------
class HybridHDCBandit:
    """
    A hybrid model that:

    * Represents the current context as a hyperdimensional decision‑hygiene vector.
    * Holds a prototype hyperdimensional vector for each bandit action.
    * Computes RBF similarity between context and prototypes.
    * Modulates the similarity with a temperature‑dependent Schoolfield rate.
    * Emits propensity scores that can be used by any downstream bandit policy.
    """

    def __init__(self, actions: List[str], sigma: float = 5.0):
        """
        Initialise the model.

        Parameters
        ----------
        actions: list of action identifiers (strings).
        sigma: bandwidth for the RBF kernel.
        """
        self.sigma = sigma
        # Create a deterministic prototype vector per action
        self.prototypes: Dict[str, np.ndarray] = {
            act: _bipolar_vector(seed=hash(act) & 0xFFFFFFFF) for act in actions
        }

    def score_actions(
        self,
        context_text: str,
        temperature_k: float,
        params: SchoolfieldParams = SchoolfieldParams(),
    ) -> List[BanditAction]:
        """
        Compute temperature‑aware propensity scores for all actions.

        Returns a list of ``BanditAction`` objects with the ``propensity``
        field populated.
        """
        # 1️⃣ Hyperdimensional representation of the textual context
        ctx_vec = compute_hdc_feature_vector(context_text)

        # 2️⃣ Temperature modulation factor
        temp_factor = schoolfield_rate(temperature_k, params)

        # 3️⃣ Compute RBF similarity and apply modulation
        actions_scored: List[BanditAction] = []
        for aid, proto in self.prototypes.items():
            sim = rbf_similarity(ctx_vec, proto, sigma=self.sigma)
            propensity = sim * temp_factor
            actions_scored.append(
                BanditAction(
                    action_id=aid,
                    propensity=propensity,
                    algorithm="HybridHDCBandit",
                )
            )
        # Sort descending by propensity for convenience
        actions_scored.sort(key=lambda a: a.propensity, reverse=True)
        return actions_scored

    def select_best(self, scored_actions: List[BanditAction]) -> BanditAction:
        """Return the action with the highest propensity."""
        if not scored_actions:
            raise ValueError("no actions to select from")
        return scored_actions[0]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example textual context
    example_text = (
        "The audit confirmed the source and provided a detailed plan. "
        "However, we should pause before executing due to risk and scarcity of resources."
    )

    # Define a small action space
    action_ids = ["deploy", "review", "abort"]

    # Instantiate the hybrid model
    model = HybridHDCBandit(actions=action_ids, sigma=10.0)

    # Choose an ambient temperature (e.g., 295 K ≈ 22 °C)
    temperature = 295.0

    # Compute scores
    scored = model.score_actions(example_text, temperature)

    # Display results
    print("Propensity scores (temperature = {:.1f} K):".format(temperature))
    for a in scored:
        print(f"  Action '{a.action_id}': propensity = {a.propensity:.6f}")

    # Pick the best action
    best = model.select_best(scored)
    print(f"\nSelected best action: {best.action_id} (propensity = {best.propensity:.6f})")