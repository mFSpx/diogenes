# DARWIN HAMMER — match 879, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_decisi_m260_s0.py (gen3)
# parent_b: hybrid_hybrid_workshare_all_hybrid_krampus_brain_m23_s2.py (gen2)
# born: 2026-05-29T23:31:29Z

import hashlib
import random
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Configuration & Constants
# ----------------------------------------------------------------------
CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}
LOCAL_PATTERNS = [
    "*bitnet*",
    "*BitNet*",
    "*fairyfuse*",
    "*FairyFuse*",
    "*lora*",
    "*LoRA*",
    "*adapter*",
]

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)

# Feature keys – order is important for reproducible vector generation
_FEATURE_KEYS = [
    "operator_visceral_ratio",
    "operator_tech_ratio",
    "operator_legal_osint_ratio",
    "operator_ledger_density",
    "operator_recursion_score",
    "operator_directive_ratio",
    "operator_target_density",
    "psyche_forensic_shield_ratio",
    "psyche_poetic_entropy",
    "psyche_dissociative_index",
    "psyche_wrath_velocity",
    "resilience_bureaucratic_weaponization_index",
    "resilience_resource_exhaustion_metric",
    "resilience_swarm_orchestration_density",
    "resilience_logic_crucifixion_index",
    "resilience_conspiracy_grounding_ratio",
    "resilience_chaotic_good_tax",
    "rainmaker_corporate_grit_tension",
    "rainmaker_countdown_density",
    "rainmaker_asset_structuring_weight",
    "rainmaker_pitch_formatting_ratio",
]

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def _rng_from_text(text: str) -> random.Random:
    """Create a deterministic RNG seeded from the SHA‑256 hash of *text*."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


def _features_vector(text: str) -> np.ndarray:
    """Generate a deterministic feature vector (0‑1 range) from *text*."""
    rng = _rng_from_text(text)
    return np.array([rng.random() for _ in _FEATURE_KEYS], dtype=np.float64)


def _evidence_tokens(text: str) -> List[str]:
    """Return a list of evidence‑related tokens found in *text*."""
    return EVIDENCE_RE.findall(text)


def _doomsday_factor() -> float:
    """
    Compute a deterministic factor based on today's weekday.
    The factor is normalised to the interval [0, 1].
    """
    weekday = (date.today().weekday() + 1) % 7  # 0 = Sunday … 6 = Saturday
    return weekday / 6.0


# ----------------------------------------------------------------------
# Core Hybrid System
# ----------------------------------------------------------------------
class HybridSystem:
    """
    A mathematically tighter integration of lens‑evaluation,
    evidence extraction and work‑share allocation.
    """

    def __init__(
        self,
        evidence_weight: float = 0.6,
        feature_weight: float = 0.4,
        allocation_scaling: float = 0.2,
    ):
        """
        Parameters
        ----------
        evidence_weight : float
            Relative importance of evidence count in the final score.
        feature_weight : float
            Relative importance of the aggregated feature vector in the final score.
        allocation_scaling : float
            Maximum proportional change applied to each allocation entry
            based on the ``operator_visceral_ratio`` and the day factor.
        """
        if not (0 <= evidence_weight <= 1 and 0 <= feature_weight <= 1):
            raise ValueError("Weights must be within [0, 1].")
        if evidence_weight + feature_weight == 0:
            raise ValueError("At least one of the weights must be non‑zero.")
        self.evidence_weight = evidence_weight
        self.feature_weight = feature_weight
        self.allocation_scaling = allocation_scaling

    # ------------------------------------------------------------------
    # Lens Evaluation
    # ------------------------------------------------------------------
    def evaluate_lens_candidate(self, text: str) -> Dict:
        """
        Produce a robust score that blends evidence occurrence with
        a normalised aggregation of the feature vector.
        """
        evidence = _evidence_tokens(text)
        evidence_score = len(evidence) / max(1, len(evidence) + len(_FEATURE_KEYS))

        features = _features_vector(text)
        # Normalise feature vector to mean 0.5 (since values are uniform [0,1])
        feature_score = np.mean(features)

        # Weighted combination
        combined = (
            self.evidence_weight * evidence_score
            + self.feature_weight * feature_score
        ) / (self.evidence_weight + self.feature_weight)

        return {
            "score": _pct(combined),
            "evidence_count": len(evidence),
            "evidence_tokens": evidence,
            "feature_mean": _pct(feature_score),
        }

    # ------------------------------------------------------------------
    # Workshare Allocation
    # ------------------------------------------------------------------
    def adjust_workshare_allocation(self, text: str, allocation: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust *allocation* using a deterministic factor derived from
        ``operator_visceral_ratio`` and the day‑of‑week factor.
        The result is re‑normalised to sum to 1.0.
        """
        if not allocation:
            raise ValueError("Allocation dictionary must not be empty.")
        if any(v < 0 for v in allocation.values()):
            raise ValueError("Allocation values must be non‑negative.")

        features = dict(zip(_FEATURE_KEYS, _features_vector(text)))
        visceral = features["operator_visceral_ratio"]
        day_factor = _doomsday_factor()

        scaling = 1 + self.allocation_scaling * visceral * day_factor

        adjusted = {k: v * scaling for k, v in allocation.items()}
        total = sum(adjusted.values())
        if total == 0:
            # Fallback to original allocation if scaling collapses everything
            return {k: v / sum(allocation.values()) for k, v in allocation.items()}
        return {k: _pct(v / total) for k, v in adjusted.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def hybrid_operation(self, text: str, allocation: Dict[str, float]) -> Dict:
        """
        Run the full hybrid pipeline: evaluate the lens candidate and
        adjust the work‑share allocation.
        """
        lens = self.evaluate_lens_candidate(text)
        adjusted = self.adjust_workshare_allocation(text, allocation)
        return {"lens_candidate": lens, "adjusted_allocation": adjusted}


# ----------------------------------------------------------------------
# CLI / Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "This is a sample text for evaluation. Evidence: verified source."
    sample_allocation = {"codex": 0.5, "groq": 0.3, "cohere": 0.2}

    system = HybridSystem()
    result = system.hybrid_operation(sample_text, sample_allocation)
    print(result)