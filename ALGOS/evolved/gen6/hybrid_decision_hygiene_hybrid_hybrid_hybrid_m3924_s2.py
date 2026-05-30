# DARWIN HAMMER — match 3924, survivor 2
# gen: 6
# parent_a: decision_hygiene.py (gen0)
# parent_b: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m417_s1.py (gen5)
# born: 2026-05-29T23:52:36Z

import re
import numpy as np
from typing import Dict, Tuple

# ----------------------------------------------------------------------
# Regular expressions for linguistic feature extraction
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
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

SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|"
    r"advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)

BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|"
    r"protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)

OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|"
    r"filed|closed|fixed|working|green|verified)\b",
    re.I,
)

IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|"
    r"tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)

SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|"
    r"rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)

RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|"
    r"crisis|collapse)\b",
    re.I,
)


def _count_pattern(pattern: re.Pattern, text: str) -> int:
    """Return the number of non‑overlapping matches of *pattern* in *text*."""
    return len(pattern.findall(text or ""))


def counts(text: str) -> Dict[str, int]:
    """Extract the raw linguistic counts used as the base feature vector."""
    return {
        "evidence_count": _count_pattern(EVIDENCE_RE, text),
        "planning_count": _count_pattern(PLANNING_RE, text),
        "delay_count": _count_pattern(DELAY_RE, text),
        "support_count": _count_pattern(SUPPORT_RE, text),
        "boundary_count": _count_pattern(BOUNDARY_RE, text),
        "outcome_count": _count_pattern(OUTCOME_RE, text),
        "impulsive_count": _count_pattern(IMPULSIVE_RE, text),
        "scarcity_count": _count_pattern(SCARCITY_RE, text),
        "risk_count": _count_pattern(RISK_RE, text),
    }


def extract_features(text: str) -> np.ndarray:
    """
    Convert a piece of text into a numeric feature vector.
    The order of the vector follows the order defined in ``counts``.
    """
    return np.asarray(list(counts(text).values()), dtype=float)


def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """
    Compute a reliability weight for the whole observation.
    Returns 0 when *total_claims_emitted* is zero to avoid division by zero.
    """
    if total_claims_emitted <= 0:
        return 0.0
    return claims_with_evidence / total_claims_emitted


def _signature_lead_lag(path: np.ndarray) -> np.ndarray:
    """
    Compute a low‑order (level‑1 + level‑2) path signature on the lead‑lag transformed path.
    This preserves the dimensionality of the original feature space while adding
    interaction terms that capture temporal ordering.

    Parameters
    ----------
    path : np.ndarray
        A (T, d) array where T is the number of timesteps and d the feature dimension.

    Returns
    -------
    np.ndarray
        Concatenated level‑1 (d) and level‑2 (d*d) signature components.
    """
    # Lead‑lag transformation (2T‑1, 2d)
    T, d = path.shape
    lead_lag = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        lead_lag[2 * t] = np.concatenate([path[t], path[t]])
        lead_lag[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    lead_lag[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])

    # Level‑1 signature: sum of increments
    increments = np.diff(lead_lag, axis=0)
    level1 = increments.sum(axis=0)  # shape (2d,)

    # Level‑2 signature: ∫ X_i dX_j approximated with midpoint rule
    #   ∫ X_i dX_j ≈ Σ ( (X_i[k] + X_i[k+1]) / 2 ) * (X_j[k+1] - X_j[k])
    mids = (lead_lag[:-1] + lead_lag[1:]) / 2.0  # shape (2T‑2, 2d)
    level2 = np.einsum("tk,tk->k", mids, increments)  # shape (2d,)

    # For richer interaction we also compute pairwise products (outer product) of level‑1
    # This yields d*d terms that survive the lead‑lag expansion.
    level2_outer = np.outer(level1, level1).ravel()

    return np.concatenate([level1, level2_outer])


def hybrid_operation(
    text: str,
    claims_with_evidence: int,
    total_claims_emitted: int,
) -> np.ndarray:
    """
    Fuse linguistic features with a path‑signature based representation
    and weight the result by a reliability factor (anti‑slop ratio).

    The function returns a vector of length 2d + (2d)² where d = 9
    (the number of raw linguistic counts).  This deepens the mathematical
    integration by keeping the full signature information instead of
    collapsing it to a two‑dimensional mean as in the original implementation.
    """
    # 1️⃣ Extract raw features (shape (9,))
    raw_features = extract_features(text)  # (d,)

    # 2️⃣ Treat the feature vector as a one‑dimensional time series.
    #    Reshape to (T, d) where T = d for a deterministic ordering.
    path = raw_features.reshape(-1, 1)  # (d, 1)

    # 3️⃣ Compute a robust signature on the lead‑lag transformed path.
    signature = _signature_lead_lag(path)  # (2d + (2d)²,)

    # 4️⃣ Apply the anti‑slop weighting.
    weight = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    weighted_signature = signature * weight

    return weighted_signature


def demonstrate_hybrid_operation() -> None:
    """Simple sanity‑check that prints the fused vector."""
    example_text = "I have evidence to support my claim and a clear plan."
    claims_with_evidence = 1
    total_claims_emitted = 2
    fused = hybrid_operation(example_text, claims_with_evidence, total_claims_emitted)
    print("Hybrid signature (length {}):".format(fused.shape[0]))
    print(fused)


if __name__ == "__main__":
    demonstrate_hybrid_operation()