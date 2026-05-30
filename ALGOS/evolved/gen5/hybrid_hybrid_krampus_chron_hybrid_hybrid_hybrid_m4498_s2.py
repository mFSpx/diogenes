# DARWIN HAMMER — match 4498, survivor 2
# gen: 5
# parent_a: hybrid_krampus_chrono_infotaxis_m67_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_semant_m997_s1.py (gen4)
# born: 2026-05-29T23:56:16Z

"""Hybrid Date-Decision Entropy Module

Parents:
- hybrid_krampus_chrono_infotaxis_m67_s0.py (date parsing with entropy)
- hybrid_hybrid_hybrid_pherom_hybrid_hybrid_semant_m997_s1.py (decision hygiene, semantic neighbors, pheromone entropy)

Mathematical Bridge:
Both parents compute Shannon entropy on a probability distribution.  
Parent A quantifies uncertainty of ambiguous date parses; Parent B quantifies uncertainty of pheromone signals derived from decision‑hygiene scores and semantic neighbor weights.  
The hybrid algorithm treats the date‑parse entropy as a prior that weights the decision‑hygiene‑based pheromone distribution, then evaluates a joint entropy that reflects combined temporal‑semantic uncertainty.
"""

import re
import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Date parsing and entropy utilities
# ----------------------------------------------------------------------
CONTENT_DATE_PATTERNS = [
    re.compile(
        r"(?im)^\s*(?:date|created|created_at|created at|generated|timestamp|time|filed|updated|modified)\s*[:=]\s*[\"']?"
        r"((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)"
    ),
    re.compile(
        r"\b((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)\b"
    ),
    re.compile(
        r"\b((?:20|19)\d{2})(\d{2})(\d{2})(?:[T_-]?(\d{2})(\d{2})(\d{2})?)?Z?\b"
    ),
]

MONTH_NAME_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+([0-3]?\d)(?:st|nd|rd|th)?,?\s+((?:20|19)\d{2})\b",
    re.I,
)

def _shannon_entropy(probs: List[float], eps: float = 1e-12) -> float:
    """Return Shannon entropy (base‑2) of a probability list."""
    total = sum(probs)
    if total <= eps:
        raise ValueError("Probability mass must be positive")
    probs = np.array(probs) / total
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log2(probs)))

def _extract_date_candidates(text: str) -> List[str]:
    """Return all raw date strings matched by the defined patterns."""
    candidates = []
    for pat in CONTENT_DATE_PATTERNS:
        for m in pat.finditer(text):
            # Join groups if the pattern has multiple capture groups
            if m.lastindex and m.lastindex > 1:
                cand = "".join(m.groups())
            else:
                cand = m.group(1)
            candidates.append(cand)
    # Also try month‑name style
    for m in MONTH_NAME_RE.finditer(text):
        month, day, year = m.groups()
        month_num = datetime.strptime(month[:3], "%b").month
        cand = f"{int(year):04d}-{month_num:02d}-{int(day):02d}"
        candidates.append(cand)
    return candidates

def parse_date_with_entropy(text: str) -> Tuple[datetime | None, float]:
    """
    Parse the most plausible date from *text* and return a tuple:
        (datetime object or None, entropy of the date‑candidate distribution)

    Entropy is computed assuming a uniform prior over all extracted candidates.
    """
    candidates = _extract_date_candidates(text)
    if not candidates:
        return None, 0.0
    # Uniform probability over candidates
    entropy_val = _shannon_entropy([1.0] * len(candidates))
    # Choose the first candidate as the "parsed" date (could be refined)
    raw = candidates[0]
    # Attempt ISO‑like parsing
    try:
        dt_obj = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        # Fallback: try common delimiters
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y_%m_%d", "%Y%m%d"):
            try:
                dt_obj = datetime.strptime(raw[:10], fmt)
                break
            except Exception:
                dt_obj = None
    return dt_obj, entropy_val

# ----------------------------------------------------------------------
# Parent B – Decision hygiene, semantic neighbors, pheromone entropy
# ----------------------------------------------------------------------
def decision_hygiene_score(vector: List[float]) -> float:
    """Simplified decision hygiene score: sum of squares of the feature vector."""
    return float(sum(x ** 2 for x in vector))

def semantic_neighbors(doc_id: str, vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
    """
    Generate *k* synthetic semantic neighbors for *doc_id*.
    Each neighbor is a tuple (neighbor_id, similarity_score) where the score is a random float.
    """
    return [(f"{doc_id}_nbr_{i}", random.random()) for i in range(k)]

def pheromone_signal(neighbor: Tuple[str, float], hygiene_score: float) -> float:
    """
    Compute a pheromone signal for a single neighbor.
    The signal is the product of the hygiene score and the neighbor similarity.
    """
    _, sim = neighbor
    return hygiene_score * sim

def pheromone_entropy(signals: List[float]) -> float:
    """Shannon entropy of the pheromone signal distribution (base‑2)."""
    if not signals:
        return 0.0
    return _shannon_entropy(signals)

# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def hybrid_date_decision_entropy(
    doc_id: str,
    text: str,
    vector: List[float],
    k: int = 5,
) -> Dict[str, Any]:
    """
    Combine date‑parsing entropy with decision‑hygiene pheromone entropy.

    Steps
    -----
    1. Parse dates from *text* → (date_obj, date_entropy)
    2. Compute hygiene score from *vector*
    3. Retrieve semantic neighbors and turn each into a pheromone signal
    4. Compute pheromone entropy
    5. Fuse the two entropies into a joint metric:
           joint_entropy = date_entropy + pheromone_entropy
       (simple additive bridge; other weightings are possible)

    Returns a dictionary with all intermediate values for inspection.
    """
    date_obj, date_entropy = parse_date_with_entropy(text)
    hygiene = decision_hygiene_score(vector)
    neighbors = semantic_neighbors(doc_id, vector, k)
    signals = [pheromone_signal(nbr, hygiene) for nbr in neighbors]
    pheromone_ent = pheromone_entropy(signals)

    joint_entropy = date_entropy + pheromone_ent

    return {
        "doc_id": doc_id,
        "parsed_date": date_obj,
        "date_entropy": date_entropy,
        "hygiene_score": hygiene,
        "neighbors": neighbors,
        "pheromone_signals": signals,
        "pheromone_entropy": pheromone_ent,
        "joint_entropy": joint_entropy,
    }

def best_action_based_on_entropy(
    joint_entropy: float,
    low_threshold: float = 1.0,
    high_threshold: float = 3.0,
) -> str:
    """
    Decide an action from the joint entropy:
        - low entropy   → "accept"
        - moderate      → "review"
        - high entropy  → "reject"
    Thresholds are tunable.
    """
    if joint_entropy < low_threshold:
        return "accept"
    elif joint_entropy < high_threshold:
        return "review"
    else:
        return "reject"

def hybrid_process_document(
    doc_id: str,
    text: str,
    vector: List[float],
    k: int = 5,
) -> Tuple[Dict[str, Any], str]:
    """
    Full pipeline for a single document:
        1. Compute the hybrid entropy dictionary.
        2. Derive the best action.
    Returns (result_dict, action).
    """
    result = hybrid_date_decision_entropy(doc_id, text, vector, k)
    action = best_action_based_on_entropy(result["joint_entropy"])
    return result, action

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = """
    Created: 2023-07-15T14:23:00Z
    The experiment was conducted on July 20, 2023 and recorded.
    """
    sample_vector = [0.2, 0.5, -0.1, 0.3]
    doc_identifier = "doc_42"

    result, decision = hybrid_process_document(doc_identifier, sample_text, sample_vector)
    print("Hybrid Result Summary:")
    for key, val in result.items():
        if key in ("neighbors", "pheromone_signals"):
            continue  # keep output concise
        print(f"  {key}: {val}")
    print(f"Derived action: {decision}")