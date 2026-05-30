# DARWIN HAMMER — match 4498, survivor 3
# gen: 5
# parent_a: hybrid_krampus_chrono_infotaxis_m67_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_semant_m997_s1.py (gen4)
# born: 2026-05-29T23:56:16Z

import re
import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any
import numpy as np

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
    total = sum(probs)
    if total <= eps:
        raise ValueError("Probability mass must be positive")
    probs = np.array(probs) / total
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log2(probs)))

def _extract_date_candidates(text: str) -> List[str]:
    candidates = []
    for pat in CONTENT_DATE_PATTERNS:
        for m in pat.finditer(text):
            if m.lastindex and m.lastindex > 1:
                cand = "".join(m.groups())
            else:
                cand = m.group(1)
            candidates.append(cand)
    for m in MONTH_NAME_RE.finditer(text):
        month, day, year = m.groups()
        month_num = datetime.strptime(month[:3], "%b").month
        cand = f"{int(year):04d}-{month_num:02d}-{int(day):02d}"
        candidates.append(cand)
    return candidates

def parse_date_with_entropy(text: str) -> Tuple[datetime | None, float]:
    candidates = _extract_date_candidates(text)
    if not candidates:
        return None, 0.0
    entropy_val = _shannon_entropy([1.0] * len(candidates))
    raw = candidates[0]
    try:
        dt_obj = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y_%m_%d", "%Y%m%d"):
            try:
                dt_obj = datetime.strptime(raw[:10], fmt)
                break
            except Exception:
                dt_obj = None
    return dt_obj, entropy_val

def decision_hygiene_score(vector: List[float]) -> float:
    return float(sum(x ** 2 for x in vector))

def semantic_neighbors(doc_id: str, vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
    return [(f"{doc_id}_nbr_{i}", random.random()) for i in range(k)]

def pheromone_signal(neighbor: Tuple[str, float], hygiene_score: float) -> float:
    _, sim = neighbor
    return hygiene_score * sim

def pheromone_entropy(signals: List[float]) -> float:
    if not signals:
        return 0.0
    return _shannon_entropy(signals)

def hybrid_date_decision_entropy(
    doc_id: str,
    text: str,
    vector: List[float],
    k: int = 5,
) -> Dict[str, Any]:
    date_obj, date_entropy = parse_date_with_entropy(text)
    hygiene = decision_hygiene_score(vector)
    neighbors = semantic_neighbors(doc_id, vector, k)
    signals = [pheromone_signal(nbr, hygiene) for nbr in neighbors]
    pheromone_ent = pheromone_entropy(signals)
    joint_entropy = (date_entropy * pheromone_ent) / (date_entropy + pheromone_ent + 1e-12)
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
    low_threshold: float = 0.5,
    high_threshold: float = 0.8,
) -> str:
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
    result = hybrid_date_decision_entropy(doc_id, text, vector, k)
    action = best_action_based_on_entropy(result["joint_entropy"])
    return result, action

if __name__ == "__main__":
    sample_text = """
    Created: 2023-07-15T14:23:00Z
    The experiment
    """
    result, action = hybrid_process_document("doc1", sample_text, [1.0, 2.0, 3.0])
    print(result)
    print(action)