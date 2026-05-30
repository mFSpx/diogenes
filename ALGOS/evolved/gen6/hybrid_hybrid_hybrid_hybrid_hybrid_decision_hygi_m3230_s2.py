# DARWIN HAMMER — match 3230, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2621_s0.py (gen5)
# parent_b: hybrid_decision_hygiene_shannon_entropy_m12_s0.py (gen1)
# born: 2026-05-29T23:48:36Z

"""Hybrid algorithm merging:
- PARENT ALGORITHM A: MinHash‑based morphological similarity with diffusion‑rate modulation.
- PARENT ALGORITHM B: Decision‑hygiene token counting plus Shannon entropy analysis.

Mathematical bridge:
1. The Shannon entropy of the decision‑hygiene token distribution of a document is used as a
   *temperature* that scales the diffusion rate produced by the noise schedule of Algorithm A.
2. The MinHash Jaccard‑like similarity between two documents is multiplied by this
   temperature‑scaled diffusion rate to obtain a unified “hybrid similarity” that drives
   recovery‑priority and pheromone‑update calculations.

The result is a single system where textual similarity and information‑theoretic uncertainty
co‑determine the dynamics of a diffusion‑based decision‑support process.
"""

import hashlib
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Set

import numpy as np

MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of a token with a seed."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def jaccard_like_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def shingles(text: str, width: int = 5) -> Set[str]:
    """Extract width‑wise word shingles from a string."""
    words = text.split()
    if width <= 0:
        raise ValueError("width must be positive")
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + width]) for i in range(len(words) - width + 1)}


def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """
    Temperature‑aware diffusion schedule.
    For the cosine schedule we use a half‑cosine that starts at 0 and ends at 1.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    t = np.arange(T)
    if schedule == "cosine":
        return (1 - np.cos(np.pi * t / T)) / 2
    # fallback to linear
    return t / T


# ----------------------------------------------------------------------
# Decision‑hygiene utilities (from Algorithm B)
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
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|first|after|review)\b", re.I
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
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
    re.I,
)


def hygiene_counts(text: str) -> Counter:
    """Count occurrences of each decision‑hygiene category."""
    return Counter(
        {
            "evidence": len(EVIDENCE_RE.findall(text or "")),
            "planning": len(PLANNING_RE.findall(text or "")),
            "delay": len(DELAY_RE.findall(text or "")),
            "support": len(SUPPORT_RE.findall(text or "")),
            "boundary": len(BOUNDARY_RE.findall(text or "")),
            "outcome": len(OUTCOME_RE.findall(text or "")),
            "impulsive": len(IMPULSIVE_RE.findall(text or "")),
            "scarcity": len(SCARCITY_RE.findall(text or "")),
            "risk": len(RISK_RE.findall(text or "")),
        }
    )


def shannon_entropy(counts: Counter) -> float:
    """Shannon entropy of the hygiene count distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values() if c > 0])
    return -np.sum(probs * np.log2(probs))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_diffusion_rate(text: str, T: int = 100) -> np.ndarray:
    """
    Produce a diffusion rate schedule whose amplitude is scaled by the
    Shannon entropy of the decision‑hygiene profile of *text*.
    """
    base = noise_schedule(T, schedule="cosine")
    entropy = shannon_entropy(hygiene_counts(text))
    # Scale factor: map entropy in [0, log2(N_categories)] -> [0.5, 1.5]
    max_entropy = math.log2(9)  # nine categories
    scale = 0.5 + (entropy / max_entropy)
    return base * scale


def hybrid_similarity(text_a: str, text_b: str, k: int = 128) -> float:
    """
    Compute the hybrid similarity:
        Jaccard‑like MinHash similarity  ×  mean diffusion temperature
    """
    sig_a = minhash_signature(list(shingles(text_a)), k=k)
    sig_b = minhash_signature(list(shingles(text_b)), k=k)
    base_sim = jaccard_like_similarity(sig_a, sig_b)

    # average diffusion temperature from both documents
    temp_a = hybrid_diffusion_rate(text_a, T=50).mean()
    temp_b = hybrid_diffusion_rate(text_b, T=50).mean()
    temperature = (temp_a + temp_b) / 2.0

    return base_sim * temperature


def recovery_priority_matrix(current: str, goal: str, categories: List[str] = None) -> np.ndarray:
    """
    Build a priority matrix whose rows correspond to decision‑hygiene categories
    and whose columns correspond to similarity‑weighted recovery scores.
    The matrix entry (i, j) = similarity * (1 + normalized category count).
    """
    if categories is None:
        categories = [
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

    sim = hybrid_similarity(current, goal)
    cur_counts = hygiene_counts(current)
    max_count = max(cur_counts.values()) if cur_counts else 1

    # Normalize counts to [0,1]
    norm = np.array([cur_counts.get(cat, 0) / max_count for cat in categories])
    # Priority = similarity * (1 + normalized count)
    priority = sim * (1.0 + norm)
    # Return as a column vector (categories × 1)
    return priority.reshape(-1, 1)


def pheromone_update(
    current_pheromone: np.ndarray,
    text: str,
    evaporation_rate: float = 0.2,
    deposit_factor: float = 1.0,
) -> np.ndarray:
    """
    Update pheromone levels using hybrid similarity as deposit strength.
    The update follows:
        τ' = (1 - ρ) τ + Δτ
    where Δτ = deposit_factor × hybrid_similarity(text, reference) × similarity_matrix.
    For simplicity we treat the reference as the same *text* (self‑reinforcement).
    """
    if not 0 <= evaporation_rate <= 1:
        raise ValueError("evaporation_rate must be in [0,1]")
    # Self‑similarity is 1, but we still apply diffusion scaling.
    temperature = hybrid_diffusion_rate(text, T=30).mean()
    deposit = deposit_factor * temperature
    return (1 - evaporation_rate) * current_pheromone + deposit


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_a = (
        "We have gathered evidence and verified the source. The plan includes a timeline and "
        "budget. Please review the checklist before proceeding."
    )
    sample_b = (
        "The outcome was successful, the shipment was shipped and verified. No further risk "
        "detected, but we should plan the next phase."
    )

    print("Hybrid similarity:", hybrid_similarity(sample_a, sample_b))
    print("Recovery priority matrix (first 3 categories):")
    print(recovery_priority_matrix(sample_a, sample_b)[:3])
    pheromone = np.full((1,), 0.5)
    pheromone = pheromone_update(pheromone, sample_a)
    print("Updated pheromone level:", pheromone.item())