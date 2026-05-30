# DARWIN HAMMER — match 19, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py (gen3)
# parent_b: hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s4.py (gen3)
# born: 2026-05-29T23:26:23Z

"""
This module fuses the stylometry and LSM utilities from hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py
with the deterministic feature extraction and master vector generation from hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s4.py.
The mathematical bridge between the two parents lies in the use of hash-based seeding for pseudo-random number generation.
The hybrid algorithm combines the stylometry features with the master vector generation, using the hash-based seeding to ensure reproducibility.

Parents:
- hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py (gen: 3)
- hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s4.py (gen: 3)
"""

import numpy as np
import random
import math
import hashlib
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path
from __future__ import annotations

FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

@dataclass
class HybridFeatures:
    stylometry: np.ndarray
    master_vector: Dict[str, float]

def _deterministic_hash(text: str) -> int:
    """Return a stable 64-bit integer hash for *text* using SHA-256."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)

def words(text: str) -> List[str]:
    """Lower-case alphabetic tokens (apostrophe-aware)."""
    return [w for w in text.lower().split() if w.isalpha()]

def stylometry_features(text: str) -> np.ndarray:
    """
    Produce a deterministic 96-dimensional numeric representation of *text*.
    The implementation uses a SHA-256 hash to seed a pseudo-random generator,
    guaranteeing reproducibility without external corpora.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.random(96)

def extract_full_features(text: str) -> Dict[str, float]:
    """
    Produce a reproducible pseudo-random feature vector from *text*.
    The same input always yields the same output across Python runs.
    """
    seed = _deterministic_hash(text) % (2**32)
    rnd = random.Random(seed)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight",
        "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline",
        "telemetry_manic_velocity",
    ]
    return {k: rnd.random() for k in keys}

def extract_master_vector(text: str) -> Dict[str, float]:
    """
    Reduce the full feature set to a compact *master vector*.
    The selection mirrors the original implementation but remains deterministic.
    """
    f = extract_full_features(text)
    return {
        "visceral_ratio": f["operator_visceral_ratio"],
        "tech_ratio": f["operator_tech_ratio"],
        "legal_osint_ratio": f["operator_legal_osint_ratio"],
        "forensic_shield_ratio": f["psyche_forensic_shield_ratio"],
        "poetic_entropy": f["psyche_poetic_entropy"],
        "dissociative_index": f["psyche_dissociative_index"],
        "bureaucratic_weaponization_index": f[
            "resilience_bureaucratic_weaponization_index"
        ],
        "resource_exhaustion_metric": f["resilience_resource_exhaustion_metric"],
    }

def lsm_vector(text: str) -> np.ndarray:
    """
    Compute the proportion of words belonging to each FUNCTION_CAT.
    Return a numpy array representing the LSM vector.
    """
    w = words(text)
    counts = {cat: sum(1 for word in w if word in cats) for cat, cats in FUNCTION_CATS.items()}
    return np.array(list(counts.values())) / len(w)

def hybrid_operation(text: str) -> HybridFeatures:
    """
    Perform the hybrid operation, combining stylometry features with master vector generation.
    """
    stylometry = stylometry_features(text)
    master_vector = extract_master_vector(text)
    return HybridFeatures(stylometry, master_vector)

def print_hybrid_features(hybrid_features: HybridFeatures):
    print("Stylometry Features:")
    print(hybrid_features.stylometry)
    print("Master Vector:")
    for key, value in hybrid_features.master_vector.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    text = "This is a test sentence for the hybrid algorithm."
    hybrid_features = hybrid_operation(text)
    print_hybrid_features(hybrid_features)