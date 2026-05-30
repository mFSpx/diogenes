# DARWIN HAMMER — match 2093, survivor 2
# gen: 5
# parent_a: hybrid_krampus_brainmap_hybrid_pheromone_inf_m37_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m385_s3.py (gen4)
# born: 2026-05-29T23:40:51Z

"""
Hybrid Algorithm: krampus_brainmap + hybrid_pheromone_infotaxis

Parents:
- hybrid_krampus_brainmap_hybrid_pheromone_inf_m37_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m385_s3.py (Algorithm B)

Mathematical Bridge:
Both parents employ Shannon entropy as a core quantitative measure.
Algorithm A uses entropy of pheromone signal distributions to guide decay and decision‑making.
Algorithm B builds categorical frequency distributions (function‑word categories) whose entropy
characterises stylistic complexity.  The fusion therefore maps the stylometric probability
vector from Algorithm B onto the pheromone space of Algorithm A, using the shared entropy
to scale pheromone signal strength and to adapt half‑life dynamically.  A 3‑axis linear
projection (as in the original krampus_brainmap) is then applied to the weighted feature
vector, yielding a unified state vector that simultaneously encodes textual style and
pheromone dynamics.
"""

import uuid
import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Pheromone subsystem (from Algorithm A)
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """In‑memory singleton store."""
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_all(cls) -> None:
        for e in list(cls._entries.values()):
            e.apply_decay()
            if e.signal_value < 1e-9:  # prune near‑zero entries
                del cls._entries[e.uuid]

    @classmethod
    def snapshot(cls) -> List[Tuple[str, float]]:
        return [(e.uuid, e.signal_value) for e in cls._entries.values()]

# ----------------------------------------------------------------------
# Stylometry subsystem (from Algorithm B)
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
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
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def _words(text: str) -> List[str]:
    """Lower‑case, split on whitespace, keep alphabetic tokens."""
    return [w for w in (text or "").lower().split() if w.isalpha()]


def lsm_vector(text: str) -> Dict[str, float]:
    """Low‑dimensional semantic vector over FUNCTION_CATS."""
    ws = _words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """Hand‑crafted numeric features (mirrors parent B)."""
    ws = _words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))

    handcrafted = [
        total_words / 500.0,
        sum(len(w) for w in ws) / total_words / 12.0,
        (text.count("\n") + 1) / 200.0,
        sum(text.count(p) for p in "!?") / total_chars,
        sum(text.count(p) for p in ";:") / total_chars,
        sum(text.count(p) for p in "-—") / total_chars,
        sum(1 for ch in text if ch.isupper()) / total_words,
    ]

    # Pad / truncate to requested dimension using a deterministic hash‑based filler
    np.random.seed(abs(hash(text)) % (2**32))
    filler = np.random.rand(max(0, dim - len(handcrafted)))
    vec = np.array(handcrafted + filler.tolist(), dtype=float)
    return vec[:dim]


# ----------------------------------------------------------------------
# Shared entropy utilities (the mathematical bridge)
# ----------------------------------------------------------------------
def shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy (base‑e) of a probability vector."""
    probs = probs[probs > 0]  # avoid log(0)
    return -float(np.sum(probs * np.log(probs)))


def normalize_distribution(vec: np.ndarray) -> np.ndarray:
    """Return a probability distribution (L1‑norm = 1)."""
    total = np.sum(vec)
    if total == 0:
        return np.full_like(vec, 1.0 / vec.size)
    return vec / total


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def extract_combined_feature_vector(text: str) -> np.ndarray:
    """
    Produce a single feature vector by concatenating:
    1. LSM categorical probabilities (len(FUNCTION_CATS))
    2. Stylometry numeric features (fixed dimension)
    The result is L2‑normalized.
    """
    lsm = np.array(list(lsm_vector(text).values()), dtype=float)
    sty = stylometry_features(text, dim=96)
    combined = np.concatenate([lsm, sty])
    norm = np.linalg.norm(combined)
    return combined if norm == 0 else combined / norm


def project_to_3d(vector: np.ndarray) -> np.ndarray:
    """
    Linear 3‑axis projection emulating the krampus_brainmap step.
    A fixed random orthonormal matrix (seeded for reproducibility) is used.
    """
    dim = vector.shape[0]
    rng = np.random.default_rng(seed=42)
    # Generate a random 3 x dim matrix and orthonormalize rows via QR
    random_mat = rng.standard_normal((3, dim))
    q, _ = np.linalg.qr(random_mat.T)
    proj_mat = q.T[:3]  # shape (3, dim)
    return proj_mat @ vector  # shape (3,)


def update_pheromone_with_text(surface_key: str, text: str) -> None:
    """
    1. Extract combined feature vector.
    2. Compute its probability distribution and entropy.
    3. Use entropy to scale the signal value.
    4. Derive a half‑life inversely proportional to entropy (more uncertain => longer persistence).
    5. Store the entry in the global PheromoneStore.
    """
    vec = extract_combined_feature_vector(text)
    prob_dist = normalize_distribution(vec)
    entropy = shannon_entropy(prob_dist)

    # Signal strength: entropy multiplied by L2 norm of original (pre‑norm) vector
    raw_norm = np.linalg.norm(vec)  # should be 1 after normalization, but keep for safety
    signal_value = entropy * raw_norm

    # Half‑life: map entropy in [0, log(N)] to seconds in [30, 300]
    max_entropy = math.log(prob_dist.size)
    min_hl, max_hl = 30, 300  # seconds
    half_life = int(min_hl + (entropy / max_entropy) * (max_hl - min_hl))

    entry = PheromoneEntry(
        surface_key=surface_key,
        signal_kind="text_style",
        signal_value=signal_value,
        half_life_seconds=half_life,
    )
    PheromoneStore.add(entry)


def decay_and_report(surface_key: str) -> List[Tuple[str, float]]:
    """
    Apply decay to all entries, then return a snapshot of entries for the given surface.
    """
    PheromoneStore.decay_all()
    return [(e.uuid, e.signal_value) for e in PheromoneStore.get_by_surface(surface_key)]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "In the quiet of the night, the owl hooted softly while the wind whispered "
        "through the ancient oaks. It was a moment of serene contemplation."
    )
    surface = "sample_001"

    # Step 1: create/update pheromone based on text
    update_pheromone_with_text(surface, sample_text)

    # Step 2: project the underlying feature vector to 3‑D (demonstration)
    vec = extract_combined_feature_vector(sample_text)
    proj = project_to_3d(vec)
    print("3‑D projection:", proj)

    # Step 3: decay and report
    report = decay_and_report(surface)
    print("Pheromone snapshot after decay:", report)

    # Ensure the store is not empty
    assert len(report) > 0, "Pheromone store should contain at least one entry."