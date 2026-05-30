# DARWIN HAMMER — match 1232, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_krampus_stick_hybrid_hybrid_hybrid_m1008_s1.py (gen5)
# parent_b: hybrid_hard_truth_math_model_pool_m8_s5.py (gen1)
# born: 2026-05-29T23:34:34Z

"""
This module fuses the governing equations of hybrid_hybrid_krampus_stick_hybrid_hybrid_hybrid_m1008_s1.py (Parent A)
and hybrid_hard_truth_math_model_pool_m8_s5.py (Parent B) by integrating the pheromone-based social interaction 
model of Parent A with the stylometry and LSM utilities of Parent B. The mathematical bridge between the two 
parents lies in the use of frequency vectors to represent text data. Parent A's pheromone model uses word 
frequencies to compute pheromone values, while Parent B's stylometry features use frequency vectors to represent 
text stylometry.

The hybrid model combines these two approaches by using the stylometry features of Parent B as input to the 
pheromone model of Parent A. This allows the hybrid model to capture both the stylometric properties of text 
and the dynamic behavior of pheromone signals.

Imports: numpy, standard library, math, random, sys, pathlib.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from datetime import datetime, timezone

Vector = list[float]

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(random.getrandbits(128))
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
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    _entries: dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> list[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> list[dict]:
        rows = []
        for entry in cls.get_by_surface(surface_key):
            before = entry.signal_value
            entry.apply_decay()
            rows.append({"surface_key": surface_key, "before": before, "after": entry.signal_value})
        return rows


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
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> list[str]:
    """Return a list of lowercase words (ASCII letters + optional apostrophe)."""
    return [w for w in text.lower().split() if w.isalpha() or "'" in w]


def lsm_vector(text: str) -> dict[str, float]:
    """Return a normalized frequency vector for each FUNCTION_CATS."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """Produce a 96‑dimensional stylometric fingerprint."""
    lsm = lsm_vector(text)
    return np.array(list(lsm.values()) + [0.0] * (dim - len(lsm)))


def social_interaction_and_pheromone(text: str, g_best: Vector, k: int = 1, r1: float | None = None, seed: int | str | None = None) -> list[float]:
    words_list = words(text)
    word_counts = Counter(words_list)
    word_frequencies = [word_counts[word] / len(words_list) for word in word_counts]
    pheromone_value = np.mean(word_frequencies)
    return [pheromone_value * f for f in stylometry_features(text)]


def hybrid_pheromone_stylometry(text: str) -> list[dict]:
    pheromone_values = social_interaction_and_pheromone(text, [])
    lsm = lsm_vector(text)
    return [{"feature": cat, "pheromone_value": pheromone_values[i], "lsm_value": value} 
            for i, (cat, value) in enumerate(lsm.items())]


def main():
    text = "This is a sample text for testing the hybrid pheromone stylometry model."
    print(hybrid_pheromone_stylometry(text))


if __name__ == "__main__":
    main()