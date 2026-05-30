# DARWIN HAMMER — match 535, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_label_foundry_hybrid_hybrid_hard_t_m304_s1.py (gen3)
# parent_b: hybrid_hybrid_krampus_stick_hybrid_hybrid_hard_t_m15_s0.py (gen4)
# born: 2026-05-29T23:29:36Z

import numpy as np
from dataclasses import dataclass
from typing import List
from collections import Counter
import math
import pathlib

# Parent A – stylometry utilities (adapted)
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
        "and but or nor so yet because although whoever that which what how why when where who whom since as until long".split()
    ),
    "adverb": set(
        "how very rather more extremely somewhat fairly highly incredibly most remarkably somewhat surprisingly".split()
    ),
}

# Parent B – pheromone utilities
@dataclass
class PheromoneEntry:
    uuid: str
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: int
    created_at: pathlib.Path
    last_decay: pathlib.Path

    def __post_init__(self):
        self.uuid = str(np.random.randint(0, 1000000))
        self.created_at = pathlib.Path.cwd()
        self.last_decay = pathlib.Path.cwd()

    def age_seconds(self) -> float:
        return np.random.uniform(0, 100)

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = pathlib.Path.cwd()


class PheromoneStore:
    _entries: dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]


def styled_labelled_features(text: str) -> np.ndarray:
    tokens = text.split()
    token_counts = Counter(tokens)
    labelled_features = np.zeros(len(FUNCTION_CATS))
    for i, (cat, words) in enumerate(FUNCTION_CATS.items()):
        labelled_features[i] = sum(token_counts[word] for word in words)
    return labelled_features


def bspline_basis(x: float, knots: np.ndarray, degree: int) -> float:
    if degree == 0:
        return 1.0 if knots[0] <= x < knots[1] else 0.0
    else:
        a = (x - knots[0]) / (knots[degree + 1] - knots[0]) if knots[degree + 1] != knots[0] else 0
        b = (knots[degree + 2] - x) / (knots[degree + 2] - knots[1]) if knots[degree + 2] != knots[1] else 0
        return a * bspline_basis(x, knots[1:], degree - 1) + b * bspline_basis(x, knots[:-1], degree - 1)


def kan_layer(labelled_features: np.ndarray, pheromone_entries: List[PheromoneEntry], 
               reliability_score: float, curvature_score: float) -> np.ndarray:
    signal_values = np.array([entry.signal_value for entry in pheromone_entries])
    output = np.zeros(len(labelled_features))
    multiplicative_factor = reliability_score * curvature_score
    for i, feature in enumerate(labelled_features):
        for j, signal_value in enumerate(signal_values):
            output[i] += feature * signal_value * bspline_basis(feature, np.array([0, 0.5, 1.0]), 2) * multiplicative_factor
    return output


def hybrid_feature_vector(text: str, pheromone_entries: List[PheromoneEntry], 
                          reliability_score: float, curvature_score: float) -> np.ndarray:
    labelled_features = styled_labelled_features(text)
    return kan_layer(labelled_features, pheromone_entries, reliability_score, curvature_score)


def hybrid_predict(text: str, pheromone_entries: List[PheromoneEntry], 
                   reliability_score: float, curvature_score: float) -> np.ndarray:
    output = hybrid_feature_vector(text, pheromone_entries, reliability_score, curvature_score)
    return output / np.sum(output) if np.sum(output) != 0 else np.zeros_like(output)


if __name__ == "__main__":
    pheromone_entry = PheromoneEntry("surface_key", "signal_kind", 1.0, 100, pathlib.Path.cwd(), pathlib.Path.cwd())
    PheromoneStore.add(pheromone_entry)
    text = "This is a test sentence with some pronouns and articles."
    reliability_score = 0.9
    curvature_score = 0.8
    output = hybrid_predict(text, PheromoneStore.get_by_surface("surface_key"), reliability_score, curvature_score)
    print(output)