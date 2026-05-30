# DARWIN HAMMER — match 546, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_workshare_all_m339_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s4.py (gen4)
# born: 2026-05-29T23:29:48Z

import sys
import random
import math
import hashlib
import pathlib
from datetime import datetime, timezone
from collections import Counter
import re
from dataclasses import dataclass, field
from typing import Tuple, Dict, Callable, List

import numpy as np

FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't".split()),
}

CATEGORY_ORDER = list(FUNCTION_CATS.keys())
NUM_CATS = len(CATEGORY_ORDER)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+'\w+|\b\w+\b", text.lower())


def stylometry_features(text: str) -> np.ndarray:
    tokens = _tokenize(text)
    counts = Counter(tokens)
    vec = np.zeros(NUM_CATS, dtype=float)
    for idx, cat in enumerate(CATEGORY_ORDER):
        cat_words = FUNCTION_CATS[cat]
        cat_count = sum(counts[w] for w in cat_words if w in counts)
        vec[idx] = cat_count
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def weekday_weight_vector(pool_size: int, weekday: int | None = None) -> np.ndarray:
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    if weekday is None:
        weekday = datetime.now().weekday()  
    angles = np.linspace(0, 2 * math.pi, pool_size, endpoint=False)
    base = np.sin(angles) + 1.0  
    rot = int((weekday / 7.0) * pool_size) % pool_size
    rotated = np.roll(base, rot)
    weight = rotated / rotated.sum()
    return weight


@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    reference_tokens: Tuple[str, ...] = field(default_factory=tuple)


class HybridModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.load_timestamp: Dict[str, float] = {}

    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _check_constraints(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used_ram() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("Inputs must be 1‑D arrays")
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _model_feature(self, model: ModelTier) -> np.ndarray:
        text = " ".join(model.reference_tokens)
        return stylometry_features(text)

    def _weighted_score(self, input_feat: np.ndarray, weekday: int | None = None) -> Dict[str, float]:
        if not self.loaded:
            return {}

        pool_names = list(self.loaded.keys())
        pool_size = len(pool_names)
        w_vec = weekday_weight_vector(pool_size, weekday)

        scores = {}
        for i, name in enumerate(pool_names):
            model = self.loaded[name]
            model_feat = self._model_feature(model)
            sim = self._cosine_similarity(input_feat, model_feat)
            scores[name] = sim * w_vec[i] * (1 + 0.1 * (1 - (i / (pool_size - 1))))  # modified equation (1)
        return scores

    def load(self, model: ModelTier) -> None:
        self._check_constraints(model)
        self.loaded[model.name] = model
        self.load_timestamp[model.name] = datetime.now().timestamp()

    def unload(self, model_name: str) -> None:
        if model_name in self.loaded:
            del self.loaded[model_name]
            del self.load_timestamp[model_name]

    def get_scores(self, input_text: str, weekday: int | None = None) -> Dict[str, float]:
        input_feat = stylometry_features(input_text)
        return self._weighted_score(input_feat, weekday)


# Example usage:
if __name__ == "__main__":
    pool = HybridModelPool(ram_ceiling_mb=8000)

    model1 = ModelTier("model1", 1000, "T1", ("token1", "token2"))
    model2 = ModelTier("model2", 2000, "T2", ("token3", "token4"))
    model3 = ModelTier("model3", 3000, "T3", ("token5", "token6"))

    pool.load(model1)
    pool.load(model2)
    pool.load(model3)

    input_text = "This is an example input text."
    scores = pool.get_scores(input_text)

    print("Scores:")
    for model, score in scores.items():
        print(f"{model}: {score}")

    pool.unload("model2")

    scores = pool.get_scores(input_text)

    print("Scores after unloading model2:")
    for model, score in scores.items():
        print(f"{model}: {score}")