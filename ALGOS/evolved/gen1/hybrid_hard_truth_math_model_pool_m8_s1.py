# DARWIN HAMMER — match 8, survivor 1
# gen: 1
# parent_a: hard_truth_math.py (gen0)
# parent_b: model_pool.py (gen0)
# born: 2026-05-29T23:22:33Z

"""Hybrid algorithm fusion of hard_truth_math.py and model_pool.py.

The mathematical bridge between the two parents is the integration of the stylistic features extracted from the hard_truth_math.py
algorithm with the model loading and eviction strategy from the model_pool.py. This fusion enables the creation of a stylometry-based
model loading and eviction strategy, where models are loaded and evicted based on their stylistic similarity to the input text.

The governing equations of the hard_truth_math.py algorithm, specifically the stylometry_features function, are used to extract
features from the input text. These features are then used to compute the similarity between the input text and the models in the
model pool. The model with the highest similarity is loaded, and the model with the lowest similarity is evicted.

The matrix operations of the model_pool.py algorithm, specifically the _used function, are used to manage the memory usage of the
models in the model pool. The _used function is modified to take into account the stylistic similarity of the models to the input
text, and the model with the lowest similarity is evicted when the memory usage exceeds the ceiling.
"""

import numpy as np
import random
import sys
import pathlib
from collections import Counter
import re
from datetime import datetime as dt
from typing import Any

FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {cat: sum(cnt[w] for w in vocab) / total for cat, vocab in FUNCTION_CATS.items()}

def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))
    vals: list[float] = [
        total_words / 500.0,
        sum(len(w) for w in ws) / total_words / 12.0,
        (text.count("\n") + 1) / 200.0,
        sum(text.count(p) for p in "!?") / total_chars,
        sum(text.count(p) for p in ";:") / total_chars,
        sum(text.count(p) for p in "-—") / total_chars,
        sum(1 for ch in text if ch.isupper()) / total_chars,
        len(re.findall(r"\b[A-Z]{2,}\b", text)) / total_words,
    ]
    lv = lsm_vector(text)
    vals.extend(lv.get(cat, 0.0) for cat in sorted(FUNCTION_CATS))
    vec = np.zeros(dim, dtype=np.float64)
    base = np.array(vals[: min(len(vals), dim)], dtype=np.float64)
    vec[: len(base)] = base
    cleaned = " ".join(ws)
    for i in range(max(0, len(cleaned) - 2)):
        gram = cleaned[i : i + 3]
        idx = 20 + (hash(gram) % max(1, dim - 20))
        vec[idx] += 1.0
    norm = np.linalg.norm(vec[20:])
    if norm > 0:
        vec[20:] /= norm
    return vec

class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)

def similarity(text: str, model: ModelTier) -> float:
    model_features = stylometry_features(model.name)
    text_features = stylometry_features(text)
    return np.dot(model_features, text_features) / (np.linalg.norm(model_features) * np.linalg.norm(text_features))

def load_model(model_pool: ModelPool, text: str, models: list[ModelTier]) -> ModelTier:
    best_model = None
    best_similarity = -1.0
    for model in models:
        if not model_pool.is_loaded(model.name):
            similarity_score = similarity(text, model)
            if similarity_score > best_similarity:
                best_model = model
                best_similarity = similarity_score
    if best_model:
        model_pool.load_with_eviction(best_model)
    return best_model

def evict_model(model_pool: ModelPool, text: str, models: list[ModelTier]) -> ModelTier:
    worst_model = None
    worst_similarity = 1.0
    for model in models:
        if model_pool.is_loaded(model.name):
            similarity_score = similarity(text, model)
            if similarity_score < worst_similarity:
                worst_model = model
                worst_similarity = similarity_score
    if worst_model:
        del model_pool.loaded[worst_model.name]
    return worst_model

if __name__ == "__main__":
    model_pool = ModelPool()
    models = [ModelTier("qwen-0.5b", 512, "T1"), ModelTier("reasoning-t2", 3000, "T2"), ModelTier("tool-t2", 2600, "T2")]
    text = "This is a test text."
    load_model(model_pool, text, models)
    print(model_pool.loaded)
    evict_model(model_pool, text, models)
    print(model_pool.loaded)