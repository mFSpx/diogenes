# DARWIN HAMMER — match 1875, survivor 6
# gen: 3
# parent_a: hybrid_hard_truth_math_model_pool_m8_s1.py (gen1)
# parent_b: hybrid_sketches_hybrid_bandit_router_m31_s0.py (gen2)
# born: 2026-05-29T23:39:32Z

import sys
import random
import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple
import numpy as np

FUNCTION_CATS = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot cant wont dont didnt isnt arent wasnt werent".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

@dataclass(frozen=True)
class BanditAction:
    model_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "epsilon_greedy"

_POLICY: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])  

def extract_stylometry(text: str) -> Counter:
    tokens = [t.lower().strip(".,!?;:()[]\"'") for t in text.split()]
    cat_counts = Counter()
    for token in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                cat_counts[cat] += 1
    cat_counts["word_count"] = len(tokens)
    return cat_counts

def count_min_sketch(counter: Counter, width: int = 64, depth: int = 4, seed: int | str | None = None) -> np.ndarray:
    rng = random.Random(seed)
    sketch = np.zeros((depth, width), dtype=np.int32)
    for key, freq in counter.items():
        key_str = str(key)
        for d in range(depth):
            h = hash(str(d) + key_str + str(rng.random()))
            col = h % width
            sketch[d, col] += freq
    return sketch

def sketch_estimate(sketch: np.ndarray) -> np.ndarray:
    return sketch.min(axis=0)

def sketch_similarity(sketch_a: np.ndarray, sketch_b: np.ndarray) -> float:
    est_a = sketch_estimate(sketch_a)
    est_b = sketch_estimate(sketch_b)
    return float(np.dot(est_a, est_b))

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(model_id: str, reward: float) -> None:
    stats = _POLICY[model_id]
    stats[0] += reward
    stats[1] += 1.0

def expected_reward(model_id: str) -> float:
    total, cnt = _POLICY[model_id]
    return total / cnt if cnt else 0.0

def select_action_eps_greedy(candidates: List[str], epsilon: float = 0.1, seed: int | str | None = None) -> str:
    rng = random.Random(seed)
    if rng.random() < epsilon:
        return rng.choice(candidates)
    best = max(candidates, key=lambda m: expected_reward(m))
    return best

@dataclass
class ModelEntry:
    model_id: str
    sketch: np.ndarray
    metadata: dict

class ModelPool:
    def __init__(self, capacity: int = 5, width: int = 64, depth: int = 4):
        self.capacity = capacity
        self.width = width
        self.depth = depth
        self._entries: Dict[str, ModelEntry] = {}

    def add_model(self, model_id: str, text_corpus: str, metadata: dict | None = None, seed: int | str | None = None) -> None:
        metadata = metadata or {}
        feats = extract_stylometry(text_corpus)
        sketch = count_min_sketch(feats, self.width, self.depth, seed)
        entry = ModelEntry(model_id=model_id, sketch=sketch, metadata=metadata)
        if len(self._entries) >= self.capacity:
            self.evict_least_valuable()
        self._entries[model_id] = entry
        _POLICY.setdefault(model_id, [0.0, 0.0])

    def evict_least_valuable(self) -> None:
        if not self._entries:
            return
        victim_id = min(self._entries.keys(), key=lambda m: expected_reward(m))
        del self._entries[victim_id]
        _POLICY.pop(victim_id, None)

    def most_similar(self, text: str, seed: int | str | None = None) -> Tuple[str, float]:
        query_feats = extract_stylometry(text)
        query_sketch = count_min_sketch(query_feats, self.width, self.depth, seed)
        best_id, best_sim = None, -float('inf')
        for mid, entry in self._entries.items():
            sim = sketch_similarity(query_sketch, entry.sketch)
            if sim > best_sim:
                best_id, best_sim = mid, sim
        return best_id, best_sim

    def select_model(self, text: str, epsilon: float = 0.1, seed: int | str | None = None) -> BanditAction:
        similarities = {}
        query_feats = extract_stylometry(text)
        query_sketch = count_min_sketch(query_feats, self.width, self.depth, seed)
        for mid, entry in self._entries.items():
            similarities[mid] = sketch_similarity(query_sketch, entry.sketch)
        top_k = sorted(similarities, key=similarities.get, reverse=True)[:3]
        if not top_k:
            raise ValueError("No models in the pool")
        chosen_id = select_action_eps_greedy(top_k, epsilon, seed)
        prop = 1.0 / len(top_k) if chosen_id in top_k else 0.0
        exp_rw = expected_reward(chosen_id)
        n = _POLICY[chosen_id][1]
        total_n = sum(v[1] for v in _POLICY.values())
        cb = exp_rw + np.sqrt(2 * np.log(total_n + 1) / (n + 1)) if n > 0 else exp_rw
        return BanditAction(
            model_id=chosen_id,
            propensity=prop,
            expected_reward=exp_rw,
            confidence_bound=cb,
        )