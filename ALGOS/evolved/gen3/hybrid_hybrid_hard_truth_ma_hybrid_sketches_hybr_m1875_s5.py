# DARWIN HAMMER — match 1875, survivor 5
# gen: 3
# parent_a: hybrid_hard_truth_math_model_pool_m8_s1.py (gen1)
# parent_b: hybrid_sketches_hybrid_bandit_router_m31_s0.py (gen2)
# born: 2026-05-29T23:39:32Z

"""Hybrid model selection using stylometry and sketch-based similarity.

Parents:
- hybrid_hard_truth_math_model_pool_m8_s1.py (stylometry feature extraction + model pool eviction)
- hybrid_sketches_hybrid_bandit_router_m31_s0.py (count‑min sketch, MinHash LSH and bandit policy)

Mathematical bridge:
Stylometric features are categorical counts.  They are embedded into a Count‑Min sketch,
producing a compact linear projection  S ∈ ℕ^{d×w}.  Two sketches  S₁, S₂  admit an
inner‑product estimate

    ⟨S₁,S₂⟩̂ = Σ_{j=1}^{w} min_{i=1..d} S₁[i,j] · min_{i=1..d} S₂[i,j]

which approximates the true dot‑product of the underlying count vectors.
This estimate serves as a similarity metric for model selection.
A lightweight ε‑greedy bandit updates expected reward for each model based on the
observed performance, guiding eviction when memory pressure exceeds a ceiling.
"""

import sys
import random
import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Stylometry extraction (Parent A core)
# ----------------------------------------------------------------------
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

def extract_stylometry(text: str) -> Counter:
    """Return a Counter of stylometric category occurrences in *text*."""
    tokens = [t.lower().strip(".,!?;:()[]\"'") for t in text.split()]
    cat_counts = Counter()
    for token in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                cat_counts[cat] += 1
    # also count total words as a baseline feature
    cat_counts["word_count"] = len(tokens)
    return cat_counts

# ----------------------------------------------------------------------
# Count‑Min sketch utilities (Parent B core)
# ----------------------------------------------------------------------
def count_min_sketch(counter: Counter, width: int = 64, depth: int = 4) -> np.ndarray:
    """Encode a Counter into a Count‑Min sketch matrix."""
    sketch = np.zeros((depth, width), dtype=np.int32)
    for key, freq in counter.items():
        key_str = str(key)
        for d in range(depth):
            h = hash(str(d) + key_str)
            col = h % width
            sketch[d, col] += freq
    return sketch

def sketch_estimate(sketch: np.ndarray) -> np.ndarray:
    """Recover an approximate frequency vector from a sketch (min across rows)."""
    return sketch.min(axis=0)

def sketch_similarity(sketch_a: np.ndarray, sketch_b: np.ndarray) -> float:
    """Estimate inner‑product similarity between two sketches."""
    est_a = sketch_estimate(sketch_a)
    est_b = sketch_estimate(sketch_b)
    return float(np.dot(est_a, est_b))

# ----------------------------------------------------------------------
# Bandit policy (Parent B core, simplified ε‑greedy)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    model_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "epsilon_greedy"

_POLICY: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])  # model_id → [total_reward, count]

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
    # exploit: pick candidate with highest expected reward
    best = max(candidates, key=lambda m: expected_reward(m))
    return best

# ----------------------------------------------------------------------
# Model pool structures (fusion of both parents)
# ----------------------------------------------------------------------
@dataclass
class ModelEntry:
    model_id: str
    sketch: np.ndarray
    metadata: dict

class ModelPool:
    """Manages a bounded pool of models using sketch similarity and bandit rewards."""
    def __init__(self, capacity: int = 5, width: int = 64, depth: int = 4):
        self.capacity = capacity
        self.width = width
        self.depth = depth
        self._entries: Dict[str, ModelEntry] = {}

    def add_model(self, model_id: str, text_corpus: str, metadata: dict | None = None) -> None:
        """Create a model from *text_corpus* and insert into the pool, evicting if necessary."""
        metadata = metadata or {}
        feats = extract_stylometry(text_corpus)
        sketch = count_min_sketch(feats, self.width, self.depth)
        entry = ModelEntry(model_id=model_id, sketch=sketch, metadata=metadata)
        if len(self._entries) >= self.capacity:
            self.evict_least_valuable()
        self._entries[model_id] = entry
        # initialise bandit stats
        _POLICY.setdefault(model_id, [0.0, 0.0])

    def evict_least_valuable(self) -> None:
        """Evict the model with the lowest bandit expected reward."""
        if not self._entries:
            return
        victim_id = min(self._entries.keys(), key=lambda m: expected_reward(m))
        del self._entries[victim_id]
        _POLICY.pop(victim_id, None)

    def most_similar(self, text: str) -> Tuple[str, float]:
        """Return (model_id, similarity) of the most similar model to *text*."""
        query_feats = extract_stylometry(text)
        query_sketch = count_min_sketch(query_feats, self.width, self.depth)
        best_id, best_sim = None, -float('inf')
        for mid, entry in self._entries.items():
            sim = sketch_similarity(query_sketch, entry.sketch)
            if sim > best_sim:
                best_id, best_sim = mid, sim
        return best_id, best_sim

    def select_model(self, text: str, epsilon: float = 0.1, seed: int | str | None = None) -> BanditAction:
        """Hybrid selection: similarity filtering + ε‑greedy bandit."""
        # Gather top‑k similar candidates (k=3 or all if fewer)
        similarities = {}
        query_feats = extract_stylometry(text)
        query_sketch = count_min_sketch(query_feats, self.width, self.depth)
        for mid, entry in self._entries.items():
            similarities[mid] = sketch_similarity(query_sketch, entry.sketch)
        top_k = sorted(similarities, key=similarities.get, reverse=True)[:3]
        chosen_id = select_action_eps_greedy(top_k, epsilon, seed)
        prop = 1.0 / len(top_k) if chosen_id in top_k else 0.0
        exp_rw = expected_reward(chosen_id)
        return BanditAction(
            model_id=chosen_id,
            propensity=prop,
            expected_reward=exp_rw,
            confidence_bound=exp_rw + np.sqrt(2 * np.log(max(1, sum(_POLICY[mid][1] for mid in _POLICY))) / max(1, _POLICY[chosen_id][1])),
        )

    def record_outcome(self, model_id: str, reward: float) -> None:
        """Feed back the observed reward to the bandit policy."""
        update_policy(model_id, reward)

    def list_models(self) -> List[str]:
        return list(self._entries.keys())

# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def demo_build_pool() -> ModelPool:
    """Create a pool with three dummy models built from sample texts."""
    pool = ModelPool(capacity=4)
    samples = {
        "model_A": "I think therefore I am. The quick brown fox jumps over the lazy dog.",
        "model_B": "She sells sea shells on the sea shore. And then she bought a new car.",
        "model_C": "Quantum mechanics is the study of the very small. It is fascinating.",
    }
    for mid, txt in samples.items():
        pool.add_model(mid, txt, metadata={"source": "demo"})
    return pool

def demo_select_and_reward(pool: ModelPool, input_text: str, reward: float) -> None:
    """Select a model for *input_text*, print choice, and record *reward*."""
    action = pool.select_model(input_text, epsilon=0.2, seed=42)
    print(f"Selected model: {action.model_id}")
    print(f"  Propensity: {action.propensity:.3f}")
    print(f"  Expected reward (pre‑update): {action.expected_reward:.3f}")
    pool.record_outcome(action.model_id, reward)
    print(f"  Expected reward (post‑update): {expected_reward(action.model_id):.3f}")

def demo_eviction_policy(pool: ModelPool) -> None:
    """Add a fourth model to trigger eviction and display remaining models."""
    extra_text = "Artificial intelligence blends computer science with linguistics."
    pool.add_model("model_D", extra_text, metadata={"source": "extra"})
    print("Current models after possible eviction:", pool.list_models())

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    reset_policy()
    pool = demo_build_pool()
    demo_select_and_reward(pool, "The fox is quick and jumps over lazy dogs.", reward=1.0)
    demo_select_and_reward(pool, "She bought a new quantum computer.", reward=0.5)
    demo_eviction_policy(pool)
    # Verify that policy dict aligns with pool contents
    assert set(pool.list_models()) == set(_POLICY.keys()), "Policy and pool mismatch"
    print("Smoke test completed successfully.")