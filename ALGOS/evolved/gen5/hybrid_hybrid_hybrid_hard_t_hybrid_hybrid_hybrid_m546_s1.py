# DARWIN HAMMER — match 546, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_workshare_all_m339_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s4.py (gen4)
# born: 2026-05-29T23:29:48Z

"""Hybrid Stylometry‑Weekday Model Pool

Parents:
- hybrid_hard_truth_math_model_pool_m8_s1.py (stylometry feature extraction,
  similarity‑based loading/eviction)
- hybrid_workshare_allocator_doomsday_calendar_m14_s2.py (weekday‑dependent
  sinusoidal weight vector used as a linear operator)

Mathematical bridge:
Let **f** ∈ ℝ^C be the stylometric feature vector of an input text (C = number of
function categories). For every model *m* in the pool we compute its own
feature vector **gₘ** from the stored reference tokens. The cosine similarity

    sₘ = (f·gₘ) / (‖f‖‖gₘ‖)

produces a similarity scalar for each model.  
A weekday‑dependent weight vector **w** ∈ ℝ^N (N = current pool size) is built by
rotating a sinusoidal pattern by an angle proportional to the weekday (0‑6).  

The hybrid score for model *m* occupying position *i* in the ordered pool is

    scoreₘ = sₘ · w_i                                    (1)

Equation (1) fuses the two parent topologies: the stylometry‑driven similarity
provides the “hard truth” core, while the weekday‑derived weights act as a
linear operator mapping the similarity scores onto the pool lanes.  The model
with the maximal score is admitted, and the model with the minimal score is
evicted, all while respecting the RAM‑ceiling and tier‑exclusivity constraints
from the original ModelPool implementation.
"""

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

# ----------------------------------------------------------------------
# Stylometry – function word categories (parent A)
# ----------------------------------------------------------------------
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
    """Very simple word tokenizer."""
    return re.findall(r"\b\w+'\w+|\b\w+\b", text.lower())


def stylometry_features(text: str) -> np.ndarray:
    """
    Extract a normalized frequency vector over FUNCTION_CATS.
    Returns a (NUM_CATS,) float array that sums to 1 (or zeros if no matches).
    """
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


# ----------------------------------------------------------------------
# Weekday‑dependent sinusoidal weight vector (parent A)
# ----------------------------------------------------------------------
def weekday_weight_vector(pool_size: int, weekday: int | None = None) -> np.ndarray:
    """
    Generate a stochastic weight vector of length `pool_size` by rotating a
    sinusoidal pattern. `weekday` is 0 (Monday) … 6 (Sunday). If None, the current
    weekday is used.
    """
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    if weekday is None:
        weekday = datetime.now().weekday()  # Monday=0
    # Base sinusoid pattern
    angles = np.linspace(0, 2 * math.pi, pool_size, endpoint=False)
    base = np.sin(angles) + 1.0  # shift to be non‑negative
    # Rotation amount proportional to weekday
    rot = int((weekday / 7.0) * pool_size) % pool_size
    rotated = np.roll(base, rot)
    # Stochastic normalization (sum to 1)
    weight = rotated / rotated.sum()
    return weight


# ----------------------------------------------------------------------
# Core data structures (parent B)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Hybrid ModelPool
# ----------------------------------------------------------------------
class HybridModelPool:
    """
    Model pool that fuses stylometric similarity with weekday‑dependent
    weighting.  Loading/eviction obeys the RAM ceiling and tier exclusivity
    constraints from the original ModelPool.
    """

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.load_timestamp: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Constraint handling (parent B)
    # ------------------------------------------------------------------
    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _check_constraints(self, model: ModelTier) -> None:
        # Mutual exclusivity: T3 cannot coexist with any T2
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used_ram() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")

    # ------------------------------------------------------------------
    # Core hybrid scoring (bridge equation)
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Safe cosine similarity for 1‑D vectors."""
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("Inputs must be 1‑D arrays")
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _model_feature(self, model: ModelTier) -> np.ndarray:
        """Compute stylometry vector from stored reference tokens."""
        text = " ".join(model.reference_tokens)
        return stylometry_features(text)

    def _weighted_score(self, input_feat: np.ndarray, weekday: int | None = None) -> Dict[str, float]:
        """
        Compute hybrid scores for all loaded models using equation (1).
        Returns a dict name → score.
        """
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
            scores[name] = sim * w_vec[i]          # Equation (1)
        return scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, model: ModelTier) -> None:
        """Load a model respecting constraints."""
        self._check_constraints(model)
        self.loaded[model.name] = model
        self.load_timestamp[model.name] = datetime.now(tz=timezone.utc).timestamp()

    def evict_lowest(self, score_fn: Callable[[ModelTier], float]) -> None:
        """Evict the model with the lowest score according to `score_fn`."""
        if not self.loaded:
            return
        victim_name = min(self.loaded, key=lambda n: score_fn(self.loaded[n]))
        del self.loaded[victim_name]
        del self.load_timestamp[victim_name]

    def hybrid_adapt(self, input_text: str, weekday: int | None = None) -> str:
        """
        Given new input text, compute hybrid scores, load the highest‑scoring
        model (if not already loaded), and evict the lowest‑scoring one when
        RAM constraints would be violated.
        Returns the name of the selected model.
        """
        input_feat = stylometry_features(input_text)

        # If pool empty, cannot select – caller must pre‑load at least one model.
        if not self.loaded:
            raise RuntimeError("Model pool is empty; load at least one model first.")

        scores = self._weighted_score(input_feat, weekday)

        # Identify best and worst models according to hybrid score
        best_name = max(scores, key=scores.get)
        worst_name = min(scores, key=scores.get)

        # No eviction needed if best already loaded and RAM is OK
        # (All models are already loaded; we just report the best)
        # In a more dynamic setting we could load a new candidate here.
        # For demonstration, we evict the worst if its score is below a threshold.
        if scores[worst_name] < 0.05:  # arbitrary low‑score cutoff
            # Evict the worst model
            del self.loaded[worst_name]
            del self.load_timestamp[worst_name]

        return best_name

    # ------------------------------------------------------------------
    # Utility for debugging / inspection
    # ------------------------------------------------------------------
    def snapshot(self) -> Tuple[List[str], List[float]]:
        """Return ordered model names and their current hybrid scores (weekday‑today)."""
        dummy_feat = np.zeros(NUM_CATS)  # zero vector yields zero similarity
        scores = self._weighted_score(dummy_feat)  # only weights matter
        names = list(scores.keys())
        values = [scores[n] for n in names]
        return names, values


# ----------------------------------------------------------------------
# Demonstration functions (require at least three)
# ----------------------------------------------------------------------
def demo_extract_and_weight(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return stylometry vector and weekday weight vector for a pool of size 5."""
    feat = stylometry_features(text)
    w = weekday_weight_vector(5)
    return feat, w


def demo_hybrid_score_example():
    """Show computation of hybrid scores for a tiny synthetic pool."""
    pool = HybridModelPool(ram_ceiling_mb=2000)

    # Create three dummy models with distinct reference token sets
    models = [
        ModelTier(name="alpha", ram_mb=400, tier="T1", reference_tokens=("i", "you", "we")),
        ModelTier(name="beta", ram_mb=600, tier="T2", reference_tokens=("the", "and", "or")),
        ModelTier(name="gamma", ram_mb=500, tier="T3", reference_tokens=("in", "on", "under")),
    ]

    for m in models:
        pool.load(m)

    sample = "I think we should go on and on, under the sky."
    best = pool.hybrid_adapt(sample)
    return best


def demo_pool_snapshot():
    """Create a pool, load models, and print the weight‑only snapshot."""
    pool = HybridModelPool()
    pool.load(ModelTier(name="A", ram_mb=300, tier="T1", reference_tokens=("i",)))
    pool.load(ModelTier(name="B", ram_mb=300, tier="T1", reference_tokens=("the",)))
    names, weights = pool.snapshot()
    return list(zip(names, weights))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Extract features and weight vector
    fvec, wvec = demo_extract_and_weight("The quick brown fox jumps over the lazy dog.")
    assert fvec.shape == (NUM_CATS,)
    assert wvec.shape == (5,)

    # 2. Run hybrid scoring demo
    chosen = demo_hybrid_score_example()
    print(f"Hybrid selected model: {chosen}")

    # 3. Show snapshot of weight‑only scores
    snap = demo_pool_snapshot()
    print("Pool snapshot (model, weight‑only score):")
    for name, score in snap:
        print(f"  {name}: {score:.4f}")

    print("Smoke test completed successfully.")