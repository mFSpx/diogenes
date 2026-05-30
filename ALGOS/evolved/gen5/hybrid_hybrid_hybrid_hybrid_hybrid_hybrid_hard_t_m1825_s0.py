# DARWIN HAMMER — match 1825, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s2.py (gen4)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_bayes_update__m123_s1.py (gen3)
# born: 2026-05-29T23:38:59Z

"""Hybrid Model Selection Algorithm
Parents:
- hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s2 (ModelPool resource management)
- hybrid_hybrid_hard_truth_ma_hybrid_bayes_update__m123_s1 (Bayesian compatibility of text features with model resources)

Mathematical Bridge:
The bridge is the scalar compatibility score  

    s = vᵀ P m  

where `v ∈ ℝⁿ` is a stylometry‑derived feature vector from text, `m ∈ ℝ²` encodes a model’s
resource attributes (RAM in MB and tier rank), and `P ∈ ℝⁿˣ²` projects the first two
dimensions of `v` onto the model space.  
The score `s` is turned into a likelihood via a sigmoid, and a Bayesian update
produces posterior probabilities for each candidate model.  The posterior‑weighted
scores drive the decision of which model to load into a `ModelPool`, respecting
RAM ceilings and tier exclusivity constraints.
"""

import hashlib
import sys
import re
import random
import math
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – ModelPool and related data structures
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
    tier: str   # e.g. "T1", "T2", "T3"

class ModelPool:
    """Manages a set of loaded models under RAM and tier constraints."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.access_timestamps: Dict[str, datetime] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model; raises if constraints are violated."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model
        self.access_timestamps[model.name] = datetime.now(timezone.utc)

    def load_with_eviction(self, model: ModelTier) -> None:
        """Evict least‑recently‑used models until the new model fits."""
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            lru_model = self._get_lru_model()
            self.loaded.pop(lru_model.name)
            self.access_timestamps.pop(lru_model.name)
        self.load(model)

    def _get_lru_model(self) -> ModelTier:
        return min(self.loaded.values(),
                   key=lambda x: self.access_timestamps[x.name])

    def access(self, model_name: str) -> None:
        if model_name in self.access_timestamps:
            self.access_timestamps[model_name] = datetime.now(timezone.utc)

# ----------------------------------------------------------------------
# Parent B – Stylometry extraction, Bayesian update, compatibility
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def words(text: str) -> List[str]:
    """Extract lowercase alphabetic tokens."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def _category_counts(tokens: List[str]) -> Dict[str, int]:
    """Count occurrences of each functional category."""
    counts = {cat: 0 for cat in FUNCTION_CATS}
    for token in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                counts[cat] += 1
    return counts

def lsm_vector(text: str) -> np.ndarray:
    """
    Compute a simple stylometry vector:
    [mean_word_len, total_word_ratio, pronoun_ratio, article_ratio]
    """
    toks = words(text)
    if not toks:
        return np.zeros(4)
    mean_len = np.mean([len(t) for t in toks])
    total_ratio = len(toks) / (len(text) + 1e-9)

    cat_counts = _category_counts(toks)
    pronoun_ratio = cat_counts["pronoun"] / (len(toks) + 1e-9)
    article_ratio = cat_counts["article"] / (len(toks) + 1e-9)

    return np.array([mean_len, total_ratio, pronoun_ratio, article_ratio], dtype=float)

def model_resource_vector(model: ModelTier) -> np.ndarray:
    """
    Encode a model as a 2‑dimensional resource vector:
    [ram_mb, tier_rank] where tier_rank maps T1→0, T2→1, T3→2.
    """
    tier_map = {"T1": 0, "T2": 1, "T3": 2}
    return np.array([model.ram_mb, tier_map.get(model.tier, 0)], dtype=float)

def compatibility_score(v: np.ndarray, m: np.ndarray) -> float:
    """
    Compute the scalar compatibility s = vᵀ P m.
    P projects the first two dimensions of v onto the model space.
    """
    # Projection matrix P (2x2) extracts the first two components.
    P = np.eye(2, dtype=float)  # Identity on the subspace
    v_proj = v[:2]               # shape (2,)
    s = float(v_proj @ P @ m)    # scalar
    return s

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def bayesian_posterior(prior: np.ndarray, likelihoods: np.ndarray) -> np.ndarray:
    """Standard Bayesian update (unnormalized posterior then normalize)."""
    unnorm = prior * likelihoods
    total = unnorm.sum()
    if total == 0:
        # Avoid division by zero – fall back to uniform
        return np.full_like(prior, 1.0 / prior.size)
    return unnorm / total

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def extract_features(text: str) -> np.ndarray:
    """High‑level wrapper returning the stylometry vector."""
    return lsm_vector(text)

def compute_compatibilities(v: np.ndarray, models: List[ModelTier]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        scores  – raw compatibility scalars for each model
        likes   – likelihoods obtained by passing scores through a sigmoid
    """
    scores = np.array([compatibility_score(v, model_resource_vector(m)) for m in models], dtype=float)
    likes = np.vectorize(sigmoid)(scores)
    return scores, likes

def select_and_load(text: str, candidates: List[ModelTier], pool: ModelPool) -> ModelTier:
    """
    Perform Bayesian model selection based on text, then load the chosen model
    into the supplied ModelPool (evicting if necessary). Returns the loaded model.
    """
    v = extract_features(text)

    # Uniform prior over candidates
    prior = np.full(len(candidates), 1.0 / len(candidates), dtype=float)

    # Compatibility‑derived likelihoods
    _, likes = compute_compatibilities(v, candidates)

    posterior = bayesian_posterior(prior, likes)

    # Choose the model with highest posterior probability
    best_idx = int(np.argmax(posterior))
    chosen = candidates[best_idx]

    # Load respecting pool constraints
    try:
        pool.load(chosen)
    except Exception:
        # If direct load fails, attempt eviction‑based load
        pool.load_with_eviction(chosen)

    # Record access for LRU policy
    pool.access(chosen.name)

    return chosen

def evaluate_pool_state(pool: ModelPool) -> Dict[str, Tuple[int, str]]:
    """
    Helper to inspect the current pool: returns a dict mapping model name to
    (ram_mb, tier) for quick debugging.
    """
    return {name: (model.ram_mb, model.tier) for name, model in pool.loaded.items()}

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a few dummy models
    models = [
        ModelTier(name="alpha", ram_mb=1500, tier="T1"),
        ModelTier(name="beta",  ram_mb=2500, tier="T2"),
        ModelTier(name="gamma", ram_mb=3000, tier="T3"),
        ModelTier(name="delta", ram_mb=1200, tier="T1"),
    ]

    # Create a pool with a modest RAM ceiling
    pool = ModelPool(ram_ceiling_mb=5000)

    sample_text = (
        "The quick brown fox jumps over the lazy dog. "
        "I think therefore I am. "
        "She sells seashells by the seashore."
    )

    # Run the hybrid selection several times to see eviction in action
    for _ in range(5):
        loaded = select_and_load(sample_text, models, pool)
        print(f"Loaded model: {loaded.name} (RAM {loaded.ram_mb} MB, Tier {loaded.tier})")
        print("Current pool state:", evaluate_pool_state(pool))
        print("-" * 60)