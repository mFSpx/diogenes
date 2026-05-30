# DARWIN HAMMER — match 546, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_workshare_all_m339_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s4.py (gen4)
# born: 2026-05-29T23:29:48Z

import sys
import math
import hashlib
import pathlib
from datetime import datetime
from collections import Counter
import re
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional

import numpy as np

# ----------------------------------------------------------------------
# Stylometry – function word categories (parent A)
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
    if total > 0.0:
        vec /= total
    return vec


# ----------------------------------------------------------------------
# Weekday‑dependent sinusoidal linear operator (enhanced)
# ----------------------------------------------------------------------
def _base_sinusoid(pool_size: int) -> np.ndarray:
    """Create the non‑negative sinusoid pattern used as a kernel."""
    angles = np.linspace(0, 2 * math.pi, pool_size, endpoint=False)
    return np.sin(angles) + 1.0  # shift to be non‑negative (range [0,2])


def weekday_weight_matrix(pool_size: int, weekday: int | None = None) -> np.ndarray:
    """
    Build a circulant matrix W ∈ ℝ^{N×N} whose rows are rotated copies of a
    sinusoidal pattern. The rotation offset is proportional to the weekday,
    providing a weekday‑dependent linear operator.
    Each row is normalised to sum to 1, so that W acts as a stochastic
    transformation on the similarity vector.
    """
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    if weekday is None:
        weekday = datetime.now().weekday()  # Monday=0 … Sunday=6

    base = _base_sinusoid(pool_size)
    # Global rotation determined by weekday
    global_rot = int((weekday / 7.0) * pool_size) % pool_size

    # Build circulant matrix by rolling the base vector for each row
    W = np.empty((pool_size, pool_size), dtype=float)
    for i in range(pool_size):
        row_rot = (global_rot + i) % pool_size
        W[i] = np.roll(base, row_rot)

    # Row‑wise stochastic normalisation (avoid division by zero)
    row_sums = W.sum(axis=1, keepdims=True)
    # If a row sum is zero (theoretically impossible with the shifted sinusoid),
    # replace it with 1 to keep the row as zeros.
    row_sums[row_sums == 0] = 1.0
    W /= row_sums
    return W


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
# Hybrid ModelPool – improved integration
# ----------------------------------------------------------------------
class HybridModelPool:
    """
    Model pool that fuses stylometric similarity with a weekday‑dependent
    circulant linear operator.  Loading/eviction respects RAM ceiling and tier
    exclusivity constraints, while the scoring mechanism now mixes similarity
    across all models, yielding a deeper mathematical integration.
    """

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.load_timestamp: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Constraint handling (parent B)
    # ------------------------------------------------------------------
    def _used_ram(self) -> int:
        """Current RAM consumption of the loaded pool."""
        return sum(m.ram_mb for m in self.loaded.values())

    def _check_constraints(self, model: ModelTier) -> None:
        """Raise if adding *model* would violate any pool constraint."""
        # Tier exclusivity: T3 cannot coexist with any T2
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.tier == "T2" and any(m.tier == "T3" for m in self.loaded.values()):
            raise RuntimeError("T2 mutually exclusive with T3 resident")
        # RAM ceiling
        if model.ram_mb + self._used_ram() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Safe cosine similarity for 1‑D vectors."""
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("Inputs must be 1‑D arrays")
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _model_feature(self, model: ModelTier) -> np.ndarray:
        """Compute stylometry vector from stored reference tokens."""
        text = " ".join(model.reference_tokens)
        return stylometry_features(text)

    # ------------------------------------------------------------------
    # Scoring – deep fusion via circulant matrix
    # ------------------------------------------------------------------
    def _similarity_vector(self, input_feat: np.ndarray) -> np.ndarray:
        """Vector of cosine similarities between *input_feat* and each loaded model."""
        sims = np.empty(len(self.loaded), dtype=float)
        for idx, model in enumerate(self.loaded.values()):
            model_feat = self._model_feature(model)
            sims[idx] = self._cosine_similarity(input_feat, model_feat)
        return sims

    def _weighted_scores(
        self, input_feat: np.ndarray, weekday: int | None = None
    ) -> Dict[str, float]:
        """
        Compute hybrid scores for all loaded models.
        The similarity vector is transformed by a weekday‑dependent stochastic
        circulant matrix, mixing information across the whole pool.
        Returns a mapping ``model_name → score``.
        """
        if not self.loaded:
            return {}

        pool_names = list(self.loaded.keys())
        pool_size = len(pool_names)

        # 1️⃣ similarity vector (s₁,…,s_N)
        sims = self._similarity_vector(input_feat)  # shape (N,)

        # 2️⃣ weekday‑dependent stochastic matrix W (N×N)
        W = weekday_weight_matrix(pool_size, weekday)  # each row sums to 1

        # 3️⃣ final scores: 𝑠̂ = W · s   (mixing across positions)
        mixed = W @ sims  # shape (N,)

        return {name: float(score) for name, score in zip(pool_names, mixed)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, model: ModelTier) -> None:
        """
        Load *model* into the pool after constraint validation.
        If the model already exists, its timestamp is refreshed.
        """
        if model.name in self.loaded:
            # Refresh timestamp – no constraint re‑check needed
            self.load_timestamp[model.name] = datetime.now().timestamp()
            return

        self._check_constraints(model)
        self.loaded[model.name] = model
        self.load_timestamp[model.name] = datetime.now().timestamp()

    def evict(self, name: str) -> None:
        """Remove the model identified by *name* from the pool."""
        if name in self.loaded:
            del self.loaded[name]
            del self.load_timestamp[name]

    def evict_lowest(self, input_text: str, weekday: int | None = None) -> Optional[str]:
        """
        Compute scores for the current pool given *input_text* and evict the
        model with the lowest score.  Returns the evicted model name or ``None``
        if the pool is empty.
        """
        if not self.loaded:
            return None
        input_feat = stylometry_features(input_text)
        scores = self._weighted_scores(input_feat, weekday)
        # Identify model with minimal score
        evict_name = min(scores, key=scores.get)
        self.evict(evict_name)
        return evict_name

    def recommend(self, input_text: str, weekday: int | None = None) -> Optional[str]:
        """
        Given *input_text*, return the name of the model with the highest hybrid
        score.  If the pool is empty, ``None`` is returned.
        """
        if not self.loaded:
            return None
        input_feat = stylometry_features(input_text)
        scores = self._weighted_scores(input_feat, weekday)
        return max(scores, key=scores.get)

    def current_state(self) -> Dict[str, Dict]:
        """
        Diagnostic snapshot of the pool: for each model we expose its tier,
        RAM usage and load timestamp.
        """
        state = {}
        for name, model in self.loaded.items():
            state[name] = {
                "tier": model.tier,
                "ram_mb": model.ram_mb,
                "loaded_at": self.load_timestamp.get(name, None),
            }
        return state

    # ------------------------------------------------------------------
    # Utility – for testing / debugging
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<HybridModelPool loaded={len(self.loaded)} "
            f"used_ram={self._used_ram()}/{self.ram_ceiling_mb} MB>"
        )