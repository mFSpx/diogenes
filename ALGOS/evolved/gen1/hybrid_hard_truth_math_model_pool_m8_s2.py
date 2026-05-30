# DARWIN HAMMER — match 8, survivor 2
# gen: 1
# parent_a: hard_truth_math.py (gen0)
# parent_b: model_pool.py (gen0)
# born: 2026-05-29T23:22:33Z

"""Hybrid module merging hard_truth_math.py (A) and model_pool.py (B).

Mathematical bridge:
- A produces high‑dimensional numeric representations of text:
    * `stylometry_features` → ℝ⁹⁶
    * `lsm_vector`   → ℝ^|FUNCTION_CATS|
- B describes models by a low‑dimensional resource vector:
    * RAM (scaled) and tier (encoded as an integer).
- The hybrid treats a text‑derived feature vector **v** ∈ ℝⁿ and a model‑resource vector **m** ∈ ℝ².
  Their compatibility is measured by a bilinear form **s = vᵀ P m**, where **P** projects the first two
  dimensions of **v** (mean stylometry and total word‑ratio) onto the model space.
  This single scalar **s** drives model‑selection under RAM and tier constraints.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry / LSM utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
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
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> Dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))
    vals: List[float] = [
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
        idx = 20 + (stable_hash(gram) % max(1, dim - 20))
        vec[idx] += 1.0
    norm = np.linalg.norm(vec[20:])
    if norm > 0:
        vec[20:] /= norm
    return vec


# ----------------------------------------------------------------------
# Parent B – model pool utilities
# ----------------------------------------------------------------------
class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded due to RAM or tier constraints."""
    pass


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g., "T1", "T2", "T3"


# Example tier constants (could be extended by users)
TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")


class ModelPool:
    """Manages a collection of loaded models respecting RAM ceiling and tier exclusivity."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise ModelLoadError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise ModelLoadError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # Evict the oldest entry (simple FIFO)
            evicted_name = next(iter(self.loaded))
            del self.loaded[evicted_name]
        self.load(model)


# ----------------------------------------------------------------------
# Hybrid mathematics
# ----------------------------------------------------------------------
_TIER_TO_NUM = {"T1": 1, "T2": 2, "T3": 3}


def hybrid_vector(text: str) -> np.ndarray:
    """
    Combine stylometry (96‑dim) with the LSM categorical profile.
    The resulting vector has length 96 + |FUNCTION_CATS|, normalized
    on the LSM tail to keep scales compatible.
    """
    sty = stylometry_features(text)                     # ℝ⁹⁶
    lsm = np.array([lsm_vector(text).get(c, 0.0) for c in sorted(FUNCTION_CATS)], dtype=np.float64)
    # Normalize LSM part to unit length
    norm = np.linalg.norm(lsm)
    if norm > 0:
        lsm /= norm
    return np.concatenate([sty, lsm])                   # ℝ^(96+|cats|)


def model_characteristic_vector(model: ModelTier) -> np.ndarray:
    """
    Encode a model as a 2‑dim resource vector:
        [scaled RAM (GB), tier number]
    Scaling RAM by 1 GB keeps numbers order‑of‑magnitude similar to stylometry.
    """
    ram_gb = model.ram_mb / 1024.0
    tier_num = _TIER_TO_NUM.get(model.tier, 0)
    return np.array([ram_gb, tier_num], dtype=np.float64)


def compatibility_score(text: str, model: ModelTier) -> float:
    """
    Bilinear compatibility: s = vᵀ P m
    where v = hybrid_vector(text), m = model_characteristic_vector(model)
    and P extracts the first two dimensions of v (mean‑stylometry and word‑ratio)
    and rescales them to comparable magnitude.
    """
    v = hybrid_vector(text)
    # Simple projection matrix P (2×n) that picks first two entries and applies a small scaling
    P = np.zeros((2, v.shape[0]), dtype=np.float64)
    P[0, 0] = 0.5   # weight for total_words/500
    P[1, 1] = 0.5   # weight for avg_word_len/12
    m = model_characteristic_vector(model)
    score = float(v @ P.T @ m)  # scalar
    return score


def rank_models(text: str, candidates: List[ModelTier]) -> List[Tuple[ModelTier, float]]:
    """
    Return candidates sorted by descending compatibility_score.
    """
    scored = [(model, compatibility_score(text, model)) for model in candidates]
    scored.sort(key=lambda x: -x[1])
    return scored


def allocate_models(text: str, pool: ModelPool, candidates: List[ModelTier]) -> List[ModelTier]:
    """
    Load the best‑scoring models into the pool respecting RAM and tier rules.
    Evicts older models if necessary.
    Returns the list of successfully loaded models.
    """
    loaded: List[ModelTier] = []
    for model, _ in rank_models(text, candidates):
        try:
            pool.load_with_eviction(model)
            loaded.append(model)
        except ModelLoadError:
            # Incompatible model – skip it
            continue
    return loaded


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In a world of AI, models compete for RAM and tier privileges."
    )
    pool = ModelPool(ram_ceiling_mb=8000)
    candidates = [
        TIER_T1_QWEN_0_5B,
        TIER_T2_REASONING,
        TIER_T2_TOOL,
        TIER_T3_QWEN_7B,
    ]
    loaded_models = allocate_models(sample_text, pool, candidates)
    print("Loaded models:")
    for m in loaded_models:
        print(f"- {m.name} (RAM {m.ram_mb} MB, tier {m.tier})")
    print("\nCurrent RAM usage:", pool._used(), "MB")
    # Verify that the hybrid vector can be computed without error
    hv = hybrid_vector(sample_text)
    print("\nHybrid vector shape:", hv.shape)