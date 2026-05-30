# DARWIN HAMMER — match 8, survivor 5
# gen: 1
# parent_a: hard_truth_math.py (gen0)
# parent_b: model_pool.py (gen0)
# born: 2026-05-29T23:22:33Z

import datetime as dt
import hashlib
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry / LSM utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
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


def words(text: str) -> List[str]:
    """Return a list of lowercase words (ASCII letters + optional apostrophe)."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> Dict[str, float]:
    """Return a normalized frequency vector for each FUNCTION_CATS."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stable_hash(text: str) -> int:
    """Deterministic hash used for trigram indexing."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """Produce a 96‑dimensional stylometric fingerprint."""
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))
    vals: List[float] = [
        total_words / 500.0,  # normalized word count
        sum(len(w) for w in ws) / total_words / 12.0,  # avg word length
        (text.count("\n") + 1) / 200.0,  # line density
        sum(text.count(p) for p in "!?") / total_chars,  # exclamation/question density
        sum(text.count(p) for p in ";:") / total_chars,  # semicolon/colon density
        sum(text.count(p) for p in "-—") / total_chars,  # dash density
        sum(1 for ch in text if ch.isupper()) / total_chars,  # uppercase ratio
        len(re.findall(r"\b[A-Z]{2,}\b", text)) / total_words,  # acronym density
    ]
    # Append LSM categories (will be re‑normalized later)
    lv = lsm_vector(text)
    vals.extend(lv.get(cat, 0.0) for cat in sorted(FUNCTION_CATS))

    vec = np.zeros(dim, dtype=np.float64)
    base = np.array(vals[: min(len(vals), dim)], dtype=np.float64)
    vec[: len(base)] = base

    # Add trigram counts to the tail (indices >= 20)
    cleaned = " ".join(ws)
    for i in range(max(0, len(cleaned) - 2)):
        gram = cleaned[i : i + 3]
        idx = 20 + (stable_hash(gram) % max(1, dim - 20))
        vec[idx] += 1.0

    # Normalise the high‑dimensional tail (indices >=20) to unit length
    tail_norm = np.linalg.norm(vec[20:])
    if tail_norm > 0:
        vec[20:] /= tail_norm
    return vec


# ----------------------------------------------------------------------
# Parent B – model pool utilities
# ----------------------------------------------------------------------
class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded due to RAM or tier constraints."""


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
        # OrderedDict preserves insertion order for FIFO eviction
        self.loaded: "OrderedDict[str, ModelTier]" = OrderedDict()

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model if constraints allow; otherwise raise ModelLoadError."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise ModelLoadError("T3 models are mutually exclusive with any T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise ModelLoadError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """Load a model, evicting the oldest entries until constraints are satisfied."""
        while (
            model.ram_mb + self._used() > self.ram_ceiling_mb
            or (model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()))
        ):
            if not self.loaded:
                raise ModelLoadError("Cannot satisfy constraints even after full eviction")
            evicted_name, _ = self.loaded.popitem(last=False)  # FIFO eviction
        self.load(model)


# ----------------------------------------------------------------------
# Hybrid mathematics – deeper integration
# ----------------------------------------------------------------------
_TIER_TO_NUM = {"T1": 1, "T2": 2, "T3": 3}


def hybrid_vector(text: str) -> np.ndarray:
    """
    Combine stylometry (96‑dim) with the LSM categorical profile.
    The resulting vector has length 96 + |FUNCTION_CATS| and is L2‑normalised.
    """
    sty = stylometry_features(text)  # ℝ⁹⁶
    lsm = np.array(
        [lsm_vector(text).get(c, 0.0) for c in sorted(FUNCTION_CATS)], dtype=np.float64
    )
    # Normalise LSM part to unit length (preserves relative importance)
    lsm_norm = np.linalg.norm(lsm)
    if lsm_norm > 0:
        lsm /= lsm_norm
    combined = np.concatenate([sty, lsm])
    # Global normalisation makes the vector comparable to model embeddings
    total_norm = np.linalg.norm(combined)
    if total_norm > 0:
        combined /= total_norm
    return combined


def model_characteristic_vector(model: ModelTier) -> np.ndarray:
    """
    Encode a model as a dense vector of the same dimensionality as `hybrid_vector`.
    The first 96 entries encode scaled RAM and tier information; the remaining
    entries embed the same RAM/tier signal uniformly across the LSM dimensions.
    This creates a *shared* representation space, allowing the full text fingerprint
    to interact with model resources.
    """
    # Scale RAM to GB and map tier to a small numeric factor
    ram_gb = model.ram_mb / 1024.0
    tier_num = _TIER_TO_NUM.get(model.tier, 0)

    # First 96 dims: two informative slots, rest zero‑filled
    model_vec = np.zeros(96 + len(FUNCTION_CATS), dtype=np.float64)
    model_vec[0] = ram_gb / 16.0  # normalise assuming <16 GB typical ceiling
    model_vec[1] = tier_num / 3.0  # normalise tier to [0,1]

    # Distribute the same scaled signal across the LSM tail
    lsm_signal = (ram_gb / 16.0 + tier_num / 3.0) / 2.0
    model_vec[96:] = lsm_signal
    # Normalise to unit length for cosine similarity
    norm = np.linalg.norm(model_vec)
    if norm > 0:
        model_vec /= norm
    return model_vec


def compatibility_score(text: str, model: ModelTier) -> float:
    """
    Deep compatibility metric: cosine similarity between the full hybrid
    text vector and the model's characteristic vector.
    Returns a value in [-1, 1]; higher is better.
    """
    v = hybrid_vector(text)
    m = model_characteristic_vector(model)
    # Cosine similarity = dot product because both vectors are L2‑normalised
    return float(np.dot(v, m))


def rank_models(text: str, candidates: List[ModelTier]) -> List[Tuple[ModelTier, float]]:
    """
    Return candidates sorted by descending compatibility_score.
    """
    scored = [(model, compatibility_score(text, model)) for model in candidates]
    scored.sort(key=lambda x: -x[1])
    return scored


def allocate_models(
    text: str,
    pool: ModelPool,
    candidates: List[ModelTier],
) -> List[ModelTier]:
    """
    Attempt to load the highest‑scoring models into `pool` while respecting
    RAM and tier constraints. Models that cannot be loaded after eviction are
    skipped. The function returns the list of models that ended up loaded
    (preserving the order of successful loading).
    """
    loaded_models: List[ModelTier] = []
    for model, _ in rank_models(text, candidates):
        if pool.is_loaded(model.name):
            # Already present – keep it and move on
            loaded_models.append(model)
            continue
        try:
            pool.load_with_eviction(model)
            loaded_models.append(model)
        except ModelLoadError:
            # If a single model cannot be accommodated even after full eviction,
            # we simply skip it and continue with the next candidate.
            continue
    return loaded_models

# End of improved module.