# DARWIN HAMMER — match 1142, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_bayes__m19_s0.py (gen4)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s1.py (gen2)
# born: 2026-05-29T23:33:02Z

"""
Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hard_t_hybrid_hybrid_bayes__m19_s0.py
- Parent B: hybrid_sparse_wta_hybrid_privacy_model_m29_s1.py

Mathematical Bridge:
Parent A creates a deterministic feature vector **f_det** from text and
adds a pseudo‑random perturbation **f_rand** that is seeded by a
deterministic SHA‑256 hash of the same text.  
Parent B evaluates a reconstruction‑risk score **R = u / N** (unique
quasi‑identifiers over total records) and applies a differentially
private aggregate **DP(R) = R + Laplace(0, 1/ε)**.

The hybrid algorithm fuses these structures by:
1. Producing a combined feature vector **f = f_det + f_rand** (addition of
   deterministic stylometry counts and hash‑seeded random noise).
2. Applying a Sparse Winner‑Take‑All (WTA) operator **W_k(f)** that keeps
   the top‑k components, yielding a sparse representation **s**.
3. Interpreting the sparsity (number of non‑zero components) as the
   number of unique quasi‑identifiers **u**, feeding it into the risk
   function **R(u, N)**, and finally into the DP aggregate **DP(R)**.
4. Using the DP‑protected risk to decide whether a model tier may be
   loaded into the pool, respecting RAM constraints and tier exclusivity.

Thus the governing equations of both parents are mathematically
integrated:

f = f_det(text) + f_rand(text, seed = H(text))
s = W_k(f)                              # sparse winner‑take‑all
u = ||s||_0                              # ℓ0‑norm = count non‑zero entries
R = reconstruction_risk_score(u, N)
DP_R = dp_aggregate([R], ε)

The resulting system provides robust, privacy‑aware feature handling
and resource‑constrained model management.
"""

import sys
import math
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Set, List, Iterable, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Stylometry feature extraction (deterministic + seeded noise)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, Set[str]] = {
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
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def _clean_token(tok: str) -> str:
    """Remove surrounding punctuation and lower‑case."""
    return tok.strip(PUNCT).lower()


def deterministic_hash(text: str) -> int:
    """SHA‑256 based deterministic integer hash (first 8 bytes)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def _deterministic_features(text: str) -> np.ndarray:
    """
    Compute deterministic stylometry proportions for each FUNCTION_CATS entry.
    Returns a vector of length len(FUNCTION_CATS) with values in [0,1].
    """
    tokens = [_clean_token(t) for t in text.split()]
    total = max(len(tokens), 1)
    proportions = []
    for cat_set in FUNCTION_CATS.values():
        count = sum(1 for t in tokens if t in cat_set)
        proportions.append(count / total)
    return np.array(proportions, dtype=np.float64)


def _seeded_noise(features_len: int, seed: int) -> np.ndarray:
    """
    Generate zero‑mean Gaussian noise seeded by the deterministic hash.
    """
    rng = random.Random(seed)
    # Use Python's random.gauss for reproducibility; convert to numpy array.
    noise = np.fromiter((rng.gauss(0.0, 1.0) for _ in range(features_len)), dtype=np.float64)
    return noise


def extract_features(text: str) -> np.ndarray:
    """
    Hybrid feature extraction:
        f = f_det + f_rand
    where f_det is deterministic stylometry and f_rand is hash‑seeded noise.
    """
    f_det = _deterministic_features(text)
    seed = deterministic_hash(text)
    f_rand = _seeded_noise(len(f_det), seed)
    return f_det + f_rand


# ----------------------------------------------------------------------
# Parent B – Sparse Winner‑Take‑All and Privacy‑Aware Model Pool
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g., "T1", "T2", "T3"


class ModelPool:
    """
    Manages a collection of ModelTier objects under RAM constraints and
    tier‑exclusivity rules.
    """

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model respecting exclusivity and RAM ceiling."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """
        Evict oldest loaded models until enough RAM is freed, then load.
        Eviction order follows insertion order (Python 3.7+ dict preserves order).
        """
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            oldest_key = next(iter(self.loaded))
            self.loaded.pop(oldest_key)
        self.load(model)


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk = u / N bounded to [0,1]; zero when no records."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """
    Differential‑privacy Laplace mechanism.
    Returns sum(values) + Laplace(0, sensitivity/ε).
    """
    total = sum(values)
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale)
    return total + noise


def sparse_wta(features: np.ndarray, k: int) -> np.ndarray:
    """
    Sparse Winner‑Take‑All: keep the top‑k largest (by magnitude) entries,
    zero out the rest. Returns a new array.
    """
    if k <= 0:
        return np.zeros_like(features)
    if k >= features.size:
        return features.copy()
    # Find threshold value: kth largest absolute value
    abs_vals = np.abs(features)
    threshold = np.partition(abs_vals, -k)[-k]
    mask = abs_vals >= threshold
    result = np.where(mask, features, 0.0)
    return result


def decide_and_load_model(
    pool: ModelPool,
    model: ModelTier,
    sparse_features: np.ndarray,
    total_records: int,
    epsilon: float = 1.0,
) -> bool:
    """
    Uses the sparsity of `sparse_features` as the number of unique quasi‑identifiers,
    computes a DP‑protected reconstruction risk, and loads the model if the
    protected risk is below a permissive threshold (0.6). Returns True on success.
    """
    u = int(np.count_nonzero(sparse_features))
    risk = reconstruction_risk_score(u, total_records)
    dp_risk = dp_aggregate([risk], epsilon=epsilon)

    # Threshold chosen to balance privacy and utility
    if dp_risk < 0.6:
        try:
            pool.load_with_eviction(model)
            return True
        except RuntimeError:
            return False
    return False


# ----------------------------------------------------------------------
# Additional Helper – Similarity via Sparse Features
# ----------------------------------------------------------------------
def sparse_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity on sparse vectors (zero entries ignored).
    Returns 0.0 for zero vectors.
    """
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "In the midst of uncertainty, the scientist carefully documented each observation. "
        "She noted the temperature, humidity, and the subtle changes in the specimen's behavior."
    )
    # 1. Feature extraction (deterministic + seeded noise)
    feats = extract_features(sample_text)
    print("Raw hybrid features:", feats)

    # 2. Sparse Winner‑Take‑All (keep top 4)
    sparse_feats = sparse_wta(feats, k=4)
    print("Sparse features (k=4):", sparse_feats)

    # 3. Model pool handling
    pool = ModelPool(ram_ceiling_mb=2000)
    model = ModelTier(name="EnvModel_v1", ram_mb=500, tier="T2")
    total_records = 1200  # hypothetical dataset size

    loaded = decide_and_load_model(pool, model, sparse_feats, total_records, epsilon=0.8)
    print(f"Model loaded: {loaded}")
    print("Current pool contents:", list(pool.loaded.keys()))

    # 4. Similarity demo between two texts
    other_text = "The researcher recorded the ambient conditions and the organism's response."
    other_feats = extract_features(other_text)
    other_sparse = sparse_wta(other_feats, k=4)
    sim = sparse_cosine_similarity(sparse_feats, other_sparse)
    print(f"Similarity between samples (sparse cosine): {sim:.4f}")

    sys.exit(0)