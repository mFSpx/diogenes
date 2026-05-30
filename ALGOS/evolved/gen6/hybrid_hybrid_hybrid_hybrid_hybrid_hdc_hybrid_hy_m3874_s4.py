# DARWIN HAMMER — match 3874, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s3.py (gen4)
# parent_b: hybrid_hdc_hybrid_hybrid_ternar_m418_s1.py (gen5)
# born: 2026-05-29T23:52:19Z

"""
Hybrid Algorithm Fusion of:
- Parent A (hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s3.py): MinHash signatures, ModelPool resource management.
- Parent B (hybrid_hdc_hybrid_hybrid_ternar_m418_s1.py): Hyperdimensional bipolar vectors, ternary vectors, binding/bundling operations.

Mathematical Bridge:
The MinHash signature (a deterministic integer array) is used as a binary mask that modulates the element‑wise binding of a bipolar hyperdimensional vector with a ternary vector.  Each signature entry determines whether the corresponding dimension of the bound vector is kept or flipped, thus fusing the set‑similarity space of Parent A with the high‑dimensional algebra of Parent B.
"""

import sys
import math
import random
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Iterable, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (trimmed)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

class ModelPool:
    """Simple LRU‑style pool respecting a RAM ceiling."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.load_time: Dict[str, datetime] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model
        self.load_time[model.name] = datetime.now(timezone.utc)

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            lru = min(self.loaded, key=lambda n: self.load_time[n])
            del self.loaded[lru]
            del self.load_time[lru]
        self.load(model)

def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Classic MinHash signature (k integer minima)."""
    seeds = list(range(k))
    sig = []
    for seed in seeds:
        min_h = sys.maxsize
        for token in tokens:
            h = _hash(seed, token)
            if h < min_h:
                min_h = h
        sig.append(min_h)
    return sig

# ----------------------------------------------------------------------
# Parent B components (trimmed)
# ----------------------------------------------------------------------
HD_DIM = 10000          # dimensionality of bipolar vectors
TERNARY_DIMS = 12       # base dimensionality for ternary patterns

def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def payload_hash(raw_command: str, normalized_intent: str, context: dict) -> str:
    """Deterministic SHA‑256 of a command envelope."""
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

def random_bipolar_vector(dim: int = HD_DIM, seed: int | str | None = None) -> np.ndarray:
    rng = random.Random(seed)
    return np.array([1 if rng.getrandbits(1) else -1 for _ in range(dim)], dtype=np.int8)

def symbol_vector(symbol: str, dim: int = HD_DIM) -> np.ndarray:
    """Deterministic bipolar vector derived from a symbol string."""
    seed = int.from_bytes(hashlib.sha256(symbol.encode('utf-8')).digest()[:8], 'big')
    return random_bipolar_vector(dim, seed)

def ternary_pattern(dim: int = HD_DIM, seed: int | str | None = None) -> np.ndarray:
    """Generate a ternary vector (values ∈ {‑1,0,1}) by repeating a short pattern."""
    rng = random.Random(seed)
    base = np.array([rng.choice([-1, 0, 1]) for _ in range(TERNARY_DIMS)], dtype=np.int8)
    repeats = dim // TERNARY_DIMS + 1
    tiled = np.tile(base, repeats)[:dim]
    return tiled

def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element‑wise multiplication (binding) of two equal‑length vectors."""
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape for binding")
    return a * b

def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """Superposition (bundling) via majority vote."""
    if not vectors:
        raise ValueError("At least one vector required for bundling")
    stacked = np.stack(vectors, axis=0)
    sums = np.sum(stacked, axis=0)
    return np.where(sums >= 0, 1, -1).astype(np.int8)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized dot product (range [-1, 1])."""
    if a.shape != b.shape:
        raise ValueError("Shape mismatch for similarity")
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm != 0 else 0.0

# ----------------------------------------------------------------------
# Hybrid Functions (the fusion)
# ----------------------------------------------------------------------
def hybrid_mask_from_signature(sig: List[int], dim: int = HD_DIM) -> np.ndarray:
    """
    Convert a MinHash signature into a binary mask of length `dim`.
    Even hash values → 1 (keep), odd → -1 (flip).
    The signature is truncated or repeated to match `dim`.
    """
    mask_vals = np.array([1 if (h % 2 == 0) else -1 for h in sig], dtype=np.int8)
    repeats = dim // len(mask_vals) + 1
    mask = np.tile(mask_vals, repeats)[:dim]
    return mask

def hybrid_vector(tokens: Iterable[str], symbol: str, dim: int = HD_DIM) -> np.ndarray:
    """
    Core hybrid operation:
    1. Compute a MinHash signature from `tokens`.
    2. Build a bipolar hyperdimensional vector for `symbol`.
    3. Build a ternary pattern vector.
    4. Bind bipolar and ternary vectors.
    5. Modulate the bound vector with the mask derived from the signature.
    """
    sig = minhash_signature(tokens, k=dim // 64)          # modest k to keep runtime sane
    mask = hybrid_mask_from_signature(sig, dim=dim)

    bipolar = symbol_vector(symbol, dim=dim)
    ternary = ternary_pattern(dim=dim, seed=symbol)      # deterministic seed

    bound = bind(bipolar, ternary)
    hybrid = bound * mask                                   # element‑wise modulation
    return hybrid.astype(np.int8)

def hybrid_similarity(tokens_a: Iterable[str], symbol_a: str,
                      tokens_b: Iterable[str], symbol_b: str,
                      dim: int = HD_DIM) -> float:
    """
    Compute similarity between two hybrid vectors constructed from
    (tokens, symbol) pairs.  The result is a cosine similarity in [-1, 1].
    """
    vec_a = hybrid_vector(tokens_a, symbol_a, dim=dim)
    vec_b = hybrid_vector(tokens_b, symbol_b, dim=dim)
    return cosine_similarity(vec_a, vec_b)

def hybrid_risk_score(actions: List[Dict], pool: ModelPool) -> float:
    """
    Example higher‑level operation:
    - Each action dict must contain `tokens` (list of str) and `symbol` (str).
    - The function builds hybrid vectors for all actions, bundles them,
      and returns a risk metric defined as 1 − average similarity to the bundle.
    - Vectors are cached in the ModelPool to avoid recomputation.
    """
    dim = HD_DIM
    vectors = []
    for idx, act in enumerate(actions):
        key = f"hybrid_{idx}"
        if pool.is_loaded(key):
            vec = pool.loaded[key]  # type: ignore
        else:
            vec = hybrid_vector(act["tokens"], act["symbol"], dim=dim)
            # Store a lightweight ModelTier placeholder; RAM usage is mocked.
            pool.load(ModelTier(name=key, ram_mb=dim // 8000, tier="T1"))
            pool.loaded[key] = vec  # overwrite with actual ndarray for demo
        vectors.append(vec)

    bundle_vec = bundle(vectors)
    sims = [cosine_similarity(v, bundle_vec) for v in vectors]
    avg_sim = sum(sims) / len(sims) if sims else 0.0
    risk = 1.0 - avg_sim  # higher similarity → lower risk
    return risk

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal demonstration that exercises all hybrid functions.
    tokens1 = ["alpha", "beta", "gamma"]
    tokens2 = ["delta", "epsilon", "zeta"]
    symbol1 = "sensor_A"
    symbol2 = "sensor_B"

    vec1 = hybrid_vector(tokens1, symbol1)
    vec2 = hybrid_vector(tokens2, symbol2)

    print("Hybrid vector 1 norm:", np.linalg.norm(vec1))
    print("Hybrid vector 2 norm:", np.linalg.norm(vec2))

    sim = hybrid_similarity(tokens1, symbol1, tokens2, symbol2)
    print("Hybrid similarity between A and B:", sim)

    # Risk scoring demo
    pool = ModelPool(ram_ceiling_mb=200)  # small ceiling for test
    actions = [
        {"tokens": tokens1, "symbol": symbol1},
        {"tokens": tokens2, "symbol": symbol2},
        {"tokens": ["theta", "iota"], "symbol": "sensor_C"},
    ]
    risk = hybrid_risk_score(actions, pool)
    print("Computed hybrid risk score:", risk)