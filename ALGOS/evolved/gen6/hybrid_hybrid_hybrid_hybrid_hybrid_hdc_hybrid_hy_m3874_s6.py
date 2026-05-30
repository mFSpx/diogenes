# DARWIN HAMMER — match 3874, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s3.py (gen4)
# parent_b: hybrid_hdc_hybrid_hybrid_ternar_m418_s1.py (gen5)
# born: 2026-05-29T23:52:19Z

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

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.load_time: Dict[str, datetime] = {}
        self.vector_cache: Dict[str, np.ndarray] = {}

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
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
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

HD_DIM = 10000
TERNARY_DIMS = 12

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def payload_hash(raw_command: str, normalized_intent: str, context: dict) -> str:
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
    seed = int.from_bytes(hashlib.sha256(symbol.encode('utf-8')).digest()[:8], 'big')
    return random_bipolar_vector(dim, seed)

def ternary_pattern(dim: int = HD_DIM, seed: int | str | None = None) -> np.ndarray:
    rng = random.Random(seed)
    base = np.array([rng.choice([-1, 0, 1]) for _ in range(TERNARY_DIMS)], dtype=np.int8)
    repeats = dim // TERNARY_DIMS + 1
    tiled = np.tile(base, repeats)[:dim]
    return tiled

def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape for binding")
    return a * b

def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("At least one vector required for bundling")
    stacked = np.stack(vectors, axis=0)
    sums = np.sum(stacked, axis=0)
    return np.where(sums >= 0, 1, -1).astype(np.int8)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Shape mismatch for similarity")
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm != 0 else 0.0

def hybrid_mask_from_signature(sig: List[int], dim: int = HD_DIM) -> np.ndarray:
    mask_vals = np.array([1 if (h % 2 == 0) else -1 for h in sig], dtype=np.int8)
    repeats = dim // len(mask_vals) + 1
    mask = np.tile(mask_vals, repeats)[:dim]
    return mask

def hybrid_vector(tokens: Iterable[str], symbol: str, pool: ModelPool, dim: int = HD_DIM) -> np.ndarray:
    cache_key = f"{symbol}:{','.join(tokens)}"
    if cache_key in pool.vector_cache:
        return pool.vector_cache[cache_key]
    
    sig = minhash_signature(tokens, k=dim // 64)
    mask = hybrid_mask_from_signature(sig, dim=dim)

    bipolar = symbol_vector(symbol, dim=dim)
    ternary = ternary_pattern(dim=dim, seed=symbol)

    bound = bind(bipolar, ternary)
    hybrid = bound * mask
    pool.vector_cache[cache_key] = hybrid.astype(np.int8)
    return hybrid

def hybrid_similarity(tokens_a: Iterable[str], symbol_a: str, tokens_b: Iterable[str], symbol_b: str, pool: ModelPool, dim: int = HD_DIM) -> float:
    vec_a = hybrid_vector(tokens_a, symbol_a, pool, dim=dim)
    vec_b = hybrid_vector(tokens_b, symbol_b, pool, dim=dim)
    return cosine_similarity(vec_a, vec_b)

def hybrid_risk_score(actions: List[Dict], pool: ModelPool) -> float:
    vectors = []
    for action in actions:
        tokens = action["tokens"]
        symbol = action["symbol"]
        vec = hybrid_vector(tokens, symbol, pool)
        vectors.append(vec)
    bundle_vec = bundle(vectors)
    similarities = [cosine_similarity(vec, bundle_vec) for vec in vectors]
    return 1 - sum(similarities) / len(similarities)