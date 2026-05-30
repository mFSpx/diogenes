# DARWIN HAMMER — match 1206, survivor 8
# gen: 4
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_ssim_doomsday_m214_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_infotaxis_min_m106_s1.py (gen3)
# born: 2026-05-29T23:34:36Z

import numpy as np
import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple, Sequence
from dataclasses import dataclass

def now_z() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _rel(path: pathlib.Path | str) -> str:
    try:
        root = pathlib.Path(__file__).resolve().parents[2]
        return str(pathlib.Path(path).resolve().relative_to(root))
    except Exception:
        return str(path)

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  

TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")

def generate_gpu_memory_signal(length: int = 256,
                               base_mb: int = 4096,
                               variance_mb: int = 512,
                               seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    sinusoid = np.sin(2 * math.pi * t / length)
    noise = rng.normal(0, variance_mb * 0.05, size=length)
    signal = base_mb + variance_mb * sinusoid + noise
    return signal.astype(np.float64)

def generate_periodic_signal(length: int = 256,
                             amplitude: float = 500.0,
                             frequency: float = 1.0,
                             phase: float = 0.0,
                             seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    signal = amplitude * np.sin(2 * math.pi * frequency * t / length + phase)
    jitter = rng.normal(0, amplitude * 0.01, size=length)
    return (signal + jitter).astype(np.float64)

def _ssim_component(x: np.ndarray, y: np.ndarray, C1: float, C2: float) -> Tuple[float, float, float]:
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    l = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    c = (2 * math.sqrt(sigma_x) * math.sqrt(sigma_y) + C2) / (sigma_x + sigma_y + C2)
    s = (sigma_xy + C2 / 2) / (math.sqrt(sigma_x) * math.sqrt(sigma_y) + C2 / 2)
    return l, c, s

def compute_ssim(x: np.ndarray, y: np.ndarray,
                 K1: float = 0.01, K2: float = 0.03,
                 L: float = 65535) -> float:
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    l, c, s = _ssim_component(x, y, C1, C2)
    return float(l * c * s)

def minhash_signature(x: np.ndarray, 
                      hash_count: int = 10, 
                      seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hash_values = np.empty((hash_count, len(x)), dtype=np.uint8)
    for i in range(hash_count):
        threshold = np.percentile(x, rng.uniform(0, 100, size=1)[0])
        hash_values[i] = (x >= threshold).astype(np.uint8)
    return np.mean(hash_values, axis=0).astype(np.uint8)

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.signatures: Dict[str, np.ndarray] = {}  

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _entropy(self) -> float:
        if not self.signatures:
            return 0.0
        counts = np.array([np.sum(sig) for sig in self.signatures.values()], dtype=float)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))

    def _hybrid_score(self,
                      gpu_signal: np.ndarray,
                      periodic_signal: np.ndarray,
                      model_sig: np.ndarray,
                      alpha: float = 0.5,
                      beta: float = 0.5) -> float:
        ssim_val = compute_ssim(gpu_signal, periodic_signal)
        intersect = np.sum(np.minimum(model_sig, minhash_signature(gpu_signal)))
        union = np.sum(np.maximum(model_sig, minhash_signature(gpu_signal)))
        jaccard_est = intersect / (union + 1e-12)

        H = self._entropy()
        Hmax = math.log2(len(self.signatures)) if self.signatures else 1.0
        w = H / (Hmax + 1e-12)  
        a = alpha * (1 - w)
        b = beta * w

        return a * ssim_val + b * jaccard_est

    def load(self,
             model: ModelTier,
             gpu_signal: np.ndarray,
             periodic_signal: np.ndarray) -> None:
        if model.name not in self.signatures:
            rng = np.random.default_rng(hash(model.name) & 0xffffffff)
            self.signatures[model.name] = rng.integers(0, 2, size=128, dtype=np.uint8)

        if model.ram_mb + self._used() <= self.ram_ceiling_mb:
            self.loaded[model.name] = model
            return

        scores = {
            name: self._hybrid_score(gpu_signal, periodic_signal, self.signatures[name])
            for name in self.loaded
        }
        evict_name = min(scores, key=scores.get)
        del self.loaded[evict_name]
        del self.signatures[evict_name]
        self.load(model, gpu_signal, periodic_signal)

    def info(self) -> str:
        lines = [
            f"RAM ceiling: {self.ram_ceiling_mb} MiB",
            f"Used RAM   : {self._used()} MiB",
            f"Loaded models: {len(self.loaded)}",
        ]
        for model_name, model_tier in self.loaded.items():
            lines.append(f"  - {model_name} ({model_tier.tier}, {model_tier.ram_mb} MiB)")
        return "\n".join(lines)

def main():
    gpu_signal = generate_gpu_memory_signal()
    periodic_signal = generate_periodic_signal()

    model_pool = ModelPool()
    model_pool.load(TIER_T1_QWEN_0_5B, gpu_signal, periodic_signal)
    model_pool.load(TIER_T2_REASONING, gpu_signal, periodic_signal)
    model_pool.load(TIER_T3_QWEN_7B, gpu_signal, periodic_signal)

    print(model_pool.info())

if __name__ == "__main__":
    main()