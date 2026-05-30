# DARWIN HAMMER — match 4109, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_ternar_m1215_s3.py (gen4)
# parent_b: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hybrid_m1904_s3.py (gen5)
# born: 2026-05-29T23:53:43Z

import math
import random
import sys
import pathlib
import numpy as np

class Multivector:
    def __init__(self, scalar, vector, bivector):
        self.scalar = scalar
        self.vector = vector
        self.bivector = bivector

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def minhash_signature(tokens: list, k: int = 128) -> list:
    def _hash(seed: int, token: str) -> int:
        data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
        import hashlib
        return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")
    return [_hash(i, token) for i, token in enumerate(tokens[:k])]

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     np.zeros(d)])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], np.zeros(d)])
    return out

def compute_ssim(
    x: list,
    y: list,
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    vxy = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    ssim = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return ssim

def hybrid_signature(tokens: list, path: np.ndarray) -> Multivector:
    minhash_sig = minhash_signature(tokens)
    lead_lag_path = lead_lag_transform(path)
    sim = compute_ssim(minhash_sig, lead_lag_path.mean(axis=1).tolist())
    scalar = sim
    vector = lead_lag_path.mean(axis=0)
    bivector = np.cov(lead_lag_path.T)
    return Multivector(scalar, vector, bivector)

def hybrid_score(theta: float, center: float, width: float, tokens: list, path: np.ndarray) -> float:
    fisher_inf = fisher_score(theta, center, width)
    hybrid_sig = hybrid_signature(tokens, path)
    return fisher_inf * hybrid_sig.scalar * (1 + np.linalg.norm(hybrid_sig.vector))

def hybrid_transform(path: np.ndarray, tokens: list) -> np.ndarray:
    lead_lag_path = lead_lag_transform(path)
    minhash_sig = minhash_signature(tokens)
    return np.concatenate([lead_lag_path, np.array(minhash_sig).reshape(-1, 1)], axis=1)

def improved_hybrid_signature(tokens: list, path: np.ndarray) -> Multivector:
    minhash_sig = minhash_signature(tokens)
    lead_lag_path = lead_lag_transform(path)
    sim = compute_ssim(minhash_sig, lead_lag_path.mean(axis=1).tolist())
    scalar = sim
    vector = lead_lag_path.mean(axis=0)
    bivector = np.cov(lead_lag_path.T)
    return Multivector(scalar, vector, bivector)

def improved_hybrid_score(theta: float, center: float, width: float, tokens: list, path: np.ndarray) -> float:
    fisher_inf = fisher_score(theta, center, width)
    hybrid_sig = improved_hybrid_signature(tokens, path)
    return fisher_inf * hybrid_sig.scalar * (1 + np.linalg.norm(hybrid_sig.vector))

def improved_hybrid_transform(path: np.ndarray, tokens: list) -> np.ndarray:
    lead_lag_path = lead_lag_transform(path)
    minhash_sig = minhash_signature(tokens)
    return np.concatenate([lead_lag_path, np.array(minhash_sig).reshape(-1, 1)], axis=1)

if __name__ == "__main__":
    tokens = ["token1", "token2", "token3"]
    path = np.random.rand(10, 5)
    hybrid_sig = hybrid_signature(tokens, path)
    hybrid_sc = hybrid_score(0.5, 0.0, 1.0, tokens, path)
    hybrid_tra = hybrid_transform(path, tokens)
    improved_hybrid_sig = improved_hybrid_signature(tokens, path)
    improved_hybrid_sc = improved_hybrid_score(0.5, 0.0, 1.0, tokens, path)
    improved_hybrid_tra = improved_hybrid_transform(path, tokens)
    print("Hybrid signature scalar:", hybrid_sig.scalar)
    print("Hybrid score:", hybrid_sc)
    print("Hybrid transformation shape:", hybrid_tra.shape)
    print("Improved Hybrid signature scalar:", improved_hybrid_sig.scalar)
    print("Improved Hybrid score:", improved_hybrid_sc)
    print("Improved Hybrid transformation shape:", improved_hybrid_tra.shape)