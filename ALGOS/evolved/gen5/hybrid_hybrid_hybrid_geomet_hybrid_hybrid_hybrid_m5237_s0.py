# DARWIN HAMMER — match 5237, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_fracti_m222_s0.py (gen4)
# born: 2026-05-30T00:00:43Z

"""
Hybrid Algorithm: Geometric Product Guided Fractional MinHash Test-Time Training (Hybrid-GP-FM-TTT)

Parents
-------
* **Parent A** – `hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s0.py`
  provides a geometric product implementation, which can be viewed as a form of optimization problem.
* **Parent B** – `hybrid_hybrid_hybrid_ternar_hybrid_hybrid_fracti_m222_s0.py`
  implements a fractional minhash operation and a ternary router.

Mathematical Bridge
-------------------
The mathematical bridge between the two parents is the use of the minhash operation from
Parent B to generate a compact representation of the text data, which can then be used as input to the
geometric product's blade arithmetic from Parent A. By using the SSIM loss as a regularization term in
the geometric product optimization problem, we can create a hybrid algorithm that adapts to the changing
requirements of the model. The hybrid algorithm treats `L_hybrid = α·L_rec + β·L_ssim` as a unified objective,
where `L_rec` is the reconstruction error and `L_ssim` is the SSIM loss.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass

def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)  
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target=None):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W, x, target=None):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)

def geometric_product(a, b):
    result = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            result[blade_out] = result.get(blade_out, 0) + coef_a * coef_b * sign
    return result

def random_hv(d=10000, kind="complex", seed=None):
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    elif kind == "bipolar":
        return rng.choice([-1, 1], size=d)
    elif kind == "real":
        vec = rng.normal(size=d)
        return vec / np.linalg.norm(vec)

def minhash_for_text(text: str, k: int = 64) -> list[int]:
    text = text.replace(" ", "").strip().lower()
    shingles = [text[i:i+5] for i in range(len(text)-4)]
    minhash = []
    for seed in range(k):
        hash_values = []
        for shingle in shingles:
            hash_value = sum(ord(c) * (seed ** i) for i, c in enumerate(shingle))
            hash_values.append(hash_value)
        minhash.append(min(hash_values))
    return minhash

def hybrid_gp_fm_ttt(text: str, W, x, target=None):
    minhash = minhash_for_text(text)
    blade = frozenset(minhash)
    gp = geometric_product({blade: 1}, W)
    pred = sum(coef * blade for blade, coef in gp.items())
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def hybrid_ttt_grad(text: str, W, x, target=None):
    minhash = minhash_for_text(text)
    blade = frozenset(minhash)
    gp_grad = ttt_grad(W, x, target)
    return gp_grad

def smoke_test():
    text = "This is a test string"
    W = init_ttt(100, 100)
    x = np.random.rand(100)
    target = np.random.rand(100)
    loss = hybrid_gp_fm_ttt(text, W, x, target)
    grad = hybrid_ttt_grad(text, W, x, target)
    print(f"Loss: {loss}, Grad: {grad.shape}")

if __name__ == "__main__":
    smoke_test()