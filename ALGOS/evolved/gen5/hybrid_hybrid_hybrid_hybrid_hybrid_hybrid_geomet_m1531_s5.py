# DARWIN HAMMER — match 1531, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_fracti_m222_s3.py (gen4)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s1.py (gen4)
# born: 2026-05-29T23:37:09Z

"""Hybrid Algorithm: SSIM‑Guided Fractional‑Power Geometric Product (SFGP)

Parents
-------
- **Parent A** (`hybrid_hybrid_hybrid_ternar_hybrid_hybrid_fracti_m222_s3.py`):
  Generates a MinHash‑seeded complex hypervector from text, computes an
  SSIM‑like similarity *s* ∈ [0,1] between the routed response and the input,
  and raises the hypervector to the fractional power *s*.

- **Parent B** (`hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s1.py`):
  Implements a Test‑Time‑Training (TTT) linear model whose weight update is
  driven by an SSIM loss, and uses the geometric product (blade arithmetic) as
  the core algebraic operation.

Mathematical Bridge
-------------------
Both parents expose **SSIM** as a scalar that measures structural similarity.
In this fusion the SSIM score *s* plays a dual role:

1. **Fractional‑Power Binding** – the hypervector *v* is transformed by
   `v⁽ˢ⁾ = |v|ˢ · exp(i·arg(v)·s)`, embedding the similarity directly into the
   hypervector’s magnitude and phase.

2. **Gradient‑Scaling in TTT** – the weight update of the linear model is
   multiplied by *(1‑s)*, so higher similarity yields smaller corrective steps.

The bound hypervector is then combined with the model’s weight matrix via the
**geometric product** `G(v, W) = v·W + v∧W`, providing a unified representation
that carries both the encoded similarity (through the fractional power) and
the adaptive optimisation (through the TTT update).

The resulting pipeline:


text ──► minhash ──► seed ──► complex hv v
      │
      └─► route → response
            │
            └─► s = ssim(text, response)

v⁽ˢ⁾ = fractional_power(v, s)
ŷ   = geometric_product(v⁽ˢ⁾, W)          # model output
W ← ttt_update(W, v⁽ˢ⁾, ŷ, lr, s)        # adapt weights


The code below implements this unified system with three core functions and a
convenient `HybridModel` class."""


import sys
import math
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np


# ----------------------------------------------------------------------
# 1. SSIM‑like similarity (scalar ∈ [0,1])
# ----------------------------------------------------------------------
def ssim_scalar(a: np.ndarray, b: np.ndarray) -> float:
    """Return a simplified SSIM‑like similarity between two 1‑D arrays.

    The implementation follows the classic SSIM formula but works on
    flattened vectors.  The result is clipped to [0, 1].
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a2 = ((a - mu_a) ** 2).mean()
    sigma_b2 = ((b - mu_b) ** 2).mean()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()

    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2)
    ssim = numerator / denominator
    return float(np.clip(ssim, 0.0, 1.0))


# ----------------------------------------------------------------------
# 2. MinHash → seed → complex hypervector
# ----------------------------------------------------------------------
def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """Return a compact MinHash signature for *text*.

    The text is split into 5‑character shingles, each hashed with Python's
    built‑in hash.  For each of *k* buckets the minimum hash value is kept.
    """
    shingle_len = 5
    shingles = [text[i:i + shingle_len] for i in range(len(text) - shingle_len + 1)]
    if not shingles:
        shingles = [text]  # fallback for very short strings

    # initialise buckets with large positive numbers
    buckets = [2 ** 63 - 1] * k
    for sh in shingles:
        h = hash(sh)
        for i in range(k):
            # a simple mixing of bucket index into the hash
            mixed = (h ^ (i * 0x9e3779b97f4a7c15)) & ((1 << 63) - 1)
            if mixed < buckets[i]:
                buckets[i] = mixed
    return buckets


def complex_hypervector(signature: List[int], dim: int = 1024) -> np.ndarray:
    """Generate a complex hypervector seeded by *signature*.

    The signature is collapsed to a single integer seed via XOR folding.
    """
    seed = 0
    for v in signature:
        seed ^= v & 0xFFFFFFFFFFFFFFFF
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=dim)
    return np.exp(1j * theta)


# ----------------------------------------------------------------------
# 3. Fractional‑power binding
# ----------------------------------------------------------------------
def fractional_power(v: np.ndarray, power: float) -> np.ndarray:
    """Raise each complex component of *v* to the fractional *power*.

    For a complex number z = r·e^{iθ}, z^p = r^p·e^{i pθ}.
    """
    r = np.abs(v)
    theta = np.angle(v)
    r_pow = np.power(r, power, where=r > 0, out=np.ones_like(r))
    return r_pow * np.exp(1j * theta * power)


# ----------------------------------------------------------------------
# 4. Geometric product for vectors (scalar + bivector)
# ----------------------------------------------------------------------
def geometric_product(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the geometric product a·b + a∧b.

    Returns a tuple (scalar_part, bivector_part).  For real vectors the scalar
    part is the inner product and the bivector part is the outer (antisymmetric)
    component.
    """
    # scalar (inner) product
    scalar = np.dot(a.real, b.real) - np.dot(a.imag, b.imag)

    # bivector (wedge) – antisymmetric outer product flattened
    outer = np.outer(a, b) - np.outer(b, a)
    bivector = outer.reshape(-1)  # flatten to 1‑D for simplicity
    return scalar, bivector


# ----------------------------------------------------------------------
# 5. TTT‑style weight update driven by SSIM
# ----------------------------------------------------------------------
def ttt_update(
    W: np.ndarray,
    x: np.ndarray,
    y_pred: np.ndarray,
    lr: float,
    ssim: float,
) -> np.ndarray:
    """Perform one TTT update step.

    The gradient is (W·x - y_pred)·xᵀ and is scaled by (1‑ssim) so that
    higher similarity yields smaller updates.
    """
    # Ensure column vectors
    x_col = x.reshape(-1, 1)
    y_col = y_pred.reshape(-1, 1)

    pred = W @ x_col
    error = pred - y_col
    grad = error @ x_col.T
    scale = lr * (1.0 - ssim)
    return W - scale * grad


# ----------------------------------------------------------------------
# 6. High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_step(
    text: str,
    model: "HybridModel",
    response_generator: callable,
    lr: float = 0.001,
) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    """Execute one fusion step for *text*.

    Returns:
        bound_vector – the fractional‑power hypervector,
        ssim_score   – similarity used as bridge,
        (scalar, bivector) – geometric product of bound_vector with current weights.
    """
    # 1️⃣ route → response (placeholder)
    response = response_generator(text)

    # 2️⃣ similarity
    txt_vec = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(float)
    rsp_vec = np.frombuffer(response.encode("utf-8"), dtype=np.uint8).astype(float)
    s = ssim_scalar(txt_vec, rsp_vec)

    # 3️⃣ hypervector generation & fractional binding
    sig = minhash_for_text(text)
    hv = complex_hypervector(sig, dim=model.dim)
    bound = fractional_power(hv, s)

    # 4️⃣ geometric product with weight matrix (flattened)
    scalar, bivector = geometric_product(bound, model.W.flatten())

    # 5️⃣ weight update (using bound as input, scalar as pseudo‑target)
    model.W = ttt_update(model.W, bound, np.full_like(bound, scalar), lr, s)

    return bound, s, (scalar, bivector)


# ----------------------------------------------------------------------
# 7. Model container
# ----------------------------------------------------------------------
class HybridModel:
    """Container for the weight matrix and hypervector dimension."""

    def __init__(self, dim: int = 1024, seed: int = 0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        # Initialise a real‑valued weight matrix; shape (dim, dim)
        self.W = rng.standard_normal((dim, dim))

    def __repr__(self) -> str:
        return f"<HybridModel dim={self.dim}>"


# ----------------------------------------------------------------------
# 8. Minimal response generator (simulates the ternary router)
# ----------------------------------------------------------------------
def dummy_response_generator(text: str) -> str:
    """A deterministic stub that returns a transformed version of *text*."""
    # Simple reversible transformation: reverse words and toggle case
    words = text.split()
    transformed = [w.swapcase()[::-1] for w in words]
    return " ".join(transformed)


# ----------------------------------------------------------------------
# 9. Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "The quick brown fox jumps over the lazy dog."
    model = HybridModel(dim=512, seed=42)

    bound_vec, sim_score, (scalar_part, bivector_part) = hybrid_step(
        sample_text,
        model,
        dummy_response_generator,
        lr=0.005,
    )

    print("SSIM score:", sim_score)
    print("Bound vector shape:", bound_vec.shape)
    print("Geometric product scalar part:", scalar_part)
    print("Bivector part length:", bivector_part.shape[0])
    print("Updated weight matrix norm:", np.linalg.norm(model.W))