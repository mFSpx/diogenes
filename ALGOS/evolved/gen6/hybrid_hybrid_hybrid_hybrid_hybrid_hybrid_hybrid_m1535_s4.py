# DARWIN HAMMER — match 1535, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m416_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_sparse_hybrid_fisher_locali_m406_s4.py (gen4)
# born: 2026-05-29T23:37:20Z

"""Hybrid Sparse‑WTA / Tropical‑Fisher Engine Selector

Parents
-------
* **Parent A** – Hybrid Endpoint‑Morphology & Tropical Hoefding Split.
  Provides a tropical (max‑plus) network, morphology‑based engine descriptors,
  and a structural‑similarity (SSIM) metric.

* **Parent B** – Hybrid Sparse‑WTA / Fisher‑Weighted SSIM.
  Supplies a hash‑based sparse expansion, a confidence scalar, Gaussian‑beam
  weighting, Fisher‑information weighting and a weighted‑SSIM routine.

Mathematical Bridge
-------------------
Both parents expose a *scalar confidence* `c` derived from the input signal.
In the Sparse‑WTA side `c` rescales random coefficients; in the Fisher side it
parameterises a Gaussian‑beam weight vector `w(θ)`.  The Tropical network of
Parent A consumes a weight matrix `W`.  By constructing `W` as the outer product
of the Gaussian‑beam vector with the Fisher‑information scalar, we embed the
information‑theoretic weighting directly into the tropical (max‑plus) evaluation.
The resulting tropical output is then compared to the original sparse signal
with a **Fisher‑weighted SSIM**.  The confidence also modulates the engine‑
selection priority derived from morphology.

The module implements three core hybrid functions that realise this bridge
and a smoke‑test at the end.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

# ----------------------------------------------------------------------
# Tropical (max‑plus) network – Parent A
# ----------------------------------------------------------------------
class TropicalNetwork:
    """Max‑plus linear layer: y_i = max_j (W_ij + x_j) + b_i"""
    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        self.weights = weights          # shape (out_dim, in_dim)
        self.biases = biases            # shape (out_dim,)

    def evaluate(self, input_vector: np.ndarray) -> np.ndarray:
        """Apply max‑plus affine transformation."""
        # broadcasting: (out_dim, in_dim) + (in_dim,) -> (out_dim, in_dim)
        affine = self.weights + input_vector   # max‑plus uses addition
        max_part = np.max(affine, axis=1)      # max over inputs per output
        return max_part + self.biases

# ----------------------------------------------------------------------
# SSIM utilities – simplified version (Parent A)
# ----------------------------------------------------------------------
def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))

def _var(x: np.ndarray) -> float:
    return float(np.var(x))

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Classic SSIM (luminance‑contrast‑structure) for 1‑D signals."""
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    mu_x = _mean(x)
    mu_y = _mean(y)
    sigma_x2 = _var(x)
    sigma_y2 = _var(y)
    sigma_xy = float(np.mean((x - mu_x) * (y - mu_y)))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return numerator / denominator if denominator != 0 else 0.0

# ----------------------------------------------------------------------
# Sparse‑WTA utilities – Parent B
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = "") -> np.ndarray:
    """Hash‑based sparse expansion of ``values`` into a length‑``m`` vector."""
    rng = np.random.default_rng(seed=hash(salt) & 0xffffffff)
    out = np.zeros(m, dtype=float)
    for v in values:
        idx = rng.integers(0, m)
        out[idx] += v
    return out

def top_k_mask(vector: np.ndarray, k: int) -> np.ndarray:
    """Return a mask with ``1`` at the positions of the top‑k values."""
    if k <= 0:
        return np.zeros_like(vector, dtype=bool)
    if k >= len(vector):
        return np.ones_like(vector, dtype=bool)
    thresh = np.partition(vector, -k)[-k]
    return vector >= thresh

def gaussian_beam(theta: float, size: int) -> np.ndarray:
    """Gaussian weighting centred at ``size//2`` with spread proportional to ``theta``."""
    x = np.arange(size) - size // 2
    sigma = max(theta, 1e-6)  # avoid division by zero
    return np.exp(-0.5 * (x / sigma) ** 2)

def fisher_score(theta: float) -> float:
    """Fisher information for a 1‑D Gaussian with variance = theta^2."""
    sigma2 = theta ** 2
    return 1.0 / sigma2 if sigma2 > 0 else 0.0

def weighted_ssim(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    """Weight each component of SSIM by ``weight`` (normalized)."""
    if weight.sum() == 0:
        return 0.0
    w = weight / weight.sum()
    # component‑wise SSIM (using same formula as ssim but on single values)
    eps = 1e-12
    mu_x = np.sum(w * x)
    mu_y = np.sum(w * y)
    sigma_x2 = np.sum(w * (x - mu_x) ** 2)
    sigma_y2 = np.sum(w * (y - mu_y) ** 2)
    sigma_xy = np.sum(w * (x - mu_x) * (y - mu_y))
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return num / (den + eps)

# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def compute_confidence(signal: np.ndarray) -> float:
    """Scalar confidence = (max−min)/stddev of the signal."""
    if signal.size == 0:
        return 0.0
    rng = signal.max() - signal.min()
    std = signal.std()
    return float(rng / std) if std > 0 else 0.0

def build_tropical_weights(confidence: float,
                           dim_in: int,
                           dim_out: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct tropical weights and biases from the confidence scalar.

    * Gaussian‑beam vector `g` (length = dim_in) using θ = confidence.
    * Fisher scalar `F = fisher_score(confidence)`.
    * Weight matrix `W = outer(F * g, ones(dim_out))`  (broadcasted across rows).
    * Biases are small random offsets scaled by `confidence`.
    """
    g = gaussian_beam(confidence, dim_in)                # (dim_in,)
    F = fisher_score(confidence)                        # scalar
    W = np.outer(F * g, np.ones(dim_out))               # shape (dim_in, dim_out)
    W = W.T  # tropical network expects (out_dim, in_dim)
    rng = np.random.default_rng(seed=int(confidence * 1e6) & 0xffffffff)
    b = rng.normal(loc=0.0, scale=0.1 * confidence, size=dim_out)
    return W, b

def hybrid_engine_score(endpoint: EngineEndpoint,
                        input_signal: List[float],
                        sparse_dim: int = 256,
                        top_k: int = 20) -> Dict[str, float]:
    """
    End‑to‑end hybrid evaluation:

    1. Expand the raw ``input_signal`` into a high‑dimensional sparse vector.
    2. Derive confidence ``c`` from the sparse vector.
    3. Build a tropical network whose weights embed the Gaussian‑beam and Fisher
       information derived from ``c``.
    4. Evaluate the network on the sparse vector.
    5. Apply a top‑k winner‑take‑all mask to the tropical output.
    6. Compute a Fisher‑weighted SSIM between the masked tropical output and the
       original sparse vector.
    7. Combine the SSIM score with a morphology‑based recovery priority that is
       also scaled by ``c``.

    Returns a dictionary with the intermediate and final scores.
    """
    # 1. Sparse expansion
    sparse_vec = expand(input_signal, sparse_dim, salt=endpoint.engine_id)

    # 2. Confidence
    c = compute_confidence(sparse_vec)

    # 3. Tropical network construction
    W, b = build_tropical_weights(c, dim_in=sparse_dim, dim_out=sparse_dim)
    tropical_net = TropicalNetwork(weights=W, biases=b)

    # 4. Tropical evaluation
    trop_out = tropical_net.evaluate(sparse_vec)

    # 5. Winner‑take‑all mask
    mask = top_k_mask(trop_out, top_k)
    masked_trop = trop_out * mask

    # 6. Weighted SSIM
    weight_vec = gaussian_beam(c, sparse_dim) * fisher_score(c)
    ssim_score = weighted_ssim(sparse_vec, masked_trop, weight_vec)

    # 7. Morphology‑derived recovery priority
    morph = endpoint.morphology
    shape_factor = (morph.length * morph.width * morph.height) ** (1/3)
    mass_factor = morph.mass / (shape_factor + 1e-6)
    recovery_priority = (mass_factor / (c + 1e-6)) * ssim_score

    return {
        "confidence": c,
        "ssim_score": ssim_score,
        "recovery_priority": recovery_priority,
        "masked_tropical_sum": float(masked_trop.sum()),
        "sparse_norm": float(np.linalg.norm(sparse_vec)),
    }

def select_best_endpoint(endpoints: List[EngineEndpoint],
                         input_signal: List[float]) -> EngineEndpoint:
    """
    Evaluate a list of ``EngineEndpoint`` objects with ``hybrid_engine_score``
    and return the one with the highest ``recovery_priority``.
    """
    best = None
    best_score = -math.inf
    for ep in endpoints:
        scores = hybrid_engine_score(ep, input_signal)
        if scores["recovery_priority"] > best_score:
            best_score = scores["recovery_priority"]
            best = ep
    return best

def hybrid_pipeline(values: List[float],
                    endpoints: List[EngineEndpoint]) -> Tuple[EngineEndpoint, Dict[str, float]]:
    """
    High‑level pipeline:

    * ``values`` – raw input (e.g., sensor readings or encoded text).
    * ``endpoints`` – candidate engine descriptors.

    Returns the selected endpoint together with its evaluation dictionary.
    """
    selected = select_best_endpoint(endpoints, values)
    evaluation = hybrid_engine_score(selected, values)
    return selected, evaluation

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a couple of mock endpoints
    morph_a = Morphology(length=2.0, width=1.0, height=1.5, mass=3.0)
    ep_a = EngineEndpoint(
        engine_id="engine_A",
        channel="alpha",
        residency="us-east",
        runtime="python3.11",
        resource_class="standard",
        always_on=True,
        endpoint="https://example.com/a",
        capabilities=["compute", "storage"],
        morphology=morph_a,
    )

    morph_b = Morphology(length=1.2, width=0.8, height=1.0, mass=1.5)
    ep_b = EngineEndpoint(
        engine_id="engine_B",
        channel="beta",
        residency="eu-west",
        runtime="python3.11",
        resource_class="highmem",
        always_on=False,
        endpoint="https://example.com/b",
        capabilities=["compute"],
        morphology=morph_b,
    )

    # Random input signal (could be derived from text, sensor, etc.)
    random.seed(42)
    raw_signal = [random.random() for _ in range(10)]

    # Run the hybrid pipeline
    best_ep, scores = hybrid_pipeline(raw_signal, [ep_a, ep_b])

    print("Selected Engine:", best_ep.engine_id)
    print("Evaluation Scores:")
    for k, v in scores.items():
        print(f"  {k}: {v:.6f}")