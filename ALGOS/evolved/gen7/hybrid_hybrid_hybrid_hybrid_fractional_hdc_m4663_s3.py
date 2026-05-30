# DARWIN HAMMER — match 4663, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hoeffding_tre_m1469_s2.py (gen6)
# parent_b: fractional_hdc.py (gen0)
# born: 2026-05-29T23:57:16Z

"""Hybrid Hoeffding‑Tree / Fractional Hyperdimensional Computing (HT‑FHC)

This module fuses the core mathematics of two ancestors:

* **Parent A** – a Hoeffding decision tree that uses the Gini coefficient
  to drive split decisions on streaming text features.
* **Parent B** – Fractional binding in Hyperdimensional Computing (HDC) where
  vectors are combined by circular convolution and the binding strength is
  controlled by a fractional power α (phase scaling in the Fourier domain).

**Mathematical bridge**

The bridge is the *Gini‑weighted fractional binding*:

1. Text is encoded into a high‑dimensional hypervector **h** by bundling
   random hypervectors for each token (Parent B).
2. For a node of the Hoeffding tree we keep a histogram of class counts.
   The Gini impurity **G** of this histogram (Parent A) measures class
   heterogeneity.
3. The Hoeffding bound **ε** tells us how many samples are needed before a
   split decision is statistically reliable.
4. When **G** is high (impure node) and the bound **ε** is small enough,
   we *strengthen* the hypervector representation by applying a fractional
   binding with α = 1‑G (more impurity → stronger binding).  This yields a
   new vector **h′ = h (*) h^α** that carries both the original content and
   a bias proportional to the impurity.

Thus the Gini coefficient directly modulates the fractional power used in
the HDC binding, and the Hoeffding bound controls when this modulation is
applied.  The three functions below illustrate this hybrid operation.

"""

import sys
import math
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Parent B – Hyperdimensional Computing primitives
# ---------------------------------------------------------------------------

def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """Generate a random hypervector.

    Parameters
    ----------
    d : int
        Dimensionality.
    kind : {"complex", "bipolar", "real"}
        Type of hypervector.
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (d,). Complex for ``kind=="complex"``, otherwise float.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * math.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice([-1.0, 1.0], size=d).astype(np.float64)
    # real Gaussian, then normalised
    vec = rng.normal(size=d)
    vec /= np.linalg.norm(vec)
    return vec.astype(np.float64)


def bind(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Circular convolution (binding) via FFT."""
    fx = np.fft.fft(x)
    fy = np.fft.fft(y)
    return np.fft.ifft(fx * fy)


def fractional_power(y: np.ndarray, alpha: float) -> np.ndarray:
    """Raise a hypervector to a fractional power by scaling its phase.

    Works for complex vectors; for real/bipolar vectors we fall back to the
    identity (alpha = 1 returns y, alpha = 0 returns unit vector).
    """
    if np.iscomplexobj(y):
        # magnitude is already 1 for complex HVs; keep it.
        phase = np.angle(y)
        new_phase = phase * alpha
        return np.exp(1j * new_phase)
    # For non‑complex vectors we approximate by linear interpolation to a unit vector.
    unit = np.ones_like(y) / np.linalg.norm(np.ones_like(y))
    return (1 - alpha) * unit + alpha * y


def bundle(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Superpose (add) a sequence of hypervectors and renormalise."""
    sum_vec = np.sum(vectors, axis=0)
    norm = np.linalg.norm(sum_vec)
    if norm == 0:
        return sum_vec
    return sum_vec / norm


def similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity (real‑valued even for complex vectors)."""
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        # Use real part of inner product for complex vectors.
        dot = np.real(np.vdot(x, y))
    else:
        dot = np.dot(x, y)
    return dot / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)


# ---------------------------------------------------------------------------
# Parent A – Text processing & Gini / Hoeffding utilities
# ---------------------------------------------------------------------------

def words(text: str) -> List[str]:
    """Simple tokeniser: lower‑case alphabetic words."""
    return [w for w in (text or "").lower().split() if w.isalpha()]

def gini_impurity(class_counts: Dict[object, int]) -> float:
    """Compute Gini impurity from a dict of class → count."""
    total = sum(class_counts.values())
    if total == 0:
        return 0.0
    prob_sq_sum = sum((cnt / total) ** 2 for cnt in class_counts.values())
    return 1.0 - prob_sq_sum

def hoeffding_bound(R: float, n: int, delta: float = 1e-7) -> float:
    """Hoeffding bound ε = sqrt( (R^2 * ln(1/δ)) / (2n) )."""
    if n <= 0:
        return float('inf')
    return math.sqrt((R * R * math.log(1.0 / delta)) / (2.0 * n))


# ---------------------------------------------------------------------------
# Hybrid components
# ---------------------------------------------------------------------------

# Global vocabulary → hypervector map (fixed random seed for reproducibility)
_VOCAB_HV: Dict[str, np.ndarray] = {}
_VOCAB_DIM = 10000
_VOCAB_SEED = 42

def _hv_for_token(token: str) -> np.ndarray:
    """Return (or create) a hypervector for a token."""
    if token not in _VOCAB_HV:
        # deterministic seed per token
        token_seed = hash(token) & 0xffffffff
        _VOCAB_HV[token] = random_hv(d=_VOCAB_DIM, kind="complex", seed=token_seed)
    return _VOCAB_HV[token]


def encode_text_to_hv(text: str) -> np.ndarray:
    """Encode a piece of text into a single hypervector by bundling token HVs."""
    token_hvs = [_hv_for_token(tok) for tok in words(text)]
    if not token_hvs:
        # fallback to zero vector (will be normalised later)
        return np.zeros(_VOCAB_DIM, dtype=np.complex128)
    return bundle(token_hvs)


def gini_weighted_fractional_bind(hv: np.ndarray, gini: float) -> np.ndarray:
    """Apply fractional binding with α = 1 - Gini.

    The more impure the node (high Gini), the larger the binding strength.
    """
    alpha = 1.0 - gini  # α ∈ [0,1]
    powered = fractional_power(hv, alpha)
    return bind(hv, powered)


class HoeffdingNode:
    """A minimal Hoeffding tree node storing class counts and a hypervector."""
    def __init__(self):
        self.class_counts: Counter = Counter()
        self.n_samples: int = 0
        self.hv: np.ndarray = np.zeros(_VOCAB_DIM, dtype=np.complex128)

    def update(self, label, text: str) -> None:
        """Incorporate a new labelled example."""
        self.class_counts[label] += 1
        self.n_samples += 1
        txt_hv = encode_text_to_hv(text)
        # incremental superposition (simple averaging)
        self.hv = (self.hv * (self.n_samples - 1) + txt_hv) / self.n_samples

    def should_split(self, R: float = 1.0, delta: float = 1e-7) -> bool:
        """Decide whether to split based on Hoeffding bound and impurity."""
        gini = gini_impurity(self.class_counts)
        eps = hoeffding_bound(R, self.n_samples, delta)
        # Heuristic: split when impurity is high enough and bound is small.
        return gini > 0.5 and eps < 0.05

    def hybrid_representation(self) -> np.ndarray:
        """Return the impurity‑weighted fractional binding of the node's HV."""
        gini = gini_impurity(self.class_counts)
        return gini_weighted_fractional_bind(self.hv, gini)


def hybrid_tree_step(node: HoeffdingNode, label, text: str) -> Tuple[bool, np.ndarray]:
    """Process one example, update the node and possibly return a split signal.

    Returns
    -------
    split_flag : bool
        True if the node meets the split criterion.
    rep : np.ndarray
        The hybrid hypervector after the update (post‑binding if split).
    """
    node.update(label, text)
    split = node.should_split()
    if split:
        # Apply impurity‑weighted binding before splitting.
        rep = node.hybrid_representation()
    else:
        rep = node.hv
    return split, rep


# ---------------------------------------------------------------------------
# Demonstration functions
# ---------------------------------------------------------------------------

def demo_hybrid_binding():
    """Show fractional binding controlled by Gini."""
    hv = encode_text_to_hv("the quick brown fox jumps over the lazy dog")
    # Simulate a node with two classes, counts {A:30, B:70}
    fake_counts = { "A": 30, "B": 70 }
    g = gini_impurity(fake_counts)
    bound_hv = gini_weighted_fractional_bind(hv, g)
    print(f"Gini: {g:.4f}, similarity before/after bind: {similarity(hv, bound_hv):.4f}")

def demo_hoeffding_node():
    """Run a tiny stream through a HoeffdingNode."""
    node = HoeffdingNode()
    data = [
        ("spam", "win money now"),
        ("ham", "meeting schedule attached"),
        ("spam", "exclusive offer just for you"),
        ("ham", "project deadline extended"),
        ("spam", "click here to claim prize"),
    ]
    for label, txt in data:
        split, rep = hybrid_tree_step(node, label, txt)
        print(f"Added [{label}] '{txt[:20]}...' | n={node.n_samples} | split={split} | rep_norm={np.linalg.norm(rep):.3f}")

def demo_full_pipeline():
    """Combine the three demo pieces into a short pipeline."""
    print("=== Demo: Hybrid Fractional Binding ===")
    demo_hybrid_binding()
    print("\n=== Demo: Hoeffding Node Streaming ===")
    demo_hoeffding_node()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_full_pipeline()