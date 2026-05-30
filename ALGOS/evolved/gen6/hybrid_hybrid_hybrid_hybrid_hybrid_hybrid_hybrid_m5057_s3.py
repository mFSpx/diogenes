# DARWIN HAMMER — match 5057, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m2197_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1811_s5.py (gen5)
# born: 2026-05-29T23:59:35Z

"""Hybrid Stylometry‑KAN‑Ternary‑Hyperdimensional Bandit Model
================================================================

Parents:
 * **Parent A** – `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m2197_s2.py`
   (stylometric feature extraction → ternary vector → ternary‑softmax → Sparse
   Winner‑Take‑All (WTA) → Kolmogorov‑Arnold Network (KAN) with learnable
   univariate B‑splines).

 * **Parent B** – `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1811_s5.py`
   (text‑hygiene extraction → high‑dimensional bipolar vector → RBF kernel
   similarity to action prototypes → modulation by a Schoolfield temperature
   model → contextual bandit decision).

Mathematical Bridge
-------------------
Both parents ultimately produce a *high‑dimensional numeric representation* of a
text fragment:

   * Parent A → ternary vector `t ∈ {−1,0,1}^T`.
   * Parent B → bipolar hyper‑dimensional vector `h ∈ {−1,+1}^D`.

The fusion maps the ternary vector onto the same algebraic space as the
hyper‑dimensional vector (by a simple linear scaling) and concatenates them,
yielding a *combined representation*  


x = [α·t , h] ∈ ℝ^{T+D}


where `α` is a scaling factor that balances the influence of the ternary part.
`x` is then processed by a shallow KAN (providing a non‑linear transformation)
and fed to an RBF‑kernel that measures similarity to a set of prototype
vectors `{p_a}` (one per bandit action).  The raw similarity scores are
multiplied by a temperature‑dependent developmental rate given by the
Schoolfield model, producing temperature‑aware propensity scores that drive a
Sparse WTA selector for the final action.

The code below implements this pipeline with three public functions that
exemplify the hybrid operation:
    1. `extract_combined_representation(text) → np.ndarray`
    2. `kan_transform(x) → np.ndarray`
    3. `select_action(x, temperature) → str`
"""

import sys
import math
import random
import hashlib
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import re

# ----------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------
TERNARY_DIMS = 2048               # dimension of ternary vector (Parent A)
HD_DIM = 10_000                   # hyper‑dimensional bipolar dimension (Parent B)
ALPHA = 0.5                       # scaling factor for ternary part in the combined vector
TOP_K_WTA = 5                     # number of winners kept by Sparse WTA
RBF_GAMMA = 1e-5                  # 1/(2σ²) for Gaussian kernel
TEMPERATURE_PARAMS = {           # simple Schoolfield‑like parameters
    "E": 1.0,
    "Ea": 8000.0,                # activation energy (J/mol)
    "kB": 8.617e-5,              # Boltzmann constant (eV/K)
    "T_opt": 310.0,              # optimal temperature (K)
    "beta": 0.1,
}

# ----------------------------------------------------------------------
# Parent A – Stylometric → ternary utilities
# ----------------------------------------------------------------------
_FUNCTION_CATS = {
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
    "conjunction": set("and but or nor so yet because although".split()),
}


def _tokenize(text: str) -> List[str]:
    """Very small tokenizer – split on non‑alphabetic characters."""
    return re.findall(r"[A-Za-z]+", text.lower())


def stylometric_vector(text: str) -> np.ndarray:
    """Return a dense real‑valued stylometric vector (dimension = number of categories)."""
    tokens = _tokenize(text)
    total = len(tokens) or 1
    vec = []
    for cat in _FUNCTION_CATS.values():
        count = sum(1 for w in tokens if w in cat)
        vec.append(count / total)
    return np.array(vec, dtype=np.float32)


def ternary_hash(vec: np.ndarray) -> np.ndarray:
    """
    Map a real‑valued vector to a ternary vector in {−1,0,1}^TERNARY_DIMS.
    The hash is deterministic: we feed the binary representation of the
    vector to SHA‑256 and interpret the digest as a stream of bits.
    """
    # Serialize the vector deterministically
    raw = vec.tobytes()
    digest = hashlib.sha256(raw).digest()
    # Expand digest to required length
    needed_bytes = (TERNARY_DIMS + 3) // 4  # 2 bits per ternary symbol
    expanded = (digest * ((needed_bytes // len(digest)) + 1))[:needed_bytes]
    bits = np.unpackbits(np.frombuffer(expanded, dtype=np.uint8))
    # Take two bits at a time → 0,1,2,3 → map to -1,0,1,0 (avoid 3)
    ternary = np.empty(TERNARY_DIMS, dtype=np.int8)
    for i in range(TERNARY_DIMS):
        b1, b2 = bits[2 * i], bits[2 * i + 1]
        code = (b1 << 1) | b2
        if code == 0:
            ternary[i] = -1
        elif code == 1:
            ternary[i] = 0
        elif code == 2:
            ternary[i] = 1
        else:  # code == 3, map to 0 to keep ternary property
            ternary[i] = 0
    return ternary


def ternary_softmax(t: np.ndarray) -> np.ndarray:
    """Apply a softmax‑like function that respects ternary values."""
    # Shift to make all entries non‑negative
    shifted = t - t.min()
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


def sparse_wta(x: np.ndarray, k: int = TOP_K_WTA) -> np.ndarray:
    """Zero‑out all but the top‑k entries (by absolute value)."""
    if k >= x.size:
        return x
    idx = np.argpartition(-np.abs(x), k - 1)[:k]
    mask = np.zeros_like(x, dtype=bool)
    mask[idx] = True
    out = np.zeros_like(x)
    out[mask] = x[mask]
    return out

# ----------------------------------------------------------------------
# Parent B – Hygiene → hyper‑dimensional utilities
# ----------------------------------------------------------------------
_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

# Very simple regex patterns (real patterns omitted for brevity)
_FEATURE_REGEX = {
    "evidence": re.compile(r"\bevidence\b", re.I),
    "planning": re.compile(r"\bplan(?:ning)?\b", re.I),
    "delay": re.compile(r"\bdelay\b", re.I),
    "support": re.compile(r"\bsupport\b", re.I),
    "boundary": re.compile(r"\bboundary\b", re.I),
    "outcome": re.compile(r"\boutcome\b", re.I),
    "impulsive": re.compile(r"\bimpuls(?:e|ive)\b", re.I),
    "scarcity": re.compile(r"\bscarcity\b", re.I),
    "risk": re.compile(r"\brisk\b", re.I),
}


def hygiene_vector(text: str) -> np.ndarray:
    """Return a bipolar hyper‑dimensional vector (±1) of dimension HD_DIM."""
    # Start from a random seed that depends on the text to keep determinism
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    # Generate a random bipolar base vector
    base = rng.choice([-1, 1], size=HD_DIM, replace=True)

    # Modulate the base vector with weighted feature counts
    for i, feat in enumerate(_FEATURE_ORDER):
        count = len(_FEATURE_REGEX[feat].findall(text))
        weight = _POSITIVE_WEIGHTS[i] - _NEGATIVE_WEIGHTS[i]
        if weight != 0:
            # Flip a proportion of bits proportional to count*weight
            flip_prob = min(1.0, (count * abs(weight)) / 1e5)
            flips = rng.random(HD_DIM) < flip_prob
            base[flips] *= -1
    return base.astype(np.int8)


# ----------------------------------------------------------------------
# Shared mathematical bridge utilities
# ----------------------------------------------------------------------
def combine_vectors(ternary: np.ndarray, hyper: np.ndarray) -> np.ndarray:
    """
    Scale the ternary part, cast to float, and concatenate with the hyper‑dimensional part.
    Resulting vector lives in ℝ^{TERNARY_DIMS + HD_DIM}.
    """
    t_scaled = ALPHA * ternary.astype(np.float32)
    h_float = hyper.astype(np.float32)  # keep bipolar values as ±1
    return np.concatenate([t_scaled, h_float], axis=0)


# ----------------------------------------------------------------------
# KAN (Kolmogorov‑Arnold Network) – simplified version
# ----------------------------------------------------------------------
@dataclass
class BSplineEdge:
    """Univariate B‑spline approximated by a set of control points (x_i, y_i)."""
    xs: np.ndarray   # shape (M,)
    ys: np.ndarray   # shape (M,)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Linear interpolation between control points (piecewise linear spline)."""
        # np.interp handles out‑of‑bounds by extrapolation with edge values
        return np.interp(z, self.xs, self.ys)


def _random_bsplines(in_dim: int, out_dim: int, order: int = 5, rng: np.random.Generator = None) -> List[List[BSplineEdge]]:
    """Create a matrix of random BSplineEdge objects."""
    if rng is None:
        rng = np.random.default_rng()
    edges = []
    for _ in range(out_dim):
        row = []
        for _ in range(in_dim):
            xs = np.linspace(-1, 1, order)
            ys = rng.uniform(-1, 1, size=order)
            row.append(BSplineEdge(xs=xs, ys=ys))
        edges.append(row)
    return edges


class SimpleKAN:
    """
    Very shallow KAN: one linear mixing followed by element‑wise univariate B‑splines.
    Input dimension = D_in, output dimension = D_out.
    """
    def __init__(self, D_in: int, D_out: int, rng: np.random.Generator = None):
        self.D_in = D_in
        self.D_out = D_out
        self.rng = rng or np.random.default_rng()
        # Linear mixing matrix
        self.W = self.rng.normal(0.0, 0.5, size=(D_out, D_in)).astype(np.float32)
        # One spline per weight entry
        self.splines = _random_bsplines(D_in, D_out, order=5, rng=self.rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x : (D_in,)
        Returns y : (D_out,)
        """
        lin = self.W @ x  # shape (D_out,)
        # Apply spline per output dimension
        y = np.empty_like(lin)
        for i in range(self.D_out):
            # For each input dimension, evaluate its spline at x_j and sum
            contributions = np.empty(self.D_in, dtype=np.float32)
            for j in range(self.D_in):
                contributions[j] = self.splines[i][j](x[j])
            y[i] = contributions.sum()
        return y


# Instantiate a global KAN (dimensions based on combined vector)
_KAN = SimpleKAN(D_in=TERNARY_DIMS + HD_DIM, D_out=512)


def kan_transform(x: np.ndarray) -> np.ndarray:
    """Public wrapper that applies the KAN and a Sparse WTA."""
    y = _KAN.forward(x)
    y_wta = sparse_wta(y, k=TOP_K_WTA)
    return y_wta

# ----------------------------------------------------------------------
# RBF kernel and Schoolfield temperature model
# ----------------------------------------------------------------------
def rbf_similarity(x: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """
    Compute Gaussian similarity between x and each prototype.
    prototypes : (N_actions, D) where D == len(x)
    Returns a vector of length N_actions.
    """
    diff = prototypes - x  # broadcast
    sq_norm = np.sum(diff ** 2, axis=1)
    return np.exp(-RBF_GAMMA * sq_norm)


def schoolfield_rate(T: float, params: Dict[str, float] = TEMPERATURE_PARAMS) -> float:
    """
    Simple implementation of a Schoolfield‑type temperature dependence.
    """
    E = params["E"]
    Ea = params["Ea"]
    kB = params["kB"]
    T_opt = params["T_opt"]
    beta = params["beta"]
    # Arrhenius term
    arr = E * math.exp(-Ea / (kB * T))
    # High‑temperature inhibition term
    inhib = 1.0 / (1.0 + math.exp(beta * (T - T_opt)))
    return arr * inhib

# ----------------------------------------------------------------------
# Bandit action selection (Sparse WTA over temperature‑modulated scores)
# ----------------------------------------------------------------------
_ACTION_PROTOTYPES: Dict[str, np.ndarray] = {}
_ACTION_NAMES: List[str] = ["accept", "reject", "review", "escalate", "archive"]


def _init_prototypes():
    """Create deterministic random prototypes for each action."""