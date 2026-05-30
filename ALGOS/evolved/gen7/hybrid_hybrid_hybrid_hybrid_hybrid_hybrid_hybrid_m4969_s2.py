# DARWIN HAMMER — match 4969, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1202_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_hybrid_m1138_s0.py (gen5)
# born: 2026-05-29T23:59:03Z

"""HybridDarwinHammer
Combines:
- Parent A (DARWIN HAMMER) – linear TTT weight matrix W used for transforming resource vectors.
- Parent B (GLINER‑Bandit) – span‑based pheromone signals and Shannon entropy of textual features.

Mathematical bridge:
1. A text passage is turned into a numeric feature vector **x** ∈ ℝ^d by concatenating:
   * load & privacy flags (Parent A),
   * Shannon entropy of the token distribution (Parent B).
2. The TTT‑Linear matrix **W** (Parent A) maps **x** to a latent representation **y = W x**.
3. A pheromone signal **ϕ** is derived from a span’s confidence score (Parent B) via
   ϕ = –log(score).  
4. The hybrid loss blends the TTT reconstruction loss with a pheromone‑weighted
   regulariser:

   L(W; x, ϕ) = ‖W x – x‖²  +  λ·ϕ·‖W‖_F²

   where λ is a small scaling constant.  
   The gradient w.r.t. W therefore contains the classic TTT term plus a
   pheromone‑scaled weight‑decay term.

The code below implements this fused system, exposing three core functions
`init_hybrid`, `hybrid_loss`, and `hybrid_step`, together with a tiny
`HybridModel` wrapper and a smoke‑test under `__main__`. """

import numpy as np
import math
import random
import sys
import pathlib
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, List

# ----------------------------------------------------------------------
# Parent A utilities (TTT‑Linear)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize a TTT‑Linear weight matrix."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    """Classic reconstruction loss ‖W x – target‖²."""
    if target is None:
        target = x
    diff = W @ x - target
    return float(np.sum(diff ** 2))


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Gradient of the reconstruction loss w.r.t. W."""
    if target is None:
        target = x
    diff = W @ x - target
    # grad = 2 * diff * xᵀ
    return 2.0 * diff[:, None] @ x[None, :]


def ttt_step(W: np.ndarray, x: np.ndarray, eta: float, target: np.ndarray = None) -> np.ndarray:
    """One SGD step on the TTT loss."""
    grad = ttt_grad(W, x, target)
    return W - eta * grad


# ----------------------------------------------------------------------
# Parent B utilities (entropy, pheromone)
# ----------------------------------------------------------------------
def shannon_entropy(tokens: List[str]) -> float:
    """Compute Shannon entropy of a token list."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    probs = np.array([c / total for c in counts.values()], dtype=float)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def compute_pheromone(score: float) -> float:
    """Pheromone signal defined in Parent B: ϕ = –log(score)."""
    # Guard against non‑positive scores
    if score <= 0.0:
        score = 1e-12
    return -math.log(score)


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float  # confidence in [0,1]


@dataclass(frozen=True)
class ResourceVector:
    load: float
    privacy: float
    entropy: float


# ----------------------------------------------------------------------
# Feature extraction (fusion of both parents)
# ----------------------------------------------------------------------
def extract_features(text: str, span: Span) -> Tuple[np.ndarray, float]:
    """
    Convert raw text + span into:
    - a column vector x ∈ ℝ³ (load, privacy, entropy)
    - a pheromone scalar ϕ derived from the span’s confidence score
    """
    # Flags from Parent A
    load = 1.0 if re.search(r"\b(evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
                               text, re.I) else 0.0
    privacy = 1.0 if re.search(r"\b(plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
                                 text, re.I) else 0.0

    # Entropy from Parent B
    tokens = re.findall(r"\w+", text.lower())
    entropy = shannon_entropy(tokens)

    # Assemble feature vector (column)
    x = np.array([load, privacy, entropy], dtype=float).reshape(-1, 1)

    # Pheromone from span
    phi = compute_pheromone(span.score)

    return x, phi


# ----------------------------------------------------------------------
# Hybrid core operations
# ----------------------------------------------------------------------
def init_hybrid(d_in: int = 3, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """
    Initialise the shared weight matrix W for the hybrid system.
    Mirrors Parent A's init_ttt but is exposed under a hybrid name.
    """
    return init_ttt(d_in, d_out, scale, seed)


def hybrid_loss(W: np.ndarray, x: np.ndarray, phi: float, lam: float = 1e-4, target: np.ndarray = None) -> float:
    """
    Hybrid loss:
        L = ‖W x – target‖² + λ·ϕ·‖W‖_F²
    """
    recon = ttt_loss(W, x, target)
    decay = lam * phi * np.sum(W ** 2)
    return recon + decay


def hybrid_grad(W: np.ndarray, x: np.ndarray, phi: float, lam: float = 1e-4, target: np.ndarray = None) -> np.ndarray:
    """
    Gradient of the hybrid loss w.r.t. W.
    The pheromone term contributes 2·λ·ϕ·W.
    """
    grad_recon = ttt_grad(W, x, target)
    grad_decay = 2.0 * lam * phi * W
    return grad_recon + grad_decay


def hybrid_step(W: np.ndarray, x: np.ndarray, phi: float, eta: float, lam: float = 1e-4,
                target: np.ndarray = None) -> np.ndarray:
    """
    Perform a single SGD update on W using the hybrid gradient.
    """
    grad = hybrid_grad(W, x, phi, lam, target)
    return W - eta * grad


# ----------------------------------------------------------------------
# Simple wrapper class for convenience
# ----------------------------------------------------------------------
class HybridModel:
    """
    Encapsulates the hybrid weight matrix and provides a `fit_step`
    method that consumes raw text and a span, computes features,
    and updates the internal matrix.
    """

    def __init__(self, d_in: int = 3, d_out: int = None, scale: float = 0.01,
                 seed: int = 0, lam: float = 1e-4, eta: float = 1e-3):
        self.W = init_hybrid(d_in, d_out, scale, seed)
        self.lam = lam
        self.eta = eta

    def fit_step(self, text: str, span: Span) -> float:
        """One optimisation step; returns the current hybrid loss."""
        x, phi = extract_features(text, span)
        loss = hybrid_loss(self.W, x, phi, self.lam)
        self.W = hybrid_step(self.W, x, phi, self.eta, self.lam)
        return loss

    def transform(self, text: str, span: Span) -> np.ndarray:
        """Apply the learned linear map to a new example."""
        x, _ = extract_features(text, span)
        return self.W @ x


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example text containing evidence and planning cues
    example_text = (
        "The system logged a hash SHA256=abc123 and the plan includes a "
        "timeline for deployment. All steps are documented."
    )
    # Dummy span with a moderate confidence score
    example_span = Span(start=0, end=len(example_text), text=example_text,
                        label="evidence_plan", score=0.78)

    model = HybridModel(seed=42)
    initial_loss = model.fit_step(example_text, example_span)
    print(f"Initial hybrid loss: {initial_loss:.6f}")

    # Run a few more steps to demonstrate learning
    for i in range(5):
        loss = model.fit_step(example_text, example_span)
        print(f"Step {i+1:02d} loss: {loss:.6f}")

    transformed = model.transform(example_text, example_span)
    print("Transformed feature vector:", transformed.ravel())