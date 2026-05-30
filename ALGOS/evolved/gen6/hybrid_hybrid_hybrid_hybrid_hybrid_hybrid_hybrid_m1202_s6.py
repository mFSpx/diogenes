# DARWIN HAMMER — match 1202, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m886_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_capybara_opti_m52_s0.py (gen3)
# born: 2026-05-29T23:34:35Z

"""Hybrid Algorithm Fusion of DARWIN HAMMER Parents A and B.

Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m28_s0):
- Provides a TTT‑Linear weight matrix **W** that linearly transforms a
  resource vector **x = [load, privacy]**.
- Defines a reconstruction loss `ttt_loss(W, x, target)` and its gradient
  `ttt_grad`.

Parent B (hybrid_hybrid_hybrid_rbf_su_hybrid_capybara_opti_m52_s0):
- Supplies a Radial‑Basis Function (RBF) surrogate model
  `RBFSurrogate` that maps a feature vector to a scalar *signal* using
  Gaussian kernels.

**Mathematical Bridge**
The bridge is the feature vector **z = W·x** produced by the TTT‑Linear
transform. This vector is fed to the RBF surrogate to obtain a signal
`s = surrogate.predict(z)`. The signal modulates the learning rate of the
TTT update, yielding an adaptive step:

    η_adapt = η * (1 + s)

Thus the surrogate’s non‑linear evaluation directly influences the
linear‑algebraic adaptation of **W**, tightly coupling both parent
topologies into a single hybrid system.
"""

import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Parent A – TTT‑Linear utilities
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize a TTT‑Linear weight matrix W with small random values."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    """Mean‑squared reconstruction loss of the linear transform."""
    if target is None:
        target = x
    diff = W @ x - target
    return float(np.sum(diff ** 2))

def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Gradient of the loss w.r.t. W."""
    if target is None:
        target = x
    diff = W @ x - target          # shape (d_out,)
    return 2.0 * np.outer(diff, x)  # shape (d_out, d_in)

def ttt_step(W: np.ndarray, x: np.ndarray, eta: float, target: np.ndarray = None) -> np.ndarray:
    """One gradient‑descent step on W."""
    grad = ttt_grad(W, x, target)
    return W - eta * grad

# ----------------------------------------------------------------------
# Parent B – RBF Surrogate utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

@dataclass(frozen=True)
class RBFSurrogate:
    """Radial‑Basis Function surrogate model."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Sequence[float]) -> float:
        """Weighted sum of Gaussian kernels evaluated at x."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass
class ResourceVector:
    """Simple resource descriptor used by Parent A."""
    load: float
    privacy: float

def extract_text_features(text: str) -> ResourceVector:
    """Regex‑based extraction of evidence and planning cues."""
    evidence = bool(re.search(
        r"\b(evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
        r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
        text, re.I))
    planning = bool(re.search(
        r"\b(plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
        r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
        text, re.I))
    # Encode booleans as 1.0 / 0.0 contributions
    load = 1.0 if evidence else 0.0
    privacy = 1.0 if planning else 0.0
    return ResourceVector(load=load, privacy=privacy)

# ----------------------------------------------------------------------
# Hybrid core functions (demonstrate the fused topology)
# ----------------------------------------------------------------------
def transform_load(W: np.ndarray, rv: ResourceVector) -> np.ndarray:
    """
    Apply the TTT‑Linear matrix W to the 2‑D resource vector.
    Returns the transformed feature vector z.
    """
    x = np.array([rv.load, rv.privacy])
    return W @ x  # shape (d_out,)

def surrogate_signal(surrogate: RBFSurrogate, z: np.ndarray) -> float:
    """
    Compute the non‑linear signal using the RBF surrogate on the transformed
    vector z.
    """
    return surrogate.predict(z.tolist())

def hybrid_update(
    W: np.ndarray,
    rv: ResourceVector,
    surrogate: RBFSurrogate,
    eta: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """
    Perform a hybrid adaptation step:
    1. Transform the resource vector with W → z.
    2. Evaluate the surrogate signal s = surrogate(z).
    3. Modulate the learning rate: η_adapt = η * (1 + s).
    4. Take a gradient step on W using the (optional) target = x.
    Returns the updated weight matrix and the surrogate signal.
    """
    x = np.array([rv.load, rv.privacy])
    z = W @ x
    s = surrogate.predict(z.tolist())
    eta_adapt = eta * (1.0 + s)  # ensure positivity
    W_new = ttt_step(W, x, eta_adapt, target=x)  # target set to identity reconstruction
    return W_new, s

def hybrid_operation(
    rv: ResourceVector,
    W: np.ndarray,
    surrogate: RBFSurrogate,
    eta: float = 0.01,
) -> Tuple[np.ndarray, float, float]:
    """
    Full hybrid pipeline:
    - Transform load (linear stage)
    - Compute surrogate signal (non‑linear stage)
    - Update the linear weights adaptively
    Returns (W_next, signal, loss_before_update).
    """
    x = np.array([rv.load, rv.privacy])
    loss_before = ttt_loss(W, x)
    W_next, signal = hybrid_update(W, rv, surrogate, eta)
    return W_next, signal, loss_before

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a 2×2 TTT‑Linear matrix
    W = init_ttt(d_in=2, d_out=2, scale=0.05, seed=42)

    # Build a tiny RBF surrogate with three centers in 2‑D space
    centers = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    weights = [0.3, -0.2, 0.5]
    surrogate = RBFSurrogate(centers=centers, weights=weights, epsilon=1.5)

    # Example textual input → resource vector
    sample_text = "The evidence was verified and the plan is ready."
    rv = extract_text_features(sample_text)

    # Run a few hybrid steps to demonstrate convergence
    for step in range(5):
        W, signal, loss = hybrid_operation(rv, W, surrogate, eta=0.02)
        print(f"Step {step+1}: loss={loss:.6f}, signal={signal:.4f}, W=\n{W}")

    sys.exit(0)