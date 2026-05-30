# DARWIN HAMMER — match 2783, survivor 5
# gen: 4
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_ternary_lens__m1552_s0.py (gen3)
# born: 2026-05-29T23:46:02Z

"""Hybrid algorithm combining the TTT‑Linear optimizer (parent A) with the
epistemic‑certainty / Shannon‑entropy framework (parent B).

Mathematical bridge
-------------------
Parent A provides a linear mapping **W** ∈ ℝ^{d_out×d_in} that is trained by
gradient descent on the reconstruction loss  

 L_rec(W, x) = ‖W x − x‖²           (1)

Parent B produces a *certainty vector* **c** ∈ ℝ^{d_in} derived from a set of
`CertaintyFlag` objects.  The components of **c** are the normalized confidence
values (basis‑points) of the flags and can be interpreted as a probability
distribution.  Its Shannon entropy  

 H(c) = −∑_i c_i log₂ c_i           (2)

is a scalar measure of epistemic uncertainty.

The hybrid model treats **c** as the input **x** of the TTT‑Linear model and
adds the entropy term as a regularizer:

 L_hybrid(W, c) = L_rec(W, c) + λ H(c)      (3)

where λ ≥ 0 controls the trade‑off between reconstruction fidelity and
epistemic‑certainty awareness.  Updating **W** with the gradient of (3) fuses
the core matrix‑operation of parent A with the information‑theoretic metric
of parent B.

The module implements:
* construction of **W** (TTT initializer),
* conversion of `CertaintyFlag` collections to a normalized certainty vector,
* hybrid loss, gradient, and a single‑step update,
* a decision function that uses the transformed vector `W c` together with
  the entropy to produce a hygiene score.

All operations rely only on NumPy and the Python standard library.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np

# ----------------------------------------------------------------------
# Parent B – epistemic certainty infrastructure
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")
        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> Dict[str, Union[str, int, Tuple[str, ...]]]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    """Factory for a `CertaintyFlag`."""
    return CertaintyFlag(
        label=label,
        confidence_bps=confidence_bps,
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(evidence_refs),
    )


def _flags_to_vector(flags: List[CertaintyFlag]) -> np.ndarray:
    """
    Convert a list of `CertaintyFlag` objects into a dense numeric vector.

    The vector length equals the number of distinct labels (|EPISTEMIC_FLAGS|).
    Each entry i contains the sum of confidence_bps for the corresponding label,
    normalized to a probability distribution (sum == 1.0).  If the total confidence
    is zero, a uniform distribution is returned.
    """
    label_to_idx = {lbl: i for i, lbl in enumerate(EPISTEMIC_FLAGS)}
    vec = np.zeros(len(EPISTEMIC_FLAGS), dtype=np.float64)
    for f in flags:
        vec[label_to_idx[f.label]] += f.confidence_bps
    total = vec.sum()
    if total > 0:
        vec /= total
    else:
        vec[:] = 1.0 / len(vec)
    return vec


def shannon_entropy(p: np.ndarray) -> float:
    """
    Compute Shannon entropy (base‑2) of a probability vector `p`.

    Zero probabilities are ignored in the sum to avoid `log(0)`.
    """
    mask = p > 0
    return -float(np.sum(p[mask] * np.log2(p[mask])))


# ----------------------------------------------------------------------
# Parent A – TTT‑Linear core
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize weight matrix **W** ∈ ℝ^{d_out×d_in} with small Gaussian noise."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    """
    Reconstruction loss ‖W x − t‖², where `t` defaults to `x`.
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """
    Gradient of the reconstruction loss with respect to **W**:
        ∂L/∂W = 2 (W x − t) xᵀ
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)


# ----------------------------------------------------------------------
# Hybrid layer
# ----------------------------------------------------------------------
def init_hybrid(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> dict:
    """
    Initialise the hybrid model.

    Returns a dict with keys:
        "W" – weight matrix (TTT‑Linear)
        "lambda_entropy" – regularisation coefficient for entropy term
    """
    return {
        "W": init_ttt(d_in, d_out, scale, seed),
        "lambda_entropy": 0.1,  # modest regularisation by default
    }


def hybrid_loss(state: dict, certainty_vec: np.ndarray) -> float:
    """
    Hybrid loss L_hybrid = reconstruction loss + λ · entropy.
    """
    W = state["W"]
    lam = state["lambda_entropy"]
    rec = ttt_loss(W, certainty_vec)
    ent = shannon_entropy(certainty_vec)
    return rec + lam * ent


def hybrid_step(state: dict, certainty_vec: np.ndarray, lr: float = 0.01) -> None:
    """
    Perform a single gradient‑descent step on the hybrid loss.

    The gradient consists of the usual TTT gradient plus the derivative of the
    entropy regulariser w.r.t. the input vector.  Since the entropy term does
    not depend on **W**, only the reconstruction part contributes to the
    gradient of **W**; nevertheless we expose the step as a hybrid operation
    for conceptual symmetry.
    """
    W = state["W"]
    # Gradient of reconstruction loss w.r.t. W
    grad_W = ttt_grad(W, certainty_vec)
    # Simple SGD update
    state["W"] = W - lr * grad_W


def hybrid_decision_score(state: dict, certainty_vec: np.ndarray) -> float:
    """
    Compute a hygiene/decision score based on the transformed certainty vector.

    Steps:
        1. Transform: y = W c
        2. Normalise y to a probability‑like vector via softmax‑ish scaling.
        3. Combine with entropy: score = (1 − entropy) · mean(y_norm)

    The score lies in [0, 1] (higher = more confident & consistent).
    """
    W = state["W"]
    y = W @ certainty_vec  # shape (d_out,)

    # Stabilised exponentiation (softmax‑like)
    max_y = np.max(y)
    exp_shifted = np.exp(y - max_y)
    y_norm = exp_shifted / np.sum(exp_shifted)

    entropy = shannon_entropy(certainty_vec)  # in bits, max = log2(len)
    max_entropy = math.log2(len(certainty_vec)) if len(certainty_vec) > 1 else 1.0
    norm_entropy = entropy / max_entropy  # in [0,1]

    score = (1.0 - norm_entropy) * float(np.mean(y_norm))
    return float(score)


# ----------------------------------------------------------------------
# Demonstration utilities
# ----------------------------------------------------------------------
def generate_dummy_flags() -> List[CertaintyFlag]:
    """Create a small random list of CertaintyFlag objects for testing."""
    random.seed(42)
    flags = []
    for _ in range(7):
        label = random.choice(EPISTEMIC_FLAGS)
        confidence = random.randint(0, 10_000)
        flags.append(
            certainty(
                label,
                confidence_bps=confidence,
                authority_class="demo",
                rationale="synthetic",
            )
        )
    return flags


if __name__ == "__main__":
    # 1. Build a dummy certainty vector from flags
    flags = generate_dummy_flags()
    c_vec = _flags_to_vector(flags)  # shape (len(EPISTEMIC_FLAGS),)

    # 2. Initialise hybrid model
    state = init_hybrid(d_in=c_vec.shape[0], d_out=4, scale=0.05, seed=123)

    # 3. Show initial loss and decision score
    print("Initial hybrid loss:", hybrid_loss(state, c_vec))
    print("Initial decision score:", hybrid_decision_score(state, c_vec))

    # 4. Run a few gradient steps
    for epoch in range(5):
        hybrid_step(state, c_vec, lr=0.05)
        print(
            f"Epoch {epoch+1:02d} – loss: {hybrid_loss(state, c_vec):.6f} – "
            f"score: {hybrid_decision_score(state, c_vec):.4f}"
        )

    # 5. Verify that the weight matrix shape matches expectations
    assert state["W"].shape == (4, c_vec.shape[0]), "Weight matrix shape mismatch"
    print("Hybrid model test completed successfully.")