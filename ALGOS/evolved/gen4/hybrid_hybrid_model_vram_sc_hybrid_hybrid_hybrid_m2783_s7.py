# DARWIN HAMMER — match 2783, survivor 7
# gen: 4
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_ternary_lens__m1552_s0.py (gen3)
# born: 2026-05-29T23:46:02Z

import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np

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
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
    score = (1 - norm_entropy) * np.mean(y_norm)
    return score


def hybrid_train(state: dict, certainty_vecs: List[np.ndarray], lr: float = 0.01, epochs: int = 100) -> None:
    """
    Train the hybrid model on a list of certainty vectors.

    Steps:
        1. For each certainty vector, compute the hybrid loss.
        2. Perform a single gradient-descent step on the hybrid loss.
    """
    for _ in range(epochs):
        for certainty_vec in certainty_vecs:
            hybrid_step(state, certainty_vec, lr)


def main():
    # Example usage:
    flags = [
        certainty("FACT", confidence_bps=1000, authority_class="HIGH", rationale="Reason 1"),
        certainty("PROBABLE", confidence_bps=500, authority_class="MEDIUM", rationale="Reason 2"),
        certainty("POSSIBLE", confidence_bps=200, authority_class="LOW", rationale="Reason 3"),
    ]
    certainty_vec = _flags_to_vector(flags)
    state = init_hybrid(d_in=len(EPISTEMIC_FLAGS), d_out=2)
    hybrid_train(state, [certainty_vec], lr=0.01, epochs=100)
    score = hybrid_decision_score(state, certainty_vec)
    print(f"Hygiene score: {score:.4f}")


if __name__ == "__main__":
    main()