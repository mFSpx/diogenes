# DARWIN HAMMER — match 2783, survivor 6
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
    return CertaintyFlag(
        label=label,
        confidence_bps=confidence_bps,
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(evidence_refs),
    )


def _flags_to_vector(flags: List[CertaintyFlag]) -> np.ndarray:
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
    mask = p > 0
    return -float(np.sum(p[mask] * np.log2(p[mask])))


def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)


def init_hybrid(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> dict:
    return {
        "W": init_ttt(d_in, d_out, scale, seed),
        "lambda_entropy": 0.1,
    }


def hybrid_loss(state: dict, certainty_vec: np.ndarray) -> float:
    W = state["W"]
    lam = state["lambda_entropy"]
    rec = ttt_loss(W, certainty_vec)
    ent = shannon_entropy(certainty_vec)
    return rec + lam * ent


def hybrid_step(state: dict, certainty_vec: np.ndarray, lr: float = 0.01) -> None:
    W = state["W"]
    grad_W = ttt_grad(W, certainty_vec)
    grad_lambda = shannon_entropy(certainty_vec)
    state["W"] = W - lr * grad_W
    state["lambda_entropy"] = state["lambda_entropy"] + lr * grad_lambda * 0.1


def hybrid_decision_score(state: dict, certainty_vec: np.ndarray) -> float:
    W = state["W"]
    y = W @ certainty_vec  
    max_y = np.max(y)
    exp_shifted = np.exp(y - max_y)
    y_norm = exp_shifted / np.sum(exp_shifted)

    entropy = shannon_entropy(certainty_vec)  
    max_entropy = math.log2(len(certainty_vec)) if len(certainty_vec) > 1 else 1.0
    norm_entropy = entropy / max_entropy
    return (1 - norm_entropy) * np.mean(y_norm)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * (np.log2(p[mask]) - np.log2(q[mask]))))


def hybrid_step_kl(state: dict, certainty_vec: np.ndarray, prior: np.ndarray, lr: float = 0.01) -> None:
    W = state["W"]
    grad_W = ttt_grad(W, certainty_vec)
    kl_div = kl_divergence(certainty_vec, prior)
    state["W"] = W - lr * grad_W
    state["lambda_entropy"] = state["lambda_entropy"] + lr * kl_div * 0.1


def hybrid_loss_kl(state: dict, certainty_vec: np.ndarray, prior: np.ndarray) -> float:
    W = state["W"]
    lam = state["lambda_entropy"]
    rec = ttt_loss(W, certainty_vec)
    ent = shannon_entropy(certainty_vec)
    kl_div = kl_divergence(certainty_vec, prior)
    return rec + lam * ent + 0.1 * kl_div