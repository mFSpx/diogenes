# DARWIN HAMMER — match 4380, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_fractional_hd_m2203_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_capybara_optimizatio_m1246_s2.py (gen5)
# born: 2026-05-29T23:55:15Z

"""Hybrid Algorithm: Fusion of DARWIN HAMMER Labeling/Binding (Parent A) and 
Capybara Optimization Load‑Privacy Social Dynamics (Parent B).

Mathematical Bridge
-------------------
Parent A provides a probabilistic labeling pipeline where each document receives
votes from labeling functions.  The confidence of a label is represented as a
scalar that can be lifted to a high‑dimensional hypervector (circular convolution
binding space).

Parent B extracts three cue counts from a text and maps them to a *load‑privacy*
vector **ℓ** = (load, privacy).  This vector is interpreted as a “social state”
that can be updated toward a global best **g_best** via a simple social‑interaction
rule:
    **x′ = x + k·r·(g_best – x)**
where *k* is a learning coefficient and *r* ∈ [0,1] a random factor.

The fusion treats the load‑privacy vector **ℓ** as a weighting operator for the
label confidence hypervectors.  Concretely, for a label hypervector **h** and a
social vector **s** (ℓ padded to the hypervector dimension) we compute a bound
hypervector **b** = **h** ⊛ **s** using circular convolution (⊛).  The magnitude of
**b** is then projected back to a scalar confidence that modulates the original
vote‑based confidence.  Thus the two parent topologies are mathematically fused:
vote aggregation ↔ scalar confidence ↔ hypervector binding ↔ social dynamics.

The module implements:
* cue extraction and load‑privacy computation (Parent B),
* labeling‑function infrastructure and vote aggregation (Parent A),
* circular‑convolution binding,
* a hybrid aggregation that blends both worlds,
* a simple social‑interaction update for the load‑privacy vector.
"""

import re
import sys
import math
import random
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Sequence, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – labeling infrastructure
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # binary label {0,1}


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # in [0,1]


def labeling_function(name: str | None = None):
    """Decorator that registers a labeling function."""
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn
    return deco


def aggregate_votes(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """Simple majority‑vote aggregation with confidence = vote proportion."""
    votes: Dict[str, List[int]] = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes[r.doc_id].append(r.label)

    prob_labels: List[ProbabilisticLabel] = []
    for doc_id, lbls in votes.items():
        if not lbls:
            continue
        count_one = sum(lbls)
        count_zero = len(lbls) - count_one
        if count_one > count_zero:
            label = 1
            confidence = count_one / len(lbls)
        elif count_zero > count_one:
            label = 0
            confidence = count_zero / len(lbls)
        else:  # tie → treat as uncertain
            label = random.choice([0, 1])
            confidence = 0.5
        prob_labels.append(ProbabilisticLabel(doc_id, label, confidence))
    return prob_labels


# ----------------------------------------------------------------------
# Parent B – cue extraction and load‑privacy social vector
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|delay|postpone|defer)\b", re.I)

W_POS = np.array([1.2, 0.8, 0.5])   # evidence, planning, delay
W_NEG = np.array([0.3, 0.2, 1.0])   # same order, penalising delay more


def _count_cues(text: str) -> np.ndarray:
    """Return raw cue counts for evidence, planning, delay."""
    return np.array([
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
    ], dtype=float)


def compute_load_privacy(text: str) -> Tuple[float, float]:
    """
    Compute a 2‑dimensional social vector (load, privacy) from textual cues.
    load  = (c · (W_POS – W_NEG))
    privacy = 0.7 × delay_count
    """
    c = _count_cues(text)
    load = float(c @ (W_POS - W_NEG))
    privacy = float(c[2] * 0.7)
    return load, privacy


def social_interaction(
    x: Sequence[float],
    g_best: Sequence[float],
    k: float = 1.0,
    r: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Update social vector x towards global best g_best.
    x' = x + k * r * (g_best – x)
    """
    if seed is not None:
        random.seed(seed)
    if r is None:
        r = random.random()
    x_arr = np.asarray(x, dtype=float)
    g_arr = np.asarray(g_best, dtype=float)
    return x_arr + k * r * (g_arr - x_arr)


# ----------------------------------------------------------------------
# Hybrid building blocks
# ----------------------------------------------------------------------
def circular_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution via FFT (binding operation)."""
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape for convolution.")
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    return np.fft.ifft(A * B).real


def label_hypervector(label: int, dim: int = 128) -> np.ndarray:
    """
    Produce a deterministic hypervector for a binary label.
    1 → +1 vector, 0 → –1 vector (bipolar encoding).
    """
    rng = np.random.default_rng(seed=label)  # deterministic per label
    hv = rng.choice([-1, 1], size=dim)
    return hv.astype(float)


def hybrid_aggregate(
    batches: List[List[LabelingFunctionResult]],
    texts: Dict[str, str],
    dim: int = 128,
    g_best_social: Tuple[float, float] = (0.0, 0.0),
) -> List[ProbabilisticLabel]:
    """
    Perform vote aggregation, compute load‑privacy vectors, bind them with label
    hypervectors, and return confidence‑adjusted probabilistic labels.
    """
    # 1. Base aggregation from Parent A
    base_labels = aggregate_votes(batches)

    # 2. Prepare output list
    hybrid_labels: List[ProbabilisticLabel] = []

    for pl in base_labels:
        doc_id = pl.doc_id
        text = texts.get(doc_id, "")
        load, privacy = compute_load_privacy(text)

        # Social dynamics update (Parent B)
        social_vec = social_interaction(
            x=(load, privacy),
            g_best=g_best_social,
            k=0.5,
            seed=hash(doc_id) & 0xffffffff,
        )
        # Normalize and pad to hypervector dimension
        social_np = np.asarray(social_vec, dtype=float)
        norm = np.linalg.norm(social_np) + 1e-12
        social_norm = social_np / norm
        # Pad with zeros to match dim
        pad = np.zeros(dim - social_norm.size, dtype=float)
        social_hv = np.concatenate([social_norm, pad])

        # 3. Bind label hypervector with social hypervector
        lbl_hv = label_hypervector(pl.label, dim=dim)
        bound_hv = circular_convolution(lbl_hv, social_hv)

        # 4. Project bound hypervector back to a scalar confidence
        bound_conf = np.tanh(bound_hv.mean())  # value in (‑1,1)
        bound_conf = (bound_conf + 1) / 2      # map to [0,1]

        # 5. Fuse original confidence with bound confidence
        fused_conf = (pl.confidence + bound_conf) / 2.0

        hybrid_labels.append(
            ProbabilisticLabel(doc_id=doc_id, label=pl.label, confidence=fused_conf)
        )
    return hybrid_labels


# ----------------------------------------------------------------------
# Example labeling functions (Parent A primitives)
# ----------------------------------------------------------------------
@labeling_function()
def lf_contains_evidence(doc: dict) -> int:
    """Return 1 if any evidence cue appears, else 0."""
    text = doc.get("text", "")
    return 1 if EVIDENCE_RE.search(text) else 0


@labeling_function()
def lf_is_planful(doc: dict) -> int:
    """Return 1 if planning cues dominate over delay cues."""
    text = doc.get("text", "")
    c = _count_cues(text)
    return 1 if c[1] > c[2] else 0


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample documents
    docs = {
        "doc1": {"text": "The evidence was verified and the plan is ready."},
        "doc2": {"text": "We need to wait for the report, delay is inevitable."},
        "doc3": {"text": "Proof of concept is logged, schedule updated."},
    }

    # Run labeling functions
    all_batches: List[List[LabelingFunctionResult]] = []
    for doc_id, payload in docs.items():
        batch: List[LabelingFunctionResult] = []
        for lf in (lf_contains_evidence, lf_is_planful):
            label = lf(payload)
            batch.append(LabelingFunctionResult(lf_name=lf.lf_name, doc_id=doc_id, label=label))
        all_batches.append(batch)

    # Extract raw texts for hybrid processing
    raw_texts = {doc_id: payload["text"] for doc_id, payload in docs.items()}

    # Define a dummy global best social vector (e.g., low load, low privacy)
    global_best = (0.0, 0.0)

    # Hybrid aggregation
    result = hybrid_aggregate(all_batches, raw_texts, dim=128, g_best_social=global_best)

    # Print results
    for pl in result:
        print(f"Doc: {pl.doc_id}, Label: {pl.label}, Confidence: {pl.confidence:.3f}")