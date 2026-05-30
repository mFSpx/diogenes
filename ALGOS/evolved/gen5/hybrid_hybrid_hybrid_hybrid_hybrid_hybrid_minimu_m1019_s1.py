# DARWIN HAMMER — match 1019, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_ollivier_ricci_curva_m393_s0.py (gen4)
# parent_b: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s4.py (gen2)
# born: 2026-05-29T23:32:20Z

"""
Hybrid Algorithm: Stylometry‑Weighted Ollivier‑Ricci Curvature → Epistemic Certainty

Parents:
- hybrid_hybrid_hybrid_hard_t_ollivier_ricci_curva_m393_s0.py (stylometry → graph edge weights → Ollivier‑Ricci curvature)
- hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s4.py (certainty flag dataclass and confidence mapping)

Mathematical Bridge:
1. Text → stylometry feature vector **f** (word‑category counts).
2. Feature vector defines a node weight w_i = Σ_c f_i(c).  Nodes are the distinct
   documents (or text fragments).  Edge weight between nodes i and j is the
   cosine similarity of their feature vectors, yielding a symmetric weighted
   adjacency matrix **W**.
3. Ollivier‑Ricci curvature κ_{ij} on edge (i,j) is approximated by  

   κ_{ij} = 1 - |w_i - w_j| / (w_i + w_j + ε),

   which is a closed‑form surrogate of the optimal‑transport based definition
   when neighbor measures are taken as Dirac masses weighted by node strengths.
4. Curvature values lie in [‑1, 1]; a linear map  

   confidence_bps = int((κ_{ij}+1)/2 * 10000)

   translates curvature into basis‑points confidence.
5. Confidence is wrapped in the `CertaintyFlag` structure from the epistemic
   module, using a simple threshold scheme to select a label.

The three core functions below realise this pipeline:
- `stylometry_features(text)`: returns a dict of category counts.
- `build_weighted_graph(features_list)`: builds the adjacency matrix **W** from a list
  of feature dicts using cosine similarity and also returns node strengths.
- `curvature_to_certainty(W, strengths)`: computes κ per edge, maps to confidence,
  and yields a dict of `CertaintyFlag` objects keyed by edge tuples.
"""

import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry utilities (trimmed to essentials)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
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
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Extract lowercase word tokens (ASCII letters + optional apostrophe)."""
    return re.findall(r"\b\w+'\w+|\b\w+\b", text.lower())


def stylometry_features(text: str) -> Dict[str, int]:
    """
    Count occurrences of each functional category in `text`.
    Returns a dict mapping category name → count.
    """
    token_counts = Counter(words(text))
    cat_counts: Dict[str, int] = {cat: 0 for cat in FUNCTION_CATS}
    for token, cnt in token_counts.items():
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                cat_counts[cat] += cnt
    # also include total word count as a fallback feature
    cat_counts["total_words"] = sum(token_counts.values())
    return cat_counts


# ----------------------------------------------------------------------
# Parent B – epistemic certainty flag (unchanged)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


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
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", "2024-01-01T00:00:00Z")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------


def build_weighted_graph(
    features_list: List[Dict[str, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a symmetric weighted adjacency matrix W from a list of feature dicts.
    Edge weight w_ij = cosine_similarity(f_i, f_j) ∈ [0,1].
    Returns:
        W          – (n, n) ndarray of edge weights (zero diagonal)
        strengths  – (n,) ndarray where s_i = Σ_j w_ij (node strength)
    """
    n = len(features_list)
    if n == 0:
        raise ValueError("features_list must contain at least one element")

    # Convert dicts to dense vectors aligned on the union of keys
    all_keys = sorted({k for f in features_list for k in f.keys()})
    vecs = np.zeros((n, len(all_keys)), dtype=float)
    key_index = {k: i for i, k in enumerate(all_keys)}
    for i, feat in enumerate(features_list):
        for k, v in feat.items():
            vecs[i, key_index[k]] = float(v)

    # Normalize vectors to unit length (avoid division by zero)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_vecs = vecs / norms

    # Cosine similarity matrix
    W = unit_vecs @ unit_vecs.T
    np.fill_diagonal(W, 0.0)  # no self‑loops

    strengths = W.sum(axis=1)
    return W, strengths


def ollivier_ricci_curvature(
    strengths: np.ndarray, epsilon: float = 1e-8
) -> np.ndarray:
    """
    Approximate edge‑wise Ollivier‑Ricci curvature using node strengths.
    For edge (i,j):
        κ_ij = 1 - |s_i - s_j| / (s_i + s_j + ε)

    Returns a symmetric matrix κ with zeros on the diagonal.
    """
    s_i = strengths[:, None]  # column vector
    s_j = strengths[None, :]  # row vector
    numerator = np.abs(s_i - s_j)
    denominator = s_i + s_j + epsilon
    kappa = 1.0 - numerator / denominator
    np.fill_diagonal(kappa, 0.0)
    return kappa


def curvature_to_certainty(
    kappa: np.ndarray,
    authority_class: str = "stylometry_curvature",
) -> Dict[Tuple[int, int], CertaintyFlag]:
    """
    Translate curvature matrix κ into epistemic certainty flags per edge.
    Mapping:
        κ ≥ 0.8  → FACT
        0.5 ≤ κ < 0.8 → PROBABLE
        0.0 ≤ κ < 0.5 → POSSIBLE
        -0.5 ≤ κ < 0.0 → BULLSHIT
        κ < -0.5 → SURE_MAYBE

    Confidence (basis points) = int((κ+1)/2 * 10000)
    """
    n = kappa.shape[0]
    flags: Dict[Tuple[int, int], CertaintyFlag] = {}
    for i in range(n):
        for j in range(i + 1, n):
            curv = float(kappa[i, j])
            confidence = int((curv + 1.0) / 2.0 * 10000)
            if curv >= 0.8:
                label = "FACT"
            elif curv >= 0.5:
                label = "PROBABLE"
            elif curv >= 0.0:
                label = "POSSIBLE"
            elif curv >= -0.5:
                label = "BULLSHIT"
            else:
                label = "SURE_MAYBE"

            rationale = f"Derived from Ollivier‑Ricci curvature κ={curv:.4f}"
            evidence = (f"edge:({i},{j})", f"curvature:{curv:.4f}")
            flags[(i, j)] = certainty(
                label,
                confidence_bps=confidence,
                authority_class=authority_class,
                rationale=rationale,
                evidence_refs=evidence,
            )
    return flags


def hybrid_process(texts: List[str]) -> Dict[Tuple[int, int], CertaintyFlag]:
    """
    End‑to‑end pipeline:
        1. Extract stylometry features from each text.
        2. Build weighted graph and obtain node strengths.
        3. Approximate curvature.
        4. Map curvature to epistemic certainty flags.

    Returns a mapping edge→CertaintyFlag.
    """
    feature_dicts = [stylometry_features(t) for t in texts]
    W, strengths = build_weighted_graph(feature_dicts)
    kappa = ollivier_ricci_curvature(strengths)
    return curvature_to_certainty(kappa)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I think that the quick brown fox jumps over the lazy dog.",
        "You are certainly aware that the lazy dog was jumped over by a fox.",
        "In a distant galaxy, the stars shine brighter than ever before.",
    ]

    flags = hybrid_process(sample_texts)

    for edge, flag in flags.items():
        print(f"Edge {edge}: label={flag.label}, confidence={flag.confidence_bps}bps, rationale={flag.rationale}")
    print("Smoke test completed without errors.")