# DARWIN HAMMER — match 2910, survivor 0
# gen: 6
# parent_a: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s3.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2680_s0.py (gen5)
# born: 2026-05-29T23:46:35Z

"""Hybrid Text Span Extraction with Certainty-Weighted RBF Kernel

Parents:
- hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s3.py (Span extraction)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2680_s0.py (Certainty flags & RBF kernel)

Mathematical bridge:
Each extracted Span is represented as a low‑dimensional vector  
v = [start, end, score].  The Hybrid RBF module defines a Gaussian
kernel K(i,j)=exp(-‖v_i‑v_j‖²).  The CertaintyFlag supplies a confidence
weight w∈[0,1] (confidence_bps/10000).  By scaling the kernel with the
product of the two weights we obtain a certainty‑weighted similarity
matrix S(i,j)=w_i·w_j·K(i,j).  This matrix fuses the linguistic
extraction (parent A) with the epistemic certainty modelling (parent B)
into a single unified system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Span extraction utilities
# ----------------------------------------------------------------------
DEFAULT_LABELS = [
    "Operator", "Rainmaker", "Paladin / God-Mode", "Psyche / State-Collapse",
    "Forensic Shield", "Infinite Sink", "Anchor Weight", "Server Wipe",
    "API Rate Limiting", "Environment Migration", "Cruelty Protocols",
    "Master’s Eye", "Chrono-Ledger", "KRAMPUSCHEWING", "KORPUS",
    "DIOGENES", "FairyFuse", "Job Fair Allocator", "Darwinian Surfaces",
    "Command Envelope Protocol",
]

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

def parse_labels(raw: str | None) -> List[str]:
    if not raw:
        return list(DEFAULT_LABELS)
    p = Path(raw)
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            labels = data.get("required_exact_labels") or data.get("labels") or []
        else:
            labels = data
        return [str(x).strip() for x in labels if str(x).strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]

def literal_fallback(text: str, labels: List[str], *, case_sensitive: bool = False) -> List[Span]:
    """Simple exact‑match extractor used as the backend for parent A."""
    flags = 0 if case_sensitive else re.IGNORECASE
    spans: List[Span] = []
    seen: set[Tuple[int, int, str]] = set()
    for label in labels:
        # generate a few normalised variants to increase recall
        candidates = {
            label,
            label.replace(" / ", " "),
            label.replace("-", " "),
        }
        for phrase in sorted(candidates, key=len, reverse=True):
            if not phrase.strip():
                continue
            pattern = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", flags)
            for m in pattern.finditer(text):
                key = (m.start(), m.end(), label)
                if key in seen:
                    continue
                seen.add(key)
                span = Span(
                    start=m.start(),
                    end=m.end(),
                    text=m.group(0),
                    label=label,
                    score=1.0,               # literal fallback assigns max confidence
                )
                spans.append(span)
    return spans

# ----------------------------------------------------------------------
# Parent B – Certainty flag & RBF utilities
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int               # basis points, 0..10000 → weight 0..1
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian (RBF) kernel with tunable bandwidth epsilon."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# ----------------------------------------------------------------------
# Hybrid core – bridging the two parents
# ----------------------------------------------------------------------
def span_to_vector(span: Span) -> np.ndarray:
    """Map a Span to a numeric vector for kernel computation."""
    return np.array([float(span.start), float(span.end), float(span.score)], dtype=np.float64)

def assign_certainty_flags(spans: List[Span]) -> List[CertaintyFlag]:
    """
    Very coarse heuristic: map label patterns to epistemic flags.
    This is the interface where linguistic information meets epistemic certainty.
    """
    flags: List[CertaintyFlag] = []
    for span in spans:
        # Example heuristic – labels containing “God” are treated as FACT,
        # those with “Possible” as POSSIBLE, otherwise PROBABLE.
        label_up = span.label.upper()
        if "GOD" in label_up or "MASTER" in label_up:
            flag_label = "FACT"
            confidence = 9500
        elif "POSSIBLE" in label_up or "MAYBE" in label_up:
            flag_label = "POSSIBLE"
            confidence = 6000
        else:
            flag_label = "PROBABLE"
            confidence = 8000
        flags.append(
            CertaintyFlag(
                label=flag_label,
                confidence_bps=confidence,
                authority_class="hybrid_extractor",
                rationale=f"derived from label '{span.label}'",
                generated_at=now_iso(),
            )
        )
    return flags

def certainty_weight(flag: CertaintyFlag) -> float:
    """Convert basis‑point confidence to a [0,1] weight."""
    return flag.confidence_bps / 10000.0

def compute_certainty_weighted_rbf(spans: List[Span], flags: List[CertaintyFlag],
                                   epsilon: float = 1.0) -> np.ndarray:
    """
    Build the certainty‑weighted RBF similarity matrix S where

        S_ij = w_i * w_j * exp(- (epsilon * ||v_i - v_j||)^2)

    with v_i the vector representation of span i and w_i the confidence weight.
    """
    if len(spans) != len(flags):
        raise ValueError("spans and flags must have the same length")
    n = len(spans)
    vectors = np.stack([span_to_vector(s) for s in spans])          # shape (n,3)
    weights = np.array([certainty_weight(f) for f in flags])       # shape (n,)

    # Compute pairwise Euclidean distances efficiently
    diff = vectors[:, None, :] - vectors[None, :, :]                # (n,n,3)
    dists = np.linalg.norm(diff, axis=2)                           # (n,n)

    # Gaussian kernel
    kernel = np.exp(-((epsilon * dists) ** 2))

    # Outer product of weights to scale the kernel
    weight_matrix = np.outer(weights, weights)

    return kernel * weight_matrix

def cluster_spans_by_similarity(sim_matrix: np.ndarray, threshold: float = 0.5) -> List[List[int]]:
    """
    Very simple agglomerative clustering based on the similarity matrix.
    Returns a list of clusters, each a list of span indices.
    """
    n = sim_matrix.shape[0]
    visited = [False] * n
    clusters: List[List[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        # start a new cluster
        cluster = [i]
        visited[i] = True
        # add any j with similarity >= threshold to the same cluster
        for j in range(i + 1, n):
            if not visited[j] and sim_matrix[i, j] >= threshold:
                cluster.append(j)
                visited[j] = True
        clusters.append(cluster)
    return clusters

# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def extract_and_weight(text: str, label_source: str | None = None) -> Tuple[List[Span], np.ndarray]:
    """Full pipeline: extract spans, assign certainty, compute weighted RBF matrix."""
    labels = parse_labels(label_source)
    spans = literal_fallback(text, labels)
    flags = assign_certainty_flags(spans)
    sim_matrix = compute_certainty_weighted_rbf(spans, flags)
    return spans, sim_matrix

def summarize_clusters(spans: List[Span], clusters: List[List[int]]) -> List[Dict[str, Any]]:
    """Produce a human‑readable summary of each cluster."""
    summaries = []
    for idx, cluster in enumerate(clusters, start=1):
        texts = [spans[i].text for i in cluster]
        labels = [spans[i].label for i in cluster]
        summary = {
            "cluster_id": idx,
            "size": len(cluster),
            "texts": texts,
            "labels": list(set(labels)),
        }
        summaries.append(summary)
    return summaries

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The Operator initiated the Server Wipe while the Infinite Sink "
        "was engaged. Meanwhile, the Paladin / God-Mode protocol ensured "
        "the system remained stable. Possible anomalies were logged."
    )
    # Run the full hybrid pipeline
    extracted_spans, similarity = extract_and_weight(sample_text)

    print(f"Extracted {len(extracted_spans)} spans:")
    for s in extracted_spans:
        print(f"  [{s.start}:{s.end}] '{s.text}' → {s.label} (score={s.score})")

    print("\nCertainty‑weighted similarity matrix (rounded):")
    print(np.round(similarity, 3))

    clusters = cluster_spans_by_similarity(similarity, threshold=0.6)
    summaries = summarize_clusters(extracted_spans, clusters)

    print("\nClusters:")
    for summ in summaries:
        print(f"Cluster {summ['cluster_id']} (size {summ['size']}):")
        for txt in summ["texts"]:
            print(f"  - {txt}")
        print(f"  Labels: {', '.join(summ['labels'])}")

    sys.exit(0)