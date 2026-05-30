# DARWIN HAMMER — match 3372, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_gliner_zero_s_hybrid_perceptual_de_m2393_s0.py (gen4)
# parent_b: hybrid_indy_learning_vector_hybrid_hybrid_geomet_m113_s2.py (gen3)
# born: 2026-05-29T23:49:41Z

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------
Vector = np.ndarray


@dataclass(frozen=True)
class Span:
    """A span of text with an associated label and confidence score."""

    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


DEFAULT_LABELS = [
    "Operator",
    "Rainmaker",
    "Paladin / God-Mode",
    "Psyche / State-Collapse",
    "Forensic Shield",
    "Infinite Sink",
    "Anchor Weight",
    "Server Wipe",
    "API Rate Limiting",
    "Environment Migration",
    "Cruelty Protocols",
    "Master’s Eye",
    "Chrono-Ledger",
    "KRAMPUSCHEWING",
    "KORPUS",
    "DIOGENES",
    "FairyFuse",
    "Job Fair Allocator",
    "Darwinian Surfaces",
    "Command Envelope Protocol",
]

WORD_RE = re.compile(r"\S+")

DEFAULT_TERMS = (
    "ENTITY",
    "ATTRIBUTE",
    "RELATIONSHIP",
    "ACTION",
    "EVENT",
    "TIME",
    "EVIDENCE",
    "CLAIM",
    "HYPOTHESIS",
    "SIGNAL",
    "PATTERN",
    "TOOL",
    "ALGORITHM",
    "BOOK",
    "SOURCE",
    "LEAD",
    "LOCATION",
    "LAW",
    "RULE",
)


def now_iso() -> str:
    """Current UTC timestamp in ISO‑8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    """SHA‑256 hash of the supplied Unicode text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_json(value: Any) -> str:
    """Deterministic SHA‑256 of a JSON‑serialisable value."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def parse_labels(raw: str | None) -> List[str]:
    """Parse a JSON file, comma‑separated string or fallback to defaults."""
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

    # treat as a raw comma‑separated list
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_go_terms(root: Path) -> List[str]:
    """Load ontology terms; fall back to DEFAULT_TERMS."""
    p = root / "OFFICIAL_ONTOLOGY.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        terms = data.get("active_terms") or []
        return [str(t).upper() for t in terms if str(t).strip()]
    except Exception:
        return list(DEFAULT_TERMS)


def tokenize(text: str) -> List[str]:
    """Return a list of token strings (splits on whitespace)."""
    return [m.group(0) for m in WORD_RE.finditer(text)]


# ----------------------------------------------------------------------
# Vector utilities – deeper integration
# ----------------------------------------------------------------------
def build_vocabulary(labels: List[str]) -> List[str]:
    """Create a deterministic token vocabulary from a list of labels."""
    vocab_set = set()
    for label in labels:
        vocab_set.update(tokenize(label.lower()))
    # Sort to guarantee deterministic ordering
    return sorted(vocab_set)


def vectorize_labels(labels: List[str], vocab: List[str]) -> np.ndarray:
    """
    Convert each label into a term‑frequency vector over ``vocab``.
    Returns an ``(n_labels, vocab_size)`` matrix.
    """
    vocab_index = {token: i for i, token in enumerate(vocab)}
    vectors = np.zeros((len(labels), len(vocab)), dtype=float)

    for row, label in enumerate(labels):
        tokens = tokenize(label.lower())
        for token in tokens:
            idx = vocab_index.get(token)
            if idx is not None:
                vectors[row, idx] += 1.0

    # Normalise each row to unit length (L2) to obtain cosine‑compatible vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for empty vectors
    norms[norms == 0] = 1.0
    return vectors / norms


def cosine_similarity(vec_a: Vector, vec_b: Vector) -> float:
    """Cosine similarity between two 1‑D vectors."""
    return float(np.dot(vec_a, vec_b))


def compute_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Efficiently compute the full pairwise cosine similarity matrix.
    ``vectors`` must be L2‑normalised row‑wise.
    """
    return np.clip(vectors @ vectors.T, -1.0, 1.0)


# ----------------------------------------------------------------------
# Clustering – more principled than the original ad‑hoc approach
# ----------------------------------------------------------------------
def cluster_labels(
    labels: List[str],
    similarity_threshold: float = 0.6,
) -> Dict[int, List[int]]:
    """
    Cluster label indices using a simple greedy threshold algorithm.

    Parameters
    ----------
    labels:
        Original label strings.
    similarity_threshold:
        Minimum cosine similarity required to join an existing cluster.

    Returns
    -------
    Dict[int, List[int]]
        Mapping ``cluster_id -> list_of_label_indices``.
    """
    if not labels:
        return {}

    vocab = build_vocabulary(labels)
    vectors = vectorize_labels(labels, vocab)
    sim_matrix = compute_similarity_matrix(vectors)

    clusters: Dict[int, List[int]] = {}
    label_to_cluster: Dict[int, int] = {}

    next_cluster_id = 0
    for i in range(len(labels)):
        # Find best existing cluster
        best_cluster = None
        best_score = -1.0
        for cid, members in clusters.items():
            # similarity of i to the centroid (mean vector) of the cluster
            member_vecs = vectors[members]
            centroid = member_vecs.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm == 0:
                continue
            centroid = centroid / centroid_norm
            score = cosine_similarity(vectors[i], centroid)
            if score > best_score:
                best_score = score
                best_cluster = cid

        if best_cluster is not None and best_score >= similarity_threshold:
            clusters[best_cluster].append(i)
            label_to_cluster[i] = best_cluster
        else:
            clusters[next_cluster_id] = [i]
            label_to_cluster[i] = next_cluster_id
            next_cluster_id += 1

    return clusters


# ----------------------------------------------------------------------
# RBF surrogate model – now uses actual distances
# ----------------------------------------------------------------------
def rbf_kernel(distance: float, sigma: float) -> float:
    """Radial‑Basis‑Function kernel."""
    if sigma == 0:
        return 1.0 if distance == 0 else 0.0
    return math.exp(-((distance) / sigma) ** 2)


def apply_rbf(
    label_indices: List[int],
    vectors: np.ndarray,
    labels: List[str],
) -> List[Span]:
    """
    Within a cluster, compute an RBF‑based confidence for each label.

    The centroid of the cluster is used as the RBF centre.
    ``sigma`` is set to the mean Euclidean distance of members to the centroid.
    """
    if not label_indices:
        return []

    cluster_vecs = vectors[label_indices]
    centroid = cluster_vecs.mean(axis=0)
    # Euclidean distances to centroid (vectors are already normalised)
    dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
    sigma = dists.mean() if dists.size > 0 else 0.0

    spans: List[Span] = []
    for idx, dist in zip(label_indices, dists):
        score = rbf_kernel(dist, sigma)
        # Dummy span – in a real system this would be derived from the source text
        span = Span(
            start=0,
            end=0,
            text="",
            label=labels[idx],
            score=score,
            backend="rbf_surrogate",
        )
        spans.append(span)

    return spans


# ----------------------------------------------------------------------
# Public API – hybrid operation
# ----------------------------------------------------------------------
def hybrid_operation(
    labels: List[str],
    similarity_threshold: float = 0.6,
) -> List[Span]:
    """
    Perform the full hybrid pipeline:
    1. Vectorise labels.
    2. Cluster them using cosine similarity.
    3. Apply an RBF surrogate model inside each cluster to obtain scores.
    """
    if not labels:
        return []

    vocab = build_vocabulary(labels)
    vectors = vectorize_labels(labels, vocab)

    # Cluster on the pre‑computed vectors
    raw_clusters = cluster_labels(labels, similarity_threshold=similarity_threshold)

    all_spans: List[Span] = []
    for cluster_indices in raw_clusters.values():
        all_spans.extend(apply_rbf(cluster_indices, vectors, labels))

    return all_spans


# ----------------------------------------------------------------------
# CLI entry‑point
# ----------------------------------------------------------------------
def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid label clustering + RBF scoring."
    )
    parser.add_argument(
        "-l",
        "--labels",
        help="Path to JSON file or comma‑separated list of labels. "
        "If omitted, defaults are used.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.6,
        help="Cosine similarity threshold for clustering (default: 0.6).",
    )
    args = parser.parse_args()

    label_list = parse_labels(args.labels)
    spans = hybrid_operation(label_list, similarity_threshold=args.threshold)

    for span in spans:
        print(asdict(span))


if __name__ == "__main__":
    _cli()