# DARWIN HAMMER — match 716, survivor 2
# gen: 3
# parent_a: hybrid_ternary_lens_audit_decreasing_pruning_m15_s2.py (gen1)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s3.py (gen2)
# born: 2026-05-29T23:30:30Z

"""
Hybrid audit-pruning module.

Parent A: ternary_lens_audit.py – builds a detailed audit report from a vendor manifest,
          classifying candidates and scanning local filesystem references.
Parent B: hybrid_krampus_brainmap_ollivier_ricci_curva.py – implements a hybrid model for
          VRAM scheduling and brain mapping, utilizing Ollivier-Ricci curvature.
Mathematical bridge:
Each candidate from the vendor manifest can be viewed as a "node" in a weighted graph.
The audit report yields a count vector **s** ∈ ℝ^k (k = number of classifications). 
  Normalising **s** gives a weight vector **w** whose components reflect the prevalence of
  each classification.  This weight vector serves as the "curvature" in the brain mapping
  context.  For a given time-step *t*, a scalar prune probability p(t) is computed (Parent A),
  and modulated per-node by the corresponding weight w_i (Parent B).  The resulting probability
  matrix

    P_i(t) = p(t) · w_i

is used to stochastically drop (prune) candidates, producing a hybrid filtered audit report
that respects both the rule-based audit and the brain-mapped pruning schedule.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Hashable, List, Mapping

import numpy as np

# ----------------------------------------------------------------------
# Constants (shared with Parent A)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "services" / "ternary_lab" / "vendor_manifest.json"
DEFAULT_OUTPUT = ROOT / "05_OUTPUTS" / "ternary_lab" / "lens_audit_report.json"
CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}
LOCAL_PATTERNS = [
    "*bitnet*",
    "*BitNet*",
    "*fairyfuse*",
    "*FairyFuse*",
    "*lora*",
    "*LoRA*",
    "*adapter*",
]

# ----------------------------------------------------------------------
# Parent B helpers (trimmed for relevance)
# ----------------------------------------------------------------------
def ollivier_ricci_curvature(graph: np.ndarray, weights: np.ndarray, t: float) -> np.ndarray:
    """
    Compute the Ollivier-Ricci curvature for a weighted graph at a given time-step t.

    Args:
    graph: Adjacency matrix of the graph.
    weights: Weight vector corresponding to each node.
    t: Time-step at which to compute the curvature.

    Returns:
    Curvature vector of the graph.
    """
    n = graph.shape[0]
    curvature = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if graph[i, j] > 0:
                curvature[i] += weights[j] * np.exp(-t * graph[i, j])
    return curvature

def hybrid_prune(graph: np.ndarray, weights: np.ndarray, t: float) -> np.ndarray:
    """
    Compute the hybrid prune probability matrix for a weighted graph at a given time-step t.

    Args:
    graph: Adjacency matrix of the graph.
    weights: Weight vector corresponding to each node.
    t: Time-step at which to compute the prune probability.

    Returns:
    Hybrid prune probability matrix.
    """
    p = np.exp(-t * np.sum(graph, axis=0))  # Parent A prune probability
    curvature = ollivier_ricci_curvature(graph, weights, t)  # Parent B curvature
    return p * curvature

# ----------------------------------------------------------------------
# Hybrid audit-pruning module
# ----------------------------------------------------------------------
def hybrid_audit_pruning(manifest: str, t: float) -> None:
    """
    Perform hybrid audit-pruning on a vendor manifest.

    Args:
    manifest: Path to the vendor manifest.
    t: Time-step at which to compute the prune probability.
    """
    # Load manifest and compute count vector
    data = json.loads(manifest.read_text(encoding="utf-8"))
    count_vector = np.array([data[classification] for classification in CLASSIFICATIONS])
    # Normalize count vector to obtain weight vector
    weight_vector = count_vector / np.sum(count_vector)
    # Compute hybrid prune probability matrix
    prune_prob = hybrid_prune(manifest, weight_vector, t)
    # Stochastically drop candidates using the prune probability matrix
    candidates = []
    for i in range(len(weight_vector)):
        if random.random() < prune_prob[i]:
            candidates.append(i)
    # Save filtered audit report
    filtered_report = {classification: count_vector[i] for i in candidates}
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUTPUT.open("w", encoding="utf-8") as fh:
        json.dump(filtered_report, fh, indent=4)

def hybrid_audit_pruning_batch(manifest_path: Path, t: float, batch_size: int) -> None:
    """
    Perform hybrid audit-pruning on a batch of vendor manifests.

    Args:
    manifest_path: Path to the directory containing the vendor manifests.
    t: Time-step at which to compute the prune probability.
    batch_size: Number of manifests to process in parallel.
    """
    with open(manifest_path, "r") as fh:
        manifests = [json.loads(line) for line in fh.readlines()]
    for i in range(0, len(manifests), batch_size):
        batch = manifests[i:i+batch_size]
        hybrid_audit_pruning_batch_helper(batch, t)

def hybrid_audit_pruning_batch_helper(batch: List[Mapping[str, Any]], t: float) -> None:
    """
    Perform hybrid audit-pruning on a batch of vendor manifests.

    Args:
    batch: Batch of vendor manifests.
    t: Time-step at which to compute the prune probability.
    """
    # Compute count vector for the batch
    count_vector = np.array([manifest[classification] for manifest in batch for classification in CLASSIFICATIONS])
    # Normalize count vector to obtain weight vector
    weight_vector = count_vector / np.sum(count_vector)
    # Compute hybrid prune probability matrix
    prune_prob = hybrid_prune(manifest_path, weight_vector, t)
    # Stochastically drop candidates using the prune probability matrix
    candidates = []
    for i in range(len(weight_vector)):
        if random.random() < prune_prob[i]:
            candidates.append(i)
    # Save filtered audit report
    filtered_report = {classification: count_vector[i] for i in candidates}
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUTPUT.open("w", encoding="utf-8") as fh:
        json.dump(filtered_report, fh, indent=4)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    manifest_path = Path("vendor_manifest.json")
    t = 1.0  # Time-step
    batch_size = 10  # Batch size
    hybrid_audit_pruning_batch(manifest_path, t, batch_size)