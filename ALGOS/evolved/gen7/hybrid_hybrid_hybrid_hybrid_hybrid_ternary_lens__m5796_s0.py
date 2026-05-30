# DARWIN HAMMER — match 5796, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2709_s2.py (gen6)
# parent_b: hybrid_ternary_lens_router_hybrid_hybrid_hard_t_m1255_s1.py (gen4)
# born: 2026-05-30T00:04:42Z

"""Hybrid Algorithm: Fusion of hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m442_s0.py and hybrid_ternary_lens_router_hybrid_hybrid_hard_t_m1255_s1.py.
Mathematical Bridge:
- Parent A supplies epistemic certainty flags and a Bayesian marginal risk that weights Euclidean distances.
- Parent B supplies ternary vector representations derived from command/intents and a stylometric (LSM) fingerprint.
The bridge is built by (1) converting epistemic flags into numeric confidence weights, (2) using the Bayesian marginal (risk) to scale a ternary vector, and (3) appending the scaled ternary vector to a 2‑D spatial point. Distances are then computed in this augmented space, allowing a Euclidean Minimum‑Cost Spanning Tree (MST) that respects both spatial and semantic (ternary‑stylometric) information.
"""

import math
import random
import sys
import json
import hashlib
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Epistemic flag handling (Parent A)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
_EPISTEMIC_WEIGHTS: Dict[str, float] = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.1,
    "SURE_MAYBE": 0.3,
}


def epistemic_weight(flag: str) -> float:
    """Return a numeric weight for an epistemic certainty flag."""
    return _EPISTEMIC_WEIGHTS.get(flag.upper(), 0.0)


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """
    Compute the Bayesian marginal probability:
        P(E) = prior * likelihood + (1 - prior) * false_positive
    All inputs must be in [0, 1].
    """
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("All probability arguments must be in [0, 1].")
    return prior * likelihood + (1.0 - prior) * false_positive


def reconstruction_risk(
    prior: float,
    likelihood: float,
    false_positive: float,
    flag: str,
) -> float:
    """
    Combine Bayesian marginal with epistemic confidence.
    The risk score is the marginal multiplied by the epistemic weight.
    """
    marginal = bayes_marginal(prior, likelihood, false_positive)
    weight = epistemic_weight(flag)
    return marginal * weight


# ----------------------------------------------------------------------
# Ternary & Stylometric utilities (Parent B)
# ----------------------------------------------------------------------
TERNARY_DIMS = 12


def ternary_vector(
    raw_command: str,
    normalized_intent: str,
    context: Dict[str, Any],
    dims: int = TERNARY_DIMS,
) -> List[int]:
    """
    Produce a ternary vector (elements in {-1, 0, 1}) from the SHA‑256 digest of the inputs.
    """
    digest = hashlib.sha256(
        (raw_command + "\0" + normalized_intent + "\0" + json.dumps(context, sort_keys=True)).encode()
    ).digest()
    return [((digest[i] % 3) - 1) for i in range(dims)]


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Simple lexical style model: frequency of each word normalized by the most frequent word.
    """
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())
    if not words:
        return {}
    freq = Counter(words)
    max_freq = max(freq.values())
    return {w: c / max_freq for w, c in freq.items()}


def hybrid_ternary_lsm(
    text: str,
    raw_command: str,
    normalized_intent: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge stylometric fingerprint with ternary representation.
    """
    return {"lsm": lsm_vector(text), "ternary": ternary_vector(raw_command, normalized_intent, context)}


# ----------------------------------------------------------------------
# Geometric fusion (core of the hybrid)
# ----------------------------------------------------------------------
def augment_point(
    point: Tuple[float, float],
    ternary: List[int],
    risk: float,
) -> np.ndarray:
    """
    Create an augmented vector by concatenating the 2‑D spatial point with a
    risk‑scaled ternary vector.
    """
    scaled = np.array(ternary, dtype=float) * risk
    return np.concatenate([np.array(point, dtype=float), scaled])


def hybrid_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    ternary1: List[int],
    ternary2: List[int],
    risk1: float,
    risk2: float,
) -> float:
    """
    Euclidean distance in the augmented space where each ternary component
    is weighted by its respective risk score.
    """
    a = augment_point(p1, ternary1, risk1)
    b = augment_point(p2, ternary2, risk2)
    return np.linalg.norm(a - b)


def prim_mst(
    points: List[Tuple[float, float]],
    ternaries: List[List[int]],
    risks: List[float],
) -> List[Tuple[int, int, float]]:
    """
    Compute a Minimum Spanning Tree (MST) over the augmented points using Prim's algorithm.
    Returns a list of edges (i, j, weight) where i < j.
    """
    n = len(points)
    if n == 0:
        return []

    in_tree = [False] * n
    key = [math.inf] * n
    parent = [-1] * n

    key[0] = 0.0

    for _ in range(n):
        # Select the vertex with the smallest key not yet in the tree
        u = min((i for i in range(n) if not in_tree[i]), key=lambda i: key[i])
        in_tree[u] = True

        for v in range(n):
            if in_tree[v]:
                continue
            w = hybrid_distance(
                points[u],
                points[v],
                ternaries[u],
                ternaries[v],
                risks[u],
                risks[v],
            )
            if w < key[v]:
                key[v] = w
                parent[v] = u

    edges = []
    for v in range(1, n):
        i, j = parent[v], v
        if i > j:
            i, j = j, i
        edges.append((i, j, key[v]))
    return edges


# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_analysis(
    text: str,
    raw_command: str,
    normalized_intent: str,
    context: Dict[str, Any],
    point: Tuple[float, float],
    epistemic: str,
    prior: float,
    likelihood: float,
    false_positive: float,
) -> Dict[str, Any]:
    """
    Perform a full hybrid evaluation:
      1. Compute epistemic‑weighted reconstruction risk.
      2. Generate ternary vector.
      3. Produce hybrid point and distance to origin (as a simple proxy).
      4. Build an MST over a small synthetic set (origin + this point) to showcase integration.
    Returns a dictionary with intermediate results.
    """
    risk = reconstruction_risk(prior, likelihood, false_positive, epistemic)
    tern = ternary_vector(raw_command, normalized_intent, context)
    hybrid_pt = augment_point(point, tern, risk)

    # Distance to origin in augmented space
    origin = (0.0, 0.0)
    origin_tern = [0] * len(tern)
    dist_to_origin = hybrid_distance(point, origin, tern, origin_tern, risk, 0.0)

    # Build MST with just two points (origin and current)
    mst = prim_mst(
        points=[origin, point],
        ternaries=[origin_tern, tern],
        risks=[0.0, risk],
    )

    return {
        "risk_score": risk,
        "ternary_vector": tern,
        "augmented_point": hybrid_pt.tolist(),
        "distance_to_origin": dist_to_origin,
        "mst_edges": mst,
        "stylometric_fingerprint": lsm_vector(text),
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "The quick brown fox jumps over the lazy dog."
    cmd = "move"
    intent = "relocate"
    ctx = {"user": "alice", "timestamp": datetime.now(timezone.utc).isoformat()}
    pt = (3.5, -2.1)
    epi = "PROBABLE"
    prior = 0.6
    likelihood = 0.9
    false_pos = 0.05

    result = hybrid_analysis(
        text=sample_text,
        raw_command=cmd,
        normalized_intent=intent,
        context=ctx,
        point=pt,
        epistemic=epi,
        prior=prior,
        likelihood=likelihood,
        false_positive=false_pos,
    )
    print("Hybrid analysis result:")
    for k, v in result.items():
        print(f"{k}: {v}")