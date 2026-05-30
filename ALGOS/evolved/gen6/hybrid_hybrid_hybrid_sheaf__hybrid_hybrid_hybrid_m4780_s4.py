# DARWIN HAMMER — match 4780, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_sheaf_cohomol_hybrid_ternary_lens__m23_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m417_s2.py (gen5)
# born: 2026-05-29T23:58:09Z

import numpy as np
import math
import random
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable, Any, Optional


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Immutable representation of a candidate action."""
    id: str
    tokens: Tuple[str, ...]
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def load_manifest(path: str) -> Dict[str, Any]:
    """Load a JSON manifest from *path*."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _simple_embedding(tokens: Tuple[str, ...]) -> np.ndarray:
    """
    Produce a deterministic low‑dimensional embedding from a tuple of strings.
    The embedding is based on character code sums and is therefore reproducible.
    """
    if not tokens:
        return np.zeros(4)
    # 4‑dimensional embedding: sum of ord, sum of ord², length, variance of ord
    codes = np.fromiter((ord(ch) for token in tokens for ch in token), dtype=np.int32)
    s1 = codes.sum()
    s2 = (codes ** 2).sum()
    length = codes.size
    var = codes.var() if length > 1 else 0.0
    return np.array([s1, s2, length, var], dtype=float)


def _rbf_kernel(X: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Compute a dense RBF (Gaussian) kernel matrix for rows of *X*.
    """
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
    dists = sq_norms + sq_norms.T - 2.0 * X @ X.T
    return np.exp(-dists / (2 * sigma ** 2))


def enforce_fast_path_rule(candidate: Dict[str, Any]) -> List[str]:
    """Validate fast‑path related fields and return a list of violations."""
    findings: List[str] = []
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    notes = candidate.get("notes", "")

    if re.search(r"standard.*lora|peft|qlora", f"{key} {family}", re.I):
        if candidate.get("classification") != "unsafe_for_fastpath" or candidate.get("fast_path_compatible"):
            findings.append(
                "STANDARD_LORA_RULE_VIOLATION: normal PEFT/QLoRA must be "
                "unsafe_for_fastpath unless benchmark proves hot‑path safety"
            )

    if re.search(r"fp16|fp32", notes, re.I) and candidate.get("fast_path_compatible"):
        findings.append(
            "FP_HOTPATH_CONFLICT: FP16/FP32 adapter note cannot be fast_path_compatible "
            "without benchmark evidence"
        )

    if candidate.get("fast_path_compatible") and candidate.get("benchmark_required") and not candidate.get("benchmark_evidence"):
        findings.append("FAST_PATH_CLAIM_NEEDS_BENCHMARK_EVIDENCE")

    return findings


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """
    Compute a probability in [0, 1] that a candidate survives the pruning step.
    The functional form is `min(1, lam * exp(-alpha * t))`.
    """
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Perform the classic lead‑lag transformation on a time‑series *path*.
    The result has shape (2·T‑1, 2·d) where T is the number of timestamps
    and d the ambient dimension.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (T, d)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)

    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])

    out[2 * (T - 1)] = np.concatenate([path[-1], path[-1]])
    return out


# ----------------------------------------------------------------------
# Core fusion logic
# ----------------------------------------------------------------------
def _build_sheaf_cohomology_matrix(candidates: List[Dict[str, Any]],
                                   sigma: float = 1.0) -> np.ndarray:
    """
    Construct a deterministic “sheaf cohomology” matrix using an RBF kernel
    on simple embeddings derived from each candidate’s token list.
    """
    embeddings = np.stack(
        [_simple_embedding(tuple(candidate.get("tokens", ()))) for candidate in candidates],
        axis=0,
    )
    return _rbf_kernel(embeddings, sigma=sigma)


def _aggregate_path_features(lead_lag_path: np.ndarray) -> np.ndarray:
    """
    Produce a fixed‑size feature vector from a lead‑lag transformed path.
    Here we use the mean and standard deviation across the temporal axis,
    yielding a vector of length 2·d.
    """
    mean_feat = lead_lag_path.mean(axis=0)
    std_feat = lead_lag_path.std(axis=0)
    return np.concatenate([mean_feat, std_feat])


def hybrid_fusion(candidates: List[Dict[str, Any]],
                  path: np.ndarray,
                  t: float,
                  lam: float = 1.0,
                  alpha: float = 0.2,
                  sigma: float = 1.0,
                  seed: Optional[int] = None) -> float:
    """
    Deeply integrate the sheaf‑cohomology view of candidates with the
    lead‑lag signature of a path.

    Steps
    -----
    1. Build a deterministic sheaf‑cohomology matrix *S* from candidate
       embeddings.
    2. Transform the vector of expected values `e` via `v = S @ e`.
    3. Compute the lead‑lag transform of *path* and aggregate it to a
       feature vector `p_feat`.
    4. Align dimensions of `v` and `p_feat` by projecting `v` onto the
       subspace spanned by the first `len(v)` components of `p_feat`.
    5. Compute a regret‑weighted utility `u = v · p_proj`.
    6. Scale `u` by the pruning probability `p = prune_probability(t, lam, alpha)`.

    The function returns the final scalar utility.
    """
    rng = np.random.default_rng(seed)

    # 1‑2. Sheaf‑cohomology transformation
    S = _build_sheaf_cohomology_matrix(candidates, sigma=sigma)
    expected_vals = np.array([c.get("expected_value", 0.0) for c in candidates], dtype=float)
    v = S @ expected_vals  # shape (n,)

    # 3. Path feature extraction
    lead_lag_path = lead_lag_transform(path)
    p_feat = _aggregate_path_features(lead_lag_path)  # shape (2·d,)

    # 4. Dimension alignment
    if v.size > p_feat.size:
        # Pad the path feature vector with zeros
        p_proj = np.pad(p_feat, (0, v.size - p_feat.size), constant_values=0.0)
    else:
        p_proj = p_feat[: v.size]

    # 5. Regret‑weighted utility (scalar)
    u = float(np.dot(v, p_proj))

    # 6. Scale by pruning probability
    p = prune_probability(t, lam, alpha)
    return u * p


def prune_candidates(candidates: List[Dict[str, Any]],
                     t: float,
                     lam: float = 1.0,
                     alpha: float = 0.2,
                     seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Prune candidates using a probability that depends on *t* and on
    any rule violations detected by :func:`enforce_fast_path_rule`.

    Candidates that violate fast‑path rules are *always* removed.
    The remaining candidates survive with probability `p = prune_probability(t, lam, alpha)`.
    """
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)

    # Remove rule violators first
    clean_candidates = [
        c for c in candidates if not enforce_fast_path_rule(c)
    ]

    # Stochastic thinning based on `p`
    return [c for c in clean_candidates if rng.random() < p]


# ----------------------------------------------------------------------
# Example usage (executed when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    example_candidates = [
        {
            "candidate_key": "key1",
            "family": "family1",
            "notes": "notes1",
            "classification": "classification1",
            "expected_value": 1.0,
            "tokens": ("a", "b", "c"),
        },
        {
            "candidate_key": "key2",
            "family": "family2",
            "notes": "notes2",
            "classification": "classification2",
            "expected_value": 2.0,
            "tokens": ("x", "y"),
        },
    ]

    # Random path with 10 timestamps in 2‑D space
    rng = np.random.default_rng(42)
    example_path = rng.random((10, 2))

    t_val = 1.0
    lam_val = 1.0
    alpha_val = 0.2
    seed_val = 42

    utility = hybrid_fusion(
        example_candidates,
        example_path,
        t_val,
        lam=lam_val,
        alpha=alpha_val,
        sigma=1.0,
        seed=seed_val,
    )
    print("Scaled utility:", utility)

    pruned = prune_candidates(
        example_candidates,
        t_val,
        lam=lam_val,
        alpha=alpha_val,
        seed=seed_val,
    )
    print("Remaining candidates after pruning:", pruned)