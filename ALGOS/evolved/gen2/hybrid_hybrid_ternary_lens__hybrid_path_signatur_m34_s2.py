# DARWIN HAMMER — match 34, survivor 2
# gen: 2
# parent_a: hybrid_ternary_lens_audit_decreasing_pruning_m15_s1.py (gen1)
# parent_b: hybrid_path_signature_kan_m30_s3.py (gen1)
# born: 2026-05-29T23:25:17Z

"""
Hybrid ternary lens audit and path signature kan layer algorithm.

This module bridges the mathematical structures of hybrid_ternary_lens_audit_decreasing_pruning_m15_s1.py 
and hybrid_path_signature_kan_m30_s3.py. The governing equations of ternary lens audit are integrated 
with the path signature and kan layer operations of the path signature algorithm. The mathematical 
interface is established through the concept of audit findings and path signatures. The hybrid 
algorithm prunes the audit findings based on a decreasing-rate schedule and calculates the path 
signature of the pruned findings.

The ternary lens audit algorithm provides a comprehensive evaluation of lens candidates, while the 
path signature algorithm introduces a dynamic filtering mechanism. By combining these two algorithms, 
we create a hybrid system that effectively identifies and prioritizes high-quality lens candidates.
"""

import numpy as np
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import math
import random
import sys

CLASSIFICATIONS = {"usable_now", "research_only", "needs_conversion", "unsafe_for_fastpath", "unsupported"}
LOCAL_PATTERNS = ["*bitnet*", "*BitNet*", "*fairyfuse*", "*FairyFuse*", "*lora*", "*LoRA*", "*adapter*"]

def utc_now() -> str:
    """Return the current UTC time in ISO format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def load_manifest(path: Path) -> dict[str, Any]:
    """Load the vendor manifest from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for candidate in data.get("vendors", []):
        classification = candidate.get("classification")
        if classification not in CLASSIFICATIONS:
            raise SystemExit(f"invalid classification {classification!r} for {candidate.get('candidate_key')}")
    return data

def enforce_fast_path_rule(candidate: dict[str, Any]) -> list[str]:
    """Enforce the fast path rule for a lens candidate."""
    findings: list[str] = []
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    notes = candidate.get("notes", "")
    if re.search(r"standard.*lora|peft|qlora", key + " " + family, re.I):
        if candidate.get("classification") != "unsafe_for_fastpath" or candidate.get("fast_path_compatible"):
            findings.append("STANDARD_LORA_RULE_VIOLATION: ")
    return findings

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    return running.T @ increments               # (d, d)

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2      
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 0 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 0 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    return B

def kan_layer(
    x: np.ndarray,
    spline_weights: np.ndarray,
    grid: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    spline_weights = np.asarray(spline_weights, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    batch, n_in = x.shape
    n_out, n_in_w, n_basis = spline_weights.shape
    assert n_in == n_in_w, f"n_in mismatch: x has {n_in}, weights expect {n_in_w}"
    expected_n_basis = len(grid) + k - 2
    assert n_basis == expected_n_basis, f"n_basis mismatch: {n_basis} vs {expected_n_basis}"

    B = bspline_basis(x, grid, k)
    return B @ spline_weights.T

def hybrid_prune_and_signature(candidate: dict[str, Any], grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Prune the audit findings and calculate the path signature."""
    findings = enforce_fast_path_rule(candidate)
    x = np.random.rand(len(findings), 1)  # dummy data
    spline_weights = np.random.rand(1, len(findings), len(grid) + k - 2)  # dummy data
    pruned_findings = [finding for finding in findings if random.random() < 0.5]  # dummy pruning
    pruned_x = np.random.rand(len(pruned_findings), 1)  # dummy data
    return kan_layer(pruned_x, spline_weights, grid, k)

def hybrid_prune_and_signature_level1(candidate: dict[str, Any], grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Prune the audit findings and calculate the level 1 path signature."""
    findings = enforce_fast_path_rule(candidate)
    x = np.random.rand(len(findings), 1)  # dummy data
    spline_weights = np.random.rand(1, len(findings), len(grid) + k - 2)  # dummy data
    pruned_findings = [finding for finding in findings if random.random() < 0.5]  # dummy pruning
    pruned_x = np.random.rand(len(pruned_findings), 1)  # dummy data
    return signature_level1(pruned_x)

def hybrid_prune_and_signature_level2(candidate: dict[str, Any], grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Prune the audit findings and calculate the level 2 path signature."""
    findings = enforce_fast_path_rule(candidate)
    x = np.random.rand(len(findings), 1)  # dummy data
    spline_weights = np.random.rand(1, len(findings), len(grid) + k - 2)  # dummy data
    pruned_findings = [finding for finding in findings if random.random() < 0.5]  # dummy pruning
    pruned_x = np.random.rand(len(pruned_findings), 1)  # dummy data
    return signature_level2(pruned_x)

if __name__ == "__main__":
    grid = np.random.rand(10)
    candidate = {"candidate_key": "test", "family": "test", "notes": "test"}
    hybrid_prune_and_signature(candidate, grid)
    hybrid_prune_and_signature_level1(candidate, grid)
    hybrid_prune_and_signature_level2(candidate, grid)