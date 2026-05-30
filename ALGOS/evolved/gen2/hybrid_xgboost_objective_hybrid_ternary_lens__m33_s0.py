# DARWIN HAMMER — match 33, survivor 0
# gen: 2
# parent_a: xgboost_objective.py (gen0)
# parent_b: hybrid_ternary_lens_audit_decreasing_pruning_m15_s1.py (gen1)
# born: 2026-05-29T23:23:14Z

"""
Hybrid XGBoost and Ternary Lens Audit algorithm.

This module integrates the mathematical structures of XGBoost and Ternary Lens Audit. 
The governing equations of XGBoost are used to optimize the parameters of the Ternary Lens Audit algorithm. 
The mathematical interface is established through the concept of audit findings and pruning probabilities. 
The hybrid algorithm prunes the audit findings based on a decreasing-rate schedule, allowing for adaptive filtering of lens candidates.

The XGBoost algorithm provides a comprehensive evaluation of the relationship between the features and the target variable, 
while the Ternary Lens Audit algorithm introduces a dynamic filtering mechanism. 
By combining these two algorithms, we create a hybrid system that effectively identifies and prioritizes high-quality lens candidates.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

CLASSIFICATIONS = {"usable_now", "research_only", "needs_conversion", "unsafe_for_fastpath", "unsupported"}
LOCAL_PATTERNS = ["*bitnet*", "*BitNet*", "*fairyfuse*", "*FairyFuse*", "*lora*", "*LoRA*", "*adapter*"]

def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-margin))

def binary_logistic_grad_hess(y_true: np.ndarray, margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h

def optimal_leaf_weight(gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0) -> float:
    return -float(gradient_sum) / (float(hessian_sum) + float(reg_lambda))

def split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
) -> float:
    gl, hl = float(left_gradient), float(left_hessian)
    gr, hr = float(right_gradient), float(right_hessian)
    parent = (gl + gr) ** 2 / (hl + hr + reg_lambda)
    children = gl**2 / (hl + reg_lambda) + gr**2 / (hr + reg_lambda)
    return 0.5 * (children - parent) - gamma

def enforce_fast_path_rule(candidate: dict[str, any]) -> list[str]:
    findings: list[str] = []
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    notes = candidate.get("notes", "")
    if key + " " + family:
        if "standard" in key.lower() + family.lower() and ("lora" in key.lower() + family.lower() or "peft" in key.lower() + family.lower() or "qlora" in key.lower() + family.lower()):
            if candidate.get("classification") != "unsafe_for_fastpath" or candidate.get("fast_path_compatible"):
                findings.append("STANDARD_LORA_RULE_VIOLATION: normal PEFT/QLoRA must be unsafe_for_fastpath unless benchmark proves hot-path safety")
        if "fp16" in notes.lower() or "fp32" in notes.lower() and candidate.get("fast_path_compatible"):
            findings.append("FP_HOTPATH_CONFLICT: FP16/FP32 adapter note cannot be fast_path_compatible without benchmark evidence")
        if candidate.get("fast_path_compatible") and candidate.get("benchmark_required") and not candidate.get("benchmark_evidence"):
            findings.append("FAST_PATH_CLAIM_NEEDS_BENCHMARK_EVIDENCE")
    return findings

def prune_findings(findings: list[str], t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> list[str]:
    rng = random.Random(seed)
    p = min(1.0, lam * math.exp(-alpha * t))
    return [finding for finding in findings if rng.random() >= p]

def hybrid_prune_and_optimize(candidate: dict[str, any], t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> dict[str, any]:
    findings = enforce_fast_path_rule(candidate)
    pruned_findings = prune_findings(findings, t, lam, alpha, seed)
    gradient_sum = sum([1 for finding in pruned_findings if "STANDARD_LORA_RULE_VIOLATION" in finding])
    hessian_sum = sum([1 for finding in pruned_findings if "FP_HOTPATH_CONFLICT" in finding])
    optimal_weight = optimal_leaf_weight(gradient_sum, hessian_sum)
    return {
        "candidate_key": candidate.get("candidate_key"),
        "display_name": candidate.get("display_name"),
        "family": candidate.get("family"),
        "classification": candidate.get("classification"),
        "fast_path_compatible": bool(candidate.get("fast_path_compatible")),
        "benchmark_required": bool(candidate.get("benchmark_required")),
        "license_id": candidate.get("license_id", "unknown"),
        "source_uri": candidate.get("source_uri", ""),
        "notes": candidate.get("notes", ""),
        "audit_findings": pruned_findings,
        "optimal_weight": optimal_weight,
    }

def build_pruned_report(manifest: dict[str, any], root: Path, t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> dict[str, any]:
    candidates = []
    summary = {classification: 0 for classification in sorted(CLASSIFICATIONS)}
    hard_rule_violations = []
    for candidate in manifest.get("vendors", []):
        pruned_candidate = hybrid_prune_and_optimize(candidate, t, lam, alpha, seed)
        classification = candidate["classification"]
        summary[classification] += 1
        if pruned_candidate["audit_findings"]:
            hard_rule_violations.append({"candidate_key": candidate.get("candidate_key"), "findings": pruned_candidate["audit_findings"]})
        candidates.append(pruned_candidate)
    local_hits = []
    for pattern in LOCAL_PATTERNS:
        for path in root.rglob(pattern):
            local_hits.append({"path": str(path.relative_to(root)), "kind": "dir" if path.is_dir() else "file"})
    return {
        "schema": "lucidota.ternary_lab.prune_lens_audit_report.v1",
        "created_at": "2024-09-16T14:30:00Z",
        "workstream": "Ternary Lens Lab",
        "separate_from_chrono_phase_c": True,
        "fast_path_contract": manifest.get("fast_path_contract"),
        "summary": summary,
        "hard_rule_violations": hard_rule_violations,
        "candidates": candidates,
        "local_reference_scan": {
            "root": str(root),
            "hit_count": len(local_hits),
            "hits": local_hits,
        },
        "next_actions": [
            "Seed lucidota_ternary.lens_registry only after explicit execute/apply step.",
            "Benchmark any adapter candidate before setting fast_path_compatible=true.",
            "Build tiny Command Envelope Router backend before attempting general ternary LoRA."
        ],
    }

if __name__ == "__main__":
    manifest = {
        "vendors": [
            {"candidate_key": "candidate1", "display_name": "Candidate 1", "family": "Family 1", "classification": "usable_now", "fast_path_compatible": True, "benchmark_required": False, "license_id": "license1", "source_uri": "https://example.com", "notes": "Notes 1"},
            {"candidate_key": "candidate2", "display_name": "Candidate 2", "family": "Family 2", "classification": "research_only", "fast_path_compatible": False, "benchmark_required": True, "license_id": "license2", "source_uri": "https://example.com", "notes": "Notes 2"},
        ],
        "fast_path_contract": "fast_path_contract"
    }
    root = Path("/path/to/root")
    report = build_pruned_report(manifest, root, 1.0)
    print(report)