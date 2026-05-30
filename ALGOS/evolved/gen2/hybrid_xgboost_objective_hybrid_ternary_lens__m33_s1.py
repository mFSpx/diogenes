# DARWIN HAMMER — match 33, survivor 1
# gen: 2
# parent_a: xgboost_objective.py (gen0)
# parent_b: hybrid_ternary_lens_audit_decreasing_pruning_m15_s1.py (gen1)
# born: 2026-05-29T23:23:14Z

"""
Hybrid XGBoost and Ternary Lens Audit Algorithm.

This module bridges the mathematical structures of xgboost_objective.py and hybrid_ternary_lens_audit_decreasing_pruning_m15_s1.py.
The governing equations of XGBoost are integrated with the ternary lens audit findings and pruning schedule of the hybrid ternary lens audit algorithm.
The mathematical interface is established through the concept of loss functions and pruning probabilities.
The hybrid algorithm uses the XGBoost loss function to evaluate the audit findings and the pruning schedule to filter the lens candidates.

The XGBoost algorithm provides a comprehensive evaluation of the loss function, while the pruning algorithm introduces a dynamic filtering mechanism.
By combining these two algorithms, we create a hybrid system that effectively identifies and prioritizes high-quality lens candidates based on their loss function and pruning schedule.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

CLASSIFICATIONS = {"usable_now", "research_only", "needs_conversion", "unsafe_for_fastpath", "unsupported"}
LOCAL_PATTERNS = ["*bitnet*", "*BitNet*", "*fairyfuse*", "*FairyFuse*", "*lora*", "*LoRA*", "*adapter*"]

def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-margin))

def binary_logistic_grad_hess(y_true: np.ndarray, margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and second gradients for binary logistic loss in margin space."""
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h

def optimal_leaf_weight(gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0) -> float:
    """Optimal leaf weight for XGBoost."""
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
    """Split gain for XGBoost."""
    gl, hl = float(left_gradient), float(left_hessian)
    gr, hr = float(right_gradient), float(right_hessian)
    parent = (gl + gr) ** 2 / (hl + hr + reg_lambda)
    children = gl**2 / (hl + reg_lambda) + gr**2 / (hr + reg_lambda)
    return 0.5 * (children - parent) - gamma

def enforce_fast_path_rule(candidate: dict[str, any]) -> list[str]:
    """Enforce the fast path rule for a lens candidate."""
    findings: list[str] = []
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    notes = candidate.get("notes", "")
    if any(re.search(pattern, key + " " + family, re.I) for pattern in ["standard.*lora", "peft", "qlora"]):
        if candidate.get("classification") != "unsafe_for_fastpath" or candidate.get("fast_path_compatible"):
            findings.append("STANDARD_LORA_RULE_VIOLATION: normal PEFT/QLoRA must be unsafe_for_fastpath unless benchmark proves hot-path safety")
    if any(re.search(pattern, notes, re.I) for pattern in ["fp16", "fp32"]) and candidate.get("fast_path_compatible"):
        findings.append("FP_HOTPATH_CONFLICT: FP16/FP32 adapter note cannot be fast_path_compatible without benchmark evidence")
    if candidate.get("fast_path_compatible") and candidate.get("benchmark_required") and not candidate.get("benchmark_evidence"):
        findings.append("FAST_PATH_CLAIM_NEEDS_BENCHMARK_EVIDENCE")
    return findings

def prune_findings(findings: list[str], t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> list[str]:
    """Prune the audit findings based on a decreasing-rate schedule."""
    rng = random.Random(seed)
    p = min(1.0, lam * math.exp(-alpha * t))
    return [finding for finding in findings if rng.random() >= p]

def evaluate_candidate(candidate: dict[str, any], t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> float:
    """Evaluate a lens candidate using the XGBoost loss function and pruning schedule."""
    findings = enforce_fast_path_rule(candidate)
    pruned_findings = prune_findings(findings, t, lam, alpha, seed)
    loss = 0.0
    for finding in pruned_findings:
        loss += split_gain(1.0, 1.0, 0.0, 0.0, reg_lambda=1.0, gamma=0.0)
    return loss

def build_pruned_report(manifest: dict[str, any], root: Path, t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> dict[str, any]:
    """Build a pruned report based on the audit findings and pruning schedule."""
    candidates = []
    summary = {classification: 0 for classification in sorted(CLASSIFICATIONS)}
    hard_rule_violations = []
    for candidate in manifest.get("vendors", []):
        findings = enforce_fast_path_rule(candidate)
        pruned_findings = prune_findings(findings, t, lam, alpha, seed)
        classification = candidate["classification"]
        summary[classification] += 1
        if pruned_findings:
            hard_rule_violations.append({"candidate_key": candidate.get("candidate_key"), "findings": pruned_findings})
        loss = evaluate_candidate(candidate, t, lam, alpha, seed)
        candidates.append({
            "candidate_key": candidate.get("candidate_key"),
            "display_name": candidate.get("display_name"),
            "family": candidate.get("family"),
            "classification": classification,
            "fast_path_compatible": bool(candidate.get("fast_path_compatible")),
            "benchmark_required": bool(candidate.get("benchmark_required")),
            "license_id": candidate.get("license_id", "unknown"),
            "source_uri": candidate.get("source_uri", ""),
            "notes": candidate.get("notes", ""),
            "audit_findings": pruned_findings,
            "loss": loss,
        })
    local_hits = []
    for pattern in LOCAL_PATTERNS:
        for path in root.rglob(pattern):
            local_hits.append({"path": str(path.relative_to(root)), "kind": "dir" if path.is_dir() else "file"})
    return {
        "schema": "lucidota.ternary_lab.prune_lens_audit_report.v1",
        "created_at": "2026-05-29T23:19:56Z",
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
    root = Path(__file__).parent
    manifest = {"vendors": [{"candidate_key": "test", "display_name": "Test", "family": "Test", "classification": "usable_now", "fast_path_compatible": True, "benchmark_required": False, "license_id": "MIT", "source_uri": "https://example.com", "notes": ""}]}
    report = build_pruned_report(manifest, root, 1.0)
    print(report)