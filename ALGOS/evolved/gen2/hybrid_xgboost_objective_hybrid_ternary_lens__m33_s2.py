# DARWIN HAMMER — match 33, survivor 2
# gen: 2
# parent_a: xgboost_objective.py (gen0)
# parent_b: hybrid_ternary_lens_audit_decreasing_pruning_m15_s1.py (gen1)
# born: 2026-05-29T23:23:14Z

"""Hybrid module combining XGBoost objective mathematics with ternary lens audit pruning.

Parent A (xgboost_objective.py) provides gradient/hessian computation, optimal leaf
weight, and split‑gain formulas for a second‑order Taylor approximation of a
logistic loss.  
Parent B (hybrid_ternary_lens_audit_decreasing_pruning_…) implements an
exponential‑decay pruning schedule for audit findings.

Mathematical bridge:
- Each audit finding is treated as a positive binary label (y=1).
- A pruning “margin’’ is derived from the decreasing probability p(t)=λ·exp(−αt)
  via the logit function, turning the schedule into a logistic‑loss margin.
- Using the binary logistic gradient/hessian from Parent A we obtain aggregate
  G and H for the whole set of findings.
- XGBoost’s split‑gain formula is then employed to modulate the pruning
  probability: a larger gain (i.e. more informative findings) reduces the chance
  of being pruned, while a small gain leaves the original exponential schedule
  essentially unchanged.

The resulting functions expose a hybrid pruning operation that is fully
differentiable in the XGBoost sense yet respects the original decreasing‑rate
policy."""
from __future__ import annotations

import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – XGBoost objective utilities
# ----------------------------------------------------------------------


def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-margin))


def binary_logistic_grad_hess(
    y_true: np.ndarray, margin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """First and second gradients for binary logistic loss in margin space."""
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h


def optimal_leaf_weight(
    gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0
) -> float:
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


# ----------------------------------------------------------------------
# Parent B – Ternary lens audit & decreasing pruning utilities
# ----------------------------------------------------------------------


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


def utc_now() -> str:
    """Return the current UTC time in ISO format."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the vendor manifest from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for candidate in data.get("vendors", []):
        classification = candidate.get("classification")
        if classification not in CLASSIFICATIONS:
            raise SystemExit(
                f"invalid classification {classification!r} for {candidate.get('candidate_key')}"
            )
    return data


def enforce_fast_path_rule(candidate: dict[str, Any]) -> list[str]:
    """Enforce the fast path rule for a lens candidate."""
    findings: list[str] = []
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    notes = candidate.get("notes", "")
    if re.search(r"standard.*lora|peft|qlora", key + " " + family, re.I):
        if (
            candidate.get("classification") != "unsafe_for_fastpath"
            or candidate.get("fast_path_compatible")
        ):
            findings.append(
                "STANDARD_LORA_RULE_VIOLATION: normal PEFT/QLoRA must be unsafe_for_fastpath unless benchmark proves hot-path safety"
            )
    if re.search(r"fp16|fp32", notes, re.I) and candidate.get("fast_path_compatible"):
        findings.append(
            "FP_HOTPATH_CONFLICT: FP16/FP32 adapter note cannot be fast_path_compatible without benchmark evidence"
        )
    if (
        candidate.get("fast_path_compatible")
        and candidate.get("benchmark_required")
        and not candidate.get("benchmark_evidence")
    ):
        findings.append("FAST_PATH_CLAIM_NEEDS_BENCHMARK_EVIDENCE")
    return findings


# ----------------------------------------------------------------------
# Hybrid mathematics
# ----------------------------------------------------------------------


def compute_pruning_margin(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """
    Convert the decreasing pruning probability p(t)=lam·exp(-alpha·t) into a
    logistic margin via the logit function.
    """
    p = min(1.0, lam * math.exp(-alpha * t))
    eps = 1e-12
    p = max(min(p, 1.0 - eps), eps)
    return math.log(p / (1.0 - p))


def pruning_grad_hess_from_findings(
    findings: list[str], t: float, lam: float = 1.0, alpha: float = 0.2
) -> tuple[float, float]:
    """
    Treat each finding as a positive binary label (y=1) and compute the summed
    gradient and hessian of the logistic loss at the margin derived from the
    pruning schedule.
    """
    if not findings:
        return 0.0, 0.0
    margin = compute_pruning_margin(t, lam, alpha)
    y = np.ones(len(findings))
    margins = np.full(len(findings), margin)
    g, h = binary_logistic_grad_hess(y, margins)
    return float(g.sum()), float(h.sum())


def hybrid_prune_findings(
    findings: list[str],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    seed: int | str | None = None,
) -> list[str]:
    """
    Hybrid pruning that blends the exponential schedule with XGBoost's
    split‑gain. A larger gain (i.e. more informative findings) lowers the
    effective pruning probability.
    """
    # Aggregate gradient/hessian for the whole set of findings
    G, H = pruning_grad_hess_from_findings(findings, t, lam, alpha)

    # Compute the gain of a hypothetical split: keep (left) vs discard (right)
    gain = split_gain(
        left_gradient=G,
        left_hessian=H,
        right_gradient=0.0,
        right_hessian=0.0,
        reg_lambda=reg_lambda,
        gamma=gamma,
    )

    # Base pruning probability from the original schedule
    base_p = min(1.0, lam * math.exp(-alpha * t))

    # Adjust probability: higher gain → less pruning
    adj_p = base_p * math.exp(-gain)
    adj_p = max(0.0, min(1.0, adj_p))

    rng = random.Random(seed)
    return [f for f in findings if rng.random() >= adj_p]


def build_hybrid_pruned_report(
    manifest: dict[str, Any],
    root: Path,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    seed: int | str | None = None,
) -> dict[str, Any]:
    """
    Produce a report where each candidate's audit findings are filtered through
    the hybrid pruning mechanism.
    """
    candidates = []
    summary = {c: 0 for c in sorted(CLASSIFICATIONS)}
    hard_rule_violations = []

    for candidate in manifest.get("vendors", []):
        raw_findings = enforce_fast_path_rule(candidate)
        pruned_findings = hybrid_prune_findings(
            raw_findings,
            t,
            lam=lam,
            alpha=alpha,
            reg_lambda=reg_lambda,
            gamma=gamma,
            seed=seed,
        )
        classification = candidate["classification"]
        summary[classification] += 1
        if pruned_findings:
            hard_rule_violations.append(
                {"candidate_key": candidate.get("candidate_key"), "findings": pruned_findings}
            )
        candidates.append(
            {
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
            }
        )

    # Local reference scan (unchanged from parent B)
    local_hits = []
    for pattern in LOCAL_PATTERNS:
        for path in root.rglob(pattern):
            local_hits.append(
                {
                    "path": str(path.relative_to(root)),
                    "kind": "dir" if path.is_dir() else "file",
                }
            )

    return {
        "schema": "lucidota.ternary_lab.prune_lens_audit_report.v1",
        "created_at": utc_now(),
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
            "Build tiny Command Envelope Router backend before attempting general ternary LoRA.",
        ],
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Minimal inline manifest for demonstration
    demo_manifest = {
        "fast_path_contract": "example_contract",
        "vendors": [
            {
                "candidate_key": "adapter_xyz",
                "display_name": "Adapter XYZ",
                "family": "LoRA",
                "classification": "needs_conversion",
                "fast_path_compatible": True,
                "benchmark_required": True,
                "benchmark_evidence": False,
                "notes": "fp16 version available",
                "license_id": "MIT",
                "source_uri": "https://example.com/adapter_xyz",
            },
            {
                "candidate_key": "standard_lora_123",
                "display_name": "Standard LoRA 123",
                "family": "standard_lora",
                "classification": "unsupported",
                "fast_path_compatible": False,
                "benchmark_required": False,
                "notes": "",
                "license_id": "Apache-2.0",
                "source_uri": "https://example.com/standard_lora_123",
            },
        ],
    }

    # Use current directory as dummy root (no actual file scanning needed)
    root_path = Path(".")
    report = build_hybrid_pruned_report(
        manifest=demo_manifest,
        root=root_path,
        t=0.7,
        lam=1.0,
        alpha=0.2,
        reg_lambda=1.0,
        gamma=0.0,
        seed=42,
    )
    # Print a concise summary to verify execution
    print(json.dumps(report["summary"], indent=2))
    print("Hard rule violations after hybrid pruning:")
    for vr in report["hard_rule_violations"]:
        print(vr)