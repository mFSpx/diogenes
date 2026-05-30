# DARWIN HAMMER — match 15, survivor 2
# gen: 1
# parent_a: ternary_lens_audit.py (gen0)
# parent_b: decreasing_pruning.py (gen0)
# born: 2026-05-29T23:19:56Z

"""Hybrid audit‑pruning module.

Parent A: ternary_lens_audit.py – builds a detailed audit report from a vendor
manifest, classifying candidates and scanning local filesystem references.

Parent B: decreasing_pruning.py – provides a decreasing‑rate pruning schedule
with probability p(t)=min(1, λ·exp(-α·t)).

Mathematical bridge:
Each candidate from the manifest can be viewed as an “edge” in a graph.
The audit summary yields a count vector **s** ∈ ℝ^k (k = number of
classifications).  Normalising **s** gives a weight vector **w** whose
components reflect the prevalence of each classification.  For a given
time‑step *t* we compute a scalar prune probability p(t) (Parent B) and
modulate it per‑candidate by the corresponding weight w_i.  The resulting
probability matrix

    P_i(t) = p(t) · w_{class(i)}

is used to stochastically drop (prune) candidates, producing a hybrid
filtered audit report that respects both the rule‑based audit and the
time‑decaying pruning schedule.
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
# Parent A helpers (trimmed for relevance)
# ----------------------------------------------------------------------
def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for candidate in data.get("vendors", []):
        classification = candidate.get("classification")
        if classification not in CLASSIFICATIONS:
            raise SystemExit(
                f"invalid classification {classification!r} for {candidate.get('candidate_key')}"
            )
    return data


def enforce_fast_path_rule(candidate: Mapping[str, Any]) -> List[str]:
    findings: List[str] = []
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


def scan_local(root: Path, max_hits: int = 200) -> List[dict[str, str]]:
    hits: List[dict[str, str]] = []
    seen: set[Path] = set()
    skip_parts = {".git", "target", "node_modules", ".cache"}
    for pattern in LOCAL_PATTERNS:
        for path in root.rglob(pattern):
            if len(hits) >= max_hits:
                return hits
            if any(part in skip_parts for part in path.parts):
                continue
            if path in seen:
                continue
            seen.add(path)
            kind = "dir" if path.is_dir() else "file"
            hits.append({"path": str(path.relative_to(root)), "kind": kind})
    return hits


# ----------------------------------------------------------------------
# Parent B functions (pruning schedule)
# ----------------------------------------------------------------------
def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


def prune_edges(
    edges: List[Hashable],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
) -> List[Hashable]:
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]


# ----------------------------------------------------------------------
# Hybrid core (mathematical fusion)
# ----------------------------------------------------------------------
def classification_summary_vector(manifest: Mapping[str, Any]) -> np.ndarray:
    """Return a normalised count vector over CLASSIFICATIONS."""
    counts = np.array(
        [
            sum(1 for c in manifest.get("vendors", []) if c.get("classification") == cls)
            for cls in sorted(CLASSIFICATIONS)
        ],
        dtype=float,
    )
    total = counts.sum()
    return counts / total if total > 0 else counts


def per_candidate_prune_mask(
    candidates: List[Mapping[str, Any]],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
) -> List[bool]:
    """
    Compute a boolean mask indicating which candidates survive pruning.

    The base prune probability p(t) from Parent B is scaled by the
    normalised prevalence of the candidate's classification (weight w_i).
    The final probability for candidate i is  p_i = p(t) * w_i .
    """
    rng = random.Random(seed)
    base_p = prune_probability(t, lam, alpha)

    # Build weight map: classification -> weight
    weight_vec = classification_summary_vector({"vendors": candidates})
    class_to_weight = {
        cls: weight_vec[idx] for idx, cls in enumerate(sorted(CLASSIFICATIONS))
    }

    mask: List[bool] = []
    for cand in candidates:
        cls = cand.get("classification")
        weight = class_to_weight.get(cls, 0.0)
        prob = base_p * weight
        keep = rng.random() >= prob
        mask.append(keep)
    return mask


def build_hybrid_report(
    manifest: Mapping[str, Any],
    root: Path,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
) -> dict[str, Any]:
    """
    Produce a report identical to Parent A but with candidates filtered by the
    hybrid pruning schedule.
    """
    # Apply rule‑based findings first
    raw_candidates = manifest.get("vendors", [])
    for cand in raw_candidates:
        cand["audit_findings"] = enforce_fast_path_rule(cand)

    # Determine which candidates survive the hybrid prune
    keep_mask = per_candidate_prune_mask(
        raw_candidates, t=t, lam=lam, alpha=alpha, seed=seed
    )
    filtered_candidates = [
        cand for cand, keep in zip(raw_candidates, keep_mask) if keep
    ]

    # Build summary counts from the filtered set
    summary = {cls: 0 for cls in sorted(CLASSIFICATIONS)}
    hard_rule_violations: List[dict[str, Any]] = []
    for cand in filtered_candidates:
        cls = cand["classification"]
        summary[cls] += 1
        if cand["audit_findings"]:
            hard_rule_violations.append(
                {"candidate_key": cand.get("candidate_key"), "findings": cand["audit_findings"]}
            )

    local_hits = scan_local(root)

    return {
        "schema": "lucidota.ternary_lab.lens_audit_report.v1",
        "created_at": utc_now(),
        "workstream": "Ternary Lens Lab (Hybrid)",
        "fast_path_contract": manifest.get("fast_path_contract"),
        "summary": summary,
        "hard_rule_violations": hard_rule_violations,
        "candidates": filtered_candidates,
        "local_reference_scan": {
            "root": str(root),
            "hit_count": len(local_hits),
            "hits": local_hits,
        },
        "pruning_parameters": {
            "time": t,
            "lambda": lam,
            "alpha": alpha,
            "seed": seed,
        },
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
def _smoke_test() -> int:
    # Create a minimal in‑memory manifest
    dummy_manifest = {
        "fast_path_contract": "v1",
        "vendors": [
            {
                "candidate_key": "adapterA",
                "display_name": "Adapter A",
                "family": "standard_lora",
                "classification": "unsafe_for_fastpath",
                "fast_path_compatible": False,
                "benchmark_required": True,
                "benchmark_evidence": None,
                "license_id": "MIT",
                "source_uri": "https://example.com/A",
                "notes": "fp16 adapter",
            },
            {
                "candidate_key": "adapterB",
                "display_name": "Adapter B",
                "family": "custom",
                "classification": "usable_now",
                "fast_path_compatible": True,
                "benchmark_required": False,
                "benchmark_evidence": None,
                "license_id": "Apache-2.0",
                "source_uri": "https://example.com/B",
                "notes": "",
            },
            {
                "candidate_key": "adapterC",
                "display_name": "Adapter C",
                "family": "peft",
                "classification": "research_only",
                "fast_path_compatible": False,
                "benchmark_required": False,
                "benchmark_evidence": None,
                "license_id": "GPL-3.0",
                "source_uri": "https://example.com/C",
                "notes": "",
            },
        ],
    }

    # Use the repository root as the scan directory (no heavy I/O)
    report = build_hybrid_report(
        manifest=dummy_manifest,
        root=ROOT,
        t=5.0,          # arbitrary time step
        lam=0.9,
        alpha=0.15,
        seed=42,
    )

    # Simple sanity checks – they should never raise
    assert "candidates" in report
    assert isinstance(report["summary"], dict)
    assert isinstance(report["pruning_parameters"], dict)

    # Print a concise summary for visual confirmation (optional)
    print(json.dumps(
        {
            "candidate_count": len(report["candidates"]),
            "hard_rule_violations": len(report["hard_rule_violations"]),
            "summary": report["summary"],
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(_smoke_test())