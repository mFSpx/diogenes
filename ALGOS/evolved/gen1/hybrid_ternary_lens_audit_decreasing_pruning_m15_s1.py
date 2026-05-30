# DARWIN HAMMER — match 15, survivor 1
# gen: 1
# parent_a: ternary_lens_audit.py (gen0)
# parent_b: decreasing_pruning.py (gen0)
# born: 2026-05-29T23:19:56Z

"""
Hybrid ternary lens audit and pruning algorithm.

This module bridges the mathematical structures of ternary_lens_audit.py and decreasing_pruning.py.
The governing equations of ternary lens audit are integrated with the decreasing-rate pruning schedule
of the pruning algorithm. The mathematical interface is established through the concept of audit findings
and pruning probabilities. The hybrid algorithm prunes the audit findings based on a decreasing-rate schedule,
allowing for adaptive filtering of lens candidates.

The ternary lens audit algorithm provides a comprehensive evaluation of lens candidates, while the pruning algorithm
introduces a dynamic filtering mechanism. By combining these two algorithms, we create a hybrid system that
effectively identifies and prioritizes high-quality lens candidates.
"""

import argparse
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
            findings.append("STANDARD_LORA_RULE_VIOLATION: normal PEFT/QLoRA must be unsafe_for_fastpath unless benchmark proves hot-path safety")
    if re.search(r"fp16|fp32", notes, re.I) and candidate.get("fast_path_compatible"):
        findings.append("FP_HOTPATH_CONFLICT: FP16/FP32 adapter note cannot be fast_path_compatible without benchmark evidence")
    if candidate.get("fast_path_compatible") and candidate.get("benchmark_required") and not candidate.get("benchmark_evidence"):
        findings.append("FAST_PATH_CLAIM_NEEDS_BENCHMARK_EVIDENCE")
    return findings

def prune_findings(findings: list[str], t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> list[str]:
    """Prune the audit findings based on a decreasing-rate schedule."""
    rng = random.Random(seed)
    p = min(1.0, lam * math.exp(-alpha * t))
    return [finding for finding in findings if rng.random() >= p]

def build_pruned_report(manifest: dict[str, Any], root: Path, t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> dict[str, Any]:
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
        })
    local_hits = []
    for pattern in LOCAL_PATTERNS:
        for path in root.rglob(pattern):
            local_hits.append({"path": str(path.relative_to(root)), "kind": "dir" if path.is_dir() else "file"})
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
            "Build tiny Command Envelope Router backend before attempting general ternary LoRA."
        ],
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="Offline prune lens audit for LUCIDOTA/FairyFuse ternary lens candidates")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "services" / "ternary_lab" / "vendor_manifest.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "05_OUTPUTS" / "ternary_lab" / "prune_lens_audit_report.json")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    report = build_pruned_report(manifest, args.root.resolve(), args.t, args.lam, args.alpha, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "report": str(args.output),
        "candidate_count": len(report["candidates"]),
        "hard_rule_violations": len(report["hard_rule_violations"]),
        "local_hit_count": report["local_reference_scan"]["hit_count"],
    }, indent=2, sort_keys=True))
    return 0 if not report["hard_rule_violations"] else 4

if __name__ == "__main__":
    sys.exit(main())