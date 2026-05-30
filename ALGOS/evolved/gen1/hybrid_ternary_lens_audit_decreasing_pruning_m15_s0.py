# DARWIN HAMMER — match 15, survivor 0
# gen: 1
# parent_a: ternary_lens_audit.py (gen0)
# parent_b: decreasing_pruning.py (gen0)
# born: 2026-05-29T23:19:56Z

#!/usr/bin/env python3
"""
This module fuses the ternary_lens_audit.py and decreasing_pruning.py algorithms.
The mathematical bridge between the two is the concept of pruning, which can be applied to the lens audit report.
The ternary lens audit report contains a list of candidates, each with a classification and a set of findings.
The decreasing pruning schedule can be used to prune the list of candidates based on their classification and findings.
The governing equation for the pruning probability is integrated into the lens audit report to create a hybrid algorithm.
"""

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Hashable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "services" / "ternary_lab" / "vendor_manifest.json"
DEFAULT_OUTPUT = ROOT / "05_OUTPUTS" / "ternary_lab" / "lens_audit_report.json"
CLASSIFICATIONS = {"usable_now", "research_only", "needs_conversion", "unsafe_for_fastpath", "unsupported"}
LOCAL_PATTERNS = ["*bitnet*", "*BitNet*", "*fairyfuse*", "*FairyFuse*", "*lora*", "*LoRA*", "*adapter*"]

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for candidate in data.get("vendors", []):
        classification = candidate.get("classification")
        if classification not in CLASSIFICATIONS:
            raise SystemExit(f"invalid classification {classification!r} for {candidate.get('candidate_key')}")
    return data

def enforce_fast_path_rule(candidate: dict[str, Any]) -> list[str]:
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

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non-negative')
    return min(1.0, lam * math.exp(-alpha * t))

def prune_candidates(candidates: list[dict[str, Any]], t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [candidate for candidate in candidates if rng.random() >= p]

def build_report(manifest: dict[str, Any], root: Path, t: float = 0.0, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> dict[str, Any]:
    candidates = []
    summary = {classification: 0 for classification in sorted(CLASSIFICATIONS)}
    hard_rule_violations = []
    for candidate in manifest.get("vendors", []):
        findings = enforce_fast_path_rule(candidate)
        classification = candidate["classification"]
        summary[classification] += 1
        if findings:
            hard_rule_violations.append({"candidate_key": candidate.get("candidate_key"), "findings": findings})
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
            "audit_findings": findings,
        })
    pruned_candidates = prune_candidates(candidates, t, lam, alpha, seed)
    local_hits = scan_local(root)
    return {
        "schema": "lucidota.ternary_lab.lens_audit_report.v1",
        "created_at": utc_now(),
        "workstream": "Ternary Lens Lab",
        "separate_from_chrono_phase_c": True,
        "fast_path_contract": manifest.get("fast_path_contract"),
        "summary": summary,
        "hard_rule_violations": hard_rule_violations,
        "candidates": pruned_candidates,
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

def scan_local(root: Path, max_hits: int = 200) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
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

def main() -> int:
    parser = sys.argv
    if len(parser) < 2:
        print("Usage: python3 hybrid_algorithm.py <manifest> <output> <root> <t> <lam> <alpha> <seed>")
        return 1
    manifest_path = Path(parser[1])
    output_path = Path(parser[2])
    root_path = Path(parser[3])
    t = float(parser[4])
    lam = float(parser[5])
    alpha = float(parser[6])
    seed = parser[7]
    manifest = load_manifest(manifest_path)
    report = build_report(manifest, root_path, t, lam, alpha, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "report": str(output_path),
        "candidate_count": len(report["candidates"]),
        "hard_rule_violations": len(report["hard_rule_violations"]),
        "local_hit_count": report["local_reference_scan"]["hit_count"],
    }, indent=2, sort_keys=True))
    return 0 if not report["hard_rule_violations"] else 4

if __name__ == "__main__":
    sys.exit(main())