# DARWIN HAMMER — match 19, survivor 2
# gen: 2
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s2.py (gen1)
# parent_b: ternary_lens_audit.py (gen0)
# born: 2026-05-29T23:23:03Z

"""Hybrid Ternary Lens Audit & Decision‑Hygiene Module.

Parents:
* **decision_hygiene.py** – extracts a 9‑dimensional feature‑count vector from free‑text
  and computes a weighted hygiene score via a dot‑product with positive/negative
  weight vectors.
* **ternary_lens_audit.py** – loads a vendor manifest, validates classifications,
  scans the local repository and builds a structured audit report.

Mathematical bridge:
For every vendor *candidate* we treat the concatenated textual fields
(`candidate_key`, `display_name`, `family`, `notes`) as a document.
The same regex‑based extraction that yields the feature‑count vector **v** ∈ ℕ⁹
is applied.  This vector participates in two coupled operations:

1. **Hygiene score**   s = w⁺·v − w⁻·v   (dot‑products with the parent‑A weight vectors)
2. **Shannon entropy** H = −∑ pᵢ log₂ pᵢ,  where p = v / Σv  (parent‑B supplies the probability
   interpretation)

The hybrid candidate metric is defined as  

  Sₕ = s · (1 + H / Hₘₐₓ)  

with Hₘₐₓ = log₂ 9 (entropy of a uniform distribution over the nine features).
Thus a candidate receives a higher overall rating when it is both
well‑hygienic *and* information‑rich.

The module combines the per‑candidate hybrid scores with the original
classification summary of the ternary‑lens audit, producing a single
unified JSON report."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Regex feature set – identical to parent A
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
    re.I,
)

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

# ----------------------------------------------------------------------
# Parent‑B constants
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

# ----------------------------------------------------------------------
# Utility: Shannon entropy (parent A)
# ----------------------------------------------------------------------
def shannon_entropy(observations: Iterable[float | Any], is_distribution: bool = False) -> float:
    xs = list(observations)
    if not xs:
        return 0.0
    if is_distribution:
        probs = [float(x) for x in xs]
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        cnt = Counter(xs)
        total = float(sum(cnt.values()))
        probs = [v / total for v in cnt.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0.0)


# ----------------------------------------------------------------------
# Feature extraction (parent A core)
# ----------------------------------------------------------------------
def _raw_counts(text: str) -> dict[str, int]:
    return {
        "evidence_count": len(EVIDENCE_RE.findall(text or "")),
        "planning_count": len(PLANNING_RE.findall(text or "")),
        "delay_count": len(DELAY_RE.findall(text or "")),
        "support_count": len(SUPPORT_RE.findall(text or "")),
        "boundary_count": len(BOUNDARY_RE.findall(text or "")),
        "outcome_count": len(OUTCOME_RE.findall(text or "")),
        "impulsive_count": len(IMPULSIVE_RE.findall(text or "")),
        "scarcity_count": len(SCARCITY_RE.findall(text or "")),
        "risk_count": len(RISK_RE.findall(text or "")),
    }


def feature_vector(text: str) -> np.ndarray:
    c = _raw_counts(text)
    return np.array(
        [
            c["evidence_count"],
            c["planning_count"],
            c["delay_count"],
            c["support_count"],
            c["boundary_count"],
            c["outcome_count"],
            c["impulsive_count"],
            c["scarcity_count"],
            c["risk_count"],
        ],
        dtype=np.int64,
    )


def hygiene_score(vector: np.ndarray) -> Tuple[int, str]:
    positive = int(np.dot(vector, _POSITIVE_WEIGHTS))
    negative = int(np.dot(vector, _NEGATIVE_WEIGHTS))
    raw = max(-10000, min(10000, positive - negative))
    if raw > 6000:
        label = "EXCELLENT"
    elif raw > 2000:
        label = "GOOD"
    elif raw > -2000:
        label = "NEUTRAL"
    elif raw > -6000:
        label = "POOR"
    else:
        label = "DISASTROUS"
    return raw, label


# ----------------------------------------------------------------------
# Parent‑B helpers (trimmed to essentials)
# ----------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for cand in data.get("vendors", []):
        classification = cand.get("classification")
        if classification not in CLASSIFICATIONS:
            raise SystemExit(f"invalid classification {classification!r} for {cand.get('candidate_key')}")
    return data


def enforce_fast_path_rule(candidate: dict[str, Any]) -> List[str]:
    findings: List[str] = []
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    notes = candidate.get("notes", "")
    if re.search(r"standard.*lora|peft|qlora", key + " " + family, re.I):
        if candidate.get("classification") != "unsafe_for_fastpath" or candidate.get("fast_path_compatible"):
            findings.append(
                "STANDARD_LORA_RULE_VIOLATION: normal PEFT/QLoRA must be unsafe_for_fastpath unless benchmark proves hot-path safety"
            )
    if re.search(r"fp16|fp32", notes, re.I) and candidate.get("fast_path_compatible"):
        findings.append("FP_HOTPATH_CONFLICT: FP16/FP32 adapter note cannot be fast_path_compatible without benchmark evidence")
    if candidate.get("fast_path_compatible") and candidate.get("benchmark_required") and not candidate.get("benchmark_evidence"):
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
# Hybrid operations
# ----------------------------------------------------------------------
_MAX_ENTROPY = math.log2(len(_FEATURE_ORDER))  # log2(9)


def candidate_feature_vector(candidate: dict[str, Any]) -> np.ndarray:
    """Create a feature vector from the free‑text fields of a manifest candidate."""
    text_parts = [
        str(candidate.get("candidate_key", "")),
        str(candidate.get("display_name", "")),
        str(candidate.get("family", "")),
        str(candidate.get("notes", "")),
    ]
    return feature_vector(" ".join(text_parts))


def hybrid_candidate_score(candidate: dict[str, Any]) -> dict[str, Any]:
    """Compute the hybrid metric Sₕ for a single vendor candidate."""
    vec = candidate_feature_vector(candidate)
    hygiene, label = hygiene_score(vec)

    total = int(vec.sum())
    if total == 0:
        entropy = 0.0
    else:
        probs = vec.astype(float) / total
        entropy = shannon_entropy(probs, is_distribution=True)

    hybrid = hygiene * (1.0 + entropy / _MAX_ENTROPY)

    return {
        "candidate_key": candidate.get("candidate_key"),
        "hygiene_score": hygiene,
        "hygiene_label": label,
        "entropy_bits": round(entropy, 4),
        "hybrid_score": round(hybrid, 2),
        "findings": enforce_fast_path_rule(candidate),
    }


def build_hybrid_report(manifest_path: Path, root_path: Path) -> dict[str, Any]:
    """Load a ternary‑lens manifest, enrich each candidate with hybrid scores,
    and assemble a unified audit report."""
    manifest = load_manifest(manifest_path)
    enriched_candidates = [hybrid_candidate_score(c) for c in manifest.get("vendors", [])]

    # Original classification summary (parent B)
    summary = {cls: 0 for cls in sorted(CLASSIFICATIONS)}
    for cand in manifest.get("vendors", []):
        summary[cand["classification"]] += 1

    # Aggregate hybrid statistics
    hybrid_scores = [c["hybrid_score"] for c in enriched_candidates if isinstance(c["hybrid_score"], (int, float))]
    agg = {
        "mean_hybrid_score": round(float(np.mean(hybrid_scores)) if hybrid_scores else 0.0, 2),
        "max_hybrid_score": round(float(np.max(hybrid_scores)) if hybrid_scores else 0.0, 2),
        "min_hybrid_score": round(float(np.min(hybrid_scores)) if hybrid_scores else 0.0, 2),
    }

    local_hits = scan_local(root_path)

    return {
        "schema": "lucidota.hybrid.ternary_lens_audit.v1",
        "generated_at": utc_now(),
        "manifest_path": str(manifest_path),
        "summary_by_classification": summary,
        "hybrid_statistics": agg,
        "candidates": enriched_candidates,
        "local_reference_scan": {
            "root": str(root_path),
            "hit_count": len(local_hits),
            "hits": local_hits,
        },
    }


# ----------------------------------------------------------------------
# CLI entry point (smoke test)
# ----------------------------------------------------------------------
def _create_dummy_manifest(path: Path) -> None:
    """Generate a tiny manifest with random textual noise for the smoke test."""
    random.seed(0)
    dummy = {
        "fast_path_contract": "example_contract",
        "vendors": [
            {
                "candidate_key": f"model_{i}",
                "display_name": f"Model {i}",
                "family": random.choice(["lora", "adapter", "base"]),
                "classification": random.choice(list(CLASSIFICATIONS)),
                "fast_path_compatible": random.choice([True, False]),
                "benchmark_required": random.choice([True, False]),
                "benchmark_evidence": random.choice([True, False]),
                "license_id": "MIT",
                "source_uri": f"https://example.com/model_{i}",
                "notes": " ".join(random.choices(
                    ["evidence", "plan", "delay", "support", "boundary", "outcome",
                     "impulsive", "scarcity", "risk", "random", "text"], k=5)),
            }
            for i in range(5)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dummy, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid audit combining ternary‑lens rules with decision‑hygiene metrics")
    parser.add_argument("--manifest", type=Path, default=Path("dummy_manifest.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("hybrid_audit_report.json"))
    args = parser.parse_args()

    if not args.manifest.is_file():
        _create_dummy_manifest(args.manifest)

    report = build_hybrid_report(args.manifest.resolve(), args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "report_path": str(args.output),
        "candidate_count": len(report["candidates"]),
        "mean_hybrid_score": report["hybrid_statistics"]["mean_hybrid_score"],
        "local_hit_count": report["local_reference_scan"]["hit_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())