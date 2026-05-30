# DARWIN HAMMER — match 19, survivor 1
# gen: 2
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s2.py (gen1)
# parent_b: ternary_lens_audit.py (gen0)
# born: 2026-05-29T23:23:03Z

"""
Hybrid Decision Hygiene & Shannon Entropy with Ternary Lens Audit Module.

This module fuses the *Decision Hygiene* algorithm with the *Shannon Entropy* calculation 
and the *Ternary Lens Audit* framework. The mathematical bridge is the 
**feature-count vector** produced by the hygiene regexes and the **ternary lens audit report**.

The final hybrid score multiplies the hygiene score by a factor that depends on the 
normalized entropy (0 ≤ H/Hmax ≤ 1) and incorporates the ternary lens audit findings.
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Tuple
import numpy as np
import json

# ----------------------------------------------------------------------
# Parent A – regexes and raw count extraction
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
# Parent B – ternary lens audit implementation
# ----------------------------------------------------------------------
CLASSIFICATIONS = {"usable_now", "research_only", "needs_conversion", "unsafe_for_fastpath", "unsupported"}
LOCAL_PATTERNS = ["*bitnet*", "*BitNet*", "*fairyfuse*", "*FairyFuse*", "*lora*", "*LoRA*", "*adapter*"]

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

# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def _raw_counts(text: str) -> dict[str, int]:
    """Extract raw feature counts from *text* using parent‑A regexes."""
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
    """Return a 9‑element ``numpy`` array of counts ordered as _FEATURE_ORDER."""
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
    """Compute the original decision‑hygiene score and a textual label."""
    positive = int(np.dot(vector, _POSITIVE_WEIGHTS))
    negative = int(np.dot(vector, _NEGATIVE_WEIGHTS))
    raw_score = max(-10000, min(10000, positive - negative))
    if raw_score > 5000:
        label = "Good"
    elif raw_score > 0:
        label = "Fair"
    else:
        label = "Poor"
    return raw_score, label

def shannon_entropy(observations: Iterable[float | Any], is_distribution: bool = False) -> float:
    """Return Shannon entropy (bits) of a discrete distribution."""
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

def hybrid_score(text: str, audit_findings: list[str]) -> Tuple[int, str]:
    """Compute the hybrid score and a textual label."""
    vector = feature_vector(text)
    hygiene, label = hygiene_score(vector)
    entropy = shannon_entropy(vector, is_distribution=False)
    max_entropy = math.log2(len(vector))
    normalized_entropy = entropy / max_entropy
    hybrid_score = int(hygiene * normalized_entropy * (1 - len(audit_findings) / 10))
    if hybrid_score > 5000:
        label = "Good"
    elif hybrid_score > 0:
        label = "Fair"
    else:
        label = "Poor"
    return hybrid_score, label

def build_audit_report(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
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
    local_hits = scan_local(root)
    return {
        "schema": "lucidota.ternary_lab.lens_audit_report.v1",
        "created_at": "2024-01-01T00:00:00Z",
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
    text = "This is a sample text for decision hygiene and shannon entropy calculation."
    vector = feature_vector(text)
    hygiene, label = hygiene_score(vector)
    entropy = shannon_entropy(vector, is_distribution=False)
    max_entropy = math.log2(len(vector))
    normalized_entropy = entropy / max_entropy
    hybrid_score_val = int(hygiene * normalized_entropy)
    print(hygiene, label, entropy, normalized_entropy, hybrid_score_val)

    manifest = {
        "vendors": [
            {
                "candidate_key": "vendor1",
                "display_name": "Vendor 1",
                "family": "Family 1",
                "classification": "usable_now",
                "fast_path_compatible": True,
                "benchmark_required": False,
                "license_id": "MIT",
                "source_uri": "https://example.com",
                "notes": "This is a sample vendor",
            }
        ]
    }
    root = Path("/path/to/root")
    report = build_audit_report(manifest, root)
    print(json.dumps(report, indent=2))