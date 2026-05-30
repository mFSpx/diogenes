# DARWIN HAMMER — match 733, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_decision_hygi_hybrid_sketches_rlct_m6_s3.py (gen2)
# parent_b: ternary_lens_audit.py (gen0)
# born: 2026-05-29T23:30:40Z

"""Hybrid Ternary Lens & Decision-Hygiene Audit Module

Parents
-------
* **Parent A** – ``hybrid_decision_hygiene_shannon_entropy_m12_s1.py``  
  Provides regex-based counts of decision-hygiene features (evidence, planning,
  delay, …) and computes their Shannon entropy.

* **Parent B** – ``ternary_lens_audit.py``  
  Performs offline ternary lens audit using the local vendor manifest and a
  local path/reference scan.

Mathematical Bridge
-------------------
The fusion uses a novel approach that combines the entropy calculation from
Parent A with the ternary lens audit findings from Parent B. The key insight is
to use the decision-hygiene counts as a feature vector `c` in the ternary lens
audit formula.

Let `N` be the number of distinct tokens in the token stream, and `H` be the
Shannon entropy of the decision-hygiene counts. We use the ternary lens audit
findings to compute a risk score `R` for each token, where `R` is a function of
the token's classification and its compatibility with the ternary lens audit
rules.

The hybrid audit score `S` is then computed as:

`S = H + R * log(N)`

This fusion combines the uncertainty of the decision-making language (captured
by the Shannon entropy `H`) with the risk associated with the ternary lens
audit findings (captured by the risk score `R`).

The module exposes three core functions that demonstrate this fusion.
"""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import re

# ----------------------------------------------------------------------
# Decision-hygiene regexes (Parent A)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|step|plan|planning)\b", re.I)

# ----------------------------------------------------------------------
# Ternary lens audit patterns (Parent B)
# ----------------------------------------------------------------------
LOCAL_PATTERNS = ["*bitnet*", "*BitNet*", "*fairyfuse*", "*FairyFuse*", "*lora*", "*LoRA*", "*adapter*"]

def load_manifest(path: Path) -> dict[str, Any]:
    """Load the vendor manifest from the given path."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for candidate in data.get("vendors", []):
        classification = candidate.get("classification")
        if classification not in CLASSIFICATIONS:
            raise SystemExit(f"invalid classification {classification!r} for {candidate.get('candidate_key')}")
    return data

def enforce_fast_path_rule(candidate: dict[str, Any]) -> list[str]:
    """Enforce the fast path rule for the given candidate."""
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

def hybrid_audit_score(token_stream: Iterable[str], manifest_path: Path) -> float:
    """Compute the hybrid audit score for the given token stream and manifest path."""
    # Compute the decision-hygiene counts and Shannon entropy
    decision_hygiene_counts = defaultdict(int)
    for token in token_stream:
        if EVIDENCE_RE.match(token):
            decision_hygiene_counts['evidence'] += 1
        elif PLANNING_RE.match(token):
            decision_hygiene_counts['planning'] += 1
        # ... add more decision-hygiene features as needed
    entropy = -sum([count / len(token_stream) * math.log2(count / len(token_stream)) for count in decision_hygiene_counts.values()])

    # Compute the ternary lens audit findings
    manifest_data = load_manifest(manifest_path)
    findings = defaultdict(int)
    for candidate in manifest_data['vendors']:
        findings[len(enforce_fast_path_rule(candidate))] += 1

    # Compute the risk score and hybrid audit score
    risk_score = sum([findings[key] * math.log2(len(token_stream)) for key in findings.keys()])
    hybrid_score = entropy + risk_score

    return hybrid_score

def hybrid_audit(token_stream: Iterable[str], manifest_path: Path, num_tokens: int) -> float:
    """Compute the hybrid audit score for the given token stream and manifest path."""
    return hybrid_audit_score(token_stream, manifest_path) * math.log2(num_tokens)

def hybrid_audit_log(token_stream: Iterable[str], manifest_path: Path, num_tokens: int) -> float:
    """Compute the hybrid audit score for the given token stream and manifest path, returning the score as a log value."""
    return math.log2(hybrid_audit(token_stream, manifest_path, num_tokens))

if __name__ == "__main__":
    # Smoke test
    token_stream = ["token1", "token2", "token3"]
    manifest_path = Path("/path/to/manifest.json")
    num_tokens = 100
    print(hybrid_audit_score(token_stream, manifest_path))
    print(hybrid_audit(token_stream, manifest_path, num_tokens))
    print(hybrid_audit_log(token_stream, manifest_path, num_tokens))