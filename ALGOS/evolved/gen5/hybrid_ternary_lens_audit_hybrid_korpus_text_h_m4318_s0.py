# DARWIN HAMMER — match 4318, survivor 0
# gen: 5
# parent_a: ternary_lens_audit.py (gen0)
# parent_b: hybrid_korpus_text_hybrid_hybrid_regret_m21_s1.py (gen4)
# born: 2026-05-29T23:56:13Z

"""
This module fuses the governing equations of the ternary_lens_audit.py and hybrid_korpus_text_hybrid_hybrid_regret_m21_s1.py algorithms.
The mathematical bridge is established by using the regret-weighted strategy from the hybrid_korpus_text_hybrid_hybrid_regret_m21_s1.py algorithm 
to modulate the classification and fast path rule enforcement in the ternary_lens_audit.py algorithm. 
This is achieved by incorporating the MinHash similarity and Jaccard-like similarity into the audit and rule enforcement pipeline.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import math
import random
import sys

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: list[str], k: int = 128) -> list[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return []
    return sorted([_hash(i, t) for i, t in enumerate(toks)])[:k]

def minhash_for_text(text: str, k: int = 64) -> list[int]:
    return signature(re.findall(r'\b\w+\b', text.lower().split()), k=k)

def jaccard_similarity(signature1: list[int], signature2: list[int]) -> float:
    set1 = set(signature1)
    set2 = set(signature2)
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0

def regret_weighted_similarity(text1: str, text2: str, k: int = 64) -> float:
    signature1 = minhash_for_text(text1, k=k)
    signature2 = minhash_for_text(text2, k=k)
    return jaccard_similarity(signature1, signature2)

def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for candidate in data.get("vendors", []):
        classification = candidate.get("classification")
        if classification not in {"usable_now", "research_only", "needs_conversion", "unsafe_for_fastpath", "unsupported"}:
            raise SystemExit(f"invalid classification {classification!r} for {candidate.get('candidate_key')}")
    return data

def enforce_fast_path_rule(candidate: dict[str, Any], reference_text: str) -> list[str]:
    findings: list[str] = []
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    notes = candidate.get("notes", "")
    similarity = regret_weighted_similarity(key + " " + family, reference_text)
    if re.search(r"standard.*lora|peft|qlora", key + " " + family, re.I):
        if candidate.get("classification") != "unsafe_for_fastpath" or candidate.get("fast_path_compatible"):
            if similarity < 0.5:
                findings.append("STANDARD_LORA_RULE_VIOLATION: normal PEFT/QLoRA must be unsafe_for_fastpath unless benchmark proves hot-path safety")
    if re.search(r"fp16|fp32", notes, re.I) and candidate.get("fast_path_compatible"):
        if similarity < 0.5:
            findings.append("FP_HOTPATH_CONFLICT: FP16/FP32 adapter note cannot be fast_path_compatible without benchmark evidence")
    if candidate.get("fast_path_compatible") and candidate.get("benchmark_required") and not candidate.get("benchmark_evidence"):
        findings.append("FAST_PATH_CLAIM_NEEDS_BENCHMARK_EVIDENCE")
    return findings

def hybrid_audit(manifest_path: Path, reference_text: str) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    for candidate in manifest.get("vendors", []):
        findings = enforce_fast_path_rule(candidate, reference_text)
        candidate["findings"] = findings
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Audit")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "services" / "ternary_lab" / "vendor_manifest.json")
    parser.add_argument("--reference", type=str, default="example reference text")
    args = parser.parse_args()
    result = hybrid_audit(args.manifest, args.reference)
    print(json.dumps(result, indent=4))