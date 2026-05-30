# DARWIN HAMMER — match 154, survivor 2
# gen: 3
# parent_a: ternary_lens_audit.py (gen0)
# parent_b: hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s1.py (gen2)
# born: 2026-05-29T23:27:13Z

"""Hybrid audit‑scheduler module.

Parents:
- ternary_lens_audit.py (Algorithm A): scans a vendor manifest, validates classifications,
  and applies fast‑path rule checks.
- hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s1.py (Algorithm B):
  defines a ModelTier, ModelPool and a reconstruction risk score used for VRAM
  scheduling.

Mathematical bridge:
Both systems quantify a *risk* associated with an entity.
Algorithm A produces a set of audit findings per vendor candidate; each finding
can be interpreted as a binary risk indicator (1 = risk, 0 = no risk).  
Algorithm B defines a reconstruction_risk_score = unique_quasi_identifiers /
total_records ∈ [0,1].  

We fuse them by (1) converting audit findings into a numeric risk vector,
(2) aggregating that vector with the reconstruction risk score via a weighted
sum, and (3) feeding the resulting scalar into the ModelPool scheduler to decide
which candidates (treated as lightweight “models”) may be loaded under RAM/VRAM
constraints.  The combined risk therefore drives both classification‑based
validation and resource‑aware loading.

The module therefore provides three core hybrid functions:
- `candidate_risk_vector` – maps audit findings to a numeric vector.
- `compute_candidate_resource` – maps a candidate to a ModelTier based on its
  classification.
- `schedule_candidates` – orders candidates by combined risk and loads them into
  a ModelPool respecting memory ceilings, evicting lower‑priority entries when
  necessary.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

import numpy as np

# ---------- Algorithm A components ----------

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


def enforce_fast_path_rule(candidate: Mapping[str, Any]) -> list[str]:
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


# ---------- Algorithm B components ----------

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1", 1024)
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2", 2048)
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2", 2048)
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3", 4096)


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk ∈ [0,1] proportional to the fraction of quasi‑identifiers."""
    return (
        0.0
        if total_records <= 0
        else max(0.0, min(1.0, unique_quasi_identifiers / total_records))
    )


def anonymize_for_indexing(record: Mapping[str, Any], redact_keys: set[str] | None = None) -> dict[str, Any]:
    redact = redact_keys or {"email", "phone", "ssn", "secret", "token", "password"}
    return {k: ("<redacted>" if k.lower() in redact else v) for k, v in record.items()}


def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Deterministic sum; noise would be added at a runtime boundary."""
    return sum(values)


class ModelPool:
    """Tracks loaded models respecting RAM/VRAM ceilings."""

    def __init__(self, ram_ceiling_mb: int = 6000, vram_ceiling_mb: int = 4096):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.vram_ceiling_mb = vram_ceiling_mb
        self.loaded: dict[str, ModelTier] = {}
        self.sensitive_records: list[dict[str, Any]] = []

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _used_vram(self) -> int:
        return sum(m.vram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        return (
            self._used_ram() + model.ram_mb <= self.ram_ceiling_mb
            and self._used_vram() + model.vram_mb <= self.vram_ceiling_mb
        )

    def load(self, model: ModelTier) -> None:
        if not self.can_load(model):
            raise RuntimeError("Insufficient resources to load model")
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        self.loaded[model.name] = model

    def evict(self, name: str) -> None:
        self.loaded.pop(name, None)

    def snapshot(self) -> dict[str, Any]:
        return {
            "ram_used_mb": self._used_ram(),
            "vram_used_mb": self._used_vram(),
            "loaded_models": list(self.loaded.keys()),
        }


# ---------- Hybrid logic ----------

def candidate_risk_vector(findings: Sequence[str]) -> np.ndarray:
    """Convert audit findings into a binary risk vector."""
    risk_map = {
        "STANDARD_LORA_RULE_VIOLATION": 1,
        "FP_HOTPATH_CONFLICT": 1,
        "FAST_PATH_CLAIM_NEEDS_BENCHMARK_EVIDENCE": 1,
    }
    vec = np.array([risk_map.get(f.split(":")[0], 0) for f in findings], dtype=float)
    return vec


def compute_candidate_resource(candidate: Mapping[str, Any]) -> ModelTier | None:
    """Map a candidate's classification to a predefined ModelTier."""
    mapping = {
        "usable_now": TIER_T1_QWEN_0_5B,
        "research_only": TIER_T2_REASONING,
        "needs_conversion": TIER_T2_TOOL,
        "unsafe_for_fastpath": TIER_T3_QWEN_7B,
        "unsupported": None,
    }
    return mapping.get(candidate.get("classification"))


def combined_risk_score(candidate: Mapping[str, Any], findings: Sequence[str]) -> float:
    """
    Fuse audit‑derived risk (binary vector) with reconstruction risk.
    The final score is a weighted average:
        α * (mean of binary vector) + (1‑α) * reconstruction_risk_score
    where α = 0.6 gives higher importance to audit violations.
    """
    alpha = 0.6
    binary_vec = candidate_risk_vector(findings)
    audit_risk = binary_vec.mean() if binary_vec.size else 0.0
    # For demonstration we treat `unique_quasi_identifiers` as number of words in notes.
    notes = candidate.get("notes", "")
    uq = len(re.findall(r"\b\w+\b", notes))
    total = max(1, len(notes.split()))
    recon_risk = reconstruction_risk_score(uq, total)
    return alpha * audit_risk + (1 - alpha) * recon_risk


def schedule_candidates(
    candidates: Sequence[Mapping[str, Any]],
    pool: ModelPool,
) -> List[str]:
    """
    Order candidates by ascending combined risk and attempt to load them.
    If a candidate cannot be loaded due to resource limits, evict the already
    loaded candidate with the highest combined risk (if any) and retry.
    Returns the list of successfully loaded model names.
    """
    # Pre‑compute risk and associated ModelTier
    enriched = []
    for cand in candidates:
        findings = enforce_fast_path_rule(cand)
        risk = combined_risk_score(cand, findings)
        tier = compute_candidate_resource(cand)
        if tier is None:
            continue
        enriched.append((risk, tier, cand.get("candidate_key", "unknown")))

    # Sort by lowest risk first (most trustworthy)
    enriched.sort(key=lambda x: x[0])

    loaded_order: List[str] = []

    for risk, tier, key in enriched:
        try:
            pool.load(tier)
            loaded_order.append(key)
        except RuntimeError:
            # Attempt eviction of a higher‑risk model
            if not pool.loaded:
                continue
            # Find currently loaded model with highest risk
            current_risks = {
                name: combined_risk_score(
                    next(c for c in candidates if c.get("candidate_key") == name), []
                )
                for name in pool.loaded
            }
            evict_candidate = max(current_risks, key=current_risks.get)
            if current_risks[evict_candidate] > risk:
                pool.evict(evict_candidate)
                try:
                    pool.load(tier)
                    loaded_order.append(key)
                except RuntimeError:
                    # Still cannot load; give up on this candidate
                    continue
    return loaded_order


def audit_and_schedule(
    manifest_path: Path,
    root_scan_path: Path,
    pool: ModelPool,
) -> dict[str, Any]:
    """
    Full hybrid pipeline:
      1. Load and validate the vendor manifest.
      2. For each candidate, run fast‑path rule audit.
      3. Compute combined risk and attempt VRAM‑aware scheduling.
      4. Return a report containing audit findings, risk scores, and pool snapshot.
    """
    manifest = load_manifest(manifest_path)
    candidates = manifest.get("vendors", [])
    report = {
        "timestamp": utc_now(),
        "audit": {},
        "schedule": {"loaded_models": []},
    }

    for cand in candidates:
        key = cand.get("candidate_key", "unknown")
        findings = enforce_fast_path_rule(cand)
        risk = combined_risk_score(cand, findings)
        report["audit"][key] = {
            "findings": findings,
            "combined_risk": risk,
        }

    loaded = schedule_candidates(candidates, pool)
    report["schedule"]["loaded_models"] = loaded
    report["schedule"]["pool_snapshot"] = pool.snapshot()
    return report


# ---------- CLI / Smoke test ----------

def _build_dummy_manifest(tmp_path: Path) -> Path:
    """Create a tiny manifest JSON used by the smoke test."""
    data = {
        "vendors": [
            {
                "candidate_key": "alpha",
                "family": "standard-lora",
                "classification": "usable_now",
                "fast_path_compatible": False,
                "benchmark_required": False,
                "benchmark_evidence": None,
                "notes": "fp16 adapter for speed",
            },
            {
                "candidate_key": "beta",
                "family": "custom",
                "classification": "research_only",
                "fast_path_compatible": True,
                "benchmark_required": True,
                "benchmark_evidence": None,
                "notes": "experimental tool",
            },
            {
                "candidate_key": "gamma",
                "family": "legacy",
                "classification": "unsafe_for_fastpath",
                "fast_path_compatible": True,
                "benchmark_required": True,
                "benchmark_evidence": "bench_v1",
                "notes": "fp32 heavy model",
            },
        ]
    }
    manifest_path = tmp_path / "vendor_manifest.json"
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid audit‑scheduler demo")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to vendor manifest JSON (generated if omitted)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory for local scans (unused in demo)",
    )
    args = parser.parse_args(argv)

    tmp_dir = Path.cwd()
    manifest_path = args.manifest or _build_dummy_manifest(tmp_dir)

    pool = ModelPool(ram_ceiling_mb=8000, vram_ceiling_mb=8192)

    report = audit_and_schedule(manifest_path, args.root, pool)

    # Print a concise summary
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())