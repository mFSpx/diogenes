#!/usr/bin/env python3
"""LUCIDOTA Automated Metric Misattribution Guard.

Stateless O(1) token collision checks for telemetry parsing slip detection.
The gate does not mutate inputs; it returns an audit envelope that callers may
use to null out compromised metric columns.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Set


@dataclass(frozen=True)
class SanityGateReceipt:
    status: str
    collisions_detected: list[str]
    corrected_metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TelemetrySanityGate:
    def __init__(self, known_static_tokens: List[str] | None = None):
        self.forbidden_strings: Set[str] = {"1650", "i5", "tuf", "nvme", "postgres"}
        self.hardware_markers: Set[str] = {"nvidia", "geforce", "gtx", "rtx", "quadro", "radeon", "intel", "amd", "tuf"}
        if known_static_tokens:
            for token in known_static_tokens:
                clean_tok = str(token).strip().lower()
                if clean_tok:
                    self.forbidden_strings.add(clean_tok)

    def verify_metric_integrity(self, raw_source_string: str, parsed_metrics: Dict[str, Any]) -> Dict[str, Any]:
        audit_report = {
            "status": "METRIC_VERIFIED_OK",
            "collisions_detected": [],
            "corrected_metrics": {**parsed_metrics},
        }
        normalized_source = (raw_source_string or "").lower()
        looks_like_nameplate = any(marker in normalized_source for marker in self.hardware_markers)
        for metric_name, val in parsed_metrics.items():
            val_str = str(val).strip()
            if not val_str:
                continue
            if val_str.isdigit() and val_str in self.forbidden_strings and looks_like_nameplate and val_str in normalized_source:
                audit_report["status"] = "DIAGNOSTIC_WARNING_METRIC_MIRROR_DETECTED"
                audit_report["collisions_detected"].append(metric_name)
                audit_report["corrected_metrics"][metric_name] = None
        return audit_report


def parse_metric_line(raw_source_string: str, parsed_metrics: Dict[str, Any], known_static_tokens: List[str] | None = None) -> Dict[str, Any]:
    return TelemetrySanityGate(known_static_tokens=known_static_tokens).verify_metric_integrity(raw_source_string, parsed_metrics)


def safe_numeric_metrics_from_text(raw_source_string: str, metrics: Dict[str, Any], *, known_static_tokens: List[str] | None = None) -> Dict[str, Any]:
    audit = parse_metric_line(raw_source_string, metrics, known_static_tokens=known_static_tokens)
    return dict(audit["corrected_metrics"])
