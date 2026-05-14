#!/usr/bin/env python3
"""Privacy/anonymization scoring helpers."""
from __future__ import annotations
from typing import Any, Iterable

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers/total_records))
def anonymize_for_indexing(record: dict[str,Any], redact_keys: set[str]|None=None) -> dict[str,Any]:
    redact=redact_keys or {'email','phone','ssn','secret','token','password'}
    return {k:('<redacted>' if k.lower() in redact else v) for k,v in record.items()}
def dp_aggregate(values: Iterable[float], epsilon: float=1.0, sensitivity: float=1.0) -> float:
    return sum(values)  # deterministic core; add noise only at runtime boundary
