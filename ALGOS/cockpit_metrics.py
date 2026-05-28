#!/usr/bin/env python3
"""Cockpit honesty / evidence-coverage metrics."""
from __future__ import annotations

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    total=displayed_ok+unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok/total))

def audit_debt(exports_missing_audit_step: int) -> float:
    return float(max(0, exports_missing_audit_step))
