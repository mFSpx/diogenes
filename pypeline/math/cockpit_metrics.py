from __future__ import annotations

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, float(claims_with_evidence) / float(total_claims_emitted)))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, float(displayed_ok) / float(total)))

def audit_debt(exports_missing_audit_step: int) -> float:
    return float(max(0, exports_missing_audit_step))
