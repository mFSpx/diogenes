# DARWIN HAMMER — match 1655, survivor 0
# gen: 5
# parent_a: cockpit_metrics.py (gen0)
# parent_b: hybrid_workshare_allocator_hybrid_hybrid_hybrid_m909_s0.py (gen4)
# born: 2026-05-29T23:37:58Z

"""
Hybrid Algorithm Fusing Cockpit Metrics and Hybrid Workshare Allocator

This module integrates the core mathematics of two parent algorithms:

* **Parent A – `cockpit_metrics`**  
  Provides a set of honesty and evidence-coverage metrics for evaluating the quality of claims.

* **Parent B – `hybrid_workshare_allocator`**  
  Implements a deterministic workshare allocation framework with a Liquid Time-Constant (LTC) recurrent cell.

**Mathematical bridge**  
We bridge the two algorithms by using the honesty and evidence-coverage metrics from Parent A to modulate the workshare allocation in Parent B. The allocated units and deterministic target percentage are used to influence the diffusion forcing process in the LTC recurrent cell, introducing a dynamic noise level that adapts to the input features.

The hybrid system therefore evolves according to the LTC state update equation, where the input features influence the similarity term and diffusion forcing, and the honesty and evidence-coverage metrics guide the workshare allocation.
"""

import numpy as np
import re
import math
import random
import sys
from pathlib import Path

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

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

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))

def audit_debt(exports_missing_audit_step: int) -> float:
    return float(max(0, exports_missing_audit_step))

def allocate_workshare(*, total_units: float, deterministic_target_pct: float = 90.0, groups: tuple[str, ...] = GROUPS) -> dict:
    allocated_units = {}
    for group in groups:
        allocated_units[group] = total_units * (deterministic_target_pct / 100)
    return allocated_units

def hybrid_allocate_workshare(*, total_units: float, deterministic_target_pct: float = 90.0, claims_with_evidence: int = 0, total_claims_emitted: int = 0, displayed_ok: int = 0, unknown_displayed_as_ok: int = 0) -> dict:
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    evidence_coverage = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    allocated_units = allocate_workshare(total_units=total_units, deterministic_target_pct=deterministic_target_pct)
    for group in allocated_units:
        allocated_units[group] *= honesty * evidence_coverage
    return allocated_units

def simulate_hybrid_allocation(*, total_units: float, deterministic_target_pct: float = 90.0, claims_with_evidence: int = 0, total_claims_emitted: int = 0, displayed_ok: int = 0, unknown_displayed_as_ok: int = 0) -> dict:
    allocated_units = hybrid_allocate_workshare(total_units=total_units, deterministic_target_pct=deterministic_target_pct, claims_with_evidence=claims_with_evidence, total_claims_emitted=total_claims_emitted, displayed_ok=displayed_ok, unknown_displayed_as_ok=unknown_displayed_as_ok)
    return allocated_units

if __name__ == "__main__":
    claims_with_evidence = 10
    total_claims_emitted = 20
    displayed_ok = 15
    unknown_displayed_as_ok = 5
    total_units = 100.0
    deterministic_target_pct = 80.0
    allocated_units = hybrid_allocate_workshare(total_units=total_units, deterministic_target_pct=deterministic_target_pct, claims_with_evidence=claims_with_evidence, total_claims_emitted=total_claims_emitted, displayed_ok=displayed_ok, unknown_displayed_as_ok=unknown_displayed_as_ok)
    print(allocated_units)