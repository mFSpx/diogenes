# DARWIN HAMMER — match 2180, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_endpoint_circ_m314_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s4.py (gen3)
# born: 2026-05-29T23:41:13Z

import math
import numpy as np
from pathlib import Path
from collections import Counter
import re

def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[tuple[float, float]], seeds: list[tuple[float, float]], circuit_breaker_state: bool) -> dict[int, list[tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        if circuit_breaker_state:
            regions[nearest(p, seeds)].append(p)
        else:
            # Assign points to regions even when circuit breaker is open
            regions[nearest(p, seeds)].append(p)
    return regions

class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = ""

    def allow(self) -> bool:
        return not self.open

def ternary_lens_audit(text: str) -> dict[str, int]:
    evidence_re = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
    planning_re = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
    delay_re = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
    support_re = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
    boundary_re = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
    outcome_re = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
    impulsivity_re = re.compile(r"\b(?:rage|impulsive|panic|panicki)\b", re.I)
    return {
        'evidence': len(evidence_re.findall(text)),
        'planning': len(planning_re.findall(text)),
        'delay': len(delay_re.findall(text)),
        'support': len(support_re.findall(text)),
        'boundary': len(boundary_re.findall(text)),
        'outcome': len(outcome_re.findall(text)),
        'impulsivity': len(impulsivity_re.findall(text)),
    }

def hybrid_score(points: list[tuple[float, float]], seeds: list[tuple[float, float]], circuit_breaker_state: bool, text: str) -> float:
    regions = assign(points, seeds, circuit_breaker_state)
    audit_result = ternary_lens_audit(text)
    # Use a more robust way to calculate the score
    return sum(len(region) for region in regions.values()) / max(1, sum(audit_result.values()))

def weighted_node_distance(node1: str, node2: str, circuit_breaker_state: bool, text: str) -> float:
    audit_result = ternary_lens_audit(text)
    if circuit_breaker_state:
        # Use a more meaningful weight when circuit breaker is open
        return 1.0 / (1 + audit_result['evidence'] + audit_result['planning'])
    else:
        return 0.5 * (audit_result['evidence'] + audit_result['planning'])

if __name__ == "__main__":
    points = [(1, 2), (3, 4), (5, 6)]
    seeds = [(0, 0), (10, 10)]
    circuit_breaker = EndpointCircuitBreaker()
    circuit_breaker.record_success()
    text = "This is a test text with evidence and planning."
    print(hybrid_score(points, seeds, circuit_breaker.allow(), text))
    print(weighted_node_distance("node1", "node2", circuit_breaker.allow(), text))