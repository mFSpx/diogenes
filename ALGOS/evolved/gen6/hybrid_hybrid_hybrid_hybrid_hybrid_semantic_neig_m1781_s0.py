# DARWIN HAMMER — match 1781, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_percyphon_hyb_m1198_s3.py (gen5)
# parent_b: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s1.py (gen2)
# born: 2026-05-29T23:38:43Z

"""
This module fuses the core topologies of 
'hybrid_hybrid_hybrid_hybrid_hybrid_percyphon_hyb_m1198_s3.py' (Parent A) 
and 'hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s1.py' (Parent B).

The mathematical bridge between the two parents lies in the fact that 
Parent A's regex feature extraction can be viewed as a vector 
transformation, and Parent B's morphology metrics can be used to 
weight or modulate these vectors. Specifically, we use the 
sphericity index and righting time index from Parent B to 
compute a weighted recovery priority vector that influences 
the regex feature extraction.

This hybrid system integrates the governing equations of both 
parents by using the morphology metrics to modulate the 
output of the regex feature extraction.
"""

import numpy as np
import re
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from random import random
from sys import exit
from pathlib import Path
from typing import Any, Dict, Tuple, List

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(morph: Morphology) -> float:
    l, w, h = morph.length, morph.width, morph.height
    if min(l, w, h) <= 0:
        raise ValueError("dimensions must be positive")
    return (l * w * h) ** (1.0 / 3.0) / max(l, w, h)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = (m.length + m.width) / (2.0 * m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

# Regex feature extraction (Parent A core)
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

REGEX_PATTERNS = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("support", SUPPORT_RE),
    ("boundary", BOUNDARY_RE),
]

def extract_regex_features(text: str) -> np.ndarray:
    """Return a unit‑length count vector for the defined regex categories."""
    counts = np.array([len(pat.findall(text)) for _, pat in REGEX_PATTERNS], dtype=float)
    norm = np.linalg.norm(counts) + 1e-12
    return counts / norm

def hybrid_regex_features(text: str, morph: Morphology) -> np.ndarray:
    """Compute weighted regex features using morphology metrics."""
    base_features = extract_regex_features(text)
    sphericity = sphericity_index(morph)
    recovery = recovery_priority(morph)
    weights = np.array([sphericity, recovery, sphericity * recovery, 1.0, 1.0])
    return base_features * weights

def semantic_neighbors(doc_id: str, vector: list[float], k: int=5) -> List[Tuple[str, float]]:
    den=math.sqrt(sum(x*x for x in vector))*math.sqrt(sum(y*y for y in vector)); 
    return sorted(((d,_cos(vector,w)) for d,w in [(doc_id,vector)]+[("doc"+str(i),np.random.rand(len(vector))) for i in range(1,k+1)] if d!=doc_id), key=lambda x:(-x[1],x[0]))[:k]

def _cos(a,b):
    den=math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b)); 
    return 0.0 if den==0 else sum(x*y for x,y in zip(a,b))/den

if __name__ == "__main__":
    morph = Morphology(1.0, 2.0, 3.0, 1.0)
    text = "This is a test sentence for evidence and planning."
    features = hybrid_regex_features(text, morph)
    print(features)

    doc_id = "doc0"
    vector = [1.0, 2.0, 3.0, 4.0, 5.0]
    neighbors = semantic_neighbors(doc_id, vector)
    print(neighbors)