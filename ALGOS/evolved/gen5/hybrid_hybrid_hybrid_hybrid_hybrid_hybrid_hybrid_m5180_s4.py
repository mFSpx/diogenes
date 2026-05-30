# DARWIN HAMMER — match 5180, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1422_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_model_vram_sc_m2303_s1.py (gen3)
# born: 2026-05-30T00:00:29Z

"""Hybrid Algorithm integrating decision‑hygiene bandit logic (Parent A) with
hyperdimensional Doomsday/Gini mechanics (Parent B).

Mathematical bridge:
1. The regex‑based decision‑hygiene scanner produces a categorical count
   vector **c** = (c₁,…,c_K).  We apply a logarithmic transform
   ℓ_i = log(1 + c_i) to obtain a frequency vector **ℓ**.
2. Each category is mapped to a symbolic hypervector **s_i** via a deterministic
   hash‑based generator (Parent B).  The weighted bundle
   **H_counts** = Σ_i ℓ_i · s_i / Σ_i ℓ_i  (hyperdimensional encoding of the
   decision‑hygiene state).
3. The Doomsday algorithm yields the weekday **w** for a given date.  A
   symbolic hypervector **H_date** = symbol_vector(str(w)) represents the
   temporal context.
4. The Gini coefficient **G** of the raw counts **c** quantifies inequality in
   the decision‑hygiene distribution.  We scale the temporal hypervector by
   G and bind it to the count‑encoding hypervector:
   **H_fused** = bind(H_counts, bind(H_date, scalar_vector(G))).
5. The fused hypervector drives a simple multi‑armed bandit (UCB) to select an
   action, and its dot‑product with a reference hypervector defines a
   “Hybrid Free Energy”.

The code below implements the full pipeline with three core functions that
demonstrate the hybrid operation."""
import math
import random
import sys
import pathlib
import datetime as dt
import hashlib
from collections import defaultdict
from typing import Dict, List, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional primitives (Parent B)
# ----------------------------------------------------------------------
Vector = List[int]

def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    # deterministic seed from SHA‑256 hash of the symbol
    h = hashlib.sha256(symbol.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: Iterable[Vector]) -> Vector:
    vecs = list(vectors)
    if not vecs:
        return []
    dim = len(vecs[0])
    # arithmetic mean (component‑wise)
    return [sum(v[j] for v in vecs) / len(vecs) for j in range(dim)]

def scalar_vector(scalar: float, dim: int = 10000) -> Vector:
    """Create a hypervector where every component equals sign(scalar) * |scalar|."""
    sign = 1 if scalar >= 0 else -1
    return [sign * abs(scalar) for _ in range(dim)]

# ----------------------------------------------------------------------
# Decision‑hygiene scanner (Parent A)
# ----------------------------------------------------------------------
import re

EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|schedule|timetable|agenda|program|procedure|protocol|policy|provision|arrangement)\b", re.I)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)

CATEGORIES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
}

def scan_texts(texts: Iterable[str]) -> Dict[str, int]:
    """Count occurrences of each decision‑hygiene category across a corpus."""
    counts = defaultdict(int)
    for txt in texts:
        for name, regex in CATEGORIES.items():
            if regex.search(txt):
                counts[name] += 1
    return dict(counts)

def log_counts(counts: Dict[str, int]) -> Dict[str, float]:
    """Apply log(1 + n) to each count to obtain a smooth frequency vector."""
    return {k: math.log1p(v) for k, v in counts.items()}

# ----------------------------------------------------------------------
# Gini coefficient (Parent B)
# ----------------------------------------------------------------------
def gini_coefficient(values: List[float]) -> float:
    """Compute Gini coefficient for a non‑negative list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(values)
    cumulative = 0.0
    for i, val in enumerate(sorted_vals, 1):
        cumulative += i * val
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return gini

# ----------------------------------------------------------------------
# Doomsday algorithm (Parent B)
# ----------------------------------------------------------------------
def doomsday_weekday(year: int, month: int, day: int) -> int:
    """
    Return weekday (0=Sunday … 6=Saturday) using Conway's Doomsday algorithm.
    """
    # anchor days for centuries
    century = year // 100
    anchor = (5 * (century % 4) + 2) % 7

    # year within century
    y = year % 100
    doomsday = (y // 12 + y % 12 + (y % 12) // 4 + anchor) % 7

    # month offsets for common year (leap year handled separately)
    month_offsets = [0, 3, 0, 3, 2, 5, 0, 3, 6, 1, 4, 6]  # Jan..Dec
    if month < 3 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        doomsday = (doomsday - 1) % 7  # leap‑year correction for Jan/Feb

    weekday = (day - month_offsets[month - 1]) % 7
    return (weekday + doomsday) % 7

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def encode_decision_hygiene(counts: Dict[str, int], dim: int = 10000) -> Vector:
    """
    Convert log‑scaled decision‑hygiene counts into a bundled hypervector.
    Each category is mapped to a symbolic hypervector; the bundle is weighted
    by the log count.
    """
    loged = log_counts(counts)
    weighted_vectors = []
    for cat, weight in loged.items():
        sym_vec = symbol_vector(f"cat:{cat}", dim)
        # scale each component by weight
        weighted_vectors.append([weight * x for x in sym_vec])
    if not weighted_vectors:
        return [0] * dim
    bundled = bundle(weighted_vectors)
    # ensure integer components for bind compatibility
    return [int(round(v)) for v in bundled]

def temporal_hypervector(date: dt.date, dim: int = 10000) -> Vector:
    """
    Generate a hypervector representing the weekday of *date* via the Doomsday
    algorithm.
    """
    w = doomsday_weekday(date.year, date.month, date.day)
    return symbol_vector(f"weekday:{w}", dim)

def hybrid_fuse(counts: Dict[str, int],
                date: dt.date,
                dim: int = 10000) -> Vector:
    """
    Fuse decision‑hygiene encoding, temporal context, and Gini weighting into
    a single hypervector.
    """
    # 1. Encode counts
    h_counts = encode_decision_hygiene(counts, dim)

    # 2. Temporal hypervector
    h_time = temporal_hypervector(date, dim)

    # 3. Gini coefficient as a scalar hypervector
    gini = gini_coefficient(list(counts.values()))
    h_gini = scalar_vector(gini, dim)

    # 4. Bind all components (associative, commutative for ±1 vectors)
    fused = bind(h_counts, bind(h_time, h_gini))
    return fused

# ----------------------------------------------------------------------
# Bandit selection using fused hypervector (Parent A + B)
# ----------------------------------------------------------------------
class SimpleUCBBandit:
    """
    Multi‑armed bandit with Upper‑Confidence Bound (UCB1).  The reward estimate
    for each arm is taken as the dot‑product between the fused hypervector and
    a pre‑generated arm hypervector.
    """
    def __init__(self, arms: List[str], dim: int = 10000):
        self.dim = dim
        self.arms = arms
        self.arm_vectors = {a: symbol_vector(f"arm:{a}", dim) for a in arms}
        self.counts = {a: 0 for a in arms}
        self.values = {a: 0.0 for a in arms}
        self.total_pulls = 0

    def select(self, fused_vector: Vector) -> str:
        self.total_pulls += 1
        # compute UCB for each arm
        ucb_scores = {}
        for a in self.arms:
            avg_reward = self.values[a]
            n = self.counts[a]
            if n == 0:
                # force exploration
                ucb = float('inf')
            else:
                confidence = math.sqrt(2 * math.log(self.total_pulls) / n)
                ucb = avg_reward + confidence
            # reward proxy: dot product (scaled)
            dot = sum(x * y for x, y in zip(fused_vector, self.arm_vectors[a]))
            # normalize by dimension to keep values modest
            proxy = dot / self.dim
            ucb_scores[a] = ucb + proxy
        # choose arm with max UCB‑adjusted proxy
        chosen = max(ucb_scores, key=ucb_scores.get)
        # simulate stochastic reward (here deterministic proxy)
        reward = ucb_scores[chosen]  # using the same score as reward proxy
        # update estimates
        self.counts[chosen] += 1
        n = self.counts[chosen]
        self.values[chosen] = ((n - 1) * self.values[chosen] + reward) / n
        return chosen

# ----------------------------------------------------------------------
# Hybrid Free Energy computation
# ----------------------------------------------------------------------
def hybrid_free_energy(fused_vector: Vector, reference: Vector) -> float:
    """
    Define a scalar free‑energy‑like quantity as the normalized dot product
    between the fused hypervector and a reference hypervector.
    """
    if len(fused_vector) != len(reference):
        raise ValueError("vector length mismatch")
    dot = sum(x * y for x, y in zip(fused_vector, reference))
    return dot / len(fused_vector)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I have verified the source and recorded the hash.",
        "Let's plan the next steps and set a schedule.",
        "I need to wait until tomorrow before I act.",
        "I will ask a friend for support.",
        "We must respect the boundary and keep safe.",
        "The task is done and verified.",
        "I'm feeling impulsive and want to act right now.",
    ]

    # 1. Scan and encode
    counts = scan_texts(sample_texts)
    today = dt.date.today()
    fused = hybrid_fuse(counts, today)

    # 2. Bandit decision
    bandit = SimpleUCBBandit(arms=["email", "chat", "call"])
    action = bandit.select(fused)

    # 3. Free energy
    ref_vec = symbol_vector("reference", len(fused))
    energy = hybrid_free_energy(fused, ref_vec)

    # Output results
    print("Decision‑hygiene counts:", counts)
    print("Gini coefficient:", round(gini_coefficient(list(counts.values())), 4))
    print("Selected action:", action)
    print("Hybrid free energy:", round(energy, 4))