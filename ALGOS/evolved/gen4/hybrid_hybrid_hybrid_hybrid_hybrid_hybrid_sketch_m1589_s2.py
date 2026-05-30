# DARWIN HAMMER — match 1589, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_temporal_motifs_m280_s0.py (gen3)
# parent_b: hybrid_hybrid_sketches_hybr_hybrid_model_vram_sc_m44_s1.py (gen3)
# born: 2026-05-29T23:37:36Z

"""Hybrid Algorithm: Bayesian Burst Detection + Count‑Min Sketch VRAM Allocation

Parents:
- hybrid_hybrid_hybrid_minimu_temporal_motifs_m280_s0.py (Bayesian burst detection &
  temporal motif mining)
- hybrid_hybrid_sketches_hybr_hybrid_model_vram_sc_m44_s1.py (Count‑Min sketch
  + VRAM budgeting)

Mathematical bridge:
Both parents estimate event frequencies probabilistically. The count‑min sketch
provides a fast, space‑efficient estimate of the marginal frequency  P(E)  for
each key. This estimate can serve as the *prior* in the Bayesian formulas of
the burst‑detection parent:

    marginal = P(E) ≈ sketch_count / total_sketch_counts
    posterior = P(E|e) = likelihood·prior / marginal

The posterior is then combined with the classical z‑score to decide whether a
key constitutes a burst. The resulting posterior probabilities are also used
to weight the VRAM budget for storing temporal motifs, closing the loop
between the two topologies."""

import math
import random
import sys
import pathlib
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BurstSignal:
    key: str
    count: int
    z_score: float
    prior: float
    likelihood: float
    posterior: float
    false_positive: float

@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int
    posterior: float  # posterior probability that the motif is meaningful

@dataclass
class VRAMBudget:
    budget_mb: int          # total VRAM budget
    reserve_mb: int         # reserved headroom for safety
    used_mb: int = 0        # updated by allocation routine


# ----------------------------------------------------------------------
# Core probabilistic utilities (from Parent A)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E|e) = P(e|E)P(E) + P(e|~E)P(~E)"""
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """P(E|e) = P(e|E)P(E) / P(E|e)"""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


# ----------------------------------------------------------------------
# Count‑Min Sketch utilities (from Parent B)
# ----------------------------------------------------------------------
def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Return a count‑min sketch matrix for the given iterable of string items."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table


def sketch_estimate(key: str, sketch: List[List[int]]) -> int:
    """Estimate frequency of *key* from a sketch (minimum across hash rows)."""
    estimates = []
    for d, row in enumerate(sketch):
        h = hashlib.sha256(f"{d}:{key}".encode()).hexdigest()
        idx = int(h, 16) % len(row)
        estimates.append(row[idx])
    return min(estimates)


def total_sketch_count(sketch: List[List[int]]) -> int:
    """Total count stored in the sketch (sum of first row, safe proxy)."""
    return sum(sketch[0]) if sketch else 0


# ----------------------------------------------------------------------
# Sessionisation (Parent A)
# ----------------------------------------------------------------------
def sessionize_events(events: List[Dict], gap_seconds: float = 1800.0) -> List[List[Dict]]:
    """Group events into sessions separated by *gap_seconds*."""
    sessions: List[List[Dict]] = []
    cur: List[Dict] = []
    last: float | None = None
    for e in sorted(events, key=lambda x: float(x.get("t", 0))):
        t = float(e.get("t", 0))
        if cur and last is not None and t - last > gap_seconds:
            sessions.append(cur)
            cur = []
        cur.append(e)
        last = t
    if cur:
        sessions.append(cur)
    return sessions


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def build_event_sketch(events: List[Dict], key: str = "type") -> List[List[int]]:
    """Create a count‑min sketch of event keys."""
    items = [str(e.get(key, "")) for e in events]
    return count_min_sketch(items)


def detect_bursts_with_bayes(
    events: List[Dict],
    key: str = "type",
    prior_default: float = 0.5,
    likelihood: float = 0.8,
    false_positive: float = 0.1,
) -> List[BurstSignal]:
    """
    Detect bursts by combining classic z‑score with Bayesian posterior.
    The prior for each key is taken from the count‑min sketch estimate,
    normalised by the total sketch count.
    """
    if not events:
        return []

    # Build sketch once
    sketch = build_event_sketch(events, key=key)

    # Raw frequencies for z‑score
    cnt = Counter(str(e.get(key, "")) for e in events)
    mean = sum(cnt.values()) / len(cnt)
    sd = math.sqrt(sum((v - mean) ** 2 for v in cnt.values()) / len(cnt)) or 1.0

    total_counts = total_sketch_count(sketch) or 1  # avoid div‑zero

    burst_signals: List[BurstSignal] = []
    for k, v in cnt.items():
        # Prior from sketch (frequency proportion)
        sketch_freq = sketch_estimate(k, sketch)
        prior = sketch_freq / total_counts if sketch_freq >= 0 else prior_default

        # Bayesian marginal and posterior
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)

        z = (v - mean) / sd
        burst_signals.append(
            BurstSignal(
                key=k,
                count=v,
                z_score=z,
                prior=prior,
                likelihood=likelihood,
                posterior=posterior,
                false_positive=false_positive,
            )
        )
    return burst_signals


def mine_temporal_motifs(sessions: List[List[Dict]], min_support: int = 2) -> List[TemporalMotif]:
    """
    Very simple motif miner: treat each session as an ordered list of event types.
    Count occurrences of each subsequence of length 2 (pair motifs). Return those
    with support >= *min_support* and attach a posterior probability derived from
    the Bayesian burst detection on the underlying events.
    """
    # Collect all pair motifs
    pair_counter: Counter[Tuple[str, str]] = Counter()
    for sess in sessions:
        types = [str(e.get("type", "")) for e in sess]
        for i in range(len(types) - 1):
            pair_counter[(types[i], types[i + 1])] += 1

    # Estimate priors for each pair using a sketch built on flattened pairs
    flat_pairs = [f"{a}->{b}" for a, b in pair_counter.keys()]
    pair_sketch = count_min_sketch(flat_pairs)

    total_pair_counts = total_sketch_count(pair_sketch) or 1

    motifs: List[TemporalMotif] = []
    for pat, supp in pair_counter.items():
        if supp < min_support:
            continue
        # Prior from sketch
        sketch_key = f"{pat[0]}->{pat[1]}"
        sketch_freq = sketch_estimate(sketch_key, pair_sketch)
        prior = sketch_freq / total_pair_counts

        # Use a fixed likelihood & false‑positive for motif relevance
        likelihood = 0.9
        false_pos = 0.05
        marginal = bayes_marginal(prior, likelihood, false_pos)
        posterior = bayes_update(prior, likelihood, marginal)

        motifs.append(TemporalMotif(pattern=pat, support=supp, posterior=posterior))
    return motifs


def allocate_vram_for_motifs(
    motifs: List[TemporalMotif],
    budget: VRAMBudget,
    width: int = 64,
    depth: int = 4,
) -> VRAMBudget:
    """
    Estimate VRAM usage for storing the given motifs.
    Each motif contributes a weight proportional to its posterior probability.
    The weight is inserted into a count‑min sketch; the sketch total drives the
    estimated usage, which is then clamped to the available budget.
    """
    # Build a sketch where each motif contributes its posterior (scaled)
    items = []
    for m in motifs:
        # Encode motif as string and repeat proportionally to posterior
        repetitions = max(1, int(m.posterior * 100))
        items.extend([f"{m.pattern[0]}->{m.pattern[1]}"] * repetitions)

    sketch = count_min_sketch(items, width=width, depth=depth)
    # Estimate raw usage: total sketch count * a scaling factor (e.g., 0.05 MB per count)
    raw_usage = total_sketch_count(sketch) * 0.05
    # Apply reserve headroom
    usable_budget = budget.budget_mb - budget.reserve_mb
    allocated = min(int(raw_usage), max(0, usable_budget))
    budget.used_mb = allocated
    return budget


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic event stream
    random.seed(42)
    types = ["A", "B", "C", "D"]
    events = []
    timestamp = 0.0
    for _ in range(200):
        evt_type = random.choices(types, weights=[0.5, 0.2, 0.2, 0.1])[0]
        events.append({"t": timestamp, "type": evt_type})
        timestamp += random.expovariate(1 / 30)  # avg 30 s between events

    # 1) Burst detection with Bayesian + sketch priors
    bursts = detect_bursts_with_bayes(events, key="type")
    print("Burst signals (top 3 by posterior):")
    for b in sorted(bursts, key=lambda x: x.posterior, reverse=True)[:3]:
        print(b)

    # 2) Sessionisation & motif mining
    sess = sessionize_events(events, gap_seconds=180.0)
    motifs = mine_temporal_motifs(sess, min_support=3)
    print("\nTemporal motifs (top 5 by posterior):")
    for m in sorted(motifs, key=lambda x: x.posterior, reverse=True)[:5]:
        print(m)

    # 3) VRAM allocation based on motifs
    budget = VRAMBudget(budget_mb=1024, reserve_mb=100)
    budget = allocate_vram_for_motifs(motifs, budget)
    print(f"\nVRAM allocation: used {budget.used_mb} MB of {budget.budget_mb} MB (reserve {budget.reserve_mb} MB)")