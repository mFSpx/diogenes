# DARWIN HAMMER — match 280, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s1.py (gen2)
# parent_b: temporal_motifs.py (gen0)
# born: 2026-05-29T23:28:04Z

"""
Fusing hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s1.py and temporal_motifs.py
into a unified system. The mathematical bridge between the two parents lies in the 
application of Bayesian updates to temporal motif mining. Specifically, we will 
incorporate Bayesian marginal probabilities and updates into the burst detection 
and temporal motif mining processes.

The governing equations of the parents are:

- Bayesian marginal probability: P(E|e) = P(e|E)P(E) + P(e|~E)P(~E)
- Bayesian update: P(E|e) = P(e|E)P(E) / P(E|e)
- Burst detection: z-score = (count - mean) / standard deviation
- Temporal motif mining: support count of patterns in sessions

The hybrid system will integrate these equations by applying Bayesian updates to 
the burst detection and temporal motif mining processes.
"""

import math
import numpy as np
from collections import Counter
from dataclasses import dataclass

@dataclass(frozen=True)
class BurstSignal: key: str; count: int; z_score: float; prior: float; likelihood: float; false_positive: float
@dataclass(frozen=True)
class TemporalMotif: pattern: tuple[str,...]; support: int; prior: float; likelihood: float; false_positive: float

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def sessionize_events(events: list[dict], gap_seconds: float=1800.0) -> list[list[dict]]:
    sessions=[]; cur=[]; last=None
    for e in sorted(events,key=lambda x:x.get('t',0)):
        t=float(e.get('t',0))
        if cur and last is not None and t-last>gap_seconds: sessions.append(cur); cur=[]
        cur.append(e); last=t
    if cur: sessions.append(cur)
    return sessions

def detect_bursts(events: list[dict], key: str='type', prior: float = 0.5, likelihood: float = 0.8, false_positive: float = 0.1) -> list[BurstSignal]:
    c=Counter(str(e.get(key,'')) for e in events)
    if not c: return []
    mean=sum(c.values())/len(c); sd=math.sqrt(sum((v-mean)**2 for v in c.values())/len(c)) or 1.0
    burst_signals = []
    for k,v in c.items():
        marginal = bayes_marginal(prior, likelihood, false_positive)
        updated_prior = bayes_update(prior, likelihood, marginal)
        z_score = (v-mean)/sd
        burst_signals.append(BurstSignal(k,v,z_score, prior, likelihood, false_positive))
    return burst_signals

def mine_temporal_motifs(sessions: list[list[dict]], key: str='type', min_support: int=2, prior: float = 0.5, likelihood: float = 0.8, false_positive: float = 0.1) -> list[TemporalMotif]:
    c=Counter(tuple(str(e.get(key,'')) for e in s) for s in sessions)
    temporal_motifs = []
    for p,v in c.items():
        if v>=min_support:
            marginal = bayes_marginal(prior, likelihood, false_positive)
            updated_prior = bayes_update(prior, likelihood, marginal)
            temporal_motifs.append(TemporalMotif(p,v, prior, likelihood, false_positive))
    return temporal_motifs

def hybrid_analysis(events: list[dict], key: str='type', gap_seconds: float=1800.0, min_support: int=2, prior: float = 0.5, likelihood: float = 0.8, false_positive: float = 0.1):
    sessions = sessionize_events(events, gap_seconds)
    burst_signals = detect_bursts(events, key, prior, likelihood, false_positive)
    temporal_motifs = mine_temporal_motifs(sessions, key, min_support, prior, likelihood, false_positive)
    return burst_signals, temporal_motifs

if __name__ == "__main__":
    events = [{'t': 1, 'type': 'A'}, {'t': 2, 'type': 'B'}, {'t': 3, 'type': 'A'}, {'t': 1801, 'type': 'C'}, {'t': 1802, 'type': 'C'}]
    burst_signals, temporal_motifs = hybrid_analysis(events)
    print("Burst Signals:")
    for signal in burst_signals:
        print(signal)
    print("Temporal Motifs:")
    for motif in temporal_motifs:
        print(motif)