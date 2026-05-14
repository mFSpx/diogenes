#!/usr/bin/env python3
"""Temporal session, burst, and motif mining helpers."""
from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass
@dataclass(frozen=True)
class BurstSignal: key: str; count: int; z_score: float
@dataclass(frozen=True)
class TemporalMotif: pattern: tuple[str,...]; support: int
def sessionize_events(events: list[dict], gap_seconds: float=1800.0) -> list[list[dict]]:
    sessions=[]; cur=[]; last=None
    for e in sorted(events,key=lambda x:x.get('t',0)):
        t=float(e.get('t',0))
        if cur and last is not None and t-last>gap_seconds: sessions.append(cur); cur=[]
        cur.append(e); last=t
    if cur: sessions.append(cur)
    return sessions
def detect_bursts(events: list[dict], key: str='type') -> list[BurstSignal]:
    c=Counter(str(e.get(key,'')) for e in events)
    if not c: return []
    mean=sum(c.values())/len(c); sd=math.sqrt(sum((v-mean)**2 for v in c.values())/len(c)) or 1.0
    return [BurstSignal(k,v,(v-mean)/sd) for k,v in c.items() if v>=mean]
def mine_temporal_motifs(sessions: list[list[dict]], key: str='type', min_support: int=2) -> list[TemporalMotif]:
    c=Counter(tuple(str(e.get(key,'')) for e in s) for s in sessions)
    return [TemporalMotif(p,v) for p,v in c.items() if v>=min_support]
