from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import uuid, statistics
from collections import Counter

@dataclass(frozen=True)
class BurstSignal:
    signal_id: str
    entity_id: str
    event_count: int
    inter_arrival_anomaly_score: float
    hawkes_self_excitement_score: float
    is_coordinated_burst: bool

@dataclass(frozen=True)
class TemporalMotif:
    motif_id: str
    pattern: tuple[str,...]
    frequency: int
    anomaly_score: float
    false_positive_band: tuple[float,float]

def _ts(e):
    v=e.get('ts',0.0)
    if isinstance(v,(int,float)): return float(v)
    s=str(v).replace('Z','+00:00')
    return datetime.fromisoformat(s).timestamp()

def sessionize_events(events: list[dict], session_gap_seconds: float=1800.0, gap_seconds: float|None=None) -> list[list[dict]]:
    gap = session_gap_seconds if gap_seconds is None else gap_seconds
    ev=sorted(events, key=_ts)
    if not ev: return []
    sessions=[[ev[0]]]
    for e in ev[1:]:
        if _ts(e)-_ts(sessions[-1][-1]) <= gap: sessions[-1].append(e)
        else: sessions.append([e])
    return sessions

def detect_bursts(events: list[dict], entity_id: str) -> list[BurstSignal]:
    if len(events)<3: return []
    ev=sorted(events, key=_ts); intervals=[_ts(b)-_ts(a) for a,b in zip(ev,ev[1:])]
    if not intervals: return []
    med=statistics.median(intervals) or 1.0
    rapid=sum(1 for x in intervals if x<=max(2.0, med/2.0))
    score=max(0.0,min(1.0, rapid/max(1,len(intervals))))
    if len(ev)>=3 and (score>0 or max(intervals)<=5.0):
        return [BurstSignal("signal_"+uuid.uuid4().hex[:12], entity_id, len(ev), score, min(1.0, len(ev)/20.0), len(ev)>=10)]
    return []

def mine_temporal_motifs(sequences, n_gram_size: int=3, min_frequency: int=2, key: str='type') -> list[TemporalMotif]:
    cnt=Counter()
    for seq in sequences:
        vals=[]
        for x in seq:
            vals.append(x.get(key) if isinstance(x,dict) else x)
        for i in range(0, max(0, len(vals)-n_gram_size+1)):
            cnt[tuple(vals[i:i+n_gram_size])]+=1
    items=[(p,c) for p,c in cnt.items() if c>=min_frequency]
    if not items: return []
    maxc=max(c for _,c in items)
    return [TemporalMotif("motif_"+uuid.uuid4().hex[:12], p, c, 1.0-(c/maxc), (0.05,0.20)) for p,c in sorted(items, key=lambda x:(-x[1],x[0]))]
