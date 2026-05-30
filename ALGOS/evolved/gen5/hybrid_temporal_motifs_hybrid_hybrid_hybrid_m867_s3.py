# DARWIN HAMMER — match 867, survivor 3
# gen: 5
# parent_a: temporal_motifs.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m217_s1.py (gen4)
# born: 2026-05-29T23:31:21Z

import numpy as np
from collections import Counter
from dataclasses import dataclass

@dataclass(frozen=True)
class BurstSignal: 
    key: str; 
    count: int; 
    z_score: float

@dataclass(frozen=True)
class TemporalMotif: 
    pattern: tuple[str,...]; 
    support: int

class Sheaf:
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions = {}
        self._sections = {}
        self._history = {}

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node: any, value: np.ndarray) -> None:
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("Section dimension must match node dimension")
        self._sections[node] = np.asarray(value, dtype=float)
        self._history[node] = self._history.get(node, []) + [value]

    def update_section(self, node: any, value: np.ndarray) -> None:
        if node not in self._history:
            self.set_section(node, value)
        else:
            prev_values = self._history[node]
            new_value = np.mean([value] + prev_values, axis=0)
            self.set_section(node, new_value)

def sessionize_events(events: list[dict], gap_seconds: float=1800.0) -> list[list[dict]]:
    sessions=[]
    cur=[]
    last=None
    for e in sorted(events,key=lambda x:x.get('t',0)):
        t=float(e.get('t',0))
        if cur and last is not None and t-last>gap_seconds: 
            sessions.append(cur); 
            cur=[]
        cur.append(e); 
        last=t
    if cur: 
        sessions.append(cur)
    return sessions

def detect_bursts(events: list[dict], key: str='type') -> list[BurstSignal]:
    c=Counter(str(e.get(key,'')) for e in events)
    if not c: 
        return []
    mean=sum(c.values())/len(c); 
    sd=math.sqrt(sum((v-mean)**2 for v in c.values())/len(c)) or 1.0
    return [BurstSignal(k,v,(v-mean)/sd) for k,v in c.items() if v>=mean]

def mine_temporal_motifs(sessions: list[list[dict]], key: str='type', min_support: int=2) -> list[TemporalMotif]:
    c=Counter(tuple(str(e.get(key,'')) for e in s) for s in sessions)
    return [TemporalMotif(p,v) for p,v in c.items() if v>=min_support]

def sheaf_temporal_motif_integration(sheaf: Sheaf, events: list[dict], key: str='type', min_support: int=2):
    sessions = sessionize_events(events)
    motifs = mine_temporal_motifs(sessions, key, min_support)
    for motif in motifs:
        node = motif.pattern[0]
        value = np.array([1.0 if event.get(key, '') == node else 0.0 for event in events])
        sheaf.update_section(node, value)

def sheaf_burst_detection(sheaf: Sheaf, events: list[dict], key: str='type') -> list[BurstSignal]:
    bursts = detect_bursts(events, key)
    for burst in bursts:
        node = burst.key
        value = np.array([1.0 if event.get(key, '') == node else 0.0 for event in events])
        sheaf.update_section(node, value)
    return bursts

def sheaf_section_retrieval(sheaf: Sheaf, node: any) -> np.ndarray:
    return sheaf._sections.get(node, np.array([]))

if __name__ == "__main__":
    node_dims = {'A': 2, 'B': 3}
    edges = [('A', 'B')]
    sheaf = Sheaf(node_dims, edges)
    events = [{'t': 1.0, 'type': 'A'}, {'t': 2.0, 'type': 'B'}, {'t': 3.0, 'type': 'A'}]
    sheaf_temporal_motif_integration(sheaf, events)
    bursts = sheaf_burst_detection(sheaf, events)
    section = sheaf_section_retrieval(sheaf, 'A')
    print(section)