# DARWIN HAMMER — match 867, survivor 1
# gen: 5
# parent_a: temporal_motifs.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m217_s1.py (gen4)
# born: 2026-05-29T23:31:21Z

"""
This module fuses the topological structures of 
temporal_motifs.py (Temporal session, burst, and motif mining helpers) and 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m217_s1.py (DARWIN HAMMER — match 217, survivor 1).

The mathematical bridge between the two parents lies in the fact that 
the temporal motifs can be viewed as patterns in a cellular sheaf, 
where each node represents a state in the temporal motif and each edge 
represents a transition between states. The sections of the sheaf can 
be used as input to detect bursts and mine temporal motifs.

The governing equations of the hybrid system are based on the sheaf's sections 
and the burst detection process.
"""

import numpy as np
import math
from collections import Counter
from dataclasses import dataclass

@dataclass(frozen=True)
class BurstSignal: key: str; count: int; z_score: float

@dataclass(frozen=True)
class TemporalMotif: pattern: tuple[str,...]; support: int

class Sheaf:
    """
    Cellular sheaf on a directed graph.

    * Nodes carry a vector space of dimension given by ``node_dims``.
    * Each directed edge ``(u, v)`` carries a linear restriction map
      ``src_map : ℝ^{dim(u)} → ℝ^{dim(e)}`` and
      ``dst_map : ℝ^{dim(v)} → ℝ^{dim(e)}``.
    * A *section* assigns a vector to every node.
    """

    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions = {}
        self._sections = {}

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        """Register the two restriction matrices for a directed edge."""
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node: any, value: np.ndarray) -> None:
        """Assign a vector to a node."""
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("Section dimension must match node dimension")
        self._sections[node] = np.asarray(value)

def detect_bursts(events: list[dict], key: str='type', sheaf: Sheaf=None) -> list[BurstSignal]:
    c=Counter(str(e.get(key,'')) for e in events)
    if not c: return []
    mean=sum(c.values())/len(c); sd=math.sqrt(sum((v-mean)**2 for v in c.values())/len(c)) or 1.0
    bursts = [BurstSignal(k,v,(v-mean)/sd) for k,v in c.items() if v>=mean]
    if sheaf:
        for burst in bursts:
            sheaf.set_section(burst.key, np.array([burst.count, burst.z_score]))
    return bursts

def mine_temporal_motifs(sessions: list[list[dict]], key: str='type', min_support: int=2, sheaf: Sheaf=None) -> list[TemporalMotif]:
    c=Counter(tuple(str(e.get(key,'')) for e in s) for s in sessions)
    motifs = [TemporalMotif(p,v) for p,v in c.items() if v>=min_support]
    if sheaf:
        for motif in motifs:
            sheaf.set_section(motif.pattern, np.array([motif.support]))
    return motifs

def hybrid_operation(events: list[dict], sessions: list[list[dict]], key: str='type', min_support: int=2) -> (list[BurstSignal], list[TemporalMotif]):
    node_dims = {'type': 2}
    edges = [('type', 'type')]
    sheaf = Sheaf(node_dims, edges)
    bursts = detect_bursts(events, key, sheaf)
    motifs = mine_temporal_motifs(sessions, key, min_support, sheaf)
    return bursts, motifs

if __name__ == "__main__":
    events = [{'t': 1, 'type': 'A'}, {'t': 2, 'type': 'B'}, {'t': 3, 'type': 'A'}, {'t': 4, 'type': 'B'}]
    sessions = [[{'t': 1, 'type': 'A'}, {'t': 2, 'type': 'B'}], [{'t': 3, 'type': 'A'}, {'t': 4, 'type': 'B'}]]
    bursts, motifs = hybrid_operation(events, sessions)
    print(bursts)
    print(motifs)