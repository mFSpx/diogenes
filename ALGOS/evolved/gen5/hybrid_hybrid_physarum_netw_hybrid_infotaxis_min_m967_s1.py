# DARWIN HAMMER — match 967, survivor 1
# gen: 5
# parent_a: hybrid_physarum_network_hybrid_hybrid_hybrid_m64_s0.py (gen4)
# parent_b: hybrid_infotaxis_minhash_m63_s5.py (gen1)
# born: 2026-05-29T23:31:52Z

from __future__ import annotations

import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Certainty infrastructure (from epistemic_certainty.py)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat() + "Z")


# Hybrid Physarum-Sheaf and Infotaxis-Minhash Module
"""
This module fuses the hybrid_physarum_network_hybrid_hybrid_hybrid_m64_s0.py 
(Physarum-Sheaf dynamics) and hybrid_infotaxis_minhash_m63_s5.py (Infotaxis-Minhash).

The mathematical bridge between the two lies in the information-theoretic 
quantities: 
1. Physarum-Sheaf's flux |q_uv| and weighted discrepancy d_uv 
2. Infotaxis-Minhash's MinHash signatures and their Jaccard similarity.

The hybrid system integrates these by:
- Using MinHash signatures to initialize sheaf sections s_u 
- Computing flux and discrepancy using Physarum-Sheaf equations 
- Updating sheaf sections based on the weighted discrepancy 
- Using the Jaccard similarity of MinHash signatures to modulate 
  the information transport gain α in the Physarum-Sheaf update.
"""

# ----------------------------------------------------------------------
# MinHash core
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    token_set: set[str] = {t for t in tokens if t}
    if not token_set:
        return [np.iinfo(np.int64).max] * k
    return [min(_hash(i, t) for t in token_set) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have the same length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Hybrid Physarum-Sheaf and Infotaxis-Minhash
# ----------------------------------------------------------------------
@dataclass
class Node:
    pressure: float
    sheaf_section: np.ndarray
    certainty: float
    minhash_signature: List[int]


def hybrid_update(nodes: List[Node], edges: List[Tuple[int, int]], 
                  conductance: Dict[Tuple[int, int], float], 
                  alpha: float, beta: float, gamma: float, dt: float) -> None:
    for u, v in edges:
        # Compute flux and weighted discrepancy
        g_uv = conductance.get((u, v), 1.0)
        p_u, p_v = nodes[u].pressure, nodes[v].pressure
        s_u, s_v = nodes[u].sheaf_section, nodes[v].sheaf_section
        c_u, c_v = nodes[u].certainty, nodes[v].certainty
        q_uv = g_uv * (p_u - p_v)
        delta_uv = np.dot(s_u, np.array([1])) - s_v
        d_uv = np.sqrt(c_u * c_v) * np.linalg.norm(delta_uv)

        # Compute Jaccard similarity of MinHash signatures
        sig_u, sig_v = nodes[u].minhash_signature, nodes[v].minhash_signature
        sim_uv = similarity(sig_u, sig_v)

        # Update conductance
        g_uv = max(0, g_uv + dt * (alpha * sim_uv * abs(q_uv) + beta * d_uv - gamma * g_uv))
        conductance[(u, v)] = g_uv


def initialize_nodes(num_nodes: int, tokens: List[List[str]]) -> List[Node]:
    nodes = []
    for token_set in tokens:
        minhash_sig = signature(token_set)
        node = Node(pressure=random.random(), 
                    sheaf_section=np.random.rand(10), 
                    certainty=random.random(), 
                    minhash_signature=minhash_sig)
        nodes.append(node)
    return nodes


def run_hybridSimulation(num_nodes: int, num_edges: int, tokens: List[List[str]]) -> None:
    nodes = initialize_nodes(num_nodes, tokens)
    edges = [(random.randint(0, num_nodes-1), random.randint(0, num_nodes-1)) for _ in range(num_edges)]
    conductance = {(u, v): 1.0 for u, v in edges}

    alpha, beta, gamma, dt = 0.1, 0.2, 0.01, 0.01
    for _ in range(10):
        hybrid_update(nodes, edges, conductance, alpha, beta, gamma, dt)


if __name__ == "__main__":
    num_nodes, num_edges = 10, 20
    tokens = [["token1", "token2"], ["token3", "token4"], 
              ["token5", "token6"], ["token7", "token8"], 
              ["token9", "token10"], ["token11", "token12"], 
              ["token13", "token14"], ["token15", "token16"], 
              ["token17", "token18"], ["token19", "token20"]]
    run_hybridSimulation(num_nodes, num_edges, tokens)