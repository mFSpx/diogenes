# DARWIN HAMMER — match 94, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_pheromone_inf_privacy_m54_s0.py (gen2)
# parent_b: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s0.py (gen1)
# born: 2026-05-29T23:25:39Z

"""Hybrid Pheromone‑Entropy & Minimum‑Cost Tree (HybridA‑B)

This module fuses the core topologies of:

* **Parent A** – `hybrid_hybrid_pheromone_inf_privacy_m54_s0.py`  
  Provides a pheromone system where each *surface* holds a decaying signal and an
  entropy helper that measures uncertainty of a probability mass.

* **Parent B** – `hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s0.py`  
  Extracts text spans, builds a similarity graph between spans and selects a
  minimum‑cost spanning tree (MST) over that graph.

**Mathematical bridge**  
The bridge is the *entropy* of the pheromone distribution.  For every extracted
span we query the pheromone system for a probability vector (here a simple
one‑hot based on the span label).  The Shannon entropy of that vector becomes
the *node cost*.  Edge costs are defined as a combination of cosine‑similarity
between span embeddings and the average node entropy, turning the MST problem
into a minimum‑cost tree that simultaneously respects semantic similarity and
information‑theoretic uncertainty.

The resulting hybrid algorithm can be used to prioritize spans that are both
semantically coherent and situated in high‑entropy (i.e. uncertain) regions of
the pheromone field.

"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Set

import numpy as np

# ----------------------------------------------------------------------
# Pheromone system (from Parent A)
# ----------------------------------------------------------------------
class PheromoneSystem:
    """Manages decaying pheromone signals and provides entropy calculations."""
    def __init__(self):
        self.pheromones: Dict[str, Dict] = {}

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        """Create or update a pheromone entry with exponential decay."""
        now = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
        else:
            prev = self.pheromones[surface_key]
            elapsed = (now - prev["created_time"]).total_seconds()
            decayed = prev["signal_value"] * math.pow(
                0.5, elapsed / prev["half_life_seconds"]
            )
            # Update with new signal; keep decayed value as a baseline
            self.pheromones[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": signal_value + decayed,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
        return self.pheromones[surface_key]["signal_value"]

    @staticmethod
    def calculate_entropy(probabilities: List[float], eps: float = 1e-12) -> float:
        """Shannon entropy of a probability mass."""
        total = sum(probabilities)
        if total <= 0:
            raise ValueError("positive probability mass required")
        probs = [(p / total) for p in probabilities if p > 0]
        return -sum(p * math.log(max(p, eps)) for p in probs)

    def label_distribution(self, label: str) -> List[float]:
        """
        Produce a synthetic probability distribution for a given label.
        In a real system this would be derived from pheromone concentrations.
        Here we map the pheromone signal (if any) to a two‑component distribution:
        [signal, 1‑signal] after normalisation.
        """
        entry = self.pheromones.get(label)
        if not entry:
            # Uniform uncertainty when no pheromone exists
            return [0.5, 0.5]
        sig = max(entry["signal_value"], 0.0)
        # Clamp to [0, 1] for probability‑like behaviour
        sig = min(sig, 1.0)
        return [sig, 1.0 - sig]

# ----------------------------------------------------------------------
# Span extraction (lightweight stand‑in for Parent B's gliner)
# ----------------------------------------------------------------------
class Span:
    """Simple data container for a text span."""
    __slots__ = ("start", "end", "text", "label", "score", "backend")

    def __init__(self, start: int, end: int, text: str, label: str, score: float, backend: str = "hybrid"):
        self.start = start
        self.end = end
        self.text = text
        self.label = label
        self.score = score
        self.backend = backend

    def __repr__(self) -> str:
        return f"Span({self.start},{self.end},'{self.text}',label='{self.label}',score={self.score:.2f})"


def extract_spans(text: str) -> List[Span]:
    """
    Very naive zero‑shot extractor: each word becomes a span.
    A random label from a small pool and a random confidence score are assigned.
    """
    labels = ["Entity", "Action", "Property", "Quantity"]
    words = text.split()
    spans: List[Span] = []
    pos = 0
    for w in words:
        start = text.find(w, pos)
        end = start + len(w)
        label = random.choice(labels)
        score = random.uniform(0.5, 1.0)
        spans.append(Span(start, end, w, label, score))
        pos = end
    return spans

# ----------------------------------------------------------------------
# Graph construction & Minimum‑Cost Tree (from Parent B)
# ----------------------------------------------------------------------
def span_embedding(span: Span, vocab: Dict[str, int]) -> np.ndarray:
    """
    One‑hot embedding of the span text based on a shared vocabulary.
    """
    vec = np.zeros(len(vocab), dtype=np.float32)
    token = span.text.lower()
    idx = vocab.get(token)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def build_graph(
    spans: List[Span],
    pheromone_system: PheromoneSystem,
) -> Tuple[Set[int], List[Tuple[int, int, float]], List[Span]]:
    """
    Constructs an undirected weighted graph where:
    * Nodes are spans (identified by their index in ``spans``).
    * Node cost = entropy of the pheromone label distribution.
    * Edge weight = (1 - cosine_similarity) + average_node_entropy.
    Returns a tuple (node_set, edge_list, spans) suitable for MST.
    """
    # Build vocabulary for embeddings
    vocab = {}
    for s in spans:
        token = s.text.lower()
        if token not in vocab:
            vocab[token] = len(vocab)

    # Pre‑compute embeddings and node entropies
    embeddings = [span_embedding(s, vocab) for s in spans]
    node_entropies = []
    for s in spans:
        probs = pheromone_system.label_distribution(s.label)
        ent = pheromone_system.calculate_entropy(probs)
        node_entropies.append(ent)

    nodes = set(range(len(spans)))
    edges: List[Tuple[int, int, float]] = []

    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            # Cosine similarity (guard against zero vectors)
            vi, vj = embeddings[i], embeddings[j]
            norm_i = np.linalg.norm(vi)
            norm_j = np.linalg.norm(vj)
            if norm_i == 0 or norm_j == 0:
                cosine = 0.0
            else:
                cosine = np.dot(vi, vj) / (norm_i * norm_j)
            # Edge cost definition (lower is better)
            cost = (1.0 - cosine) + (node_entropies[i] + node_entropies[j]) / 2.0
            edges.append((i, j, cost))

    return nodes, edges, spans


def minimum_spanning_tree(
    nodes: Set[int], edges: List[Tuple[int, int, float]]
) -> List[Tuple[int, int, float]]:
    """
    Kruskal's algorithm for MST. Returns the list of edges selected.
    """
    # Union‑Find data structure
    parent = {n: n for n in nodes}
    rank = {n: 0 for n in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        xr, yr = find(x), find(y)
        if xr == yr:
            return False
        if rank[xr] < rank[yr]:
            parent[xr] = yr
        elif rank[xr] > rank[yr]:
            parent[yr] = xr
        else:
            parent[yr] = xr
            rank[xr] += 1
        return True

    # Sort edges by increasing cost
    sorted_edges = sorted(edges, key=lambda e: e[2])
    mst: List[Tuple[int, int, float]] = []
    for u, v, w in sorted_edges:
        if union(u, v):
            mst.append((u, v, w))
        if len(mst) == len(nodes) - 1:
            break
    return mst

# ----------------------------------------------------------------------
# Hybrid operation exposing at least three public functions
# ----------------------------------------------------------------------
def update_pheromones_from_spans(
    pheromone_system: PheromoneSystem,
    spans: List[Span],
    half_life: float = 30.0,
) -> None:
    """
    Feed each span's label into the pheromone system, using the span score as the
    signal value.  This demonstrates interaction from the NLP side back into the
    pheromone field.
    """
    for s in spans:
        pheromone_system.calculate_pheromone_signal(
            surface_key=s.label,
            signal_kind="span_score",
            signal_value=s.score,
            half_life_seconds=half_life,
        )


def select_spans_via_mst(
    text: str,
    pheromone_system: PheromoneSystem,
) -> List[Span]:
    """
    Full hybrid pipeline:
    1. Extract spans.
    2. Update pheromones with those spans.
    3. Build a graph where entropy informs edge costs.
    4. Compute the MST and return the spans that participate in it.
    """
    spans = extract_spans(text)
    update_pheromones_from_spans(pheromone_system, spans)
    nodes, edges, span_list = build_graph(spans, pheromone_system)
    mst_edges = minimum_spanning_tree(nodes, edges)

    # Collect unique span indices from MST edges
    selected_idx = {i for edge in mst_edges for i in edge[:2]}
    # Preserve original order for readability
    return [span_list[i] for i in sorted(selected_idx)]


def compute_global_entropy(pheromone_system: PheromoneSystem) -> float:
    """
    Aggregate entropy over all known pheromone labels.  This function is useful
    for monitoring the overall uncertainty of the system and is independent of
    any particular text input.
    """
    entropies = []
    for label in pheromone_system.pheromones.keys():
        probs = pheromone_system.label_distribution(label)
        entropies.append(pheromone_system.calculate_entropy(probs))
    return float(np.mean(entropies)) if entropies else 0.0

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_text = "Alice discovered the hidden treasure in the ancient ruins."
    pheromone_sys = PheromoneSystem()

    print("=== Extracted spans ===")
    raw_spans = extract_spans(demo_text)
    for sp in raw_spans:
        print(sp)

    print("\n=== Selected spans after MST ===")
    selected = select_spans_via_mst(demo_text, pheromone_sys)
    for sp in selected:
        print(sp)

    print("\n=== Global pheromone entropy ===")
    print(f"{compute_global_entropy(pheromone_sys):.4f}")