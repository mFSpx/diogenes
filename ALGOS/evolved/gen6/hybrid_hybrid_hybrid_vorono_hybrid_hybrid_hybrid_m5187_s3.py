# DARWIN HAMMER — match 5187, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_doomsd_m2387_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_krampu_m1196_s1.py (gen5)
# born: 2026-05-30T00:00:35Z

"""Hybrid Voronoi‑Sheaf / Doomsday‑Gini‑Ternary Lens (VSG‑DG‑TL) + Krampus‑Pheromone‑Sheaf (KP‑S)

Parents
-------
* **Parent A** – `hybrid_hybrid_voronoi_parti_hybrid_hybrid_doomsd_m2387_s2.py`
  (Voronoi partition → region size vector → Gini coefficient,
   sheaf restriction maps, ternary‑lens signatures).

* **Parent B** – `hybrid_hybrid_hybrid_krampu_hybrid_hybrid_krampu_m1196_s1.py`
  (Text → deterministic master vector → pheromone entries with
   entropy‑driven half‑life, sheaf co‑homology aggregation).

Mathematical Bridge
-------------------
Both parents expose a *sheaf* whose node spaces are finite‑dimensional
vectors.  The bridge is built by letting the **Voronoi region sizes**
determine a global **Gini weight** `w_G ∈ [0,1]` that scales every
restriction map of the sheaf.  Simultaneously, the **text‑derived
pheromone magnitudes** initialise the node sections, while the
**ternary‑lens** of each region supplies a compact signature that is
used as the raw vector for the corresponding node.  The resulting
system therefore couples a spatial partition, a distributional
inequality measure, a similarity‑compacting hash, and a time‑aware
pheromone dynamics within a single unified sheaf framework.

The code below implements this fusion, exposing three high‑level
functions that demonstrate the combined workflow."""
import math
import random
import hashlib
import sys
import pathlib
from typing import List, Tuple, Dict, Iterable
import numpy as np
from collections import Counter
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Core geometric utilities (Voronoi – Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the closest seed (break ties by index)."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def voronoi_partition(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the nearest seed, returning a region dict."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def region_sizes(regions: Dict[int, List[Point]]) -> List[int]:
    """Return the size (cardinality) of each Voronoi region."""
    return [len(v) for v in regions.values()]

def gini_coefficient(sizes: List[int]) -> float:
    """Standard Gini coefficient for a non‑negative list."""
    if not sizes:
        return 0.0
    arr = np.array(sizes, dtype=float)
    if np.all(arr == 0):
        return 0.0
    sorted_arr = np.sort(arr)
    n = len(arr)
    cumulative = np.cumsum(sorted_arr)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)

def ternary_lens(region_points: List[Point], k: int = 8) -> np.ndarray:
    """
    Produce a compact ternary signature of length `k`.
    Each coordinate of the hash is mapped to -1, 0, or 1.
    """
    if not region_points:
        # empty region → zero vector
        return np.zeros(k, dtype=int)
    # deterministic hash of the concatenated coordinates
    raw = ''.join(f'{x:.6f}{y:.6f}' for x, y in region_points).encode()
    h = hashlib.sha256(raw).digest()
    bits = int.from_bytes(h, 'big')
    # map groups of bits to ternary values
    vals = []
    for i in range(k):
        trit = (bits >> (2 * i)) & 0b11  # take two bits
        if trit == 0:
            vals.append(-1)
        elif trit == 1:
            vals.append(0)
        else:
            vals.append(1)
    return np.array(vals, dtype=int)

# ----------------------------------------------------------------------
# Text → feature → pheromone (Parent B)
# ----------------------------------------------------------------------
def text_entropy(text: str) -> float:
    """Shannon entropy of the character distribution."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    probs = np.array([c / total for c in counts.values()], dtype=float)
    return -np.sum(probs * np.log2(probs))

def master_vector(text: str, dim: int = 64) -> np.ndarray:
    """
    Deterministic pseudo‑random vector derived from the text.
    Each character contributes to a bit‑wise spread across `dim` slots.
    """
    vec = np.zeros(dim, dtype=float)
    for i, ch in enumerate(text):
        # hash the character together with its position
        h = hashlib.sha256(f'{ch}{i}'.encode()).digest()
        bits = int.from_bytes(h, 'big')
        for d in range(dim):
            if (bits >> d) & 1:
                vec[d] += 1.0
    # normalise to unit L2 norm
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

class PheromoneEntry:
    """A single pheromone signal attached to one dimension of a feature vector."""
    def __init__(self, index: int, value: float, half_life: float):
        self.index = index
        self.initial = value
        self.half_life = max(half_life, 1e-6)  # avoid division by zero
        self.age = 0.0

    def decay(self, dt: float = 1.0) -> float:
        """Advance time by `dt` and return the decayed signal."""
        self.age += dt
        factor = 0.5 ** (self.age / self.half_life)
        return self.initial * factor

def create_pheromones(text: str) -> List[PheromoneEntry]:
    """
    Convert a text into a list of pheromone entries.
    The magnitude of each entry is the absolute value of the master vector
    component; the half‑life is a monotonic function of the text entropy.
    """
    entropy = text_entropy(text)
    # map entropy in [0, log2(256)] ≈ [0,8] to half‑life in [1,10]
    half_life = 1.0 + 9.0 * (entropy / 8.0)
    vec = master_vector(text, dim=64)
    entries = [PheromoneEntry(i, float(abs(v)), half_life) for i, v in enumerate(vec) if v != 0]
    return entries

# ----------------------------------------------------------------------
# Hybrid Sheaf (combines Parent A & B)
# ----------------------------------------------------------------------
class HybridSheaf:
    """
    Sheaf whose nodes correspond to Voronoi regions.
    Each node stores a section vector (ternary lens) that is modulated
    by pheromone signals.  Restriction maps are identity matrices scaled
    by the global Gini weight.
    """
    def __init__(self, node_ids: Iterable[int], width: int = 64):
        self.nodes = list(node_ids)
        self.width = width
        self._sections: Dict[int, np.ndarray] = {n: np.zeros(width, dtype=float) for n in self.nodes}
        self._restrictions: Dict[Tuple[int, int], np.ndarray] = {}  # (src, dst) → matrix

    def set_section(self, node: int, vec: np.ndarray):
        if vec.shape != (self.width,):
            raise ValueError("section vector has wrong shape")
        self._sections[node] = vec.astype(float)

    def set_restriction(self, src: int, dst: int, weight: float):
        """Restriction is `weight * I` where I is the identity on `width`."""
        self._restrictions[(src, dst)] = weight * np.identity(self.width, dtype=float)

    def propagate(self, steps: int = 3, dt: float = 1.0) -> Dict[int, np.ndarray]:
        """
        Perform a simple diffusion: at each step each node receives the
        average of incoming restricted sections.  Pheromone decay is
        applied to the node's own section after every step.
        """
        for _ in range(steps):
            new_sections = {n: np.copy(v) for n, v in self._sections.items()}
            # accumulate incoming messages
            for (src, dst), mat in self._restrictions.items():
                msg = mat @ self._sections[src]
                new_sections[dst] += msg
            # average where multiple messages arrived
            incoming_counts = {n: 1 for n in self.nodes}  # start with own value
            for (src, dst) in self._restrictions:
                incoming_counts[dst] += 1
            for n in self.nodes:
                new_sections[n] /= incoming_counts[n]
            # apply pheromone decay (global, same for all nodes)
            for n in self.nodes:
                decay_factor = 0.5 ** (dt / 5.0)  # arbitrary fixed decay for demo
                new_sections[n] *= decay_factor
            self._sections = new_sections
        return {n: v.copy() for n, v in self._sections.items()}

# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------
def build_hybrid_sheaf(points: List[Point],
                      seeds: List[Point],
                      text: str,
                      ternary_dim: int = 8,
                      sheaf_width: int = 64) -> HybridSheaf:
    """
    Construct a HybridSheaf from spatial points, Voronoi seeds, and a text.
    1. Partition points → regions.
    2. Compute Gini weight from region sizes.
    3. Produce ternary lens vectors for each region.
    4. Initialise node sections with pheromone‑scaled ternary vectors.
    5. Connect nodes with edges based on seed proximity and set weighted restrictions.
    """
    # 1. Voronoi partition
    regions = voronoi_partition(points, seeds)
    sizes = region_sizes(regions)
    gini = gini_coefficient(sizes)            # scalar in [0,1]

    # 2. Ternary signatures per region
    ternary_vectors = {i: ternary_lens(regions[i], k=ternary_dim) for i in regions}

    # 3. Pheromone entries from text
    pheromones = create_pheromones(text)
    # aggregate pheromone magnitudes per dimension into a scaling vector
    pheromone_scale = np.zeros(sheaf_width, dtype=float)
    for p in pheromones:
        idx = p.index % sheaf_width
        pheromone_scale[idx] += p.initial

    # 4. Initialise sheaf
    sheaf = HybridSheaf(node_ids=regions.keys(), width=sheaf_width)

    for node_id, tern_vec in ternary_vectors.items():
        # embed ternary vector into the sheaf width (zero‑pad) and scale
        section = np.zeros(sheaf_width, dtype=float)
        section[:ternary_dim] = tern_vec.astype(float)
        section *= pheromone_scale[:sheaf_width]  # element‑wise scaling
        sheaf.set_section(node_id, section)

    # 5. Build adjacency (simple complete graph weighted by seed distance)
    for i in regions:
        for j in regions:
            if i >= j:
                continue
            w = distance(seeds[i], seeds[j])
            # restriction weight = (1 - gini) * exp(-w)  (example)
            weight = (1.0 - gini) * math.exp(-w)
            sheaf.set_restriction(i, j, weight)
            sheaf.set_restriction(j, i, weight)

    return sheaf

def hybrid_evaluate(points: List[Point],
                    seeds: List[Point],
                    text: str) -> Dict[int, np.ndarray]:
    """
    End‑to‑end evaluation: build the sheaf, run propagation and return
    the final node sections.
    """
    sheaf = build_hybrid_sheaf(points, seeds, text)
    final_sections = sheaf.propagate(steps=4, dt=1.0)
    return final_sections

def summarize_sections(sections: Dict[int, np.ndarray]) -> Dict[int, float]:
    """
    Produce a simple scalar summary (L2 norm) for each node's final section.
    """
    return {node: float(np.linalg.norm(vec)) for node, vec in sections.items()}

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # generate synthetic geometry
    random.seed(42)
    pts = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(200)]
    seeds = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(5)]

    # sample text
    sample_text = ("In the cryptic halls of the algorithmic forge, "
                   "entropy dances with deterministic hashes, "
                   "while pheromones linger like forgotten spells.")

    # run hybrid pipeline
    final = hybrid_evaluate(pts, seeds, sample_text)
    summary = summarize_sections(final)

    # output a concise report
    print("Hybrid Sheaf Node Summaries (L2 norm of final sections):")
    for node_id in sorted(summary):
        print(f"  Node {node_id}: {summary[node_id]:.4f}")

    # ensure no exception occurred
    sys.exit(0)