# DARWIN HAMMER — match 2314, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_decision_hygi_hybrid_rete_bandit_g_m544_s1.py (gen3)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s1.py (gen4)
# born: 2026-05-29T23:41:50Z

"""Hybrid Algorithm integrating Weighted Entropy Work Allocation with Geometric‑Product Voronoi Partitioning and Ternary Routing.

Parents:
- hybrid_hybrid_decision_hygi_hybrid_rete_bandit_g_m544_s1.py (weighted entropy, regex‑based evidence extraction, work allocation)
- hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s1.py (Clifford geometric product, Voronoi partition, ternary routing)

Mathematical Bridge:
The bridge is the notion of *allocation density* over a spatial partition.  
Evidence counts are transformed into a weighted entropy measure (Parent A).  
Points in Euclidean space are partitioned into Voronoi cells using distances
derived from the Clifford geometric product (Parent B).  
The entropy value determines a total work budget, which is then *distributed*
among the Voronoi cells proportionally to both the cell population and the
entropy‑derived density.  Finally, a ternary routing scheme (max 3 neighbours)
provides a path for a work‑unit to travel from a source seed to a target seed,
demonstrating the fused dynamics.

The module therefore fuses:
1. Regex‑driven evidence extraction → weighted entropy.
2. Geometric‑product based distance → Voronoi assignment.
3. Entropy‑scaled work allocation → ternary graph routing.

All operations rely only on the Python standard library and NumPy.
"""

import math
import re
import sys
import random
from collections import Counter
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex patterns and weighted entropy utilities
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|"
    r"advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|"
    r"protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|"
    r"filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|"
    r"tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|"
    r"cannot afford|rent due|no money)\b",
    re.I,
)

REGEX_CATEGORIES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
}


def extract_category_counts(text: str) -> Counter:
    """Count occurrences of each semantic regex category in *text*."""
    counts = Counter()
    for name, pattern in REGEX_CATEGORIES.items():
        matches = pattern.findall(text)
        counts[name] = len(matches)
    return counts


def weighted_entropy(counts: Counter, weights: dict) -> float:
    """
    Compute weighted Shannon entropy.

    H = - Σ_i w_i * p_i * log(p_i)    where p_i = count_i / total_count
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for key, cnt in counts.items():
        p = cnt / total
        if p > 0:
            w = weights.get(key, 1.0)
            entropy -= w * p * math.log(p, 2)
    return entropy


# ----------------------------------------------------------------------
# Parent B – Clifford geometric product helpers and Voronoi utilities
# ----------------------------------------------------------------------
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Duplicated indices cancel (e_i * e_i = 1) and are removed.
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel pair
                lst.pop(j)
                lst.pop(j)  # next element shifts into position j
                n -= 2
                i = -1  # restart outer loop because list changed
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset):
    """Multiply two basis blades represented as frozensets of indices."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def geometric_product(v1: np.ndarray, v2: np.ndarray):
    """
    Compute the Clifford geometric product of two grade‑1 vectors.

    Returns a tuple (scalar_part, bivector_part) where:
    - scalar_part = v1·v2 (dot product)
    - bivector_part = v1∧v2 encoded as an antisymmetric matrix.
    """
    dot = float(np.dot(v1, v2))
    # Bivector as antisymmetric matrix B_ij = v1_i*v2_j - v1_j*v2_i
    biv = np.outer(v1, v2) - np.outer(v2, v1)
    return dot, biv


def geometric_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Derive a distance metric from the geometric product.
    The scalar (dot) part gives cosθ·|p||q|; we use the Euclidean norm of the
    bivector as a complementary measure and combine them.
    """
    dot, biv = geometric_product(p, q)
    biv_norm = np.linalg.norm(biv)
    # Euclidean distance is a robust fallback; we blend both terms.
    euclid = np.linalg.norm(p - q)
    # Blend factor α balances algebraic vs Euclidean contribution.
    α = 0.5
    return α * euclid + (1 - α) * biv_norm


def voronoi_assign(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Assign each point to the index of its nearest seed using geometric_distance.
    Returns an array `assignments` where assignments[i] is the seed index for point i.
    """
    assignments = np.empty(len(points), dtype=int)
    for i, pt in enumerate(points):
        dists = np.array([geometric_distance(pt, seed) for seed in seeds])
        assignments[i] = int(np.argmin(dists))
    return assignments


# ----------------------------------------------------------------------
# Hybrid Functions (demonstrating the fused system)
# ----------------------------------------------------------------------
def hybrid_compute_allocation(
    points: np.ndarray,
    seeds: np.ndarray,
    evidence_text: str,
    total_work_budget: float,
    category_weights: dict,
) -> dict:
    """
    Full hybrid pipeline:

    1. Extract semantic category counts from *evidence_text* (Parent A).
    2. Compute weighted entropy H.
    3. Partition *points* into Voronoi cells defined by *seeds* using geometric distance
       (Parent B).
    4. Distribute *total_work_budget* across cells proportionally to:
         (H / ΣH) * (cell_population / total_population)
       where ΣH is the sum of entropies computed for each cell (here identical,
       so the factor reduces to the population share, but the function keeps the
       general form for future extensions).

    Returns a mapping ``seed_index -> allocated_work``.
    """
    # Step 1 & 2: evidence → entropy
    counts = extract_category_counts(evidence_text)
    entropy = weighted_entropy(counts, category_weights)

    # Guard against zero entropy
    if entropy == 0:
        entropy = 1e-12

    # Step 3: Voronoi assignment
    assignments = voronoi_assign(points, seeds)

    # Step 4: allocation
    allocation = {}
    total_points = len(points)
    for seed_idx in range(len(seeds)):
        cell_mask = assignments == seed_idx
        cell_pop = np.count_nonzero(cell_mask)
        # Simple proportional allocation using both entropy and population
        share = (entropy / (entropy * len(seeds))) * (cell_pop / total_points)
        allocation[seed_idx] = share * total_work_budget
    return allocation


def build_ternary_graph(points: np.ndarray, k: int = 3) -> dict:
    """
    Build a graph where each point connects to its *k* nearest neighbours
    (according to geometric_distance).  The graph is represented as a dict:
    ``node_index -> list of neighbour indices``.
    """
    n = len(points)
    graph = {i: [] for i in range(n)}
    for i, pt in enumerate(points):
        distances = [(geometric_distance(pt, points[j]), j) for j in range(n) if j != i]
        distances.sort(key=lambda x: x[0])
        neighbours = [idx for _, idx in distances[:k]]
        graph[i] = neighbours
    return graph


def ternary_route(
    graph: dict,
    start: int,
    goal: int,
    max_steps: int = 1000,
) -> list:
    """
    Find a path from *start* to *goal* using a breadth‑first search limited to
    ternary branching (the graph already respects that constraint). Returns a list
    of node indices representing the path, or an empty list if no path is found
    within *max_steps*.
    """
    from collections import deque

    visited = set()
    queue = deque([[start]])

    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        if len(path) > max_steps:
            continue
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return []


def hybrid_demo():
    """
    Demonstration that ties together all hybrid components:

    - Random 2‑D points and seeds.
    - Synthetic evidence text.
    - Compute work allocation per seed.
    - Build a ternary graph on the seed positions.
    - Route a work‑unit from the seed with most allocation to the one with least.
    """
    # 1. Synthetic geometry
    np.random.seed(42)
    points = np.random.rand(200, 2) * 100  # 200 points in 100×100 square
    seeds = np.random.rand(5, 2) * 100     # 5 seed locations

    # 2. Synthetic evidence
    evidence_text = """
    The team verified the source and logged the evidence.  We have a plan and a checklist.
    However, there is a delay because we need to wait for additional support.
    The outcome was successful and the document is archived.
    """
    category_weights = {
        "evidence": 1.5,
        "planning": 1.2,
        "delay": 0.8,
        "support": 1.0,
        "outcome": 1.3,
    }

    # 3. Allocation
    total_work_budget = 1000.0
    allocation = hybrid_compute_allocation(
        points, seeds, evidence_text, total_work_budget, category_weights
    )
    print("Work allocation per seed:")
    for idx, work in allocation.items():
        print(f"  Seed {idx}: {work:.2f}")

    # 4. Build ternary graph on seed positions
    seed_graph = build_ternary_graph(seeds, k=3)

    # 5. Route from most‑loaded seed to least‑loaded seed
    most_loaded = max(allocation, key=allocation.get)
    least_loaded = min(allocation, key=allocation.get)
    path = ternary_route(seed_graph, most_loaded, least_loaded)
    print(f"\nTernary route from seed {most_loaded} (most work) to seed {least_loaded} (least work):")
    if path:
        print(" -> ".join(map(str, path)))
    else:
        print("No path found within step limit.")


if __name__ == "__main__":
    hybrid_demo()