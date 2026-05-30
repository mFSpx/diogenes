# DARWIN HAMMER — match 116, survivor 2
# gen: 3
# parent_a: hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s1.py (gen2)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s3.py (gen2)
# born: 2026-05-29T23:26:55Z

"""Hybrid Semantic–Geometric Engine
Combines:
- PARENT ALGORITHM A: semantic_neighbors + pheromone‑based probabilities & entropy.
- PARENT ALGORITHM B: Voronoi partitioning + geometric (Clifford) product via Multivector algebra.

Mathematical bridge:
Each document is a point **p**∈ℝⁿ (semantic vector) and carries a pheromone vector **π**.
We first partition the point set with a Voronoi diagram (B).  Neighbourhood search is then
restricted to the same Voronoi cell, guaranteeing spatial locality.
Within a cell the pheromone mass is turned into a probability distribution; the
probabilities are interpreted as coefficients of a grade‑1 blade (∑πᵢ eᵢ).  The
semantic vector is likewise a grade‑1 blade (∑vᵢ eᵢ).  Their geometric product
gives a multivector whose scalar part is the dot product v·π and whose bivector
part encodes orientation‑weighted interaction.  The scalar part is used to
modulate the cosine similarity, while the bivector grade supplies an
entropy‑adjusted exploration term.  This yields a unified similarity
score that fuses semantic proximity, pheromone reinforcement and geometric
structure."""

import math
import random
import sys
import pathlib
import numpy as np
from collections import defaultdict

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def _cos(a, b):
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den


def pheromone_probabilities(pheromones):
    total = sum(pheromones)
    if total == 0:
        raise ValueError("pheromone vector must contain positive mass")
    return [p / total for p in pheromones]


def entropy(probabilities, eps=1e-12):
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0
    )


# ----------------------------------------------------------------------
# Utilities from Parent B (Voronoi + Multivector)
# ----------------------------------------------------------------------
Point = tuple[float, float]  # for Voronoi we work in 2‑D for simplicity


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError("seeds required")
    return min(
        range(len(seeds)),
        key=lambda i: (distance(point, seeds[i]), i)
    )


def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


def _blade_sign(indices):
    """Return a sorted tuple of unique indices and the sign of the permutation."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate index → cancel (e_i ∧ e_i = 0)
                lst.pop(j)
                lst.pop(j)
                return tuple(lst), sign
    return tuple(lst), sign


def _multiply_blades(blade_a, blade_b):
    combined = blade_a + blade_b
    result, sign = _blade_sign(combined)
    return result, sign


class Multivector:
    """Very small subset of a Clifford algebra implementation."""
    def __init__(self, components: dict[tuple[int, ...], float]):
        # remove zero coefficients
        self.components = {
            tuple(sorted(k)): float(v)
            for k, v in components.items()
            if abs(v) > 1e-15
        }

    def __add__(self, other):
        res = dict(self.components)
        for blade, coef in other.components.items():
            res[blade] = res.get(blade, 0.0) + coef
        return Multivector(res)

    def __mul__(self, other):
        """Geometric product (grade‑agnostic, returns a new Multivector)."""
        result = defaultdict(float)
        for b1, c1 in self.components.items():
            for b2, c2 in other.components.items():
                blade, sign = _multiply_blades(b1, b2)
                result[blade] += sign * c1 * c2
        return Multivector(dict(result))

    def scalar_part(self):
        return self.components.get((), 0.0)

    def __repr__(self):
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(),
                                   key=lambda x: (len(x[0]), x[0])):
            if blade:
                label = "e" + "".join(str(i) for i in blade)
            else:
                label = "1"
            terms.append(f"{coef:+.4g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"


def vector_to_multivector(vec: np.ndarray) -> Multivector:
    """Treat a real vector as a grade‑1 blade: Σ v_i e_i."""
    comps = { (i,): float(v) for i, v in enumerate(vec) if v != 0 }
    return Multivector(comps)


# ----------------------------------------------------------------------
# Hybrid Engine
# ----------------------------------------------------------------------
class HybridEngine:
    """Manages documents, pheromones and geometric partitions."""
    def __init__(self, seed_points: list[Point] | None = None):
        self.documents = {}          # doc_id → np.ndarray (semantic vector)
        self.pheromones = {}         # doc_id → list[float]
        self.positions = {}          # doc_id → Point (2‑D projection for Voronoi)
        self.seed_points = seed_points or []   # Voronoi seeds
        self._region_cache = None    # memoised region assignment

    # ------------------------------------------------------------------
    # Core registration
    # ------------------------------------------------------------------
    def register(self, doc_id: str, vector: np.ndarray,
                 pheromone: list[float], position: Point | None = None):
        self.documents[doc_id] = vector
        self.pheromones[doc_id] = pheromone
        if position is None:
            # simple projection: first two components (or random if missing)
            if len(vector) >= 2:
                position = (float(vector[0]), float(vector[1]))
            else:
                position = (random.random(), random.random())
        self.positions[doc_id] = position
        self._region_cache = None   # invalidate cache

    # ------------------------------------------------------------------
    # Voronoi region handling
    # ------------------------------------------------------------------
    def _compute_regions(self):
        if not self.seed_points:
            raise ValueError("seed points must be defined before region queries")
        points = list(self.positions.values())
        regions = assign(points, self.seed_points)
        # map doc_id → region index
        rev = {}
        for idx, pts in regions.items():
            for pt in pts:
                for did, pos in self.positions.items():
                    if pos == pt:
                        rev[did] = idx
        self._region_cache = rev

    def region_of(self, doc_id: str) -> int:
        if self._region_cache is None:
            self._compute_regions()
        return self._region_cache[doc_id]

    # ------------------------------------------------------------------
    # Geometric‑semantic similarity
    # ------------------------------------------------------------------
    def _geometric_score(self, doc_id: str, neighbor_id: str) -> float:
        """Geometric product between semantic and pheromone blades, reduced to a scalar."""
        v = self.documents[doc_id]
        w = self.documents[neighbor_id]
        pi = self.pheromones[doc_id]

        # grade‑1 blades
        mv_sem = vector_to_multivector(v)
        mv_pher = vector_to_multivector(np.array(pi[:len(v)]))  # truncate / pad

        # geometric product
        prod = mv_sem * mv_pher
        return prod.scalar_part()  # dot‑like term

    # ------------------------------------------------------------------
    # Hybrid neighbor retrieval
    # ------------------------------------------------------------------
    def hybrid_neighbors(self, doc_id: str, k: int = 5):
        """Return top‑k neighbours inside the same Voronoi cell, scored by:
        cos_similarity * pheromone_probability * (1 + geometric_score)."""
        if doc_id not in self.documents:
            raise KeyError(f"{doc_id} not registered")
        region = self.region_of(doc_id)
        # candidates: same region, different id
        candidates = [
            d for d in self.documents
            if d != doc_id and self.region_of(d) == region
        ]

        if not candidates:
            return []

        # pheromone probabilities for the query document
        probs = pheromone_probabilities(self.pheromones[doc_id])

        scores = []
        for idx, neighbor in enumerate(candidates):
            cos_sim = _cos(self.documents[doc_id], self.documents[neighbor])
            geom = self._geometric_score(doc_id, neighbor)
            # weight by pheromone probability (use index of neighbor in dict order)
            prob = probs[idx % len(probs)]
            combined = cos_sim * prob * (1.0 + geom)
            scores.append((neighbor, combined))

        return sorted(scores, key=lambda x: (-x[1], x[0]))[:k]

    # ------------------------------------------------------------------
    # Entropy‑aware action selection
    # ------------------------------------------------------------------
    def best_action(self, actions: list[str], doc_id: str, k: int = 5):
        """Choose an action based on neighbour entropy and pheromone distribution."""
        neighbors = self.hybrid_neighbors(doc_id, k)
        if not neighbors:
            return random.choice(actions)

        # build probability mass from neighbour scores
        raw = [score for _, score in neighbors]
        total = sum(raw) or 1.0
        probs = [r / total for r in raw]

        # compute entropy of the neighbour distribution
        neigh_entropy = entropy(probs)

        # combine with pheromone entropy
        pher_probs = pheromone_probabilities(self.pheromones[doc_id])
        pher_entropy = entropy(pher_probs)

        # weighted mixture (more exploration when entropy high)
        mix = (neigh_entropy + pher_entropy) / 2.0
        weights = [mix] + [ (1 - mix) / (len(actions) - 1) ] * (len(actions) - 1)

        # simple roulette selection
        choice = random.choices(actions, weights=weights, k=1)[0]
        return choice

    # ------------------------------------------------------------------
    # Pheromone update (simple evaporation + reinforcement)
    # ------------------------------------------------------------------
    def update_pheromones(self, doc_id: str, reward: float, decay: float = 0.1):
        """Increase pheromone mass proportional to reward and decay others."""
        pi = self.pheromones[doc_id]
        pi = [p * (1 - decay) for p in pi]
        # reinforce the first component (arbitrary choice for demo)
        if pi:
            pi[0] += reward
        self.pheromones[doc_id] = pi


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # create a tiny corpus of 6 documents in 4‑D space
    engine = HybridEngine(seed_points=[(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)])

    for i in range(6):
        vec = np.random.rand(4)
        pher = [random.random() for _ in range(4)]
        doc_id = f"doc{i}"
        engine.register(doc_id, vec, pher)

    # pick a document and ask for neighbours
    qid = "doc0"
    print(f"Region of {qid}: {engine.region_of(qid)}")
    print("Hybrid neighbours:", engine.hybrid_neighbors(qid, k=3))

    # action selection demo
    actions = ["read", "ignore", "bookmark"]
    print("Chosen action:", engine.best_action(actions, qid))

    # pheromone update demo
    engine.update_pheromones(qid, reward=0.5)
    print("Updated pheromones:", engine.pheromones[qid])