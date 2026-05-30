# DARWIN HAMMER — match 5576, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m784_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s2.py (gen3)
# born: 2026-05-30T00:02:59Z

"""
Hybrid Fusion of:
- hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m784_s1.py (geometric product, Voronoi, Fisher, Gaussian beam)
- hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s2.py (stylometry features, model tiering, sphericity)

Mathematical Bridge:
Stylometry categories are encoded as basis blades of a geometric algebra multivector.
The component vector of each multivector serves as a statistical sample for a Fisher
information matrix (I = Σ grad·gradᵀ).  The Gaussian beam intensity,
I_gauss(d) = exp(-2 (d/σ)²), weights the connectivity between Voronoi cells that
host model instances.  The sphericity index derived from the multivector’s grade
distribution modulates the RAM ceiling for loading a model tier.  Thus the
geometric‑algebra representation, statistical Fisher metric, and spatial Voronoi
structure are fused with the model‑loading logic of the second parent.
"""

import math
import random
import sys
import pathlib
from typing import Dict, Tuple, List, Set, FrozenSet
import numpy as np

# -------------------- Geometric Algebra Core (Parent A) --------------------

Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def _blade_sign(indices: Tuple[int, ...]) -> Tuple[List[int], int]:
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate basis vectors (e_i ∧ e_i = 0)
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(tuple(combined))
    return frozenset(result), sign

class Multivector:
    """Simple multivector for an n‑dimensional Euclidean space."""
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> 'Multivector':
        return Multivector({blade: coef for blade, coef in self.components.items()
                            if len(blade) == k}, self.n)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: 'Multivector') -> 'Multivector':
        new = self.components.copy()
        for b, c in other.components.items():
            new[b] = new.get(b, 0.0) + c
            if abs(new[b]) < 1e-15:
                del new[b]
        return Multivector(new, self.n)

    def __mul__(self, other: 'Multivector') -> 'Multivector':
        result: Dict[FrozenSet[int], float] = {}
        for ba, ca in self.components.items():
            for bb, cb in other.components.items():
                blade, sign = _multiply_blades(ba, bb)
                result[blade] = result.get(blade, 0.0) + sign * ca * cb
        return Multivector(result, self.n)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(),
                                  key=lambda item: (len(item[0]), sorted(item[0]))):
            if blade:
                basis = "e" + "^".join(str(i) for i in sorted(blade))
            else:
                basis = "1"
            terms.append(f"{coef:.3g}{basis}")
        return " + ".join(terms)

    def as_vector(self) -> np.ndarray:
        """Return the scalar + 1‑vector part as a dense vector of length n+1."""
        vec = np.zeros(self.n + 1)
        vec[0] = self.scalar_part()
        for blade, coef in self.components.items():
            if len(blade) == 1:
                idx = next(iter(blade))
                vec[idx] = coef
        return vec

    def magnitude(self) -> float:
        return math.sqrt(sum(c * c for c in self.components.values()))

# -------------------- Stylometry & Model Pool (Parent B) --------------------

FUNCTION_CATS: Dict[str, Set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

    def __repr__(self) -> str:
        return f"ModelTier(name={self.name}, ram={self.ram_mb}MB, tier={self.tier})"

class ModelPool:
    """Manages loading/unloading of models respecting a RAM ceiling."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        return self._used_ram() + model.ram_mb <= self.ram_ceiling_mb

    def load(self, model: ModelTier) -> bool:
        if self.is_loaded(model.name):
            return True
        if self.can_load(model):
            self.loaded[model.name] = model
            return True
        return False

    def unload(self, name: str) -> bool:
        if name in self.loaded:
            del self.loaded[name]
            return True
        return False

    def __repr__(self) -> str:
        return f"ModelPool(used={self._used_ram()}MB/{self.ram_ceiling_mb}MB, models={list(self.loaded)})"

# -------------------- Fusion Functions --------------------

def text_to_multivector(text: str) -> Multivector:
    """
    Encode stylometry category counts into a geometric algebra multivector.
    Each category maps to a distinct basis vector e_i (i starts at 1).
    The scalar part is the total word count.
    """
    words = [w.strip(".,!?;:()[]\"'").lower() for w in text.split()]
    counts = {cat: 0 for cat in FUNCTION_CATS}
    for w in words:
        for cat, vocab in FUNCTION_CATS.items():
            if w in vocab:
                counts[cat] += 1
    # Map categories to basis indices 1..len(FUNCTION_CATS)
    basis_index = {cat: i + 1 for i, cat in enumerate(FUNCTION_CATS)}
    components: Dict[FrozenSet[int], float] = {frozenset(): len(words)}  # scalar = total words
    for cat, cnt in counts.items():
        if cnt != 0:
            components[frozenset({basis_index[cat]})] = float(cnt)
    return Multivector(components, n=len(FUNCTION_CATS) + 1)

def fisher_information_matrix(multivectors: List[Multivector]) -> np.ndarray:
    """
    Approximate the Fisher information matrix using the outer product of
    the 1‑vector parts of the multivectors (treated as gradient samples).
    I = Σ (g_i ⊗ g_i) where g_i is the vector of grade‑1 components.
    """
    if not multivectors:
        raise ValueError("At least one multivector required")
    n = multivectors[0].n
    I = np.zeros((n, n))
    for mv in multivectors:
        g = mv.as_vector()
        I += np.outer(g, g)
    return I / len(multivectors)

def gaussian_beam_intensity(dist: float, waist: float = 1.0) -> float:
    """Gaussian beam intensity as a function of distance."""
    return math.exp(-2.0 * (dist / waist) ** 2)

def sphericity_index(mv: Multivector) -> float:
    """
    Compute a sphericity‑like index: ratio of the magnitude of the highest‑grade
    part to the total magnitude.  Values close to 1 indicate a “spherical”
    multivector (mass concentrated in high grades).
    """
    if not mv.components:
        return 0.0
    max_grade = max(len(b) for b in mv.components)
    high_grade_mag = math.sqrt(sum(c * c for b, c in mv.components.items() if len(b) == max_grade))
    return high_grade_mag / mv.magnitude()

def load_model_via_hybrid(pool: ModelPool,
                          mv: Multivector,
                          seeds: List[Point],
                          model_catalog: Dict[str, ModelTier],
                          point: Point,
                          waist: float = 1.0) -> Tuple[bool, str]:
    """
    Hybrid loading routine:
    1. Assign the point to a Voronoi seed (region = model key).
    2. Compute Gaussian intensity based on distance to seed.
    3. Adjust the model's RAM demand by the intensity and sphericity index.
    4. Attempt to load the model respecting the pool ceiling.
    Returns (success, message).
    """
    region_idx = nearest(point, seeds)
    # Use the region index to pick a model name (simple deterministic mapping)
    model_names = sorted(model_catalog.keys())
    model_name = model_names[region_idx % len(model_names)]
    model = model_catalog[model_name]

    # Spatial weighting
    dist = distance(point, seeds[region_idx])
    intensity = gaussian_beam_intensity(dist, waist)

    # Statistical weighting via sphericity
    sph = sphericity_index(mv)

    # Effective RAM demand (scaled down if intensity high and sphericity low)
    effective_ram = int(model.ram_mb * (1.0 - 0.5 * intensity) * (1.0 + 0.3 * (1 - sph)))
    effective_model = ModelTier(name=model.name, ram_mb=effective_ram, tier=model.tier)

    if pool.load(effective_model):
        return True, f"Loaded {effective_model} into region {region_idx}"
    else:
        return False, f"Insufficient RAM for {effective_model} (region {region_idx})"

def hybrid_analysis(points: List[Point],
                    seeds: List[Point],
                    texts: List[str],
                    model_catalog: Dict[str, ModelTier]) -> Dict[int, List[str]]:
    """
    Perform a full hybrid pipeline:
    - Voronoi partition the spatial points.
    - Convert each associated text into a multivector.
    - Compute the Fisher matrix for each region.
    - Return a mapping region -> list of model load messages.
    """
    regions = assign(points, seeds)
    results: Dict[int, List[str]] = {i: [] for i in range(len(seeds))}
    # Pair each point with its text (assume same length)
    for pt, txt in zip(points, texts):
        region = nearest(pt, seeds)
        mv = text_to_multivector(txt)
        # For demonstration we compute Fisher on the fly with a single sample
        fisher = fisher_information_matrix([mv])
        # Use the trace of Fisher as a proxy for “information richness”
        info_score = np.trace(fisher)
        # Adjust model selection based on info_score (higher -> prefer larger models)
        # Simple heuristic: pick the model with ram closest but not exceeding info_score*10
        candidates = [m for m in model_catalog.values() if m.ram_mb <= info_score * 10]
        if not candidates:
            chosen = min(model_catalog.values(), key=lambda m: m.ram_mb)
        else:
            chosen = max(candidates, key=lambda m: m.ram_mb)
        pool = ModelPool()
        success, msg = load_model_via_hybrid(pool, mv, seeds, {chosen.name: chosen}, pt)
        results[region].append(msg)
    return results

# -------------------- Smoke Test --------------------

if __name__ == "__main__":
    # Define a tiny spatial scenario
    seed_points = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    data_points = [(1.0, 0.5), (4.5, 4.8), (9.0, 0.2)]
    sample_texts = [
        "I am the one who loves the beautiful sky and the vast ocean.",
        "You cannot not see the article about the preposition under the table.",
        "They will not never forget the conjunction and the quantifier."
    ]

    # Model catalog (name -> ModelTier)
    catalog = {
        "tiny": ModelTier(name="tiny", ram_mb=500, tier="A"),
        "small": ModelTier(name="small", ram_mb=1500, tier="B"),
        "medium": ModelTier(name="medium", ram_mb=3000, tier="C"),
        "large": ModelTier(name="large", ram_mb=5000, tier="D")
    }

    # Run the hybrid analysis
    out = hybrid_analysis(data_points, seed_points, sample_texts, catalog)

    for region, msgs in out.items():
        print(f"Region {region}:")
        for m in msgs:
            print("  ", m)

    # Demonstrate direct functions
    mv_example = text_to_multivector(sample_texts[0])
    print("\nExample Multivector:", mv_example)
    print("Magnitude:", mv_example.magnitude())
    print("Sphericity index:", sphericity_index(mv_example))
    print("Fisher matrix from single sample:\n", fisher_information_matrix([mv_example]))