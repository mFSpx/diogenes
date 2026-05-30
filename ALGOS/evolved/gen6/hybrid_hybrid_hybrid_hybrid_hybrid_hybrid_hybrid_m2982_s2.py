# DARWIN HAMMER — match 2982, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2060_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hdc_se_hybrid_hybrid_fracti_m265_s2.py (gen4)
# born: 2026-05-29T23:47:02Z

"""Hybrid Algorithm integrating DARWIN HAMMER match 2060 (regret‑weighted model pool with Count‑Min Sketch)
and DARWIN HAMMER match 265 (hyperdimensional computing with fractional power binding).

Mathematical bridge:
- The Count‑Min Sketch provides a lightweight frequency estimate for *identifiers* (model names,
  action IDs, or hashed text signatures).  
- Hyperdimensional vectors (HDVs) are built from morphology parameters and a MinHash‑derived seed.
- Fractional power binding (`(v₁ ⊙ v₂)^α`) fuses the morphological HDV with the text‑derived HDV,
  producing a single representation whose *access frequency* is tracked by the sketch.
- Regret‑weighted selection combines the sketch estimate (proxy for model usefulness) with a
  regret term associated to each action, yielding a unified scoring function:
  `score = EV - λ·regret – μ·sketch_estimate`.

The code below implements the combined system, exposing three core functions that
demonstrate the hybrid operation."""
import sys
import pathlib
import math
import random
import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
@dataclass
class CountMinSketch:
    width: int = 2000
    depth: int = 5
    seed: int = 0
    table: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.table = np.zeros((self.depth, self.width), dtype=int)

    def _hash(self, item: str, i: int) -> int:
        # Deterministic hash using SHA‑256 seeded by row index
        h = hashlib.sha256(f"{self.seed}-{i}-{item}".encode()).digest()
        return int.from_bytes(h[:4], "big") % self.width

    def add(self, item: str) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.table[i, idx] += 1

    def estimate(self, item: str) -> int:
        return min(self.table[i, self._hash(item, i)] for i in range(self.depth))


@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Data structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)


def random_vector(dim: int = 1024, seed: int | str | None = None) -> np.ndarray:
    rng = random.Random(seed)
    return np.array([rng.random() for _ in range(dim)], dtype=float)


def morphology_vector(m: Morphology, dim: int = 1024) -> np.ndarray:
    seed_bytes = hashlib.sha256(f"{m.length}{m.width}{m.height}{m.mass}".encode()).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    base = random_vector(dim, seed)
    # Scale by the four physical attributes repeated across the vector
    scaling = np.array([m.length, m.width, m.height, m.mass], dtype=float)
    repeats = dim // scaling.size
    scaling_vec = np.tile(scaling, repeats)
    return base * scaling_vec


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    cleaned = re.sub(r"\s+", " ", text or "").strip().lower()
    shingles = [cleaned[i:i + 5] for i in range(len(cleaned) - 4)]
    hashes = []
    for _ in range(k):
        h = 0
        for sh in shingles:
            h = (h * 31 + hash(sh)) % (2 ** 32)
        hashes.append(h)
    return hashes


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def fractional_bind(v1: np.ndarray, v2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Element‑wise binding with fractional power: (v1 * v2) ** α."""
    bound = np.multiply(v1, v2)
    # Guard against negative values when α is non‑integer
    bound = np.where(bound >= 0, bound, -bound)
    return np.power(bound, alpha)


def create_hypervector(morph: Morphology, text: str, dim: int = 1024, alpha: float = 0.5) -> np.ndarray:
    """
    Build a hypervector by binding a morphology‑derived vector with a text‑derived vector.
    The text vector is generated from a deterministic seed obtained from the MinHash signature.
    """
    morph_vec = morphology_vector(morph, dim)
    mh = minhash_for_text(text, k=1)[0]  # single hash sufficient for a seed
    text_vec = random_vector(dim, mh)
    return fractional_bind(morph_vec, text_vec, alpha)


class ModelPool:
    """
    Stores hypervectors representing models. Uses a Count‑Min Sketch to track
    how often each model is accessed (frequency proxy) and integrates regret‑weighted
    selection.
    """

    def __init__(self, ram_ceiling_mb: int = 6000, sketch: CountMinSketch | None = None):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, np.ndarray] = {}
        self.sketch = sketch if sketch else CountMinSketch()
        self._usage_mb = 0

    def _vector_size_mb(self, vec: np.ndarray) -> int:
        return vec.nbytes // (1024 * 1024)

    def load(self, name: str, vector: np.ndarray) -> bool:
        """Attempt to load a model; returns True on success, False if memory limit would be exceeded."""
        size_mb = self._vector_size_mb(vector)
        if self._usage_mb + size_mb > self.ram_ceiling_mb:
            return False
        self.loaded[name] = vector
        self._usage_mb += size_mb
        self.sketch.add(name)
        return True

    def unload(self, name: str) -> None:
        if name in self.loaded:
            self._usage_mb -= self._vector_size_mb(self.loaded[name])
            del self.loaded[name]

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def get_vector(self, name: str) -> np.ndarray | None:
        return self.loaded.get(name)

    def frequency(self, name: str) -> int:
        return self.sketch.estimate(name)


def add_model_from_data(pool: ModelPool, name: str, morph: Morphology, text: str,
                       dim: int = 1024, alpha: float = 0.5) -> bool:
    """
    Constructs the hypervector for a model and attempts to load it into the pool.
    Returns True if the model was successfully loaded.
    """
    hv = create_hypervector(morph, text, dim, alpha)
    return pool.load(name, hv)


def regret_weighted_score(action: MathAction, regret: float,
                          frequency: int, lambda_reg: float = 1.0,
                          mu_freq: float = 0.1) -> float:
    """
    Unified scoring function:
        score = EV - λ·regret - μ·frequency_estimate
    """
    return action.expected_value - lambda_reg * regret - mu_freq * frequency


def select_best_action(pool: ModelPool, actions: List[MathAction],
                       regrets: Dict[str, float],
                       lambda_reg: float = 1.0,
                       mu_freq: float = 0.1) -> Tuple[str, float]:
    """
    Choose the action (model) with the highest regret‑weighted score.
    The model identifier is assumed to match the action.id.
    """
    best_id = None
    best_score = -math.inf
    for act in actions:
        freq = pool.frequency(act.id)
        reg = regrets.get(act.id, 0.0)
        sc = regret_weighted_score(act, reg, freq, lambda_reg, mu_freq)
        if sc > best_score:
            best_score = sc
            best_id = act.id
    return best_id, best_score


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a pool with modest RAM budget
    pool = ModelPool(ram_ceiling_mb=200)

    # Sample morphologies and texts
    morphs = [
        Morphology(1.2, 0.5, 0.3, 2.1),
        Morphology(0.9, 0.7, 0.4, 1.8),
        Morphology(1.5, 0.6, 0.5, 2.5),
    ]
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Evidence suggests that the algorithm converges rapidly.",
        "Verification of the model requires extensive testing."
    ]

    # Load models into the pool
    for i, (m, t) in enumerate(zip(morphs, texts), start=1):
        name = f"model_{i}"
        success = add_model_from_data(pool, name, m, t)
        if not success:
            print(f"Failed to load {name} due to memory constraints.", file=sys.stderr)

    # Define actions corresponding to model identifiers
    actions = [
        MathAction(id="model_1", expected_value=10.0, cost=1.0),
        MathAction(id="model_2", expected_value=12.0, cost=1.5),
        MathAction(id="model_3", expected_value=9.0, cost=0.8),
    ]

    # Regret values (could be derived from counterfactuals)
    regrets = {
        "model_1": 2.0,
        "model_2": 1.0,
        "model_3": 3.5,
    }

    # Perform selection
    chosen_id, score = select_best_action(pool, actions, regrets)
    print(f"Chosen model: {chosen_id} with score {score:.3f}")

    # Demonstrate retrieval of the selected hypervector
    vec = pool.get_vector(chosen_id)
    if vec is not None:
        print(f"Hypervector norm for {chosen_id}: {np.linalg.norm(vec):.3f}")
    else:
        print(f"Model {chosen_id} not loaded.", file=sys.stderr)