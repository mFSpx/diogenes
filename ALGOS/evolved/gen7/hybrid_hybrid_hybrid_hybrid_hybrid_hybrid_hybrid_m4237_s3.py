# DARWIN HAMMER — match 4237, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1205_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m1531_s4.py (gen5)
# born: 2026-05-29T23:54:37Z

import math
import random
import sys
from collections import OrderedDict
from collections.abc import Mapping, Hashable
from dataclasses import dataclass, field
from typing import Tuple, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (Probabilistic branch)
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]


@dataclass(frozen=True)
class MathAction:
    """Action whose acceptance will be evaluated."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome used to modulate acceptance."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class ModelTier:
    """Lightweight description of a model that can be loaded into the pool."""
    name: str
    ram_mb: int
    tier: str   # e.g. "T1","T2","T3"
    text: str   # descriptive text for linguistic similarity


class ModelPool:
    """
    LRU‑based model pool respecting RAM ceiling and tier exclusivity.
    The most‑recently‑used model is kept; least‑recently‑used are evicted first.
    """
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self._loaded: "OrderedDict[str, ModelTier]" = OrderedDict()

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self._loaded.values())

    def _check_constraints(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self._loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")

    def load(self, model: ModelTier) -> None:
        """Load a model; raises if constraints are violated."""
        self._check_constraints(model)
        self._loaded[model.name] = model
        self._loaded.move_to_end(model.name)   # mark as most‑recent

    def load_with_eviction(self, model: ModelTier) -> None:
        """Load a model, evicting LRU models until space is available."""
        while True:
            try:
                self.load(model)
                break
            except Exception as exc:
                if "RAM ceiling exceeded" not in str(exc):
                    raise
                # Evict the least‑recently‑used entry
                evicted_name, _ = self._loaded.popitem(last=False)
                # continue loop to retry loading after freeing space

    def touch(self, name: str) -> None:
        """Mark a model as recently used (updates LRU order)."""
        if name in self._loaded:
            self._loaded.move_to_end(name)

    @property
    def loaded(self) -> Dict[str, ModelTier]:
        """Read‑only view of loaded models."""
        return dict(self._loaded)


# ----------------------------------------------------------------------
# Geometric‑Algebra utilities (Geometric branch)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Bubble‑sort indices, tracking sign changes; cancel duplicate indices."""
    lst = indices[:]
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)
        elif lst[i] == lst[i + 1]:
            # duplicate cancels out
            lst.pop(i)
            lst.pop(i)  # second element shifts into position i
            n -= 2
            i = max(i - 1, 0)
        else:
            i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset, int]:
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def tokens_to_blades(tokens: List[str], dim: int = 256) -> List[frozenset]:
    """
    Map each token to a basis blade index (hash modulo `dim`).
    Returns a list of blades (each a frozenset of a single index).
    """
    blades = []
    for tok in tokens:
        idx = hash(tok) % dim
        blades.append(frozenset({idx}))
    return blades


def blade_similarity(blades_a: List[frozenset], blades_b: List[frozenset]) -> float:
    """
    Compute a similarity based on geometric products of token blades.
    For each pair we multiply blades and count sign‑preserving matches.
    Normalised to [0,1].
    """
    if not blades_a or not blades_b:
        return 0.0
    matches = 0
    total = len(blades_a) * len(blades_b)
    for a in blades_a:
        for b in blades_b:
            _, sign = _multiply_blades(a, b)
            if sign == 1:
                matches += 1
    return matches / total


# ----------------------------------------------------------------------
# Helper utilities (shared)
# ----------------------------------------------------------------------
def safe_logistic(x: float) -> float:
    """Logistic with overflow protection."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def acceptance_probability(action: MathAction, cf: MathCounterfactual) -> float:
    """
    Probabilistic acceptance from the original branch.
    Uses a safe logistic transform of cost/risk relative to expected value,
    then multiplies by the counterfactual probability.
    """
    eps = 1e-12
    denominator = action.expected_value + eps
    raw = -(action.cost + action.risk) / denominator
    return safe_logistic(raw) * cf.probability


def trust_weighted_linguistic_similarity(text_a: str, text_b: str, trust: float) -> float:
    """
    Token‑set Jaccard similarity scaled by a trust factor.
    Trust is clamped to [0,1] to keep the result in [0,1].
    """
    trust = max(0.0, min(1.0, trust))
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a and not set_b:
        return trust
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return trust * (inter / union)


def text_to_vector(text: str, dim: int = 128, seed: int = 0) -> np.ndarray:
    """
    Deterministic pseudo‑random embedding of a text string.
    Each token contributes a random unit vector; the sum is normalised.
    """
    rng = np.random.default_rng(seed)
    vec = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        token_seed = (hash(token) & 0xffffffff) ^ seed
        rng_token = np.random.default_rng(token_seed)
        direction = rng_token.normal(size=dim).astype(np.float32)
        direction /= np.linalg.norm(direction) + 1e-12
        vec += direction
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def ternary_similarity_score(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Normalised dot product clamped to [0,1]; maps [-1,1] → [0,1].
    """
    dot = float(np.dot(vec1, vec2))
    norm = float(np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-12)
    sim = dot / norm
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def fractional_power_binding(vector: np.ndarray, exponent: float, min_exp: float = 0.1) -> np.ndarray:
    """
    Raise each component to `exponent` while preserving sign.
    Exponent is lower‑bounded to avoid collapsing the vector.
    """
    exponent = max(min_exp, exponent)
    sign = np.sign(vector)
    magnitude = np.abs(vector) ** exponent
    return sign * magnitude


def ttt_linear_update(W: np.ndarray, inp: np.ndarray, bound: np.ndarray, lr: float = 0.01) -> np.ndarray:
    """
    TTT‑Linear weight update: W ← W + lr·(inp ⊗ bound − W)
    where ⊗ denotes the outer product.
    """
    grad = np.outer(inp, bound) - W
    return W + lr * grad


# ----------------------------------------------------------------------
# Hybrid core operation (deep integration)
# ----------------------------------------------------------------------
def hybrid_decision_and_update(
    pool: ModelPool,
    model: ModelTier,
    action: MathAction,
    cf: MathCounterfactual,
    trust: float,
    reference_text: str,
    candidate_text: str,
    W: np.ndarray,
    inp_vec: np.ndarray,
    rng: random.Random = random,
    blade_dim: int = 256,
    embed_dim: int = 128,
) -> Tuple[ModelPool, np.ndarray, float]:
    """
    Integrated decision pipeline:

    1. Probabilistic acceptance `p_a`.
    2. Linguistic similarity `ℓ` (trust‑weighted Jaccard).
    3. Geometric‑algebra blade similarity `γ`.
    4. Textual embeddings → ternary similarity `s`.
    5. Combined metric
           α = p_a · (ℓ · γ · s)
       (clamped to [0,1] for stability).
    6. If α exceeds a dynamic threshold, load the model (with LRU eviction).
    7. Bind `inp_vec` with exponent `s` (bounded away from zero).
    8. Update weight matrix `W` via TTT‑Linear rule.
    Returns updated pool, weight matrix, and the final α.
    """
    # 1. Probabilistic acceptance
    p_a = acceptance_probability(action, cf)

    # 2. Trust‑weighted linguistic similarity
    ell = trust_weighted_linguistic_similarity(reference_text, candidate_text, trust)

    # 3. Blade‑based geometric similarity
    tokens_ref = reference_text.lower().split()
    tokens_cand = candidate_text.lower().split()
    blades_ref = tokens_to_blades(tokens_ref, dim=blade_dim)
    blades_cand = tokens_to_blades(tokens_cand, dim=blade_dim)
    gamma = blade_similarity(blades_ref, blades_cand)  # ∈ [0,1]

    # 4. Embedding‑based ternary similarity
    # Deterministic probe derived from model text
    probe_vec = text_to_vector(model.text, dim=embed_dim, seed=hash(model.name) & 0xffffffff)
    s = ternary_similarity_score(inp_vec, probe_vec)  # ∈ [0,1]

    # 5. Combined acceptance metric
    alpha_raw = p_a * (ell * gamma * s)
    alpha = max(0.0, min(1.0, alpha_raw))

    # 6. Model loading decision (threshold adapts to α distribution)
    threshold = 0.4  # could be tuned or made adaptive
    if alpha >= threshold:
        pool.load_with_eviction(model)
        pool.touch(model.name)

    # 7. Fractional‑power binding (exponent bounded)
    bound_vec = fractional_power_binding(inp_vec, exponent=s)

    # 8. Weight matrix update
    W_new = ttt_linear_update(W, inp_vec, bound_vec, lr=0.01)

    return pool, W_new, alpha