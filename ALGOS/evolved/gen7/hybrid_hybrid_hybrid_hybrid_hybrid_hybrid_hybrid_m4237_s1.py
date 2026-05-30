# DARWIN HAMMER — match 4237, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1205_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m1531_s4.py (gen5)
# born: 2026-05-29T23:54:37Z

"""Hybrid Algorithm: Probabilistic‑Linguistic‑Geometric Fusion (PLGF)

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1205_s0.py (probabilistic
  acceptance + trust‑weighted linguistic similarity for model‑pool management)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m1531_s4.py (geometric‑algebra
  blade multiplication, ternary similarity exponent, fractional‑power binding,
  TTT‑Linear weight update)

Mathematical Bridge:
The ternary‑router similarity score `s ∈ [0,1]` produced in the geometric branch is
used as the exponent in a fractional‑power binding of a hypervector.  The same
score also scales the trust‑weighted linguistic similarity `ℓ` from the probabilistic
branch, yielding a combined acceptance metric  

    α = σ(‑ (c + r) / (e+ε)) · (ℓ · s)

where `c,r,e` are cost, risk and expected value of an action, `σ` is the logistic
function, and `ℓ` is the trust‑weighted textual similarity.  The metric `α`
drives model‑pool loading/eviction, while the bound hypervector participates in
the TTT‑Linear weight‑matrix adaptation.  Thus probabilistic decision‑making and
geometric binding are mathematically fused into a single unified system.
"""

import math
import random
import sys
import pathlib
from collections.abc import Mapping, Hashable
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Core data structures (from Parent A)
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]

@dataclass(frozen=True)
class MathAction:
    """An action whose acceptance will be evaluated."""
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
    """Manages a set of loaded models respecting RAM ceiling and tier exclusivity."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model; raises if constraints are violated."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """Load a model, evicting least‑recently‑added models until space is available."""
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # deterministic eviction: pop first inserted (order of dict)
            evicted_name = next(iter(self.loaded))
            self.loaded.pop(evicted_name)
        self.load(model)

# ----------------------------------------------------------------------
# Geometric‑Algebra utilities (from Parent B)
# ----------------------------------------------------------------------
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate index cancels out (e_i ^ e_i = 0)
                lst.pop(j)
                lst.pop(j)  # second element shifts into position j
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]):
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

# ----------------------------------------------------------------------
# Hybrid functional building blocks
# ----------------------------------------------------------------------
def logistic(x: float) -> float:
    """Standard logistic function σ(x) = 1 / (1 + e^{-x})."""
    return 1.0 / (1.0 + math.exp(-x))

def acceptance_probability(action: MathAction, cf: MathCounterfactual) -> float:
    """
    Probabilistic acceptance from Parent A.
    Uses a logistic transform of cost/risk relative to expected value,
    then multiplies by the counterfactual probability.
    """
    eps = 1e-9
    raw = -(action.cost + action.risk) / (action.expected_value + eps)
    return logistic(raw) * cf.probability

def trust_weighted_linguistic_similarity(text_a: str, text_b: str, trust: float) -> float:
    """
    Very simple token‑set Jaccard similarity scaled by a trust factor.
    Returns a value in [0, trust].
    """
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a and not set_b:
        return trust  # both empty -> maximal similarity
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return trust * (intersection / union)

def ternary_similarity_score(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Mimics the SSIM‑like ternary router similarity.
    Normalised dot product clamped to [0,1].
    """
    dot = float(np.dot(vec1, vec2))
    norm = float(np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-12)
    sim = dot / norm
    # map from [-1,1] to [0,1]
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))

def fractional_power_binding(vector: np.ndarray, exponent: float) -> np.ndarray:
    """
    Raise each component to `exponent` while preserving sign.
    Equivalent to sign(v) * |v|**exponent.
    """
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
# Hybrid core operation (integrates both parents)
# ----------------------------------------------------------------------
def hybrid_decision_and_update(
    pool: ModelPool,
    model: ModelTier,
    action: MathAction,
    cf: MathCounterfactual,
    trust: float,
    text_ref: str,
    text_candidate: str,
    W: np.ndarray,
    inp_vec: np.ndarray,
    rng: random.Random = random,
) -> tuple[ModelPool, np.ndarray, float]:
    """
    1. Compute probabilistic acceptance `p_a`.
    2. Compute linguistic similarity `ℓ` using trust.
    3. Compute ternary similarity `s` between `inp_vec` and a random probe.
       (In a real system the probe would be derived from the model's text.)
    4. Combine into α = p_a * (ℓ * s).
    5. If α exceeds a threshold, load the model (with eviction) into the pool.
    6. Use `s` as exponent for fractional‑power binding of `inp_vec`.
    7. Update the weight matrix `W` with the bound vector via TTT‑Linear rule.
    Returns the (potentially) updated pool, weight matrix, and α.
    """
    # 1. Probabilistic acceptance
    p_a = acceptance_probability(action, cf)

    # 2. Trust‑weighted linguistic similarity
    ell = trust_weighted_linguistic_similarity(text_ref, text_candidate, trust)

    # 3. Ternary similarity (probe is a random unit vector)
    probe = rng.normalvariate(0, 1)
    probe_vec = np.full_like(inp_vec, probe)
    s = ternary_similarity_score(inp_vec, probe_vec)

    # 4. Combined metric α
    alpha = p_a * (ell * s)

    # 5. Model‑pool decision
    threshold = 0.05  # empirically chosen small threshold
    if alpha > threshold:
        try:
            pool.load(model)
        except Exception:
            # If direct load fails, attempt eviction‑based load
            pool.load_with_eviction(model)

    # 6. Fractional‑power binding using similarity as exponent
    bound_vec = fractional_power_binding(inp_vec, exponent=s)

    # 7. Weight‑matrix adaptation
    W_new = ttt_linear_update(W, inp_vec, bound_vec, lr=0.02)

    return pool, W_new, alpha

# ----------------------------------------------------------------------
# Additional helper functions to showcase the hybrid system
# ----------------------------------------------------------------------
def generate_random_hypervector(dim: int, rng: random.Random = random) -> np.ndarray:
    """Create a random hypervector with components drawn from N(0,1)."""
    return np.array([rng.normalvariate(0, 1) for _ in range(dim)], dtype=float)

def demo_hybrid_flow():
    """Runs a short demonstration of the hybrid algorithm."""
    # Initialise model pool
    pool = ModelPool(ram_ceiling_mb=2000)

    # Define two dummy models
    model_a = ModelTier(name="model_A", ram_mb=500, tier="T1",
                        text="the quick brown fox jumps over the lazy dog")
    model_b = ModelTier(name="model_B", ram_mb=800, tier="T2",
                        text="lorem ipsum dolor sit amet consectetur adipiscing elit")

    # Action / counterfactual
    act = MathAction(id="act1", expected_value=10.0, cost=2.0, risk=1.0)
    cf = MathCounterfactual(action_id="act1", outcome_value=9.5, probability=0.9)

    # Trust and texts
    trust = 0.8
    ref_text = "machine learning model selection based on performance"
    cand_text = model_a.text

    # Random weight matrix and input vector
    dim = 64
    W = np.zeros((dim, dim))
    inp = generate_random_hypervector(dim)

    # Perform hybrid step with model_a
    pool, W, alpha = hybrid_decision_and_update(
        pool=pool,
        model=model_a,
        action=act,
        cf=cf,
        trust=trust,
        text_ref=ref_text,
        text_candidate=cand_text,
        W=W,
        inp_vec=inp,
    )
    print(f"Alpha after first step: {alpha:.4f}")
    print(f"Loaded models: {list(pool.loaded.keys())}")

    # Second step with model_b (different texts, same action)
    cand_text2 = model_b.text
    pool, W, alpha = hybrid_decision_and_update(
        pool=pool,
        model=model_b,
        action=act,
        cf=cf,
        trust=trust,
        text_ref=ref_text,
        text_candidate=cand_text2,
        W=W,
        inp_vec=inp,
    )
    print(f"Alpha after second step: {alpha:.4f}")
    print(f"Loaded models: {list(pool.loaded.keys())}")

if __name__ == "__main__":
    demo_hybrid_flow()