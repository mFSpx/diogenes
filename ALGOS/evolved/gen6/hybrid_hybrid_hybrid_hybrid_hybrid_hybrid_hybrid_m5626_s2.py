# DARWIN HAMMER — match 5626, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1073_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1722_s0.py (gen5)
# born: 2026-05-30T00:03:44Z

"""
Hybrid Algorithm Fusion of:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1073_s0.py (Parent A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1722_s0.py (Parent B)

Mathematical Bridge
-------------------
Both parents maintain a *scalar resource level* that influences
resource‑constrained decisions:

* Parent A uses a RAM ceiling and a deterministic ternary vector
  (derived from a payload hash) together with a MinHash signature to
  decide model loading/eviction.
* Parent B uses a scalar “resource level” to modulate a pruning
  probability `p_i(t)` for Bayesian updates and to scale the confidence
  term of a bandit via a Hoeffding bound.  The pruning probability is
  coupled to a Count‑Min sketch that compresses evidence frequencies.

The hybrid system unifies these ideas by letting the *resource level*
drive a **pruning probability** `p(t)` that simultaneously:

1. Scales the **eviction pressure** on the `ModelPool` (A) – a higher
   `p(t)` makes eviction more aggressive.
2. Adjusts the **confidence bound** of a `BanditAction` (B) via the
   Hoeffding inequality.
3. Modulates the **update weight** of a Count‑Min sketch that stores
   evidence frequencies.

All hash‑based structures (ternary vector, MinHash, Count‑Min sketch) share
the same deterministic 64‑bit hash function, providing a common mathematical
interface between the two parent topologies.
"""

import numpy as np
import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Tuple, Dict

# ----------------------------------------------------------------------
# Shared deterministic 64‑bit hash
# ----------------------------------------------------------------------
def _hash64(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash using SHA‑256."""
    h = hashlib.sha256()
    h.update(seed.to_bytes(4, "big"))
    h.update(token.encode("utf-8", errors="ignore"))
    # Take first 8 bytes → 64 bits
    return int.from_bytes(h.digest()[:8], "big", signed=False)


# ----------------------------------------------------------------------
# Parent‑A structures
# ----------------------------------------------------------------------
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


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


class ModelPool:
    """Manages a set of loaded models under a RAM ceiling."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.resource_level: float = 1.0  # scalar shared with Parent B

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model respecting tier exclusivity and RAM limits."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """Evict least‑recently‑added models until enough RAM is free."""
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # pop arbitrary (FIFO) entry
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)


# ----------------------------------------------------------------------
# Parent‑B structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathEvidence:
    id: str
    claim: str
    classification: str


@dataclass(frozen=True)
class MathHypothesis:
    id: str
    prior: float
    posterior: float
    evidence_ids: Tuple[str, ...] = ()


@dataclass
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float = 0.0
    algorithm: str = "hybrid"


# ----------------------------------------------------------------------
# Hash‑based vector utilities (bridge)
# ----------------------------------------------------------------------
def ternary_vector(payload: str, dim: int = 128) -> np.ndarray:
    """
    Produce a deterministic ternary vector (-1, 0, +1) from a payload string.
    The vector is built by hashing successive seeds and mapping the two
    least‑significant bits to {-1,0,+1}.
    """
    vec = np.zeros(dim, dtype=np.int8)
    for i in range(dim):
        h = _hash64(i, payload)
        bits = h & 0b11
        if bits == 0:
            vec[i] = -1
        elif bits == 1:
            vec[i] = 0
        else:
            vec[i] = 1
    return vec


def minhash_signature(tokens: List[str], num_perm: int = 64) -> np.ndarray:
    """
    Compute a MinHash signature of `tokens`.  For each permutation seed `s`
    the minimum 64‑bit hash over all tokens is taken.
    """
    sig = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    for s in range(num_perm):
        for token in tokens:
            h = _hash64(s, token)
            if h < sig[s]:
                sig[s] = h
    return sig


# ----------------------------------------------------------------------
# Count‑Min sketch (Parent B)
# ----------------------------------------------------------------------
class CountMinSketch:
    """
    Simple Count‑Min sketch with pairwise‑independent hash functions.
    """
    def __init__(self, width: int = 64, depth: int = 4):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int32)
        self.seeds = [random.randint(0, 2**31 - 1) for _ in range(depth)]

    def _hash(self, seed: int, item: str) -> int:
        return _hash64(seed, item) % self.width

    def update(self, items: Iterable[str], inc: int = 1) -> None:
        for item in items:
            for d, seed in enumerate(self.seeds):
                idx = self._hash(seed, item)
                self.table[d, idx] += inc

    def query(self, item: str) -> int:
        """Estimate frequency of `item` (upper bound)."""
        mins = [
            self.table[d, self._hash(seed, item)]
            for d, seed in enumerate(self.seeds)
        ]
        return min(mins)

    def total_count(self) -> int:
        """Total number of increments stored in the sketch."""
        return int(self.table.sum())


# ----------------------------------------------------------------------
# Hybrid mathematical operations
# ----------------------------------------------------------------------
def pruning_probability(resource: float, sketch: CountMinSketch,
                        alpha: float = 0.01, delta: float = 1e-3) -> float:
    """
    Compute a pruning probability `p(t)` that couples the scalar resource
    level with the sketch's total count.  The formulation mirrors the
    Parent B coupling:

        p(t) = exp(-α * resource) * (1 - 1 / (1 + total_sketch))

    The result is clipped to (0,1).
    """
    total = sketch.total_count() + 1  # avoid division by zero
    p = math.exp(-alpha * resource) * (1.0 - 1.0 / total)
    return max(0.0, min(1.0, p))


def hybrid_load_model(pool: ModelPool, model: ModelTier,
                      payload: str, sketch: CountMinSketch) -> None:
    """
    Decide whether to load `model` directly or to evict existing models.
    The decision uses:
      * ternary vector of the payload (Parent A)
      * MinHash signature of payload tokens (Parent A)
      * pruning probability derived from the shared resource level
        and the Count‑Min sketch (Parent B)

    If `p` exceeds a threshold proportional to the model's RAM usage,
    eviction is triggered before loading.
    """
    tokens = payload.split()
    signature = minhash_signature(tokens, num_perm=32)  # cheap signature
    ternary = ternary_vector(payload, dim=64)

    # Simple metric: overlap between signature and ternary (both int‑like)
    overlap = np.mean((signature % 3).astype(np.int8) == ternary[:32])
    p = pruning_probability(pool.resource_level, sketch)

    # Threshold mixes overlap and pruning probability
    threshold = 0.5 * (1.0 - overlap) + 0.5 * p

    if threshold > 0.4:
        # aggressive eviction path
        pool.load_with_eviction(model)
    else:
        # try direct load, may raise if RAM exceeded
        pool.load(model)


def hybrid_bandit_update(actions: List[BanditAction],
                         sketch: CountMinSketch,
                         resource: float,
                         delta: float = 1e-3) -> None:
    """
    Update the confidence bound of each `BanditAction` using a Hoeffding‑style
    term that depends on the sketch‑derived effective sample size `n_i`.
    The sample size is approximated by the sketch frequency of the action_id.
    """
    for act in actions:
        n_i = max(1, sketch.query(act.action_id))
        hoeffding = math.sqrt(math.log(1.0 / delta) / (2.0 * n_i))
        # Resource level scales the bound (more resources → tighter bound)
        act.confidence_bound = hoeffding / (1.0 + resource)


def hybrid_resource_update(pool: ModelPool,
                           reward: float,
                           regret: float,
                           learning_rate: float = 0.1) -> None:
    """
    Adjust the shared `resource_level` based on observed reward and regret.
    This mirrors the regret‑weighted update from Parent A while keeping the
    scalar compatible with Parent B's pruning probability.
    """
    delta = learning_rate * (reward - regret)
    pool.resource_level = max(0.0, pool.resource_level + delta)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise components
    pool = ModelPool(ram_ceiling_mb=2000)
    sketch = CountMinSketch(width=128, depth=4)

    # Dummy payload and models
    payload = "user:42 session:7 action:click item:1234"
    model_a = ModelTier(name="model_A", ram_mb=500, tier="T1")
    model_b = ModelTier(name="model_B", ram_mb=800, tier="T2")

    # Populate sketch with some evidence identifiers
    evidence_ids = [f"e{i}" for i in range(20)]
    sketch.update(evidence_ids, inc=1)

    # Hybrid loading
    hybrid_load_model(pool, model_a, payload, sketch)
    hybrid_load_model(pool, model_b, payload, sketch)

    # Create bandit actions
    actions = [
        BanditAction(action_id="e5", propensity=0.2, expected_reward=1.0),
        BanditAction(action_id="e12", propensity=0.5, expected_reward=0.8),
    ]

    # Update bandit confidence bounds
    hybrid_bandit_update(actions, sketch, pool.resource_level)

    # Simulate a reward/regret observation and update resource level
    hybrid_resource_update(pool, reward=0.9, regret=0.3)

    # Print summary (no external libraries used)
    print("Loaded models:", list(pool.loaded.keys()))
    print("Resource level:", pool.resource_level)
    for a in actions:
        print(f"Action {a.action_id}: confidence_bound={a.confidence_bound:.4f}")