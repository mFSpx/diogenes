# DARWIN HAMMER — match 173, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s6.py (gen3)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s1.py (gen2)
# born: 2026-05-29T23:27:28Z

import sys
import hashlib
import random
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Dict, Tuple, Callable

# ----------------------------------------------------------------------
# Core data structures
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
    tier: str
    reference_tokens: Tuple[str, ...] = field(default_factory=tuple)


# ----------------------------------------------------------------------
# Model pool with sophisticated eviction
# ----------------------------------------------------------------------
class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.load_timestamp: Dict[str, float] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _check_constraints(self, model: ModelTier) -> None:
        # Mutual exclusivity: T3 cannot coexist with any T2
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")

    def load(self, model: ModelTier) -> None:
        self._check_constraints(model)
        self.loaded[model.name] = model
        self.load_timestamp[model.name] = datetime.now(tz=timezone.utc).timestamp()

    def _evict_one(self, score_fn: Callable[[ModelTier], float]) -> None:
        """Evict the model with the lowest score according to score_fn."""
        if not self.loaded:
            return
        victim_name = min(self.loaded, key=lambda n: score_fn(self.loaded[n]))
        del self.loaded[victim_name]
        del self.load_timestamp[victim_name]

    def load_with_eviction(self, model: ModelTier,
                           score_fn: Callable[[ModelTier], float]) -> None:
        """Load model, evicting the least valuable resident until constraints hold."""
        while True:
            try:
                self.load(model)
                break
            except RuntimeError:
                self._evict_one(score_fn)

    def lru_model(self) -> ModelTier:
        """Return the least‑recently‑used model (oldest timestamp)."""
        if not self.loaded:
            raise RuntimeError("No models loaded")
        oldest_name = min(self.load_timestamp, key=self.load_timestamp.get)
        return self.loaded[oldest_name]


# ----------------------------------------------------------------------
# MinHash utilities (liquid‑time‑constant style)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length k."""
    seeds = range(k)
    sig = [sys.maxsize] * k
    for token in tokens:
        token = token.strip()
        for i, seed in enumerate(seeds):
            h = _hash(seed, token)
            if h < sig[i]:
                sig[i] = h
    return sig


def minhash_jaccard(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


# ----------------------------------------------------------------------
# Differential privacy helpers
# ----------------------------------------------------------------------
def laplace_noise(scale: float) -> float:
    return np.random.laplace(0.0, scale)


def dp_noised(value: float, epsilon: float, sensitivity: float = 1.0) -> float:
    """Add Laplace noise to a scalar value."""
    scale = sensitivity / epsilon
    return value + laplace_noise(scale)


# ----------------------------------------------------------------------
# Sparse winner‑take‑all (WTA) selector
# ----------------------------------------------------------------------
def wta_top_actions(actions: List[MathAction],
                    k: int = 3,
                    risk_weight: float = 0.5) -> List[MathAction]:
    """
    Return the top‑k actions after applying a sparse winner‑take‑all rule.
    Score = expected_value - risk_weight * risk.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    scored = [(a, a.expected_value - risk_weight * a.risk) for a in actions]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [pair[0] for pair in scored[:k]]


# ----------------------------------------------------------------------
# Hybrid decision logic
# ----------------------------------------------------------------------
class HybridSelector:
    """
    Encapsulates the fused decision process:
    1. MinHash similarity (privacy‑noised) between request tokens and tier reference.
    2. Regret‑weighted expected value from top‑k actions.
    3. Sparse WTA to focus on the most promising actions.
    4. Model‑tier selection respecting memory and exclusivity constraints.
    """

    def __init__(self,
                 tiers: List[ModelTier],
                 epsilon: float = 1.0,
                 alpha: float = 0.6,
                 wta_k: int = 3):
        """
        :param tiers: List of candidate ModelTier objects (must contain reference_tokens).
        :param epsilon: Differential privacy budget for similarity.
        :param alpha: Weight for similarity vs. regret in final score (0..1).
        :param wta_k: Number of actions kept after winner‑take‑all.
        """
        self.tiers = tiers
        self.epsilon = epsilon
        self.alpha = alpha
        self.wta_k = wta_k

    def _tier_similarity(self, tokens: Iterable[str]) -> Dict[str, float]:
        """Compute DP‑noised Jaccard similarity for each tier."""
        token_sig = minhash_signature(tokens)
        sims = {}
        for tier in self.tiers:
            ref_sig = minhash_signature(tier.reference_tokens)
            raw_sim = minhash_jaccard(token_sig, ref_sig)
            sims[tier.name] = dp_noised(raw_sim, self.epsilon, sensitivity=1.0)
        return sims

    def _regret_score(self, actions: List[MathAction]) -> float:
        """Mean expected value of the top‑k actions after WTA."""
        top = wta_top_actions(actions, k=self.wta_k)
        if not top:
            return 0.0
        return np.mean([a.expected_value for a in top])

    def select_tier(self,
                    actions: List[MathAction],
                    tokens: Iterable[str]) -> ModelTier:
        """Return the best ModelTier according to the fused objective."""
        sims = self._tier_similarity(tokens)
        regret = self._regret_score(actions)

        # Normalise similarity to [0,1] (they already are, but clipping guards noise)
        sims_norm = {k: max(0.0, min(1.0, v)) for k, v in sims.items()}

        # Compute fused score: α·similarity + (1‑α)·regret
        scores = {
            name: self.alpha * sims_norm[name] + (1 - self.alpha) * regret
            for name in sims_norm
        }

        # Choose tier with highest fused score
        best_name = max(scores, key=scores.get)
        return next(t for t in self.tiers if t.name == best_name)

    def eviction_score(self,
                       model: ModelTier,
                       actions: List[MathAction],
                       tokens: Iterable[str]) -> float:
        """
        Higher score → more valuable → less likely to be evicted.
        Combines privacy risk (inverse similarity) with memory pressure.
        """
        sim = self._tier_similarity(tokens).get(model.name, 0.0)
        privacy_risk = 1.0 - max(0.0, min(1.0, sim))  # low similarity ⇒ high risk
        memory_factor = model.ram_mb / 6000.0  # normalised RAM usage
        regret = self._regret_score(actions)
        # Weighted sum (tunable coefficients)
        return 0.4 * (1 - privacy_risk) + 0.4 * (1 - memory_factor) + 0.2 * regret


# ----------------------------------------------------------------------
# Example usage (can be removed or adapted)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define tier reference token sets (these would be derived from real data)
    tier_defs = [
        ModelTier(
            name="T1",
            ram_mb=1024,
            tier="T1",
            reference_tokens=("alpha", "beta", "gamma", "delta")
        ),
        ModelTier(
            name="T2",
            ram_mb=2048,
            tier="T2",
            reference_tokens=("epsilon", "zeta", "eta", "theta")
        ),
        ModelTier(
            name="T3",
            ram_mb=4096,
            tier="T3",
            reference_tokens=("iota", "kappa", "lambda", "mu")
        ),
    ]

    selector = HybridSelector(tiers=tier_defs, epsilon=0.8, alpha=0.7, wta_k=2)

    # Sample actions
    actions = [
        MathAction("a1", expected_value=0.9, risk=0.1),
        MathAction("a2", expected_value=0.4, risk=0.3),
        MathAction("a3", expected_value=0.6, risk=0.2),
        MathAction("a4", expected_value=0.2, risk=0.5),
    ]

    # Tokens from an incoming request
    request_tokens = ["alpha", "theta", "lambda", "unknown"]

    # Choose tier
    chosen_tier = selector.select_tier(actions, request_tokens)

    # Initialise pool and load with eviction based on the selector's eviction score
    pool = ModelPool(ram_ceiling_mb=6000)
    pool.load_with_eviction(
        chosen_tier,
        score_fn=lambda m: selector.eviction_score(m, actions, request_tokens)
    )

    # Demonstrate eviction of a less valuable model (if any)
    # Load a second model to trigger eviction logic
    other_tier = tier_defs[1] if tier_defs[0].name == chosen_tier.name else tier_defs[0]
    pool.load_with_eviction(
        other_tier,
        score_fn=lambda m: selector.eviction_score(m, actions, request_tokens)
    )

    print("Loaded models:")
    for m in pool.loaded.values():
        print(f" - {m.name} (RAM {m.ram_mb} MB)")

    # Show which model would be evicted next according to the score
    eviction_candidate = min(pool.loaded.values(),
                             key=lambda m: selector.eviction_score(m, actions, request_tokens))
    print("\nModel with lowest value (candidate for eviction):")
    print(eviction_candidate)