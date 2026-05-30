# DARWIN HAMMER — match 3381, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_infotaxis_min_m1759_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_tempor_hybrid_hybrid_doomsd_m676_s2.py (gen3)
# born: 2026-05-29T23:49:40Z

"""Hybrid Bandit‑Geometric‑Infotaxis‑Temporal Algorithm
Parent A: hybrid_hybrid_hybrid_bandit_hybrid_infotaxis_min_m1759_s0.py
Parent B: hybrid_hybrid_hybrid_tempor_hybrid_hybrid_doomsd_m676_s2.py

Mathematical Bridge
-------------------
Both parents operate on probability‑based decision making.  The bandit side
maintains a policy (propensity, expected reward, confidence bound) while the
infotaxis side evaluates information gain via entropy of a probability
distribution derived from MinHash similarity.  The temporal‑spatial side
provides a deterministic hyper‑dimensional (HD) embedding of a context
(e.g. geographic location) using the `symbol_vector` primitive.  By binding
the HD context vector with the HD action vector we obtain a composite
representation whose normalized dot product yields a similarity score.
That similarity is turned into a probability distribution over actions;
its Shannon entropy quantifies the expected information gain of choosing an
action.  The hybrid selector therefore combines the classic Upper‑Confidence
Bound (UCB) term from the bandit with an entropy‑based exploration bonus,
yielding a single scoring function:

    score(a|c) = μ_a + β·σ_a + λ·H(p_a|c)

where μ_a is the estimated reward, σ_a the confidence bound, H the
entropy of the similarity‑derived distribution, and β, λ are tunable
weights.  The update step also incorporates a spatial weighting (via
Haversine distance) and a Gini‑coefficient regularizer on the reward
distribution.

The module below implements this fused system with three public functions:
`bind_context_action`, `select_action`, and `update_policy_hybrid`.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, FrozenSet, Any
import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Parent A – Bandit / Store components
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float = 0.0
    expected_reward: float = 0.0
    confidence_bound: float = 0.0
    algorithm: str = "hybrid"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

# ----------------------------------------------------------------------
# Parent B – Infotaxis / MinHash components
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    return [_hash(i, min(toks, key=lambda t: _hash(i, t))) for i in range(k)]

def jaccard_estimate(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

def shannon_entropy(probs: Iterable[float]) -> float:
    """Standard Shannon entropy (base e)."""
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    return -sum(p * math.log(p) for p in probs)

# ----------------------------------------------------------------------
# Parent B – Hyperdimensional primitives
# ----------------------------------------------------------------------
Vector = List[int]

def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed) if seed is not None else random.Random()
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def cosine_similarity(a: Vector, b: Vector) -> float:
    """Normalized dot product for bipolar vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    norm = len(a)  # because each component is ±1, |a| = sqrt(dim) and same for b; product = dim
    return dot / norm

# ----------------------------------------------------------------------
# Spatial‑Temporal utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

def haversine_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    aa = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(aa), math.sqrt(1 - aa))
    return 6371.0 * c  # Earth radius in km

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    n = len(xs)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * xs) / (n * np.sum(xs)))

# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def bind_context_action(entity: Entity, action: BanditAction, dim: int = 10000) -> Vector:
    """
    Produce a hyper‑dimensional binding of the spatial context (entity) and the
    discrete action.  The entity is encoded by hashing its (lat,lon) pair and
    its optional address signature; the action is encoded by its identifier.
    """
    # Encode location as a deterministic string
    loc_token = f"{entity.lat:.6f},{entity.lon:.6f}"
    # Mix address signature if present
    if entity.address_signature:
        loc_token += f":{entity.address_signature}"
    ctx_vec = symbol_vector(loc_token, dim)
    act_vec = symbol_vector(action.action_id, dim)
    return bind(ctx_vec, act_vec)

def similarity_distribution(
    entity: Entity,
    actions: List[BanditAction],
    dim: int = 10000,
) -> Tuple[List[float], List[float]]:
    """
    Compute a similarity score for each action, turn it into a probability
    distribution (softmax), and also return the raw cosine similarities.
    """
    raw_sims = []
    for a in actions:
        bound_vec = bind_context_action(entity, a, dim)
        # Compare bound vector to the pure action vector (unbound) to gauge alignment
        act_vec = symbol_vector(a.action_id, dim)
        raw_sims.append(cosine_similarity(bound_vec, act_vec))
    # Shift to positive domain
    min_val = min(raw_sims)
    shifted = [s - min_val + 1e-6 for s in raw_sims]  # avoid zeros
    total = sum(shifted)
    probs = [s / total for s in shifted]
    return probs, raw_sims

def select_action(
    entity: Entity,
    actions: List[BanditAction],
    beta: float = 1.0,
    lam: float = 0.5,
    dim: int = 10000,
) -> BanditAction:
    """
    Hybrid selector that mixes Upper‑Confidence Bound (UCB) with an entropy
    exploration term derived from the similarity distribution.
    """
    # 1. UCB terms from bandit statistics
    ucb_scores = {}
    for a in actions:
        mu = _reward(a.action_id)
        n = _count(a.action_id)
        # Classic confidence bound (sqrt(log(T)/n)), T approximated by total pulls
        T = sum(_count(act.action_id) for act in actions) + 1
        sigma = math.sqrt(math.log(T) / (n + 1e-6))
        ucb_scores[a.action_id] = mu + beta * sigma

    # 2. Entropy term from similarity distribution
    probs, _ = similarity_distribution(entity, actions, dim)
    entropy = shannon_entropy(probs)
    # Distribute the entropy proportionally to each action (simple uniform split)
    entropy_bonus = lam * entropy / len(actions)

    # 3. Combine
    combined = {}
    for a in actions:
        combined[a.action_id] = ucb_scores[a.action_id] + entropy_bonus

    # 4. Choose the action with maximal combined score
    best_id = max(combined, key=combined.get)
    # Return a fresh BanditAction enriched with the computed propensity
    best_action = next(a for a in actions if a.action_id == best_id)
    return BanditAction(
        action_id=best_action.action_id,
        propensity=combined[best_id],
        expected_reward=_reward(best_id),
        confidence_bound=beta * math.sqrt(math.log(sum(_count(a.action_id) for a in actions) + 1) / (_count(best_id) + 1e-6)),
        algorithm="hybrid",
    )

def update_policy_hybrid(
    entity: Entity,
    chosen_action: BanditAction,
    reward: float,
    beta: float = 1.0,
    lam: float = 0.5,
    dim: int = 10000,
) -> None:
    """
    Perform a policy update after observing a reward.  The update records the
    raw reward (bandit side) and also adjusts the confidence bound using a
    spatial weighting factor derived from the Haversine distance to a
    reference point (here the origin (0,0) for illustration).  Additionally,
    a Gini regularizer penalises highly skewed reward distributions.
    """
    # Record bandit statistics
    update = BanditUpdate(
        context_id=entity.id,
        action_id=chosen_action.action_id,
        reward=reward,
        propensity=chosen_action.propensity,
    )
    update_policy([update])

    # Spatial weighting – actions taken far from origin receive a slight decay
    origin = (0.0, 0.0)
    dist = haversine_distance((entity.lat, entity.lon), origin)  # km
    decay = math.exp(-dist / 10000.0)  # decay over ~10,000 km scale

    # Adjust confidence bound in the stored policy (conceptual; we recompute on demand)
    # Here we simply store a scaled reward to illustrate the effect
    scaled_reward = reward * decay
    # Overwrite the latest entry with the scaled reward
    stats = _POLICY.setdefault(chosen_action.action_id, [0.0, 0.0])
    stats[0] = stats[0] - reward + scaled_reward  # replace last reward contribution
    # Gini regularization: if reward distribution becomes too unequal, dampen future propensities
    gini = gini_coefficient([s[0] for s in _POLICY.values() if s[1] > 0])
    if gini > 0.6:  # arbitrary threshold
        # Apply a global penalty to all propensities (no side‑effects on stored stats)
        for act_id in _POLICY:
            _POLICY[act_id][0] *= 0.9  # dampen total reward

# ----------------------------------------------------------------------
# Demonstration functions (required >=3)
# ----------------------------------------------------------------------
def demo_binding() -> None:
    ent = Entity(id="E1", lat=37.7749, lon=-122.4194, category="city")
    act = BanditAction(action_id="click")
    vec = bind_context_action(ent, act, dim=256)
    print(f"Binding vector length: {len(vec)}; first 5 components: {vec[:5]}")

def demo_selection() -> None:
    ent = Entity(id="E2", lat=48.8566, lon=2.3522, category="city")
    actions = [
        BanditAction(action_id="view"),
        BanditAction(action_id="click"),
        BanditAction(action_id="purchase"),
    ]
    chosen = select_action(ent, actions, beta=1.2, lam=0.8, dim=512)
    print(f"Selected action: {chosen.action_id} with propensity {chosen.propensity:.4f}")

def demo_update() -> None:
    ent = Entity(id="E3", lat=51.5074, lon=-0.1278, category="city")
    actions = [
        BanditAction(action_id="view"),
        BanditAction(action_id="click"),
        BanditAction(action_id="purchase"),
    ]
    chosen = select_action(ent, actions, beta=1.0, lam=0.5, dim=256)
    reward = random.random()  # synthetic reward
    update_policy_hybrid(ent, chosen, reward, beta=1.0, lam=0.5, dim=256)
    print(f"Updated policy for action '{chosen.action_id}' with reward {reward:.3f}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    reset_policy()
    demo_binding()
    demo_selection()
    demo_update()
    # Show final policy snapshot
    print("Final policy state:", _POLICY)