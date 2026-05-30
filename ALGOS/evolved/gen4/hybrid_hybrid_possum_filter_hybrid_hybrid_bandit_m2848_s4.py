# DARWIN HAMMER — match 2848, survivor 4
# gen: 4
# parent_a: hybrid_possum_filter_hybrid_hybrid_ternar_m1381_s1.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s3.py (gen2)
# born: 2026-05-29T23:46:23Z

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """A spatially‑located item with a categorical label."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "linucb"


@dataclass(frozen=True)
class BanditUpdate:
    """Feedback used to improve the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat, lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


# ----------------------------------------------------------------------
# VRAM‑style cache for categorical embeddings
# ----------------------------------------------------------------------
_CATEGORY_CACHE: Dict[str, np.ndarray] = {}
_EMBED_DIM: int = 64  # dimensionality of the synthetic embedding space


def _embed_category(cat: str) -> np.ndarray:
    """Deterministic pseudo‑embedding for a category string."""
    if cat not in _CATEGORY_CACHE:
        rng = np.random.default_rng(abs(hash(cat)) % (2 ** 32))
        _CATEGORY_CACHE[cat] = rng.normal(size=_EMBED_DIM).astype(np.float32)
    return _CATEGORY_CACHE[cat]


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity in [‑1, 1]."""
    if x.size == 0 or y.size == 0:
        return 0.0
    num = float(np.dot(x, y))
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return num / den if den != 0 else 0.0


# ----------------------------------------------------------------------
# Reward computation – the true “bridge” between the two subsystems
# ----------------------------------------------------------------------
def _spatial_reward(
    entity: Entity, ref_point: Tuple[float, float], delta_m: float
) -> float:
    """Inverse‑distance reward, clipped by delta_m."""
    dist = haversine_m((entity.lat, entity.lon), ref_point)
    if dist >= delta_m or dist == 0:
        return 0.0
    # Linear decay: closer → larger reward, scaled to [0,1]
    return 1.0 - (dist / delta_m)


def _category_reward(action: str, entity: Entity) -> float:
    """Similarity between the action’s implied category and the entity’s category."""
    act_vec = _embed_category(action)
    ent_vec = _embed_category(entity.category)
    # Rescale cosine similarity from [‑1,1] to [0,1]
    return (cosine_similarity(act_vec, ent_vec) + 1.0) / 2.0


def combined_reward(
    action: str,
    entities: Iterable[Entity],
    ref_point: Tuple[float, float],
    delta_m: float = 75.0,
    spatial_weight: float = 0.6,
    category_weight: float = 0.4,
) -> float:
    """
    Weighted average of spatial and categorical similarity over all
    entities that lie within ``delta_m`` of ``ref_point``.
    """
    rewards: List[float] = []
    for e in entities:
        sp = _spatial_reward(e, ref_point, delta_m)
        if sp == 0.0:
            continue
        cat = _category_reward(action, e)
        rewards.append(spatial_weight * sp + category_weight * cat)

    return float(np.mean(rewards)) if rewards else 0.0


# ----------------------------------------------------------------------
# LinUCB policy implementation (deep integration)
# ----------------------------------------------------------------------
class LinUCBPolicy:
    """
    Per‑action LinUCB parameters.
    A (d×d) matrix and b (d) vector are stored for each action.
    The context dimension ``d`` is inferred from the first call.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._A: Dict[str, np.ndarray] = {}
        self._b: Dict[str, np.ndarray] = {}
        self._dim: int | None = None

    def _ensure_action(self, action: str) -> None:
        if self._dim is None:
            raise RuntimeError("Policy not initialised – call `initialize` first.")
        if action not in self._A:
            self._A[action] = np.identity(self._dim, dtype=np.float64)
            self._b[action] = np.zeros(self._dim, dtype=np.float64)

    def initialize(self, dim: int) -> None:
        """Set the context dimensionality before the first decision."""
        if self._dim is None:
            self._dim = dim

    def predict(self, context_vec: np.ndarray, actions: List[str]) -> Tuple[str, float]:
        """Return the action with the highest LinUCB score and its score."""
        if self._dim is None:
            self.initialize(len(context_vec))

        best_action = None
        best_score = -math.inf
        for a in actions:
            self._ensure_action(a)
            A_inv = np.linalg.inv(self._A[a])
            theta = A_inv @ self._b[a]
            p = float(theta @ context_vec + self.alpha * math.sqrt(context_vec @ A_inv @ context_vec))
            if p > best_score:
                best_score, best_action = p, a
        return best_action, best_score

    def update(self, action: str, context_vec: np.ndarray, reward: float) -> None:
        """Standard LinUCB posterior update."""
        self._ensure_action(action)
        self._A[action] += np.outer(context_vec, context_vec)
        self._b[action] += reward * context_vec


# ----------------------------------------------------------------------
# Public API – hybrid selector
# ----------------------------------------------------------------------
_POLICY = LinUCBPolicy(alpha=1.0)


def _build_context_vector(
    raw_context: Dict[str, float],
    user_location: Tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Concatenate numeric context, optional (lat,lon) and a constant bias term.
    All values are cast to ``float64`` for numerical stability.
    """
    parts: List[float] = list(raw_context.values())
    if user_location is not None:
        parts.extend(user_location)
    parts.append(1.0)  # bias
    return np.array(parts, dtype=np.float64)


def select_action_hybrid(
    context: Dict[str, float],
    entities: Iterable[Entity],
    actions: List[str],
    user_location: Tuple[float, float] | None = None,
    delta_m: float = 75.0,
    alpha: float = 1.0,
) -> BanditAction:
    """
    Choose an action using LinUCB where the expected reward is the
    ``combined_reward`` defined above.
    """
    if not actions:
        raise ValueError("At least one action must be supplied.")

    # Build a deterministic context vector (includes location if provided)
    ctx_vec = _build_context_vector(context, user_location)

    # Initialise policy dimensionality on first call
    if _POLICY._dim is None:
        _POLICY.initialize(len(ctx_vec))

    # Compute LinUCB scores
    chosen, score = _POLICY.predict(ctx_vec, actions)

    # Estimate the reward for the chosen action (used for confidence bound)
    est_reward = combined_reward(
        chosen, entities, user_location or (0.0, 0.0), delta_m=delta_m
    )

    # Confidence bound derived from the LinUCB exploration term
    A_inv = np.linalg.inv(_POLICY._A[chosen])
    confidence = _POLICY.alpha * math.sqrt(ctx_vec @ A_inv @ ctx_vec)

    # Record the decision (propensity approximated as uniform for reporting)
    propensity = 1.0 / len(actions)

    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=est_reward,
        confidence_bound=confidence,
        algorithm="linucb",
    )


def update_policy(
    context: Dict[str, float],
    action: str,
    reward: float,
    user_location: Tuple[float, float] | None = None,
) -> None:
    """
    Apply a LinUCB update after observing the real reward.
    """
    ctx_vec = _build_context_vector(context, user_location)
    _POLICY.update(action, ctx_vec, reward)


def reset_policy() -> None:
    """Clear all learned statistics and the category cache."""
    _POLICY._A.clear()
    _POLICY._b.clear()
    _POLICY._dim = None
    _CATEGORY_CACHE.clear()


# ----------------------------------------------------------------------
# Convenience filter (unchanged semantics, but uses proper distance)
# ----------------------------------------------------------------------
def hybrid_filter_entities(
    entities: Iterable[Entity],
    ref_point: Tuple[float, float] | None = None,
    delta_m: float = 75.0,
    sort_by_score: bool = True,
) -> List[Entity]:
    """
    Return entities within ``delta_m`` of ``ref_point`` (defaults to (0,0))
    optionally sorted by descending score then id.
    """
    ref = ref_point or (0.0, 0.0)
    filtered = [
        e for e in entities if haversine_m((e.lat, e.lon), ref) < delta_m
    ]
    if sort_by_score:
        filtered.sort(key=lambda e: (-e.score, e.id))
    return filtered


# ----------------------------------------------------------------------
# Demo / sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_entities = [
        Entity("id1", 0.0, 0.0, "restaurant", 1.0),
        Entity("id2", 0.0005, 0.0005, "cafe", 2.0),
        Entity("id3", 0.01, 0.01, "museum", 3.0),
    ]
    demo_actions = ["restaurant", "cafe", "park"]
    demo_context = {"hour_of_day": 14.0, "user_age": 30.0}
    user_loc = (0.0, 0.0)

    # First decision
    ba = select_action_hybrid(
        context=demo_context,
        entities=demo_entities,
        actions=demo_actions,
        user_location=user_loc,
    )
    print("Chosen action:", ba)

    # Simulate observed reward (e.g., click = 1, no click = 0)
    observed = combined_reward(
        ba.action_id, demo_entities, user_loc, delta_m=75.0
    )
    update_policy(demo_context, ba.action_id, observed, user_loc)

    # Show filtered entities
    print("Nearby entities:", hybrid_filter_entities(demo_entities, user_loc))