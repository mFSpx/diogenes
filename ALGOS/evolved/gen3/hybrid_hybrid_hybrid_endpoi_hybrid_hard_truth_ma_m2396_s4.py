# DARWIN HAMMER — match 2396, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_krampus_brain_m24_s0.py (gen2)
# parent_b: hybrid_hard_truth_math_model_pool_m8_s0.py (gen1)
# born: 2026-05-29T23:42:22Z

import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core components (circuit‑breaker + morphology)
# ----------------------------------------------------------------------


class EndpointCircuitBreaker:
    """Failure counter with configurable threshold and health metric."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold < 0:
            raise ValueError("failure_threshold must be non‑negative")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        """Reset failures and close the breaker."""
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def record_failure(self) -> None:
        """Increment failure count and open the breaker if needed."""
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def health_factor(self) -> float:
        """
        Normalised health factor H ∈ [0,1].

        Uses a *smooth* decay (sigmoid) to avoid a hard zero when failures
        reach the threshold, which would otherwise nullify the whole fusion.
        """
        if self.failure_threshold == 0:
            return 0.0
        # Linear decay capped at 0, then softened with a sigmoid for smoothness
        linear = max(0.0, 1.0 - self.failures / self.failure_threshold)
        # Sigmoid scaling (steeper near 0)
        return 1.0 / (1.0 + math.exp(-12 * (linear - 0.5)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""

    length: float
    width: float
    height: float
    mass: float


def _positive_dimensions(*dims: float) -> None:
    if any(d <= 0 for d in dims):
        raise ValueError("All dimensions must be strictly positive")


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric mean divided by the longest side; ∈ (0,1]."""
    _positive_dimensions(length, width, height)
    geom_mean = (length * width * height) ** (1.0 / 3.0)
    longest = max(length, width, height)
    return geom_mean / longest


def flatness_index(length: float, width: float, height: float) -> float:
    """
    Smallest dimension divided by the average of the other two; ∈ (0,1].
    """
    _positive_dimensions(length, width, height)
    dims = sorted([length, width, height])
    smallest, mid, largest = dims
    return smallest / ((mid + largest) / 2.0)


def curvature_score(morph: Morphology) -> float:
    """
    Combined curvature C = sphericity × flatness, clipped to [0,1].
    """
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flt = flatness_index(morph.length, morph.width, morph.height)
    return max(0.0, min(1.0, sph * flt))


def global_weight(cb: EndpointCircuitBreaker, morph: Morphology) -> float:
    """
    Deeply fused weight W ∈ [0,1].

    Instead of a simple product H·C, we use the *harmonic mean* to penalise
    extreme values while still rewarding balanced health and curvature.
    """
    H = cb.health_factor()
    C = curvature_score(morph)
    if H + C == 0:
        return 0.0
    return 2.0 * H * C / (H + C)


# ----------------------------------------------------------------------
# Stylometry + Model pool components
# ----------------------------------------------------------------------


FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't wasn'tn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def extract_stylometry_features(text: str) -> np.ndarray:
    """
    Relative frequency vector for each FUNCTION_CATS category.
    Guarantees sum ≤ 1 (tokens not belonging to any category are ignored).
    """
    tokens = [
        t.lower().strip(".,!?;:()[]{}\"'")
        for t in text.split()
        if t.strip()
    ]
    total = len(tokens) or 1
    freqs = []
    for cat_words in FUNCTION_CATS.values():
        count = sum(1 for t in tokens if t in cat_words)
        freqs.append(count / total)
    return np.array(freqs, dtype=float)


def stylometry_norm(features: np.ndarray) -> float:
    """
    Normalised Euclidean norm ∈ [0,1].

    Uses the theoretical maximum sqrt(N) for a unit‑sum vector of length N.
    """
    norm = np.linalg.norm(features)
    max_norm = math.sqrt(len(features))
    return max(0.0, min(1.0, norm / max_norm))


class ModelTier:
    """Simple container for model metadata."""

    def __init__(self, name: str, ram_mb: int, tier: str):
        if ram_mb <= 0:
            raise ValueError("ram_mb must be positive")
        if tier not in {"T1", "T2", "T3"}:
            raise ValueError("tier must be one of T1, T2, T3")
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

    def __repr__(self) -> str:
        return f"ModelTier(name={self.name!r}, ram_mb={self.ram_mb}, tier={self.tier!r})"


class ModelPool:
    """
    Manages loading/unloading of models under a RAM ceiling.
    """

    def __init__(self, ram_ceiling_mb: int = 6000):
        if ram_ceiling_mb <= 0:
            raise ValueError("ram_ceiling_mb must be positive")
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _available(self) -> int:
        return self.ram_ceiling_mb - self._used()

    def _tier_rank(self, tier: str) -> int:
        """Higher numeric rank = lower priority for eviction."""
        return {"T1": 1, "T2": 2, "T3": 3}[tier]

    def _evict_until_fit(self, required_mb: int) -> bool:
        """
        Evict loaded models starting from the lowest priority (highest tier rank)
        until enough RAM is freed. Returns True if successful.
        """
        if required_mb > self.ram_ceiling_mb:
            return False  # impossible to fit

        # Sort by (tier rank desc, ram desc) – evict biggest, lowest‑tier first
        candidates = sorted(
            self.loaded.values(),
            key=lambda m: (self._tier_rank(m.tier), m.ram_mb),
            reverse=True,
        )
        freed = 0
        evicted: List[str] = []
        for model in candidates:
            if self._available() + freed >= required_mb:
                break
            freed += model.ram_mb
            evicted.append(model.name)

        for name in evicted:
            self.loaded.pop(name, None)
        return self._available() >= required_mb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def can_load(self, model: ModelTier) -> bool:
        return self._available() >= model.ram_mb

    def load(self, model: ModelTier) -> bool:
        """
        Load a model, evicting lower‑priority models if necessary.
        Returns True on success, False otherwise.
        """
        if self.is_loaded(model.name):
            return True

        if self.can_load(model):
            self.loaded[model.name] = model
            return True

        if self._evict_until_fit(model.ram_mb):
            self.loaded[model.name] = model
            return True

        return False

    def unload(self, name: str) -> bool:
        """Explicitly unload a model; returns True if it was present."""
        return self.loaded.pop(name, None) is not None

    def list_loaded(self) -> List[ModelTier]:
        """Snapshot of currently loaded models."""
        return list(self.loaded.values())


# ----------------------------------------------------------------------
# Fusion engine – deeper integration of all components
# ----------------------------------------------------------------------


class FusionEngine:
    """
    Orchestrates the hybrid decision process.

    The scoring function blends:
      • Global weight W (health × curvature)
      • Normalised stylometry norm S ∈ [0,1]
      • RAM utilisation penalty R ∈ [0,1] (higher penalty for larger RAM)

    The final score is a weighted sum:
        score = α·W + β·S + γ·(1‑R)

    where α+β+γ = 1 and can be tuned at construction time.
    """

    def __init__(
        self,
        cb: EndpointCircuitBreaker,
        morph: Morphology,
        pool: ModelPool,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
    ):
        if not math.isclose(alpha + beta + gamma, 1.0):
            raise ValueError("alpha, beta, gamma must sum to 1")
        self.cb = cb
        self.morph = morph
        self.pool = pool
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Scoring utilities
    # ------------------------------------------------------------------

    def _ram_penalty(self, model: ModelTier) -> float:
        """
        Normalised RAM usage penalty R ∈ [0,1].

        Uses a logarithmic scaling to avoid linear domination by very large
        models while still discouraging excessive consumption.
        """
        max_mb = self.pool.ram_ceiling_mb
        ratio = model.ram_mb / max_mb
        # Log‑scale: 0 → 0, 1 → 1
        return max(0.0, min(1.0, math.log1p(ratio * 9) / math.log1p(9)))

    def _hybrid_score(self, model: ModelTier, stylometry_vec: np.ndarray) -> float:
        W = global_weight(self.cb, self.morph)                # ∈ [0,1]
        S = stylometry_norm(stylometry_vec)                  # ∈ [0,1]
        R = self._ram_penalty(model)                         # ∈ [0,1]
        return self.alpha * W + self.beta * S + self.gamma * (1.0 - R)

    # ------------------------------------------------------------------
    # Public workflow
    # ------------------------------------------------------------------

    def select_and_load(
        self,
        text: str,
        candidates: Iterable[ModelTier],
    ) -> Tuple[ModelTier | None, float]:
        """
        Given raw text and a set of candidate models, compute scores,
        pick the highest‑scoring model that can be loaded (evicting if needed),
        and return the chosen model together with its score.

        If the circuit breaker is open, no model will be loaded and (None, 0.0)
        is returned.
        """
        if not self.cb.allow():
            return None, 0.0

        features = extract_stylometry_features(text)

        # Compute scores for all candidates
        scored: List[Tuple[ModelTier, float]] = [
            (m, self._hybrid_score(m, features)) for m in candidates
        ]

        # Sort descending by score
        scored.sort(key=lambda pair: pair[1], reverse=True)

        for model, score in scored:
            if self.pool.load(model):
                return model, score

        # If none could be loaded, return the best candidate with score 0
        return None, 0.0

    def evaluate_all(
        self,
        text: str,
        candidates: Iterable[ModelTier],
    ) -> List[Tuple[ModelTier, float]]:
        """
        Return a list of (model, score) for all candidates, sorted by descending
        score, without performing any loading.
        """
        features = extract_stylometry_features(text)
        results = [(m, self._hybrid_score(m, features)) for m in candidates]
        results.sort(key=lambda pair: pair[1], reverse=True)
        return results


# ----------------------------------------------------------------------
# Example usage (can be removed or adapted by downstream code)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Instantiate core components
    cb = EndpointCircuitBreaker(failure_threshold=5)
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.2)
    pool = ModelPool(ram_ceiling_mb=8000)

    # Define a few dummy models
    models = [
        ModelTier(name="tiny", ram_mb=500, tier="T1"),
        ModelTier(name="medium", ram_mb=2500, tier="T2"),
        ModelTier(name="large", ram_mb=6000, tier="T3"),
    ]

    # Create the fusion engine
    engine = FusionEngine(cb, morph, pool, alpha=0.35, beta=0.45, gamma=0.20)

    sample_text = (
        "I can't believe how quickly the system responded, but it failed "
        "when the load increased beyond expectations."
    )

    chosen, score = engine.select_and_load(sample_text, models)
    if chosen:
        print(f"Loaded model: {chosen.name} (score={score:.4f})")
        print("Currently loaded:", pool.list_loaded())
    else:
        print("No model could be loaded (circuit may be open or insufficient RAM).")