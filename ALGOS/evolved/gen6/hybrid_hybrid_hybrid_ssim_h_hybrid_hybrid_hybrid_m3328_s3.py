# DARWIN HAMMER — match 3328, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_ssim_hybrid_h_hybrid_bandit_router_m1109_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_gliner_zero_s_m697_s0.py (gen4)
# born: 2026-05-29T23:49:15Z

"""Hybrid Module combining:

- Parent A (hybrid_hybrid_ssim_hybrid_h_hybrid_bandit_router_m1109_s0.py): Multivector algebra and
  temperature‑dependent activity curve.
- Parent B (hybrid_hybrid_hybrid_hybrid_gliner_zero_s_m697_s0.py): ModelPool with RAM ceiling,
  model tiering, and entropy‑based reconstruction risk for zero‑shot label matching.

Mathematical Bridge:
Each candidate model is encoded as a **Multivector** where

* grade‑0 (scalar) = temperature‑modulated activity score.
* grade‑1 blades   = (ram_mb, tier_index) encoding resource usage.
* grade‑2 blades   = reconstruction risk derived from the entropy of input spans.

Model selection proceeds by forming these multivectors, multiplying them with a
global “decision hygiene” multivector, and finally extracting the scalar part,
which yields a temperature‑aware, risk‑aware priority used by the ModelPool for
loading/eviction decisions.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict, frozen
from typing import Dict, List, Tuple

import numpy as np

# ---------- Parent A components (Multivector) ----------
class Multivector:
    def __init__(self, components: dict, n: int = 0):
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get((), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result)

    def __mul__(self, other: "Multivector") -> "Multivector":
        result = {}
        for blade, coef in self.components.items():
            for blade2, coef2 in other.components.items():
                # Simple outer product: combine blade indices without sign handling
                new_blade = tuple(sorted(set(blade + blade2)))
                result[new_blade] = result.get(new_blade, 0.0) + coef * coef2
        return Multivector(result)


# ---------- Parent B components (ModelPool, ModelTier, Span) ----------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str


TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")


class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            evicted_name = next(iter(self.loaded))
            del self.loaded[evicted_name]
        self.load(model)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


# ---------- Fusion Functions ----------
def temperature_activity(temp_celsius: float) -> float:
    """Non‑linear temperature‑dependent activity curve.
    Uses a Gaussian centered at 37 °C with std‑dev 10 °C.
    Returns a factor in (0, 1].
    """
    sigma = 10.0
    mu = 37.0
    exponent = -((temp_celsius - mu) ** 2) / (2 * sigma ** 2)
    return math.exp(exponent)


def reconstruction_risk(spans: List[Span]) -> float:
    """Entropy‑based reconstruction risk.
    Calculates Shannon entropy of label distribution, normalised to [0,1].
    """
    if not spans:
        return 0.0
    label_counts = {}
    for sp in spans:
        label_counts[sp.label] = label_counts.get(sp.label, 0) + 1
    total = len(spans)
    entropy = -sum((cnt / total) * math.log(cnt / total + 1e-12) for cnt in label_counts.values())
    max_entropy = math.log(len(label_counts)) if len(label_counts) > 1 else 1.0
    return entropy / max_entropy


def model_to_multivector(model: ModelTier, temp_c: float, risk: float) -> Multivector:
    """Encode a model as a multivector.
    - Grade‑0 (scalar): temperature activity * base score (1.0)
    - Grade‑1 blade (1): RAM usage (scaled)
    - Grade‑1 blade (2): tier index (T1=1, T2=2, T3=3)
    - Grade‑2 blade (1,2): reconstruction risk
    """
    # Map tier string to index
    tier_index = {"T1": 1, "T2": 2, "T3": 3}.get(model.tier, 0)

    # Scale RAM to a modest magnitude (e.g., divide by 1000)
    ram_scaled = model.ram_mb / 1000.0

    components = {
        (): temperature_activity(temp_c),                      # scalar
        (1,): ram_scaled,                                      # grade‑1 blade e1
        (2,): float(tier_index),                               # grade‑1 blade e2
        (1, 2): risk,                                          # grade‑2 blade e1e2
    }
    return Multivector(components, n=2)


def hybrid_select_and_load(pool: ModelPool, candidates: List[ModelTier],
                           temp_c: float, spans: List[Span]) -> ModelTier:
    """Select the best candidate using multivector algebra and load it (with eviction)."""
    risk = reconstruction_risk(spans)

    # Decision hygiene multivector (acts as a weighting factor)
    hygiene = Multivector({(): 1.0, (1,): 0.9, (2,): 0.9, (1, 2): 0.8}, n=2)

    best_score = -math.inf
    best_model = None

    for model in candidates:
        mv = model_to_multivector(model, temp_c, risk)
        combined = mv * hygiene
        score = combined.scalar_part()  # scalar part aggregates all grades
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError("No viable model found.")

    # Load using eviction policy
    pool.load_with_eviction(best_model)
    return best_model


def report_pool_state(pool: ModelPool) -> None:
    """Utility to print current pool status."""
    used = pool._used()
    print(f"RAM used: {used}/{pool.ram_ceiling_mb} MB")
    for name, model in pool.loaded.items():
        print(f" - {name}: {model.ram_mb} MB, tier {model.tier}")


# ---------- Smoke Test ----------
if __name__ == "__main__":
    # Initialise pool with a modest ceiling
    pool = ModelPool(ram_ceiling_mb=8000)

    # Define candidate models (including a T3 that may conflict with T2)
    candidates = [
        TIER_T1_QWEN_0_5B,
        TIER_T2_REASONING,
        TIER_T2_TOOL,
        TIER_T3_QWEN_7B,
    ]

    # Simulated input spans
    spans = [
        Span(0, 5, "Hello", "greeting", 0.9),
        Span(6, 10, "world", "object", 0.8),
        Span(11, 15, "!", "punctuation", 0.7),
        Span(16, 20, "Test", "action", 0.6),
    ]

    # Random temperature between 20 and 45 °C
    temp_c = random.uniform(20.0, 45.0)
    print(f"Current temperature: {temp_c:.2f} °C")

    selected = hybrid_select_and_load(pool, candidates, temp_c, spans)
    print(f"Selected model: {selected.name} (tier {selected.tier}, {selected.ram_mb} MB)")

    report_pool_state(pool)