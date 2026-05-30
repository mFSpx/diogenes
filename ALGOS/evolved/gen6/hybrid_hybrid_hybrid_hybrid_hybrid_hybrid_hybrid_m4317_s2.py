# DARWIN HAMMER — match 4317, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_model_pool_hy_m707_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m935_s2.py (gen5)
# born: 2026-05-29T23:54:56Z

"""Hybrid algorithm merging ModelPool resource management (Parent A) with epistemic‑weighted shape descriptors (Parent B).

Mathematical bridge:
- Each loaded ModelTier is associated with a CertaintyFlag whose confidence is derived from the tier’s RAM size
  (larger RAM ⇒ higher epistemic confidence).
- From a set of CertaintyFlag objects we build a diagonal weight matrix **W** whose entries are the
  normalized confidences (confidence_bps / 10 000).
- Given a shape‑descriptor vector **x** ∈ ℝ³ (e.g., sphericity, flatness, righting‑time) the fused decision
  metric is the quadratic form **s = xᵀ W x**.
- Finally the score is modulated by the fraction of RAM actually occupied in the ModelPool,
  yielding **S = s · (memory_used / ram_ceiling)**.

Thus the resource‑tier topology of Parent A supplies epistemic confidences that weight the morphological
metrics of Parent B in a single unified computation."""


import json
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np


# ----------------------------------------------------------------------
# Parent A – Model tier & pool
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str


# Example tiers (could be extended)
TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")


class ModelPool:
    """Manages a collection of loaded models respecting a RAM ceiling."""

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def used_memory(self) -> int:
        """Public accessor for the amount of RAM currently occupied."""
        return self._used()

    def load(self, model: ModelTier) -> None:
        """Load a model if constraints allow."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def unload(self, name: str) -> None:
        """Remove a model from the pool."""
        self.loaded.pop(name, None)


# ----------------------------------------------------------------------
# Parent B – Epistemic certainty & quadratic form
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable container for an epistemic certainty label."""
    label: str
    confidence_bps: int  # 0 … 10 000 basis points = 0 % … 100 %
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")
        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> Dict[str, Union[str, int, Tuple[str, ...]]]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }


def _normalize_confidences(flags: List[CertaintyFlag]) -> List[float]:
    """Return confidences in the range [0,1]."""
    return [f.confidence_bps / 10_000.0 for f in flags]


def build_weight_matrix(flags: List[CertaintyFlag], dim: int) -> np.ndarray:
    """
    Construct a diagonal weight matrix W ∈ ℝ^{dim×dim} from a list of CertaintyFlag objects.
    If fewer flags than ``dim`` are supplied, remaining diagonal entries are set to 1.0.
    If more, excess flags are ignored.
    """
    norm = _normalize_confidences(flags)
    diag = np.ones(dim, dtype=float)
    for i in range(min(dim, len(norm))):
        diag[i] = norm[i]
    return np.diag(diag)


def quadratic_form(x: np.ndarray, W: np.ndarray) -> float:
    """Compute s = xᵀ W x."""
    return float(x.T @ W @ x)


# ----------------------------------------------------------------------
# Hybrid layer – mapping ModelTier ↔ CertaintyFlag and final score
# ----------------------------------------------------------------------
def tier_to_certainty(model: ModelTier) -> CertaintyFlag:
    """
    Derive a CertaintyFlag from a ModelTier.
    Confidence is proportional to the RAM size (larger RAM → higher confidence).
    The mapping caps at 10 000 bps (100 %).
    """
    # Scale RAM (0–10 000) linearly with a 10 000 bps ceiling.
    max_ram = 10_000  # arbitrary normalization constant
    confidence = min(int((model.ram_mb / max_ram) * 10_000), 10_000)
    # Choose a label based on confidence bands.
    if confidence >= 8_000:
        label = "FACT"
    elif confidence >= 5_000:
        label = "PROBABLE"
    elif confidence >= 2_000:
        label = "POSSIBLE"
    else:
        label = "BULLSHIT"
    return CertaintyFlag(
        label=label,
        confidence_bps=confidence,
        authority_class=model.tier,
        rationale=f"Derived from RAM size {model.ram_mb} MB",
        evidence_refs=(model.name,),
    )


def generate_flags_from_pool(pool: ModelPool) -> List[CertaintyFlag]:
    """Create a CertaintyFlag for each loaded model in the pool."""
    return [tier_to_certainty(m) for m in pool.loaded.values()]


def fused_score(pool: ModelPool, descriptors: Iterable[float]) -> float:
    """
    Compute the hybrid metric:

        1. Build CertaintyFlag objects from the loaded models.
        2. Assemble diagonal weight matrix W from their normalized confidences.
        3. Evaluate the quadratic form s = xᵀ W x for the descriptor vector x.
        4. Scale by the fraction of RAM currently used (memory_used / ram_ceiling).

    Returns the final scalar score S.
    """
    flags = generate_flags_from_pool(pool)
    x = np.array(list(descriptors), dtype=float)
    dim = x.shape[0]

    if dim == 0:
        raise ValueError("Descriptor vector must contain at least one element")

    W = build_weight_matrix(flags, dim)
    s = quadratic_form(x, W)

    mem_frac = pool.used_memory() / float(pool.ram_ceiling_mb) if pool.ram_ceiling_mb else 0.0
    return s * mem_frac


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def demo_load_models(pool: ModelPool, models: List[ModelTier]) -> None:
    """Attempt to load a list of models, printing success or failure."""
    for m in models:
        try:
            pool.load(m)
            print(f"Loaded model {m.name} (RAM {m.ram_mb} MB, tier {m.tier})")
        except RuntimeError as e:
            print(f"Failed to load {m.name}: {e}")


def demo_compute_score(pool: ModelPool) -> None:
    """
    Create a synthetic descriptor vector and compute the hybrid score.
    Descriptors: [sphericity, flatness, righting_time] – values in [0,1].
    """
    descriptors = [random.random() for _ in range(3)]
    print(f"Descriptors: {descriptors}")
    score = fused_score(pool, descriptors)
    print(f"Hybrid fused score: {score:.6f}")


def demo_full_workflow() -> None:
    """Run a complete end‑to‑end demonstration."""
    pool = ModelPool(ram_ceiling_mb=8000)
    models = [TIER_T1_QWEN_0_5B, TIER_T2_REASONING, TIER_T2_TOOL, TIER_T3_QWEN_7B]
    demo_load_models(pool, models)
    demo_compute_score(pool)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_full_workflow()