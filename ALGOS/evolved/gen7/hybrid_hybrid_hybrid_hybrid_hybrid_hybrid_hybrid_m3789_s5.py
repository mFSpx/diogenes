# DARWIN HAMMER — match 3789, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_physar_m1288_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m579_s1.py (gen5)
# born: 2026-05-29T23:51:35Z

"""
Hybrid Algorithm merging:
- Parent A: TTT‑Linear model (weight matrix updates, loss) 
- Parent B: Pheromone dynamics, risk scoring, Ollivier‑Ricci curvature, VRAM planning

Mathematical bridge:
The TTT‑Linear transformation `y = W·x` produces a deviation `Δ = y‑x`.  
In the hybrid, this deviation is scaled by the *risk score* `r` (from the
Krampus‑Stick side) and by the *pheromone signal value* `p` (from the
Honeybee‑Store side).  The product `Δ·(1‑r)·p` drives both:
1. An update of the pheromone entry (decay‑adjusted reinforcement).
2. A modulation of the conductance‑like update that would appear in a
   Physarum‑style network (here represented by a simple weight‑matrix
   adaptation).

The final scheduling/VRAM allocation score combines model health,
the risk‑adjusted pheromone strength, and the curvature `κ`:

    score = health * (1‑r) * p * κ

This code implements the fused mathematics with three core functions
demonstrating the hybrid operation.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Iterable, List, Dict

# ----------------------------------------------------------------------
# Shared primitives (unified across both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int = 0  # optional, default 0 for legacy usage

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
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)

# ----------------------------------------------------------------------
# Parent‑A‑like utilities (TTT‑Linear)
# ----------------------------------------------------------------------
def init_hybrid_weights(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize a weight matrix W ~ N(0,scale²) similar to parent A."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_linear_transform(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply the TTT linear model: y = W·x."""
    return W @ x

def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    """Mean‑squared deviation loss; if target omitted, target = x."""
    if target is None:
        target = x
    return float(np.sum((W @ x - target) ** 2))

# ----------------------------------------------------------------------
# Parent‑B‑like primitives (Pheromone, curvature, risk)
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = f"{random.getrandbits(128):032x}"
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        self.created_at = datetime.now(timezone.utc)
        self.last_decay = self.created_at

    def decay(self, now: datetime = None) -> None:
        """Exponential decay based on half‑life."""
        now = now or datetime.now(timezone.utc)
        elapsed = (now - self.last_decay).total_seconds()
        decay_factor = 0.5 ** (elapsed / self.half_life_seconds)
        self.signal_value *= decay_factor
        self.last_decay = now

def risk_score(model: ModelTier) -> float:
    """Placeholder risk estimator returning a value in [0,1]."""
    # Simple heuristic: larger RAM models are deemed riskier.
    max_ram = 10000.0
    return min(1.0, model.ram_mb / max_ram)

def curvature_stub() -> float:
    """Placeholder Ollivier‑Ricci curvature (positive for well‑connected graphs)."""
    return 0.8  # static value for demonstration

# ----------------------------------------------------------------------
# Hybrid core functions (mathematical fusion)
# ----------------------------------------------------------------------
def hybrid_pheromone_update(W: np.ndarray,
                            x: np.ndarray,
                            pher: PheromoneEntry,
                            model: ModelTier,
                            curvature: float) -> np.ndarray:
    """
    Perform a hybrid update:
    1. Compute TTT deviation Δ = (W·x) - x.
    2. Scale Δ by (1‑risk) and current pheromone signal.
    3. Use the scaled Δ to:
       a) Reinforce the pheromone entry (additive boost).
       b) Adapt the weight matrix via a simple gradient‑like step.
    The function returns the updated weight matrix.
    """
    # 1. TTT deviation
    y = ttt_linear_transform(W, x)
    delta = y - x

    # 2. Risk and pheromone scaling
    r = risk_score(model)                     # from parent B
    scale = (1.0 - r) * pher.signal_value * curvature

    # 3a. Pheromone reinforcement (linear boost)
    pher.signal_value += float(np.mean(np.abs(delta))) * scale

    # 3b. Weight adaptation (gradient descent style)
    lr = 0.01 * scale  # learning‑rate modulated by the same scale
    W_new = W - lr * np.outer(delta, x)  # simple outer‑product update
    return W_new

def compute_hybrid_score(model: ModelTier,
                         pher: PheromoneEntry,
                         curvature: float) -> float:
    """
    Compute the unified scheduling/VRAM allocation score:
        score = health * (1‑risk) * pheromone_value * curvature
    Health is defined as the fraction of RAM utilisation relative to a
    nominal ceiling (here 10 GB).
    """
    health = 1.0 - (model.ram_mb / 10000.0)          # higher RAM → lower health
    r = risk_score(model)
    score = health * (1.0 - r) * pher.signal_value * curvature
    # Clamp to [0,1] for safety
    return max(0.0, min(1.0, score))

def allocate_vram_plan(model: ModelTier,
                       score: float,
                       reason: str = "hybrid_allocation") -> Dict[str, Any]:
    """
    Produce a VRAM slot plan proportional to the hybrid score.
    The plan mimics the VramSlotPlan dataclass from parent B but is
    returned as a plain dict to stay within the allowed imports.
    """
    allocated_mb = int(model.vram_mb * score)
    plan = {
        "artifact_id": f"plan-{model.name}-{random.getrandbits(32)}",
        "artifact_kind": "vram_allocation",
        "action": "reserve",
        "estimated_mb": allocated_mb,
        "reason": reason,
        "detail": {
            "model_name": model.name,
            "tier": model.tier,
            "score": score,
            "allocated_mb": allocated_mb,
        },
    }
    return plan

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a tiny model tier
    model = ModelTier(name="demo-model", ram_mb=2000, tier="T2", vram_mb=4096)

    # Initialise weight matrix and input vector
    dim = 8
    W = init_hybrid_weights(d_in=dim, seed=42)
    x = np.random.rand(dim)

    # Create a pheromone entry
    pher = PheromoneEntry(
        surface_key="demo_surface",
        signal_kind="info",
        signal_value=0.5,
        half_life_seconds=60,
    )

    # Compute curvature (stub)
    kappa = curvature_stub()

    # Hybrid update
    W_updated = hybrid_pheromone_update(W, x, pher, model, kappa)

    # Verify loss reduction (optional)
    loss_before = ttt_loss(W, x)
    loss_after = ttt_loss(W_updated, x)
    print(f"Loss before: {loss_before:.6f}, after: {loss_after:.6f}")

    # Compute scheduling score and VRAM plan
    score = compute_hybrid_score(model, pher, kappa)
    plan = allocate_vram_plan(model, score)

    print(f"Hybrid score: {score:.4f}")
    print("VRAM allocation plan:", plan)

    # Simple sanity checks
    assert 0.0 <= score <= 1.0
    assert isinstance(plan, dict) and plan["estimated_mb"] >= 0
    print("Smoke test completed successfully.")