# DARWIN HAMMER — match 4648, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_percyp_hybrid_hybrid_hybrid_m576_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m2737_s0.py (gen5)
# born: 2026-05-29T23:57:12Z

"""Hybrid Algorithm: Morphic‑Stylometric‑TTT‑NLMS Fusion

Parents:
- hybrid_hybrid_hybrid_percyp_hybrid_hybrid_hybrid_m576_s1.py
  (Morphology indices S, F; stylometric score; store dynamics)
- hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m2737_s0.py
  (TTT‑Linear model, ternary router, NLMS‑style error correction,
   diffusion time‑constant forcing, endpoint circuit breaker)

Mathematical Bridge
------------------
The bridge is built on the observation that the morphology‑derived confidence
weights `S·F` and the stylometric score `Σ w_i·C_i` can be interpreted as adaptive
scaling factors for the TTT‑Linear reconstruction loss used inside the NLMS‑like
weight‑update.  Concretely, the hybrid weight update becomes  


ΔW = -η·[∇_W loss  +  γ·Score·S·F·∇_W loss]


i.e. the classic gradient step is multiplied by a factor  
`1 + γ·Score·S·F`.  

The ternary router, driven by the normalized stylometric score `s∈[0,1]`,
modulates the diffusion timestep  


t_i = round((1‑s)·T)·(1‑endpoint.allow())


which in turn scales the injected noise in the NLMS diffusion forcing.
The final hybrid store update combines the original store dynamics with the
morphology‑stylometry term and the TTT‑Linear contribution.

The module implements three core functions that demonstrate this fused
behaviour:
1. `compute_morphology_indices`
2. `stylometric_score`
3. `HybridModel.hybrid_step` – performs a full hybrid iteration
"""

import hashlib
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Parent‑A utilities: procedural slots, morphology, stylometry
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _uuid_from_sha256(seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _slot_name(seed: str, idx: int) -> Tuple[str, str, str]:
    h = hashlib.sha256(f"{seed}:{idx}".encode()).hexdigest()
    name = f"Villager-{int(h[:6], 16) % 5000:04d}"
    alias = f"Alias-{h[6:10]}"
    persona = random.choice(["ledger", "runner", "scholar", "artisan"])
    return name, alias, persona


def generate_procedural_slots(seed: str, count: int = 10) -> List[ProceduralSlot]:
    """Create a list of deterministic ProceduralSlot objects."""
    slots = []
    for i in range(count):
        name, alias, persona = _slot_name(seed, i)
        uuid = _uuid_from_sha256(f"{seed}:{i}")
        ternary_offset = random.choice([-1, 0, 1])
        slots.append(
            ProceduralSlot(
                slot_index=i,
                name=name,
                alias=alias,
                persona=persona,
                uuid=uuid,
                ternary_offset=ternary_offset,
            )
        )
    return slots


def compute_morphology_indices(slots: List[ProceduralSlot]) -> Tuple[float, float]:
    """
    Derive *sphericity* (S) and *flatness* (F) from the distribution of UUID hashes.
    The indices lie in (0, 1] and act as confidence weights.
    """
    if not slots:
        return 1.0, 1.0
    # Convert UUID hex to integers and compute variance‑based measures
    hashes = np.array([int(s.uuid.replace("-", ""), 16) for s in slots], dtype=np.float64)
    norm = (hashes - hashes.min()) / (hashes.max() - hashes.min() + 1e-12)
    S = 1.0 - np.std(norm)  # higher variance → lower sphericity
    F = 1.0 - np.mean(np.abs(np.diff(norm)))  # smoother sequence → higher flatness
    # Clamp to (0,1]
    S = max(min(S, 1.0), 1e-6)
    F = max(min(F, 1.0), 1e-6)
    return float(S), float(F)


def stylometric_score(categories: Dict[str, int], S: float, F: float) -> float:
    """
    Compute a weighted stylometric score.
    Each category frequency C_i is multiplied by a confidence weight w_i = S·F·log(1+C_i).
    The final score is Σ w_i·C_i normalized to [0,1].
    """
    if not categories:
        return 0.0
    weights = {k: S * F * math.log1p(v) for k, v in categories.items()}
    raw = sum(weights[k] * v for k, v in categories.items())
    max_possible = max(categories.values()) * S * F * math.log1p(max(categories.values()))
    score = raw / (max_possible + 1e-12)
    return max(min(score, 1.0), 0.0)


# ----------------------------------------------------------------------
# Parent‑B utilities: TTT‑Linear, ternary router, endpoint breaker
# ----------------------------------------------------------------------


class EndpointCircuitBreaker:
    """Simple circuit breaker that disables diffusion when failures exceed a threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        """True if diffusion is permitted."""
        return not self.open


class TTTLinearModel:
    """Linear model with a trainable weight matrix W."""

    def __init__(self, d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        if d_out is None:
            d_out = d_in
        self.W = self.rng.standard_normal((d_out, d_in)) * scale

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x

    def loss_and_grad(self, x: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Return squared reconstruction loss and its gradient w.r.t. W.
        grad = 2·(pred‑target)·xᵀ
        """
        pred = self.predict(x)
        residual = pred - target
        loss = float(residual @ residual)
        grad = 2.0 * np.outer(residual, x)
        return loss, grad


def ternary_router(s: float) -> int:
    """
    Map a normalized scalar s∈[0,1] to a ternary offset:
        -1 if s < 0.33
         0 if 0.33 ≤ s ≤ 0.66
        +1 if s > 0.66
    """
    if s < 0.33:
        return -1
    if s > 0.66:
        return 1
    return 0


# ----------------------------------------------------------------------
# Hybrid system integrating both parent topologies
# ----------------------------------------------------------------------


class HybridModel:
    """
    Combines morphology‑stylometric confidence with a TTT‑Linear/NLMS hybrid.
    The core equations are:

        η_eff = η·(1 + γ·Score·S·F)          # adaptive learning rate
        ΔW    = -η_eff·grad(loss)           # weight update
        t_i   = round((1‑s)·T)·(1‑endpoint.allow())
        x_noisy = √α[t_i]·I + √(1‑α[t_i])·ε

    where `s` is the normalized stylometric score.
    """

    def __init__(
        self,
        dim: int,
        α_schedule: List[float] | None = None,
        η: float = 0.01,
        γ: float = 0.5,
        α_store: float = 0.6,
        β_store: float = 0.4,
        T: int = 10,
    ):
        self.dim = dim
        self.model = TTTLinearModel(d_in=dim, d_out=dim, seed=42)
        self.endpoint = EndpointCircuitBreaker()
        self.η = η
        self.γ = γ
        self.α_store = α_store
        self.β_store = β_store
        self.T = T
        # Simple schedule for diffusion coefficients α[t]; default linear decay
        if α_schedule is None:
            self.α_schedule = np.linspace(0.9, 0.1, T).tolist()
        else:
            self.α_schedule = α_schedule

    def _effective_lr(self, score: float, S: float, F: float) -> float:
        """Adaptive learning rate η_eff = η·(1 + γ·Score·S·F)."""
        return self.η * (1.0 + self.γ * score * S * F)

    def _diffusion_timestep(self, s: float) -> int:
        """Compute diffusion timestep t_i using ternary router and endpoint state."""
        base = round((1.0 - s) * self.T)
        allowed = 1 if self.endpoint.allow() else 0
        return base * allowed

    def _inject_noise(self, input_vec: np.ndarray, t_i: int) -> np.ndarray:
        """Apply diffusion forcing with schedule α[t_i]."""
        if t_i >= len(self.α_schedule):
            t_i = len(self.α_schedule) - 1
        α_t = self.α_schedule[t_i]
        eps = np.random.standard_normal(self.dim)
        return math.sqrt(α_t) * input_vec + math.sqrt(1.0 - α_t) * eps

    def hybrid_step(
        self,
        x: np.ndarray,
        inflow: np.ndarray,
        outflow: np.ndarray,
        categories: Dict[str, int],
        slots: List[ProceduralSlot],
    ) -> Tuple[np.ndarray, float, float]:
        """
        Perform a single hybrid iteration.

        Returns:
            new_x        – updated state vector
            store_delta  – change applied to the abstract store
            loss         – reconstruction loss after weight update
        """
        # 1️⃣ Morphology & stylometry
        S, F = compute_morphology_indices(slots)
        score = stylometric_score(categories, S, F)

        # 2️⃣ Diffusion timestep from ternary router
        t_i = self._diffusion_timestep(score)
        x_noisy = self._inject_noise(x, t_i)

        # 3️⃣ TTT‑Linear loss & gradient
        loss, grad = self.model.loss_and_grad(x_noisy, x)  # target is the clean state

        # 4️⃣ Adaptive weight update (bridge)
        η_eff = self._effective_lr(score, S, F)
        self.model.W -= η_eff * grad

        # 5️⃣ Store dynamics with morphology‑stylometry term
        store_delta = (
            self.α_store * inflow.sum()
            - self.β_store * outflow.sum()
            + self.γ * score * S * F
        )

        # 6️⃣ Update state (simple Euler step using model prediction)
        pred = self.model.predict(x_noisy)
        new_x = x + pred * store_delta * 1e-4  # scaling factor keeps values bounded

        # 7️⃣ Endpoint bookkeeping (randomly simulate failures)
        if random.random() < 0.05:
            self.endpoint.record_failure()
        else:
            self.endpoint.record_success()

        return new_x, store_delta, loss


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Initialise hybrid model
    dim = 8
    hybrid = HybridModel(dim=dim)

    # Dummy state vector
    x = np.zeros(dim)

    # Random inflow / outflow vectors
    inflow = np.random.poisson(5, size=dim).astype(float)
    outflow = np.random.poisson(3, size=dim).astype(float)

    # Example stylometric categories
    categories = {"noun": 120, "verb": 80, "adjective": 45, "adverb": 30}

    # Generate procedural slots
    slots = generate_procedural_slots(seed="hybrid_test", count=12)

    # Run a few hybrid steps
    for step in range(5):
        x, delta, loss = hybrid.hybrid_step(x, inflow, outflow, categories, slots)
        print(
            f"Step {step+1:02d} | Store Δ={delta: .4f} | Loss={loss: .6f} | "
            f"||x||={np.linalg.norm(x): .4f}"
        )