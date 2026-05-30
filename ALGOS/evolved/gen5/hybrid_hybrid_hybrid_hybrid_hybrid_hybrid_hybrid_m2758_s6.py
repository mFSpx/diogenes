# DARWIN HAMMER — match 2758, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hdc_serpentin_m1551_s0.py (gen4)
# born: 2026-05-29T23:45:38Z

"""Hybrid Fusion of Darwin Hammer and Serpentina HDC

Parents:
- hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s3.py (resource‑vector + VRAM‑modulated bandit)
- hybrid_hybrid_hybrid_hybrid_hdc_serpentin_m1551_s0.py (epistemic certainty flags + sphericity index)

Mathematical bridge:
The sphericity index *σ* (a scalar derived from morphology) is used as a
multiplicative bridge between the two domains:

* It scales the virtual‑VRAM store that modulates the learning‑rate η of the
  bandit (parent A).
* It also scales the epistemic‑certainty weight w_f derived from a certainty
  flag (parent B).

Thus the effective learning‑rate becomes  

    η_eff = η_base · (1 + σ) / (1 + ‖store‖)

and the reward expectation for entity *i* is  

    R̂_i = (e_i · (W · store)) · w_f·(1+σ)

where e_i = [d_i, p_i, s_i] is the resource vector, W is the bandit weight
matrix and w_f is a scalar confidence factor attached to an epistemic flag.
The fusion therefore intertwines the resource‑vector dynamics of Darwin Hammer
with the high‑dimensional certainty representation of Serpentina HDC. """

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants and helper tables (parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
# Map flags to a base confidence (0..1).  Higher confidence → larger weight.
FLAG_CONFIDENCE = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.1,
    "SURE_MAYBE": 0.3,
}


def sphericity_index(length: float, width: float, height: float) -> float:
    """Return the sphericity index defined in parent B."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Meters between two lat/lon points (parent A)."""
    R = 6371000.0  # Earth radius in metres
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)

    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """A minimal entity used by the hybrid system."""
    lat: float
    lon: float
    signature: str  # arbitrary identifier
    decision_score: float  # s_i from parent A


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float = 1.0  # unused but kept for completeness


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at",
                               datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    @property
    def base_confidence(self) -> float:
        """Return the normalized confidence (0..1) from the flag label."""
        return FLAG_CONFIDENCE[self.label]


# ----------------------------------------------------------------------
# Core hybrid class
# ----------------------------------------------------------------------
class HybridFusion:
    """
    Implements the fused algorithm.
    - W: weight matrix (d_in × d_out) from the bandit side.
    - store: virtual VRAM vector (size d_out) that modulates learning.
    - dt, alpha, beta: parameters of the store differential equation (parent A).
    - sigma: sphericity index derived from morphology (parent B) used as a bridge.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        seed: int = 0,
        base_eta: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1.0,
        store_decay: float = 0.99,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(loc=0.0, scale=0.1, size=(d_in, d_out))
        self.store = np.zeros(d_out, dtype=float)
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.store_decay = store_decay

    # ------------------------------------------------------------------
    # 1️⃣ Resource‑vector construction (parent A)
    # ------------------------------------------------------------------
    def resource_vector(
        self,
        entity: Entity,
        reference: Tuple[float, float],
        collision_set: set[str],
        beta: float = 1.0,
    ) -> np.ndarray:
        """
        Build e_i = [d_i, p_i, s_i].

        - d_i: haversine distance to a reference point.
        - p_i: β·σ_i where σ_i = 1 if the signature collides with another entity.
        - s_i: decision_score from the entity.
        """
        d_i = haversine_distance(entity.lat, entity.lon, *reference)
        sigma_i = 1.0 if entity.signature in collision_set else 0.0
        p_i = beta * sigma_i
        s_i = float(entity.decision_score)
        return np.array([d_i, p_i, s_i], dtype=float)

    # ------------------------------------------------------------------
    # 2️⃣ Certainty‑weight from epistemic flag (parent B)
    # ------------------------------------------------------------------
    def certainty_weight(self, flag: CertaintyFlag, sigma: float) -> float:
        """
        Compute w_f = base_confidence × (1 + σ).

        The sphericity index σ (from morphology) bridges the two parents.
        """
        return flag.base_confidence * (1.0 + sigma)

    # ------------------------------------------------------------------
    # 3️⃣ VRAM‑modulated learning rate (parent A + bridge)
    # ------------------------------------------------------------------
    def effective_eta(self, sigma: float) -> float:
        """
        η_eff = η_base × (1 + σ) / (1 + ‖store‖)
        """
        norm = np.linalg.norm(self.store)
        return self.base_eta * (1.0 + sigma) / (1.0 + norm)

    # ------------------------------------------------------------------
    # 4️⃣ Expected reward computation (fusion core)
    # ------------------------------------------------------------------
    def expected_reward(
        self,
        e_vec: np.ndarray,
        flag: CertaintyFlag,
        sigma: float,
    ) -> float:
        """
        R̂ = (e · (W·store)) × w_f × (1+σ)

        - (W·store) yields a d_in‑dimensional vector that is dotted with e.
        - w_f incorporates epistemic confidence.
        """
        w_f = self.certainty_weight(flag, sigma)
        bandit_term = e_vec @ (self.W @ self.store)
        return bandit_term * w_f * (1.0 + sigma)

    # ------------------------------------------------------------------
    # 5️⃣ Store dynamics (parent A)
    # ------------------------------------------------------------------
    def update_store(self, reward: float, sigma: float) -> None:
        """
        Simple Euler integration of the store differential equation:

        dstore/dt = α·reward – β·store
        store ← store + dt·(α·reward – β·store)
        Then apply exponential decay.
        """
        dstore = self.alpha * reward - self.beta * self.store
        self.store += self.dt * dstore
        self.store *= self.store_decay

    # ------------------------------------------------------------------
    # 6️⃣ Weight matrix update (bandit learning)
    # ------------------------------------------------------------------
    def update_weights(
        self,
        e_vec: np.ndarray,
        flag: CertaintyFlag,
        sigma: float,
        reward: float,
    ) -> None:
        """
        Stochastic gradient ascent on the expected reward.
        ΔW ∝ η_eff * reward * (e ⊗ store)
        where ⊗ denotes the outer product.
        """
        eta = self.effective_eta(sigma)
        grad = np.outer(e_vec, self.store)  # shape (d_in, d_out)
        self.W += eta * reward * grad

    # ------------------------------------------------------------------
    # 7️⃣ One full hybrid step over a batch of entities
    # ------------------------------------------------------------------
    def step(
        self,
        entities: List[Entity],
        flags: List[CertaintyFlag],
        morphology: Morphology,
        reference: Tuple[float, float],
    ) -> List[float]:
        """
        Process a batch:
        1. Detect signature collisions.
        2. Compute sphericity σ.
        3. For each (entity, flag) pair compute expected reward,
           update store and weights.
        Returns the list of computed expected rewards.
        """
        # 1. Collision detection
        signatures = [e.signature for e in entities]
        collision_set = {sig for sig in signatures if signatures.count(sig) > 1}

        # 2. Sphericity index (bridge scalar)
        sigma = sphericity_index(
            morphology.length, morphology.width, morphology.height
        )

        rewards: List[float] = []
        for entity, flag in zip(entities, flags):
            e_vec = self.resource_vector(entity, reference, collision_set)
            r_hat = self.expected_reward(e_vec, flag, sigma)
            rewards.append(r_hat)

            # update internal dynamics
            self.update_store(r_hat, sigma)
            self.update_weights(e_vec, flag, sigma, r_hat)

        return rewards


# ----------------------------------------------------------------------
# Utility functions demonstrating the hybrid operation (≥3)
# ----------------------------------------------------------------------
def generate_dummy_entities(n: int, seed: int = 42) -> List[Entity]:
    rng = random.Random(seed)
    entities: List[Entity] = []
    for _ in range(n):
        lat = rng.uniform(-90.0, 90.0)
        lon = rng.uniform(-180.0, 180.0)
        signature = f"id_{rng.randint(0, 10)}"  # intentional collisions possible
        decision_score = rng.uniform(0.0, 1.0)
        entities.append(Entity(lat, lon, signature, decision_score))
    return entities


def generate_dummy_flags(n: int, seed: int = 99) -> List[CertaintyFlag]:
    rng = random.Random(seed)
    flags: List[CertaintyFlag] = []
    for _ in range(n):
        label = rng.choice(EPISTEMIC_FLAGS)
        confidence = int(rng.uniform(0, 10000))
        authority = "auto"
        rationale = "synthetic"
        flags.append(CertaintyFlag(label, confidence, authority, rationale))
    return flags


def smoke_test() -> None:
    """Run a quick sanity check of the hybrid system."""
    # Parameters
    d_in, d_out = 3, 5
    hf = HybridFusion(d_in, d_out, seed=123)

    # Dummy data
    entities = generate_dummy_entities(8)
    flags = generate_dummy_flags(8)
    morph = Morphology(length=2.0, width=1.5, height=1.0)
    reference_point = (0.0, 0.0)  # equator & prime meridian

    # Single step
    rewards = hf.step(entities, flags, morph, reference_point)

    # Simple assertions (non‑exceptional run)
    assert len(rewards) == len(entities)
    # Ensure internal structures have been updated
    assert hf.W.shape == (d_in, d_out)
    assert hf.store.shape == (d_out,)

    # Print a concise summary
    print("Hybrid step completed.")
    print(f"Rewards: {np.round(rewards, 3)}")
    print(f"Store norm: {np.linalg.norm(hf.store):.4f}")
    print(f"Weight matrix norm: {np.linalg.norm(hf.W):.4f}")


if __name__ == "__main__":
    smoke_test()