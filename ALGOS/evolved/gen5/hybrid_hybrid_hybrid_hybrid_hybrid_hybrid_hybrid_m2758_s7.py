# DARWIN HAMMER — match 2758, survivor 7
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hdc_serpentin_m1551_s0.py (gen4)
# born: 2026-05-29T23:45:38Z

"""Hybrid Fusion of Darwin Hammer Decision-Hygiene & Hybrid Bandit TTT with
Epistemic Certainty & Hyperdimensional Morphology.

Parents:
- hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s3.py (resource vector eᵢ,
  VRAM store, weight matrix)
- hybrid_hybrid_hybrid_hybrid_hdc_serpentin_m1551_s0.py (epistemic certainty
  flags, sphericity index, bipolar high‑dimensional vectors)

Mathematical Bridge:
The 3‑dimensional resource vector eᵢ = [dᵢ, pᵢ, sᵢ] is projected into a
high‑dimensional bipolar space using a random projection whose scale is
controlled by the sphericity index of a Morphology object.  The resulting
bipolar vector is then combined (element‑wise AND) with a bipolar
certainty‑flag vector derived from the epistemic flag’s confidence.  This
compound high‑dimensional representation modulates the virtual VRAM store,
which in turn scales the learning rate of the bandit’s weight‑matrix update.
Thus the deterministic resource‑based decision logic of Darwin Hammer and the
uncertainty‑driven hyperdimensional encoding of the serpentina module are
mathematically fused."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Iterable, Tuple, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great‑circle distance in metres between two lat/lon points."""
    R = 6371e3  # Earth radius in metres
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)

    a = (
        math.sin(Δφ / 2) ** 2
        + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # basis points, 0..10000
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
            object.__setattr__(
                self, "generated_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

    def as_dict(self) -> dict:
        return asdict(self)


def sphericity_index(length: float, width: float, height: float) -> float:
    """Dimension‑agnostic sphericity; 1.0 for a perfect cube."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


# ----------------------------------------------------------------------
# Fusion data structures
# ----------------------------------------------------------------------
@dataclass
class Entity:
    """Minimal representation of an entity processed by the hybrid system."""
    id: str
    lat: float
    lon: float
    signature: str  # arbitrary identifier used for collision detection
    score: float    # decision‑hygiene score (sᵢ)
    morphology: Tuple[float, float, float]  # (length, width, height)


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def compute_resource_vector(
    entity: Entity,
    ref_lat: float,
    ref_lon: float,
    beta: float = 1.0,
    signatures_seen: Dict[str, int] | None = None,
) -> np.ndarray:
    """
    eᵢ = [dᵢ, pᵢ, sᵢ]
    dᵢ : haversine distance to reference location
    pᵢ : β·σᵢ where σᵢ = 1 if signature collides, else 0
    sᵢ : decision‑hygiene score
    """
    d_i = haversine_distance(entity.lat, entity.lon, ref_lat, ref_lon)

    # collision detection
    if signatures_seen is None:
        signatures_seen = {}
    sigma_i = 1 if signatures_seen.get(entity.signature, 0) > 0 else 0
    signatures_seen[entity.signature] = signatures_seen.get(entity.signature, 0) + 1
    p_i = beta * sigma_i

    s_i = entity.score
    return np.array([d_i, p_i, s_i], dtype=float)


def certainty_flag_to_bipolar(flag: CertaintyFlag, dim: int = 1024) -> np.ndarray:
    """
    Convert a CertaintyFlag into a bipolar (+1 / -1) vector.
    The confidence (0..10000) controls the proportion of +1 entries.
    """
    prob_pos = flag.confidence_bps / 10000.0
    # deterministic pseudo‑random seed from label+authority to keep reproducibility
    seed = hash((flag.label, flag.authority_class)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    vec = rng.choice([1, -1], size=dim, p=[prob_pos, 1 - prob_pos])
    return vec.astype(np.int8)


def project_resource_to_hyperdim(
    resource_vec: np.ndarray,
    morphology: Tuple[float, float, float],
    proj_dim: int = 1024,
) -> np.ndarray:
    """
    Random linear projection of the 3‑D resource vector into a bipolar space.
    The projection matrix is scaled by the sphericity index of the morphology.
    """
    length, width, height = morphology
    sph = sphericity_index(length, width, height)

    rng = np.random.default_rng(int(sph * 1e6) % (2**32))
    proj_matrix = rng.normal(loc=0.0, scale=1.0, size=(proj_dim, 3))
    projected = proj_matrix @ resource_vec  # shape (proj_dim,)

    # bipolar conversion
    bipolar = np.where(projected >= 0, 1, -1).astype(np.int8)
    return bipolar


def update_vram_store(
    store: np.ndarray,
    delta: np.ndarray,
    dt: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Store dynamics:  dS/dt = α·Δ - β·S
    Integrated with explicit Euler: S_{t+Δt} = S_t + dt·(α·Δ - β·S_t)
    """
    dS = alpha * delta - beta * store
    return store + dt * dS


def hybrid_bandit_step(
    entities: List[Entity],
    ref_location: Tuple[float, float],
    weight_matrix: np.ndarray,
    vram_store: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One learning step that:
    1. Builds resource vectors and projects them to hyper‑dimensional space.
    2. Generates certainty vectors and combines them (element‑wise AND).
    3. Uses the combined vector as a modulation signal (Δ) for the VRAM store.
    4. Updates the bandit weight matrix with a learning rate scaled by the
       current store magnitude.
    Returns updated (weight_matrix, vram_store).
    """
    base_eta = params.get("base_eta", 0.01)
    alpha = params.get("alpha", 1.0)
    beta = params.get("beta", 1.0)
    dt = params.get("dt", 1.0)
    proj_dim = weight_matrix.shape[0]

    signatures_seen: Dict[str, int] = {}
    deltas = np.zeros(proj_dim, dtype=float)

    for ent in entities:
        # 1️⃣ resource → hyperdim
        res_vec = compute_resource_vector(
            ent,
            ref_lat=ref_location[0],
            ref_lon=ref_location[1],
            beta=params.get("sig_beta", 1.0),
            signatures_seen=signatures_seen,
        )
        res_hyper = project_resource_to_hyperdim(res_vec, ent.morphology, proj_dim)

        # 2️⃣ certainty → bipolar
        # For demonstration we pick a flag based on score thresholds
        if ent.score > 0.8:
            label = "FACT"
        elif ent.score > 0.5:
            label = "PROBABLE"
        else:
            label = "POSSIBLE"
        flag = CertaintyFlag(
            label=label,
            confidence_bps=int(min(max(ent.score * 10000, 0), 10000)),
            authority_class="auto",
            rationale="score‑derived",
        )
        cert_vec = certainty_flag_to_bipolar(flag, proj_dim)

        # 3️⃣ combine → modulation delta
        combined = res_hyper * cert_vec  # element‑wise AND in bipolar domain
        deltas += combined.astype(float)

    # average over entities
    if entities:
        deltas /= len(entities)

    # 4️⃣ update VRAM store
    vram_store = update_vram_store(vram_store, deltas, dt, alpha, beta)

    # 5️⃣ compute learning‑rate scaling factor
    store_norm = np.linalg.norm(vram_store)
    eta = base_eta * (1.0 + store_norm)

    # 6️⃣ weight matrix update (simple gradient ascent on expected reward)
    # Expected reward for each action i: r_i = w_i · μ where μ is mean resource vector
    mean_res = np.mean(
        [compute_resource_vector(e, *ref_location, beta=params.get("sig_beta", 1.0)) for e in entities],
        axis=0,
    )
    # project mean resource to hyperdim (same projection as above, but reuse random seed)
    mean_hyper = project_resource_to_hyperdim(mean_res, (1.0, 1.0, 1.0), proj_dim)
    reward_est = weight_matrix @ mean_hyper  # shape (proj_dim,)

    # gradient ascent: w ← w + η·(reward_est)·mean_hyperᵀ
    grad = np.outer(reward_est, mean_hyper)
    weight_matrix = weight_matrix + eta * grad

    return weight_matrix, vram_store


# ----------------------------------------------------------------------
# Utility constructors
# ----------------------------------------------------------------------
def init_weight_matrix(d_in: int, d_out: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.1, size=(d_out, d_in)).astype(float)


def init_vram_store(dim: int, decay: float = 0.99) -> np.ndarray:
    """Initialize VRAM store as a zero vector; decay is stored for external use."""
    return np.zeros(dim, dtype=float)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    D_IN = 1024
    D_OUT = 1024
    SEED = 42
    params = {
        "base_eta": 0.005,
        "alpha": 0.8,
        "beta": 0.2,
        "dt": 1.0,
        "sig_beta": 1.0,
    }

    # Initialise core structures
    W = init_weight_matrix(D_IN, D_OUT, SEED)
    S = init_vram_store(D_OUT)

    # Create a tiny synthetic population
    entities = [
        Entity(
            id="A",
            lat=37.7749,
            lon=-122.4194,
            signature="sig1",
            score=0.92,
            morphology=(1.2, 0.8, 0.6),
        ),
        Entity(
            id="B",
            lat=34.0522,
            lon=-118.2437,
            signature="sig2",
            score=0.47,
            morphology=(0.9, 1.0, 0.7),
        ),
        Entity(
            id="C",
            lat=40.7128,
            lon=-74.0060,
            signature="sig1",  # intentional collision with A
            score=0.66,
            morphology=(1.0, 1.0, 1.0),
        ),
    ]

    # Reference location (e.g., system hub)
    REF_LOC = (36.0, -120.0)

    # Perform a single hybrid learning step
    W_new, S_new = hybrid_bandit_step(entities, REF_LOC, W, S, params)

    # Simple sanity checks
    assert W_new.shape == W.shape, "Weight matrix shape changed unexpectedly"
    assert S_new.shape == S.shape, "VRAM store shape changed unexpectedly"
    print("Hybrid step completed successfully.")
    print(f"Store norm before: {np.linalg.norm(S):.4f}, after: {np.linalg.norm(S_new):.4f}")
    print(f"Weight matrix Frobenius norm change: {np.linalg.norm(W_new - W):.6f}")