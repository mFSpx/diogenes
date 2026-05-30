# DARWIN HAMMER — match 1139, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_pheromone_hyb_hybrid_hybrid_hybrid_m152_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s1.py (gen3)
# born: 2026-05-29T23:33:00Z

"""Hybrid Algorithm: Pheromone-SSIM Morphology Fusion

Parents:
- hybrid_pheromone_hybrid_distributed_l_m41_s2.py (Pheromone decay & leader election)
- hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s3.py (SSIM, morphology, endpoint)

Mathematical Bridge:
Both parents expose a *morphology* based index (righting_time_index) that yields a
scalar recovery priority p ∈ [0,1].  The pheromone branch supplies a temporal
decay function ϕ(t)=v0·0.5^{t/τ}.  The SSIM branch supplies a structural similarity
σ(x,y)∈[0,1].  By treating p as a weighting factor we can fuse the three quantities
into a single hybrid score

    H = p·ϕ(t) + (1‑p)·σ(x,y)

When many agents are present we stack their morphologies into a matrix M and
compute pair‑wise SSIM values in a similarity matrix S.  The decay is then
applied element‑wise to a pheromone grid P, yielding a fused influence matrix

    F = diag(p)·P + (I‑diag(p))·S

where diag(p) is a diagonal matrix of recovery priorities.  This formulation
combines temporal relevance, structural similarity, and physical recovery
capacity in a unified linear‑algebraic framework.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import asdict, dataclass
from typing import Dict, List

# ----------------------------------------------------------------------
# Core data structures (merged from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

    def as_dict(self) -> Dict[str, any]:
        d = asdict(self)
        d["morphology"] = asdict(self.morphology)
        return d


# ----------------------------------------------------------------------
# Morphology‑based indices (identical in both parents)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised righting‑time index in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Pheromone decay (from Parent A)
# ----------------------------------------------------------------------
def pheromone_decay(v0: float, half_life_seconds: int, delta_t: int) -> float:
    """Exponential decay using half‑life expressed in seconds."""
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be > 0")
    tau = half_life_seconds / 3600.0  # convert to hours for the original formula
    return v0 * (0.5 ** (delta_t / tau))


# ----------------------------------------------------------------------
# Structural Similarity Index (SSIM) – vector version (from Parent B)
# ----------------------------------------------------------------------
def ssim(x: List[float], y: List[float], dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Simplified SSIM for 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("Input signals must have the same length")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mu_x = x_arr.mean()
    mu_y = y_arr.mean()
    sigma_x = x_arr.var()
    sigma_y = y_arr.var()
    sigma_xy = ((x_arr - mu_x) * (y_arr - mu_y)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Hybrid core functions (demonstrate the fused topology)
# ----------------------------------------------------------------------
def hybrid_score(
    morph: Morphology,
    pheromone_initial: float,
    half_life_seconds: int,
    delta_t: int,
    signal_x: List[float],
    signal_y: List[float],
) -> float:
    """
    Compute the fused hybrid score H = p·ϕ(t) + (1‑p)·σ(x,y)
    where:
        p  – recovery priority from morphology,
        ϕ(t) – pheromone decay,
        σ(x,y) – SSIM similarity.
    """
    p = recovery_priority(morph)
    phi = pheromone_decay(pheromone_initial, half_life_seconds, delta_t)
    sigma = ssim(signal_x, signal_y)
    return p * phi + (1.0 - p) * sigma


def batch_hybrid_matrix(
    morphologies: List[Morphology],
    pheromone_grid: np.ndarray,
    half_life_seconds: int,
    delta_t: int,
    signals: List[tuple[List[float], List[float]]],
) -> np.ndarray:
    """
    Vectorised version for a population of agents.

    Parameters
    ----------
    morphologies : list of Morphology
        Physical descriptors for each agent.
    pheromone_grid : 2‑D ndarray (N×N)
        Current pheromone levels between each pair of agents.
    half_life_seconds, delta_t : int
        Decay parameters applied uniformly.
    signals : list of (x, y) tuples
        One signal pair per agent for SSIM computation.

    Returns
    -------
    fused_influence : ndarray (N×N)
        F = diag(p)·P_decay + (I‑diag(p))·S
        where P_decay is the decayed pheromone grid and S the SSIM similarity matrix.
    """
    n = len(morphologies)
    if pheromone_grid.shape != (n, n):
        raise ValueError("pheromone_grid shape must match number of morphologies")
    if len(signals) != n:
        raise ValueError("signals length must match number of morphologies")

    # 1. Recovery priorities vector p
    p_vec = np.array([recovery_priority(m) for m in morphologies], dtype=np.float64)

    # 2. Decayed pheromone matrix (element‑wise decay)
    decay_factor = 0.5 ** (delta_t / (half_life_seconds / 3600.0))
    P_decay = pheromone_grid * decay_factor

    # 3. SSIM similarity matrix S (pairwise)
    S = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        xi, yi = signals[i]
        for j in range(n):
            if i == j:
                S[i, j] = 1.0
            else:
                xj, yj = signals[j]
                # combine the two signal pairs by concatenating – a simple heuristic
                S[i, j] = ssim(xi + xj, yi + yj)

    # 4. Build diagonal matrix of priorities
    D = np.diag(p_vec)

    # 5. Fuse according to the derived formula
    fused = D @ P_decay + (np.eye(n) - D) @ S
    return fused


def endpoint_hybrid_assessment(endpoint: EngineEndpoint,
                               pheromone_initial: float,
                               half_life_seconds: int,
                               delta_t: int,
                               reference_signal: List[float]) -> Dict[str, float]:
    """
    Assess an EngineEndpoint by combining its morphology‑derived priority,
    pheromone decay, and SSIM against a reference signal.

    Returns a dictionary with intermediate values and the final hybrid score.
    """
    morph = endpoint.morphology
    p = recovery_priority(morph)
    phi = pheromone_decay(pheromone_initial, half_life_seconds, delta_t)

    # Use the endpoint's capabilities as a proxy signal (hash to floats)
    cap_signal = [float(abs(hash(c)) % 256) for c in endpoint.capabilities]
    sigma = ssim(cap_signal, reference_signal)

    hybrid = p * phi + (1.0 - p) * sigma
    return {
        "recovery_priority": p,
        "pheromone_level": phi,
        "ssim_similarity": sigma,
        "hybrid_score": hybrid,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple morphology instance
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=10.0)

    # Signals for SSIM
    sig_x = [10, 20, 30, 40, 50]
    sig_y = [12, 19, 31, 39, 48]

    # Hybrid score for a single agent
    h = hybrid_score(
        morph,
        pheromone_initial=100.0,
        half_life_seconds=3600,
        delta_t=1800,
        signal_x=sig_x,
        signal_y=sig_y,
    )
    print(f"Hybrid score (single): {h:.4f}")

    # Batch processing with three agents
    morphs = [
        Morphology(2.0, 1.0, 0.5, 10.0),
        Morphology(1.5, 0.8, 0.6, 8.0),
        Morphology(2.2, 1.1, 0.4, 12.0),
    ]
    pher_grid = np.full((3, 3), 50.0)  # uniform initial pheromone
    signals = [
        (sig_x, sig_y),
        ([15, 25, 35, 45, 55], [14, 26, 34, 46, 54]),
        ([9, 18, 27, 36, 45], [11, 17, 29, 35, 47]),
    ]
    fused = batch_hybrid_matrix(
        morphs,
        pher_grid,
        half_life_seconds=3600,
        delta_t=900,
        signals=signals,
    )
    print("Fused influence matrix:\n", np.round(fused, 4))

    # Endpoint assessment
    endpoint = EngineEndpoint(
        engine_id="eng-001",
        channel="alpha",
        residency="us-east",
        runtime="prod",
        resource_class="high",
        always_on=True,
        endpoint="http://example.com/api",
        capabilities=["compress", "encrypt", "stream"],
        morphology=morph,
    )
    ref_signal = [20, 30, 40, 50, 60]
    assessment = endpoint_hybrid_assessment(
        endpoint,
        pheromone_initial=80.0,
        half_life_seconds=7200,
        delta_t=3600,
        reference_signal=ref_signal,
    )
    print("Endpoint hybrid assessment:", assessment)