# DARWIN HAMMER — match 4599, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1350_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2643_s1.py (gen6)
# born: 2026-05-29T23:56:47Z

"""Hybrid algorithm merging Parent A (temperature‑scaled variational dynamics) and Parent B
(stylometric feature extraction with Ollivier‑Ricci curvature and pheromone‑guided geometric product).

Mathematical bridge:
- The temperature‑dependent developmental rate ρ(T) from Parent A is used as a *scalar
  scaling factor* for curvature values and for the tropical max‑plus product of Parent B.
- The Metropolis acceptance probability 𝛼(ΔE, T) from Parent A decides whether a curvature
  update (computed via the Ollivier‑Ricci formula of Parent B) is accepted.
- Pheromone levels guide the tropical max‑plus product, while the scaled curvature
  feeds back into pheromone reinforcement, creating a closed hybrid loop.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

# ----------------------------------------------------------------------
# Minimal stand‑ins for classes/functions referenced by Parent B
# ----------------------------------------------------------------------
class Span:
    def __init__(self, start: int, end: int, token: str, category: str, weight: float):
        self.start = start
        self.end = end
        self.token = token
        self.category = category
        self.weight = weight

    def __repr__(self) -> str:
        return f"Span({self.start},{self.end},'{self.token}','{self.category}',{self.weight})"


class PheromoneEntry:
    def __init__(self, i: int, j: int, level: float = 0.0):
        self.i = i
        self.j = j
        self.level = level

    def __repr__(self) -> str:
        return f"PheromoneEntry({self.i},{self.j},level={self.level})"


def krampus_ollivier_ricci_curvature(G: Any, edge: Tuple[int, int]) -> float:
    """Placeholder: return a pseudo curvature for an edge."""
    # In a real implementation this would analyse the graph G.
    # Here we return a small random number centred at 0.
    return random.uniform(-0.1, 0.1)


def krampus_update(curvature: float, pheromone: float, lr: float = 0.1) -> float:
    """Simple update rule: move pheromone towards curvature."""
    return pheromone + lr * (curvature - pheromone)


def _multiply_blades(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Placeholder geometric product (standard matrix multiplication)."""
    return a @ b

# ----------------------------------------------------------------------
# Parent A core functions (temperature‑scaled dynamics)
# ----------------------------------------------------------------------
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # universal gas constant (cal·K⁻¹·mol⁻¹)


def c_to_k(celsius: float) -> float:
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Temperature‑dependent scaling factor ρ(T)."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    num = params.rho_25 * (temp_k / 298.15) * math.exp(
        -params.delta_h_activation / params.r_cal * (1.0 / temp_k - 1.0 / 298.15)
    )
    low = 1.0 + math.exp(
        params.delta_h_low / params.r_cal * (1.0 / params.t_low - 1.0 / temp_k)
    )
    high = 1.0 + math.exp(
        params.delta_h_high / params.r_cal * (1.0 / temp_k - 1.0 / params.t_high)
    )
    return num / (low * high)


def variational_free_energy(mu: float, Wx: float) -> float:
    return (mu - Wx) ** 2


def hybrid_vfe(mu: float, Wx: float, temp_c: float) -> float:
    temp_k = c_to_k(temp_c)
    rho = developmental_rate(temp_k)
    return variational_free_energy(mu, Wx) * rho


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance rule."""
    if delta_e < 0:
        return 1.0
    return math.exp(-delta_e / temperature)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    pos = x_arr >= 0
    neg = ~pos
    out = np.empty_like(x_arr)
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return float(out) if np.isscalar(x) else out


def tropical_max_plus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tropical (max‑plus) matrix product."""
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible shapes for tropical product")
    result = np.full((A.shape[0], B.shape[1]), -np.inf)
    for k in range(A.shape[1]):
        candidate = A[:, k, None] + B[k, None, :]
        result = np.maximum(result, candidate)
    return result

# ----------------------------------------------------------------------
# Parent B core functions (stylometric extraction & pheromone‑guided geometry)
# ----------------------------------------------------------------------
_FUNCTION_CATS = {
    "pronoun": {"he", "she", "they", "it", "i", "you", "we"},
    "article": {"a", "an", "the"},
    "preposition": {"in", "on", "at", "by", "with", "about"},
}


def stylometric_feature_extraction(text: str) -> List[Span]:
    """Very simple stylometric extractor returning Span objects."""
    words = text.split()
    features: List[Span] = []
    pos = 0
    for word in words:
        cat = None
        if word.lower() in _FUNCTION_CATS["pronoun"]:
            cat = "pronoun"
        elif word.lower() in _FUNCTION_CATS["article"]:
            cat = "article"
        elif word.lower() in _FUNCTION_CATS["preposition"]:
            cat = "preposition"
        if cat:
            features.append(Span(pos, pos + len(word), word, cat, 1.0))
        pos += len(word) + 1  # +1 for the space
    return features


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_scaled_curvature(text: str, temp_c: float) -> Dict[Tuple[int, int], float]:
    """
    Extract stylometric spans, treat each consecutive pair as an edge,
    compute Ollivier‑Ricci curvature for the edge and scale it by ρ(T).
    """
    spans = stylometric_feature_extraction(text)
    temp_k = c_to_k(temp_c)
    rho = developmental_rate(temp_k)

    curvature_dict: Dict[Tuple[int, int], float] = {}
    # Build a dummy graph where node ids are indices of spans
    for i in range(len(spans) - 1):
        edge = (i, i + 1)
        raw_curv = krampus_ollivier_ricci_curvature(None, edge)
        curvature_dict[edge] = raw_curv * rho
    return curvature_dict


def pheromone_guided_tropical_product(
    A: np.ndarray,
    B: np.ndarray,
    pheromones: np.ndarray,
    temp_c: float,
) -> np.ndarray:
    """
    Perform tropical max‑plus product, then bias each entry by the corresponding
    pheromone level and by the temperature scaling factor ρ(T).
    """
    if A.shape != pheromones.shape or B.shape[1] != pheromones.shape[1]:
        raise ValueError("Pheromone matrix must match result dimensions")
    base = tropical_max_plus(A, B)                # shape (m, n)
    temp_k = c_to_k(temp_c)
    rho = developmental_rate(temp_k)

    # Convert pheromone levels to a multiplicative boost factor >0
    boost = 1.0 + sigmoid(pheromones)            # keep boost in (0,2)
    result = base * boost * rho
    return result


def hybrid_optimization_step(
    text: str,
    A: np.ndarray,
    B: np.ndarray,
    pheromones: np.ndarray,
    temp_c: float,
    learning_rate: float = 0.05,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
    """
    One hybrid iteration:
    1. Compute temperature‑scaled curvature for the text.
    2. Perform pheromone‑guided tropical product.
    3. Update pheromone levels using Metropolis acceptance on curvature change.
    Returns the updated pheromone matrix and the curvature dictionary.
    """
    # 1. curvature
    curv = compute_scaled_curvature(text, temp_c)

    # 2. product
    prod = pheromone_guided_tropical_product(A, B, pheromones, temp_c)

    # 3. pheromone update
    # For each edge (i,i+1) we compare current product entry with curvature.
    # delta_e = curvature - product entry
    for (i, j), curv_val in curv.items():
        delta_e = curv_val - prod[i, j]
        temp_k = c_to_k(temp_c)
        prob = acceptance_probability(delta_e, temp_k)
        if random.random() < prob:
            # accept: move pheromone towards curvature (scaled)
            pheromones[i, j] = krampus_update(curv_val, pheromones[i, j], lr=learning_rate)

    return pheromones, curv


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic seed for reproducibility
    random.seed(0)
    np.set_printoptions(precision=3, suppress=True)

    # Example matrices (3×3) with finite values
    A = np.array([[0.2, -0.5, 0.1],
                  [1.0, 0.3, -0.2],
                  [0.0, 0.4, 0.6]])

    B = np.array([[0.5, -0.1, 0.2],
                  [0.3, 0.8, -0.4],
                  [-0.2, 0.1, 0.9]])

    # Initialise pheromones to small random positives
    pheromones = np.abs(np.random.randn(3, 3)) * 0.1

    sample_text = "She gave the book to him in the library"

    temperature_c = 25.0  # ambient temperature

    updated_pheromones, curvature_map = hybrid_optimization_step(
        sample_text, A, B, pheromones, temperature_c
    )

    print("Updated pheromones:")
    print(updated_pheromones)
    print("\nCurvature (scaled) per edge:")
    for edge, val in curvature_map.items():
        print(f"{edge}: {val:.5f}")