# DARWIN HAMMER — match 1536, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m926_s1.py (gen5)
# parent_b: hybrid_ternary_lens_audit_hybrid_hybrid_privac_m154_s1.py (gen3)
# born: 2026-05-29T23:37:19Z

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Callable


# ----------------------------------------------------------------------
# Core chaotic graph generation (Parent A)
# ----------------------------------------------------------------------
def logistic_map(x: np.ndarray, r: float) -> np.ndarray:
    """Vectorised logistic map."""
    return r * x * (1.0 - x)


def chaotic_graph(num_nodes: int, chaos_factor: float = 3.9, steps: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a directed graph with a chaotic adjacency matrix and a latent vector.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    chaos_factor : float, optional
        Logistic‑map parameter (default 3.9, chaotic regime).
    steps : int, optional
        Number of logistic iterations used to accumulate the adjacency matrix.

    Returns
    -------
    A : np.ndarray, shape (n, n)
        Binary adjacency matrix where A[i, j] = 1 indicates an edge i→j.
    z : np.ndarray, shape (n,)
        Latent vector drawn from the final chaotic state.
    """
    # initialise with a uniform random seed in (0,1)
    x = np.random.rand(num_nodes)
    A = np.zeros((num_nodes, num_nodes), dtype=np.int8)

    for _ in range(steps):
        x = logistic_map(x, chaos_factor)
        # edge exists if current state exceeds 0.5 and differs from previous binary pattern
        edge_mask = (x > 0.5).astype(np.int8)
        A = np.clip(A + edge_mask[:, None] ^ edge_mask[None, :], 0, 1)

    z = x.copy()                     # latent vector is the final chaotic state
    return A, z


# ----------------------------------------------------------------------
# Ternary lens audit logic (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


class ModelPool:
    """Simple resource‑constrained model registry."""
    def __init__(self, ram_ceiling_mb: int = 6000, vram_ceiling_mb: int = 4096):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.vram_ceiling_mb = vram_ceiling_mb
        self.loaded: dict[str, ModelTier] = {}

    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _used_vram(self) -> int:
        return sum(m.vram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if self._used_ram() + model.ram_mb > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        if self._used_vram() + model.vram_mb > self.vram_ceiling_mb:
            raise RuntimeError("VRAM ceiling exceeded")
        self.loaded[model.name] = model


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """
    Normalised reconstruction risk (0‑1). Guard against division by zero.
    """
    if total_records <= 0:
        raise ValueError("total_records must be positive")
    return min(1.0, max(0.0, unique_quasi_identifiers / total_records))


# ----------------------------------------------------------------------
# Hybrid mathematical bridge
# ----------------------------------------------------------------------
def pheromone_probabilities(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Derive a deterministic pheromone distribution from graph topology.
    Uses out‑degree normalisation; adds epsilon to avoid 0/1 extremes.
    """
    out_deg = A.sum(axis=1).astype(np.float64)
    p = out_deg / (out_deg.max() + eps)          # scale to [0,1]
    p = np.clip(p, eps, 1.0 - eps)               # keep away from the boundaries
    return p


def fisher_information(p: np.ndarray) -> np.ndarray:
    """
    Diagonal Fisher information matrix for a Bernoulli‑like distribution.
    I_ii = 1 / (p_i (1-p_i)).
    """
    return np.diag(1.0 / (p * (1.0 - p)))


def weighted_fisher(risk_score: float, I: np.ndarray) -> np.ndarray:
    """
    Scale the Fisher matrix by the reconstruction risk.
    """
    if risk_score <= 0.0:
        # No risk → no weighting; fall back to identity to keep gradients well‑behaved
        return np.eye(I.shape[0])
    return risk_score * I


def weighted_jepa_energy(z: np.ndarray, I_w: np.ndarray, p: np.ndarray) -> float:
    """
    Energy term E^(w)(z, p) = ½ zᵀ I_w z − Σ log p_i .
    """
    quadratic = 0.5 * float(z.T @ I_w @ z)
    log_term = -np.sum(np.log(p))
    return quadratic + log_term


def hybrid_gradient_step(
    A: np.ndarray,
    z: np.ndarray,
    risk_score: float,
    lr: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single gradient descent step on the latent vector.

    Returns
    -------
    z_new : np.ndarray
        Updated latent vector.
    energy : float
        Energy after the update.
    """
    p = pheromone_probabilities(A)
    I = fisher_information(p)
    I_w = weighted_fisher(risk_score, I)

    # Gradient of the energy w.r.t. z is I_w @ z (log term independent of z)
    grad = I_w @ z

    # Gradient descent (subtract gradient)
    z_new = z - lr * grad

    # Optional: re‑project into (0,1) to keep latent values bounded
    z_new = np.clip(z_new, 0.0, 1.0)

    energy = weighted_jepa_energy(z_new, I_w, p)
    return z_new, energy


# ----------------------------------------------------------------------
# End‑to‑end hybrid routine
# ----------------------------------------------------------------------
def chaotic_ternary_hybrid(
    num_nodes: int,
    chaos_factor: float = 3.9,
    unique_quasi_identifiers: int = 10,
    total_records: int = 100,
    lr: float = 0.05,
    steps: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run the full hybrid pipeline for a configurable number of optimisation steps.

    Returns
    -------
    A : np.ndarray
        Final adjacency matrix.
    z : np.ndarray
        Optimised latent vector.
    final_energy : float
        Energy after the last optimisation step.
    """
    A, z = chaotic_graph(num_nodes, chaos_factor)
    risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)

    for _ in range(steps):
        z, _ = hybrid_gradient_step(A, z, risk, lr)

    # Compute final energy for reporting
    p = pheromone_probabilities(A)
    I_w = weighted_fisher(risk, fisher_information(p))
    final_energy = weighted_jepa_energy(z, I_w, p)

    return A, z, final_energy


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)                     # deterministic for the demo
    A, z_opt, energy = chaotic_ternary_hybrid(
        num_nodes=12,
        chaos_factor=3.9,
        unique_quasi_identifiers=12,
        total_records=150,
        lr=0.07,
        steps=5,
    )
    print("Optimised latent vector (first 5 entries):", z_opt[:5])
    print("Final weighted JEPA energy:", energy)