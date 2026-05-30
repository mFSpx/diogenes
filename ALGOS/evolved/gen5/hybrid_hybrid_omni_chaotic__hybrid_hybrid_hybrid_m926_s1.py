# DARWIN HAMMER — match 926, survivor 1
# gen: 5
# parent_a: hybrid_omni_chaotic_sprint_jepa_energy_m80_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_fisher_m22_s2.py (gen4)
# born: 2026-05-29T23:31:37Z

"""Hybrid Algorithm: Chaotic Omni-Front Synthesis Core × JEPA Energy meets Pheromone‑Fisher‑Entropy Bridge

Parents:
- hybrid_omni_chaotic_sprint_jepa_energy_m80_s2.py (Chaotic graph generation + JEPA latent‑energy prediction)
- hybrid_hybrid_hybrid_pherom_hybrid_hybrid_fisher_m22_s2.py (Pheromone probability, entropy, Fisher information, ternary lens)

Mathematical Bridge:
The JEPA energy term E(z)=½ zᵀ Λ z − ∑log p_i uses a precision matrix Λ that models uncertainty of the latent
variables *z*.  In the pheromone‑based parent the Fisher information I(p) of the pheromone probability
distribution p = (p₁,…,p_n) quantifies the amount of information each probability carries:
 I_i = 1 / (p_i · (1−p_i)).
We therefore set the JEPA precision matrix Λ to the diagonal Fisher‑information matrix of the pheromone
distribution.  This fuses the chaotic graph’s latent variables with the information‑theoretic surface
usage tracked by pheromones, while the ternary lens provides a lightweight routing index for each node.

The resulting hybrid operates in three stages:
1. Generate a chaotic adjacency matrix A and a latent vector z for each node.
2. Retrieve (or mock) pheromone probabilities, compute entropy H(p) and the diagonal Fisher matrix I(p).
3. Compute the JEPA‑style energy E(z, p) and perform a gradient step on z, optionally using the ternary
   lens index to bias updates.

All operations rely only on NumPy and the Python standard library.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Core chaotic graph generation (from Parent A)
# ----------------------------------------------------------------------
def chaotic_graph(num_nodes: int, chaos_factor: float = 3.9) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a directed graph with a chaotic adjacency matrix and a latent variable vector.

    Parameters
    ----------
    num_nodes: int
        Number of nodes in the graph.
    chaos_factor: float
        Logistic‑map parameter (default 3.9, chaotic regime).

    Returns
    -------
    A: np.ndarray shape (n, n)
        Binary adjacency matrix where A[i, j]=1 indicates an edge i→j.
    z: np.ndarray shape (n,)
        Latent variable for each node, initialized from the chaotic logistic map.
    """
    # Logistic map to generate a pseudo‑random chaotic sequence
    x = random.random()
    seq = np.empty(num_nodes)
    for i in range(num_nodes):
        x = chaos_factor * x * (1.0 - x)
        seq[i] = x

    # Use the sequence to define latent variables (scaled to mean 0, std 1)
    z = (seq - np.mean(seq)) / (np.std(seq) + 1e-12)

    # Build adjacency: connect i→j if |z_i - z_j| < threshold
    threshold = np.percentile(np.abs(np.subtract.outer(z, z)), 30)  # sparsity ~30%
    A = (np.abs(np.subtract.outer(z, z)) < threshold).astype(int)

    # Remove self‑loops
    np.fill_diagonal(A, 0)
    return A, z

# ----------------------------------------------------------------------
# Pheromone probability, entropy and Fisher information (from Parent B)
# ----------------------------------------------------------------------
def pheromone_probabilities(num_nodes: int, seed: int | None = None) -> np.ndarray:
    """
    Mock retrieval of pheromone signals for each node and convert to a probability distribution.
    In a real system this would query a database; here we use a Dirichlet draw for reproducibility.
    """
    rng = np.random.default_rng(seed)
    # Dirichlet with concentration 0.5 yields a sparse distribution (many near‑zero entries)
    probs = rng.dirichlet(alpha=np.full(num_nodes, 0.5))
    return probs

def entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """
    Shannon entropy of a probability vector.
    """
    p = np.clip(probabilities, eps, 1.0)
    return -np.sum(p * np.log(p))

def fisher_information_diagonal(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Diagonal Fisher information matrix for independent Bernoulli‑like probabilities.
    I_i = 1 / (p_i * (1 - p_i))
    """
    p = np.clip(probabilities, eps, 1.0 - eps)
    return 1.0 / (p * (1.0 - p))

# ----------------------------------------------------------------------
# Ternary lens routing (from Parent B)
# ----------------------------------------------------------------------
TERNARY_DIMS = 12

def ternary_lens_index(z: np.ndarray) -> int:
    """
    Map a latent vector to a ternary index in [0, 3**TERNARY_DIMS).
    The sign of each component determines the ternary digit:
        - negative → 0
        - near zero (|z| < 0.1) → 1
        - positive → 2
    The digits are interpreted as a base‑3 number.
    """
    digits = np.where(z < -0.1, 0,
                      np.where(z > 0.1, 2, 1))
    # Pad or truncate to TERNARY_DIMS
    if len(digits) < TERNARY_DIMS:
        digits = np.pad(digits, (0, TERNARY_DIMS - len(digits)), constant_values=1)
    else:
        digits = digits[:TERNARY_DIMS]

    index = 0
    for d in digits:
        index = index * 3 + int(d)
    return index

# ----------------------------------------------------------------------
# Hybrid energy computation and latent update
# ----------------------------------------------------------------------
def hybrid_energy(z: np.ndarray, pheromone_probs: np.ndarray) -> float:
    """
    JEPA‑style energy using the Fisher information of pheromone probabilities as the precision matrix.

    E(z, p) = 0.5 * zᵀ I(p) z  −  Σ log(p_i + ε)

    Returns the scalar energy value.
    """
    eps = 1e-12
    I_diag = fisher_information_diagonal(pheromone_probs, eps=eps)
    precision = np.diag(I_diag)
    quadratic = 0.5 * z @ (precision @ z)
    log_term = -np.sum(np.log(np.clip(pheromone_probs, eps, None)))
    return quadratic + log_term

def gradient_step(z: np.ndarray, pheromone_probs: np.ndarray, lr: float = 0.01) -> np.ndarray:
    """
    Perform one gradient descent step on the latent vector z with respect to the hybrid energy.

    ∂E/∂z = I(p) z
    """
    I_diag = fisher_information_diagonal(pheromone_probs)
    grad = I_diag * z  # element‑wise because I is diagonal
    return z - lr * grad

# ----------------------------------------------------------------------
# Demonstration functions (fulfil requirement of ≥3 functions)
# ----------------------------------------------------------------------
def run_hybrid_iteration(num_nodes: int, seed: int | None = None) -> dict:
    """
    Execute a single hybrid iteration:
    1. Generate chaotic graph and latent variables.
    2. Obtain pheromone probabilities.
    3. Compute energy, entropy, ternary index.
    4. Update latent variables.
    Returns a dictionary with diagnostics.
    """
    A, z = chaotic_graph(num_nodes)
    p = pheromone_probabilities(num_nodes, seed=seed)
    ent = entropy(p)
    energy_before = hybrid_energy(z, p)
    z_updated = gradient_step(z, p, lr=0.05)
    energy_after = hybrid_energy(z_updated, p)
    lens_idx = ternary_lens_index(z)

    diagnostics = {
        "adjacency_density": A.mean(),
        "entropy": ent,
        "energy_before": energy_before,
        "energy_after": energy_after,
        "latent_norm_before": np.linalg.norm(z),
        "latent_norm_after": np.linalg.norm(z_updated),
        "ternary_lens_index": lens_idx,
    }
    return diagnostics

def batch_hybrid(num_nodes: int, iterations: int = 5, seed: int | None = None) -> list[dict]:
    """
    Run multiple hybrid iterations, feeding the updated latent vector from the previous
    iteration into the next.  This demonstrates the dynamical coupling of the chaotic
    engine with the information‑theoretic feedback loop.
    """
    A, z = chaotic_graph(num_nodes)
    results = []
    rng = np.random.default_rng(seed)
    for it in range(iterations):
        p = pheromone_probabilities(num_nodes, seed=int(rng.integers(0, 1_000_000)))
        ent = entropy(p)
        energy = hybrid_energy(z, p)
        lens_idx = ternary_lens_index(z)
        results.append({
            "iteration": it,
            "entropy": ent,
            "energy": energy,
            "latent_norm": np.linalg.norm(z),
            "ternary_lens_index": lens_idx,
        })
        z = gradient_step(z, p, lr=0.03)  # propagate updated latents
    return results

def simulate_surface_usage(num_nodes: int, surface_key: str = "default") -> dict:
    """
    High‑level wrapper that pretends to query a surface (using the mock pheromone function),
    runs a hybrid iteration and returns a compact report.
    """
    # surface_key is unused in the mock but kept for API compatibility with Parent B
    diagnostics = run_hybrid_iteration(num_nodes)
    report = {
        "surface_key": surface_key,
        "nodes": num_nodes,
        "adjacency_density": diagnostics["adjacency_density"],
        "entropy": diagnostics["entropy"],
        "energy_reduction": diagnostics["energy_before"] - diagnostics["energy_after"],
        "ternary_lens_index": diagnostics["ternary_lens_index"],
    }
    return report

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple sanity check that runs without external resources
    NUM_NODES = 17
    print("Running single hybrid iteration...")
    diag = run_hybrid_iteration(NUM_NODES, seed=42)
    for k, v in diag.items():
        print(f"{k}: {v}")

    print("\nRunning batch hybrid (3 iterations)...")
    batch = batch_hybrid(NUM_NODES, iterations=3, seed=7)
    for entry in batch:
        print(entry)

    print("\nSimulating surface usage report...")
    report = simulate_surface_usage(NUM_NODES, surface_key="example_surface")
    print(report)