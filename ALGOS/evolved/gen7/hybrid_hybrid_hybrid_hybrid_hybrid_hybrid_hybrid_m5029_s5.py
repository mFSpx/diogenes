# DARWIN HAMMER — match 5029, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_ternar_bayes_claim_kernel_m294_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m1434_s1.py (gen6)
# born: 2026-05-29T23:59:30Z

"""Hybrid Physarum‑Bayes‑Fisher‑Ternary Router Algorithm

Parents:
- hybrid_hybrid_hybrid_ternar_bayes_claim_kernel_m294_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m1434_s1.py (Algorithm B)

Mathematical bridge:
Algorithm A updates a belief mean μ using the variational free‑energy rule  
 μ←μ+η Σ · o, where Σ is a covariance matrix and o an observation vector.  
Algorithm B provides a physics‑based conductance update for a physarum‑like network
via flux f = g/ℓ · (pₐ−p_b) and a Fisher‑information‑derived precision τ that can be
interpreted as the inverse variance of an observation.

The hybrid algorithm treats each edge conductance g as a Bayesian latent variable.
The Fisher information τ (computed from recent observations) supplies a precision
matrix that plays the role of Σ in the variational update.  Consequently the conductance
mean is updated by  

 g←g+η τ · (o−g),

where o is the measured flux on that edge.  After the update the whole flux
distribution of the network is evaluated against a reference pattern using a
Structural Similarity Index (SSIM), the performance metric from Algorithm A.

The code below implements this fusion, exposing three core functions:
 1. compute_fisher_information – converts raw observations into precisions.
 2. variational_conductance_update – Bayesian‑variational update of edge conductances.
 3. evaluate_network_ssim – SSIM comparison of current and reference flux vectors.
"""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class Edge:
    """Represents an undirected edge in the physarum network."""
    id: str
    node_a: int
    node_b: int
    length: float
    conductance: float  # mean conductance (latent variable)

@dataclass
class Graph:
    """Simple adjacency list representation."""
    nodes: List[int]
    edges: List[Edge]

# ----------------------------------------------------------------------
# Core physics (Algorithm B)
# ----------------------------------------------------------------------
def flux(conductance: float, length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    """Compute flux through an edge given conductance, length and node pressures."""
    if length <= 0:
        raise ValueError('edge length must be positive')
    return conductance / max(length, eps) * (pressure_a - pressure_b)

def compute_flux_vector(graph: Graph, pressures: Dict[int, float]) -> np.ndarray:
    """Return a vector of fluxes for all edges ordered as in graph.edges."""
    fluxes = []
    for e in graph.edges:
        p_a = pressures[e.node_a]
        p_b = pressures[e.node_b]
        fluxes.append(flux(e.conductance, e.length, p_a, p_b))
    return np.array(fluxes, dtype=float)

# ----------------------------------------------------------------------
# Fisher information (Algorithm B)
# ----------------------------------------------------------------------
def compute_fisher_information(observations: np.ndarray) -> np.ndarray:
    """
    Approximate Fisher information for a set of independent Gaussian observations.
    For a Gaussian N(μ,σ²) the Fisher information w.r.t μ is 1/σ².
    Here we estimate σ² from the sample variance and return a diagonal precision matrix.
    """
    if observations.ndim != 1:
        raise ValueError("observations must be a 1‑D array")
    var_est = np.var(observations, ddof=1) + 1e-12  # avoid division by zero
    precision = 1.0 / var_est
    # Return a scalar precision; callers will broadcast as needed.
    return np.full(observations.shape, precision, dtype=float)

# ----------------------------------------------------------------------
# Variational update (Algorithm A + B)
# ----------------------------------------------------------------------
def variational_conductance_update(
    graph: Graph,
    observed_flux: np.ndarray,
    fisher_prec: np.ndarray,
    learning_rate: float = 0.1,
) -> Graph:
    """
    Perform a Bayesian variational update of each edge's conductance.
    The update rule mirrors Algorithm A's mean update but substitutes the
    covariance matrix with the Fisher precision derived from observations.
    """
    if len(observed_flux) != len(graph.edges):
        raise ValueError("Observation length must match number of edges")
    updated_edges = []
    for idx, edge in enumerate(graph.edges):
        # Predicted flux based on current conductance and a provisional pressure field.
        # For the update we only need the discrepancy (observed - predicted).
        # Here we approximate predicted flux with the current conductance * unit pressure diff.
        # This keeps the update self‑contained without solving a global pressure system.
        predicted = edge.conductance  # unit pressure difference assumed
        delta = observed_flux[idx] - predicted
        # Variational update: g_new = g + η * τ * delta
        g_new = edge.conductance + learning_rate * fisher_prec[idx] * delta
        # Enforce positivity of conductance
        g_new = max(g_new, 1e-6)
        updated_edges.append(
            Edge(
                id=edge.id,
                node_a=edge.node_a,
                node_b=edge.node_b,
                length=edge.length,
                conductance=g_new,
            )
        )
    return Graph(nodes=graph.nodes.copy(), edges=updated_edges)

# ----------------------------------------------------------------------
# SSIM evaluation (Algorithm A)
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """
    Simplified Structural Similarity Index for 1‑D signals.
    Returns a value in [-1, 1] where 1 indicates perfect similarity.
    """
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    cov_xy = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator if denominator != 0 else 0.0

def evaluate_network_ssim(graph: Graph, pressures: Dict[int, float], reference_flux: np.ndarray) -> float:
    """
    Compute the SSIM between the current network flux vector and a reference vector.
    """
    current_flux = compute_flux_vector(graph, pressures)
    return ssim(current_flux, reference_flux)

# ----------------------------------------------------------------------
# Hybrid step tying everything together
# ----------------------------------------------------------------------
def hybrid_step(
    graph: Graph,
    pressures: Dict[int, float],
    raw_observations: np.ndarray,
    reference_flux: np.ndarray,
    learning_rate: float = 0.1,
) -> Tuple[Graph, float]:
    """
    One iteration of the hybrid algorithm:
      1. Convert raw flux observations into Fisher precisions.
      2. Update edge conductances via the variational rule.
      3. Evaluate the updated network against the reference using SSIM.
    Returns the updated graph and the SSIM score.
    """
    # 1. Fisher information from observations
    fisher_prec = compute_fisher_information(raw_observations)

    # 2. Conductance update
    updated_graph = variational_conductance_update(
        graph,
        observed_flux=raw_observations,
        fisher_prec=fisher_prec,
        learning_rate=learning_rate,
    )

    # 3. Performance evaluation
    ssim_score = evaluate_network_ssim(updated_graph, pressures, reference_flux)

    return updated_graph, ssim_score

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny triangular network (3 nodes, 3 edges)
    nodes = [0, 1, 2]
    edges = [
        Edge(id="e01", node_a=0, node_b=1, length=1.0, conductance=0.5),
        Edge(id="e12", node_a=1, node_b=2, length=1.2, conductance=0.5),
        Edge(id="e20", node_a=2, node_b=0, length=0.9, conductance=0.5),
    ]
    graph = Graph(nodes=nodes, edges=edges)

    # Assign arbitrary pressures (higher at node 0)
    pressures = {0: 1.0, 1: 0.5, 2: 0.0}

    # Simulate true fluxes using the current graph (these act as “observations”)
    true_flux = compute_flux_vector(graph, pressures)

    # Add Gaussian noise to create raw observations
    noise = np.random.normal(loc=0.0, scale=0.05, size=true_flux.shape)
    raw_obs = true_flux + noise

    # Reference flux (e.g., ideal flux pattern we would like to achieve)
    reference_flux = true_flux  # in this simple test we consider the noise‑free flux as reference

    # Run a single hybrid iteration
    updated_graph, score = hybrid_step(
        graph,
        pressures,
        raw_observations=raw_obs,
        reference_flux=reference_flux,
        learning_rate=0.2,
    )

    # Output results
    print("Initial conductances:", [e.conductance for e in graph.edges])
    print("Updated conductances:", [e.conductance for e in updated_graph.edges])
    print("SSIM score after update:", score)