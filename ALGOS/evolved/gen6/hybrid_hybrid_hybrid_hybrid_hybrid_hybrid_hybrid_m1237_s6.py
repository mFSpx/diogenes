# DARWIN HAMMER — match 1237, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1025_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_percep_hybrid_hard_truth_ma_m741_s1.py (gen5)
# born: 2026-05-29T23:34:42Z

"""Hybrid Algorithm integrating EndpointCircuitBreaker (Decision Hygiene) with
Radial‑Basis Function similarity, Hoeffding bound‑driven updates and
Minimum‑Cost Tree weighting.

Parents:
- **hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1025_s2.py** – provides
  an `EndpointCircuitBreaker` that uses regex‑based decision hygiene to open
  or close a circuit based on the presence of evidence‑related tokens.
- **hybrid_hybrid_hybrid_percep_hybrid_hard_truth_ma_m741_s1.py** – supplies
  radial‑basis function (RBF) similarity, Hoeffding‑bound modulation, and a
  minimum‑cost spanning‑tree (MCST) that is weighted by Bayesian evidence.

Mathematical Bridge:
The bridge is built on the observation that the circuit‑breaker’s *open/close*
state can act as a binary gating variable `g ∈ {0,1}` for the learning update
of the RBF similarity matrix **S**. When `g = 1` (evidence detected) the
gradient of the MCST edge‑weights is diffused with a *liquid time constant*
τ, i.e.


w_{t+1} = w_t + (1 - exp(-Δt/τ)) * ∇_w L


where `L` is the minimum‑cost loss. The Hoeffding bound


ε = sqrt( (R^2 * ln(1/δ)) / (2 n) )


(with range `R = 1` for normalized costs) supplies an adaptive step size
that caps the magnitude of the gradient. This yields a mathematically fused
system where decision hygiene regulates when RBF‑driven graph learning
occurs, and the learning itself respects statistical confidence via the
Hoeffding bound while propagating changes through a liquid‑time diffusion
process.

The module below implements this hybrid, exposing three core functions:
`rbf_similarity`, `hoeffding_bound` and `hybrid_update`. The `HybridCircuitRBF`
class ties the pieces together.
"""

import numpy as np
import re
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Regex feature set (Decision Hygiene)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------


def rbf_similarity(x: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute Gaussian RBF similarity between a vector `x` and an array of
    `centers`. Returns a 1‑D array of similarities.

    similarity_i = exp(-||x - c_i||² / (2 * sigma²))
    """
    diff = centers - x  # shape (m, d)
    sq_norm = np.einsum("ij,ij->i", diff, diff)
    return np.exp(-sq_norm / (2 * sigma ** 2))


def hoeffding_bound(n: int, delta: float = 0.05, R: float = 1.0) -> float:
    """
    Hoeffding bound ε for a bounded random variable with range R.
    Provides a confidence interval width for the empirical mean after n samples.

    ε = sqrt( (R² * ln(1/δ)) / (2n) )
    """
    if n <= 0:
        return float("inf")
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2.0 * n))


def liquid_diffusion_update(
    w: np.ndarray, grad: np.ndarray, dt: float, tau: float
) -> np.ndarray:
    """
    Perform a liquid‑time constant diffusion update.
    w_{t+1} = w_t + (1 - exp(-dt/τ)) * grad
    """
    factor = 1.0 - math.exp(-dt / tau)
    return w + factor * grad


def prim_mst(cost_matrix: np.ndarray) -> np.ndarray:
    """
    Compute a Minimum‑Cost Spanning Tree (MST) using Prim's algorithm.
    Returns an adjacency matrix `mst` where mst[i, j] = cost if edge (i, j)
    belongs to the tree, otherwise 0.
    """
    n = cost_matrix.shape[0]
    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    mst = np.zeros_like(cost_matrix)

    while not selected.all():
        # mask edges where one vertex is selected and the other is not
        mask = np.outer(selected, ~selected) | np.outer(~selected, selected)
        # set impossible edges to large number
        candidate_costs = np.where(mask, cost_matrix, np.inf)
        i, j = np.unravel_index(np.argmin(candidate_costs), cost_matrix.shape)
        mst[i, j] = mst[j, i] = cost_matrix[i, j]
        selected[i] = selected[j] = True

    return mst


def bayesian_edge_update(
    prior: np.ndarray, likelihood: np.ndarray, evidence: np.ndarray
) -> np.ndarray:
    """
    Simple Bayesian update for edge weights.
    posterior ∝ prior * likelihood / evidence
    All inputs must be positive and of the same shape.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        posterior = prior * likelihood / np.where(evidence == 0, 1, evidence)
        posterior = np.nan_to_num(posterior, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalise to keep weights in a comparable range
    total = posterior.sum()
    if total > 0:
        posterior /= total
    return posterior


# ----------------------------------------------------------------------
# Core Hybrid Class
# ----------------------------------------------------------------------


class HybridCircuitRBF:
    """
    Combines an EndpointCircuitBreaker with an RBF‑driven graph learner.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        sigma: float = 1.0,
        tau: float = 5.0,
        dt: float = 1.0,
        delta: float = 0.05,
    ):
        # Circuit‑breaker state
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

        # RBF / graph parameters
        self.sigma = sigma
        self.tau = tau
        self.dt = dt
        self.delta = delta

        # Initialise a small random graph (undirected) with 5 nodes
        self.n_nodes = 5
        rng = np.random.default_rng()
        self.centers = rng.normal(size=(self.n_nodes, 3))  # 3‑D feature centres
        self.edge_costs = np.abs(rng.normal(size=(self.n_nodes, self.n_nodes)))
        self.edge_costs = (self.edge_costs + self.edge_costs.T) / 2  # symmetric
        np.fill_diagonal(self.edge_costs, 0.0)

        # Prior for Bayesian edge updates (uniform)
        self.edge_prior = np.full_like(self.edge_costs, 1.0 / (self.n_nodes ** 2))

        # Sample counter for Hoeffding bound
        self.sample_count = 0

    # ------------------------------------------------------------------
    # Circuit‑Breaker logic
    # ------------------------------------------------------------------
    def _check_evidence(self, text: str) -> bool:
        """Return True if evidence‑related regex matches."""
        return bool(EVIDENCE_RE.search(text))

    def _update_circuit(self, evidence_present: bool) -> None:
        """Update failure count and open/close state."""
        if evidence_present:
            self.failures = 0
            self.open = False
        else:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.open = True

    # ------------------------------------------------------------------
    # Hybrid learning step
    # ------------------------------------------------------------------
    def step(self, text: str, feature_vec: np.ndarray) -> None:
        """
        Process a single observation:
        - Apply decision‑hygiene regex to possibly open/close the circuit.
        - If the circuit is closed (open == False), compute RBF similarity,
          update edge costs via liquid diffusion, and refresh the MCST
          using Bayesian evidence.
        """
        evidence = self._check_evidence(text)
        self._update_circuit(evidence)

        # Record event timestamp (simplified)
        self.last_event_at = "step"

        if self.open:
            # Circuit open → skip learning this round
            return

        # ------------------------------
        # 1. RBF similarity to all centres
        # ------------------------------
        sim = rbf_similarity(feature_vec, self.centers, self.sigma)  # shape (n_nodes,)

        # ------------------------------
        # 2. Gradient for edge costs
        #    We use (1 - similarity) as a proxy loss for each node pair.
        # ------------------------------
        loss_matrix = np.abs(1.0 - np.outer(sim, sim))
        grad = loss_matrix - self.edge_costs  # direction to reduce loss

        # ------------------------------
        # 3. Hoeffding‑bound limited step size
        # ------------------------------
        self.sample_count += 1
        epsilon = hoeffding_bound(self.sample_count, delta=self.delta)
        grad = np.clip(grad, -epsilon, epsilon)

        # ------------------------------
        # 4. Liquid‑time diffusion update of edge costs
        # ------------------------------
        self.edge_costs = liquid_diffusion_update(
            self.edge_costs, grad, dt=self.dt, tau=self.tau
        )
        # Keep costs non‑negative and symmetric
        self.edge_costs = np.maximum(self.edge_costs, 0.0)
        self.edge_costs = (self.edge_costs + self.edge_costs.T) / 2.0

        # ------------------------------
        # 5. Minimum‑Cost Spanning Tree (MST) extraction
        # ------------------------------
        mst = prim_mst(self.edge_costs)

        # ------------------------------
        # 6. Bayesian update of edge priors using MST as evidence
        # ------------------------------
        likelihood = np.where(mst > 0, 1.0, 0.1)  # edges in MST are more likely
        self.edge_prior = bayesian_edge_update(self.edge_prior, likelihood, mst)

    # ------------------------------------------------------------------
    # Inspection utilities
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """Return a snapshot of the internal state for debugging / testing."""
        return {
            "circuit_open": self.open,
            "failures": self.failures,
            "edge_costs": self.edge_costs.copy(),
            "edge_prior": self.edge_prior.copy(),
            "sample_count": self.sample_count,
        }


# ----------------------------------------------------------------------
# Demonstration Functions (required ≥3)
# ----------------------------------------------------------------------


def demo_rbf_and_circuit():
    """Showcase how evidence detection gates the RBF update."""
    model = HybridCircuitRBF(failure_threshold=2, sigma=0.8)
    # Observation without evidence → circuit will eventually open
    for i in range(3):
        model.step("Just a regular update", np.array([0.1, -0.2, 0.3]))
    # Observation with evidence → circuit closes and learning resumes
    model.step(
        "Please provide evidence and verify the source",
        np.array([0.5, 0.0, -0.1]),
    )
    return model.get_state()


def demo_hoeffding_and_diffusion():
    """Run a sequence of steps and expose the Hoeffding‑limited gradient."""
    model = HybridCircuitRBF(sigma=1.2, tau=3.0, dt=0.5, delta=0.01)
    for i in range(5):
        vec = np.random.randn(3)
        text = "evidence" if i % 2 == 0 else "no proof here"
        model.step(text, vec)
    return model.get_state()


def demo_mst_and_bayesian():
    """Extract the MST after several updates and display posterior edge weights."""
    model = HybridCircuitRBF()
    for _ in range(4):
        model.step(
            "verified source attached",
            np.random.uniform(-1, 1, size=3),
        )
    mst = prim_mst(model.edge_costs)
    posterior = model.edge_prior
    return {"mst": mst, "posterior": posterior}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Run each demo and print a concise summary to verify execution.
    state_a = demo_rbf_and_circuit()
    print("Demo 1 – Circuit state:", state_a["circuit_open"], "Failures:", state_a["failures"])

    state_b = demo_hoeffding_and_diffusion()
    print("Demo 2 – Sample count:", state_b["sample_count"], "First edge cost:", state_b["edge_costs"][0, 1])

    results_c = demo_mst_and_bayesian()
    mst_nonzero = np.count_nonzero(results_c["mst"])
    print("Demo 3 – MST edges:", mst_nonzero, "Posterior sum:", results_c["posterior"].sum())