# DARWIN HAMMER — match 1952, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_model__m860_s1.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_variational_free_ene_m21_s2.py (gen2)
# born: 2026-05-29T23:40:01Z

"""hybrid_hybrid_hoeffding_ternary_vfe.py
Hybrid algorithm merging:

* **Parent A** – probabilistic primitives, Hoeffding bound and tropical algebra.
* **Parent B** – ternary‑router, SSIM‑based observation variance and variational free‑energy.

Mathematical bridge
-------------------
Both parents evaluate a *reconstruction* error.  
Parent B maps an input text *x* to a routed output *y* and derives an observation
noise variance  


σ²_obs = ε + (1 – SSIM(x, y))·R²                (1)


where *R*≈255.  

Parent A supplies a Hoeffding bound  


ε_H = √( r²·log(1/δ) / (2·n) )                (2)


which we use as a confidence radius on the estimated variance (1).  
The bound tightens with the number of characters *n* and is injected into the
free‑energy term.  Acceptance of a belief update (new mean μ′) is then governed
by the annealed acceptance probability of Parent A.

The resulting hybrid performs:

1. **Ternary routing** of the input text.  
2. **SSIM** between input and routed text → similarity *s*.  
3. **Observation variance** σ²_obs from (1) and a Hoeffding‑corrected variance
   σ²_eff = σ²_obs + ε_H².  
4. **Variational free‑energy** F(μ) with μ the current belief (router output
   vector).  
5. **Simulated‑annealing update** of μ using the acceptance rule of Parent A.  
6. **Graph construction** where edges are kept only if the Hoeffding bound on
   pairwise similarity is below a threshold; tropical algebra combines edge
   weights.

The three public functions below illustrate the full pipeline.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Probabilistic primitives (Parent A)
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Geometric cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a bounded random variable in [0, r]."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


# Tropical (max‑plus) algebra
def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition = max."""
    return np.maximum(x, y)


def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication = plus."""
    return np.add(x, y)


def t_polyval(coeffs: List[float], x: np.ndarray) -> np.ndarray:
    """Evaluate a polynomial in tropical algebra."""
    result = np.full_like(x, -np.inf, dtype=float)  # tropical zero (−∞)
    for i, c in enumerate(reversed(coeffs)):
        term = t_mul(np.full_like(x, c, dtype=float), x * i)
        result = t_add(result, term)
    return result


# ----------------------------------------------------------------------
# Ternary router and SSIM (Parent B)
# ----------------------------------------------------------------------
def ternary_route(text: str) -> str:
    """
    Simple deterministic ternary router:
    - '0' : keep character
    - '1' : shift ASCII code +1 (wrap at 255)
    - '2' : shift ASCII code -1 (wrap at 0)
    The choice cycles every three characters.
    """
    out_chars = []
    for i, ch in enumerate(text):
        mode = i % 3
        code = ord(ch)
        if mode == 0:
            new_code = code
        elif mode == 1:
            new_code = (code + 1) % 256
        else:  # mode == 2
            new_code = (code - 1) % 256
        out_chars.append(chr(new_code))
    return ''.join(out_chars)


def _mean_std(arr: np.ndarray) -> Tuple[float, float]:
    """Utility returning mean and unbiased std."""
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def ssim_index(x: np.ndarray, y: np.ndarray, L: int = 255) -> float:
    """
    Simplified SSIM for 1‑D integer signals.
    Uses default constants C1 = (0.01·L)^2, C2 = (0.03·L)^2.
    Returns a value in [0, 1].
    """
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    mu_x, sigma_x = _mean_std(x)
    mu_y, sigma_y = _mean_std(y)
    cov = float(np.cov(x, y, ddof=1)[0, 1]) if x.size > 1 else 0.0

    numerator = (2 * mu_x * mu_y + C1) * (2 * cov + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2)
    return float(numerator / denominator) if denominator != 0 else 1.0


# ----------------------------------------------------------------------
# Variational free energy (Parent B) with Hoeffding correction
# ----------------------------------------------------------------------
def variational_free_energy(
    mu_q: np.ndarray,
    x_obs: np.ndarray,
    sigma_obs_sq: float,
    sigma_prior_sq: float = 1.0,
) -> float:
    """
    Gaussian variational free energy:
        F = 0.5 * [ (x-μ)^T Σ^{-1} (x-μ) + log|Σ| + const ] + KL(q||p)
    For diagonal Σ = σ² I we obtain the scalar form below.
    """
    if sigma_obs_sq <= 0 or sigma_prior_sq <= 0:
        raise ValueError("variances must be positive")

    # Reconstruction term (negative log‑likelihood)
    diff = x_obs - mu_q
    recon = 0.5 * np.sum(diff ** 2) / sigma_obs_sq
    recon += 0.5 * len(x_obs) * math.log(2 * math.pi * sigma_obs_sq)

    # KL divergence between q = N(μ_q, σ²_prior) and p = N(0, σ²_prior)
    kl = 0.5 * (np.sum(mu_q ** 2) / sigma_prior_sq + len(mu_q) * (sigma_prior_sq / sigma_prior_sq - 1 - math.log(sigma_prior_sq / sigma_prior_sq)))
    # The term simplifies to 0.5 * Σ μ_q² / σ²_prior
    kl = 0.5 * np.sum(mu_q ** 2) / sigma_prior_sq

    return recon + kl


def observation_variance_from_ssim(ssim: float, R: int = 255, eps: float = 1e-6) -> float:
    """Equation (1) – map SSIM to observation variance."""
    return eps + (1.0 - ssim) * (R ** 2)


def hoeffding_corrected_variance(
    sigma_obs_sq: float,
    r: float,
    delta: float,
    n: int,
) -> float:
    """
    Apply Hoeffding bound (2) as an additive uncertainty on the variance.
    The bound ε_H is interpreted as a standard deviation; we add its square.
    """
    eps_h = hoeffding_bound(r, delta, n)
    return sigma_obs_sq + eps_h ** 2


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_free_energy_step(
    mu_q: np.ndarray,
    x_obs: np.ndarray,
    temperature: float,
    r: float = 1.0,
    delta: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single Metropolis‑Hastings style update of the belief mean μ_q.
    The proposal adds Gaussian noise whose variance is the Hoeffding‑corrected
    observation variance.  The function returns the new μ_q and the free‑energy
    value after the accepted (or rejected) step.
    """
    n = x_obs.size
    # 1) compute SSIM and base observation variance
    ssim = ssim_index(x_obs, mu_q)
    sigma_obs_sq = observation_variance_from_ssim(ssim)

    # 2) Hoeffding‑corrected variance
    sigma_eff_sq = hoeffding_corrected_variance(sigma_obs_sq, r, delta, n)

    # 3) current free energy
    current_F = variational_free_energy(mu_q, x_obs, sigma_eff_sq)

    # 4) propose a new mean
    proposal = mu_q + np.random.normal(scale=math.sqrt(sigma_eff_sq), size=mu_q.shape)

    # 5) free energy of proposal
    proposal_F = variational_free_energy(proposal, x_obs, sigma_eff_sq)

    # 6) Metropolis acceptance
    delta_E = proposal_F - current_F
    if random.random() < acceptance_probability(delta_E, temperature):
        return proposal, proposal_F
    else:
        return mu_q, current_F


def hybrid_anneal(
    init_mu: np.ndarray,
    x_obs: np.ndarray,
    steps: int = 50,
    t0: float = 1.0,
    alpha: float = 0.93,
) -> Tuple[np.ndarray, List[float]]:
    """
    Simulated annealing over the belief vector using the hybrid free‑energy.
    Returns the final μ and the trajectory of free‑energy values.
    """
    mu = init_mu.copy()
    trajectory = []
    for k in range(steps):
        temp = cooling_temperature(k, t0, alpha)
        mu, F = hybrid_free_energy_step(mu, x_obs, temp)
        trajectory.append(F)
    return mu, trajectory


def hybrid_graph_construction(texts: List[str], delta: float = 0.05) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build a weighted undirected graph where nodes are the input texts.
    Edge weight = tropical addition of pairwise similarity scores.
    An edge is kept only if the Hoeffding bound on the similarity (treated as a
    bounded variable in [0, 1]) is below *delta*.
    Returns an adjacency list: {node: [(neighbor, weight), ...]}.
    """
    n = len(texts)
    adj: Dict[str, List[Tuple[str, float]]] = {txt: [] for txt in texts}
    for i in range(n):
        for j in range(i + 1, n):
            xi = np.frombuffer(texts[i].encode('utf-8'), dtype=np.uint8).astype(float)
            xj = np.frombuffer(texts[j].encode('utf-8'), dtype=np.uint8).astype(float)

            # similarity via SSIM (bounded in [0,1])
            sim = ssim_index(xi, xj)

            # Hoeffding bound on the similarity estimate
            bound = hoeffding_bound(r=1.0, delta=delta, n=min(xi.size, xj.size))

            if bound < delta:  # confidence high enough → keep edge
                # Tropical combination: weight = max(previous, sim)
                weight = sim  # single edge, tropical add later when merging
                adj[texts[i]].append((texts[j], weight))
                adj[texts[j]].append((texts[i], weight))
    # Apply tropical addition on parallel edges (not needed here, but kept for completeness)
    for node, edges in adj.items():
        # Collapse parallel edges by taking max weight per neighbor
        merged: Dict[str, float] = {}
        for nbr, w in edges:
            merged[nbr] = max(merged.get(nbr, -np.inf), w)
        adj[node] = [(nbr, w) for nbr, w in merged.items()]
    return adj


# ----------------------------------------------------------------------
# Demonstration entry point
# ----------------------------------------------------------------------
def _text_to_vector(txt: str) -> np.ndarray:
    """Encode a string as a float vector of its byte values."""
    return np.frombuffer(txt.encode('utf-8'), dtype=np.uint8).astype(float)


def main() -> None:
    # Sample input
    input_text = "The quick brown fox jumps over the lazy dog."
    routed_text = ternary_route(input_text)

    # Encode as vectors
    x_vec = _text_to_vector(input_text)
    mu_vec = _text_to_vector(routed_text)  # initial belief = routed output

    # Annealing on a single observation
    final_mu, free_energy_traj = hybrid_anneal(
        init_mu=mu_vec,
        x_obs=x_vec,
        steps=30,
        t0=1.0,
        alpha=0.90,
    )

    print("=== Hybrid Free‑Energy Annealing ===")
    print(f"Initial free energy: {free_energy_traj[0]:.4f}")
    print(f"Final free energy  : {free_energy_traj[-1]:.4f}")
    print(f"Free‑energy trajectory (last 5): {free_energy_traj[-5:]}")

    # Graph construction from a small corpus
    corpus = [
        input_text,
        routed_text,
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Pack my box with five dozen liquor jugs."
    ]
    graph = hybrid_graph_construction(corpus, delta=0.02)
    print("\n=== Hybrid Graph (Adjacency List) ===")
    for node, edges in graph.items():
        edge_str = ", ".join(f"{nbr[:15]}...:{w:.3f}" for nbr, w in edges)
        print(f"{node[:30]}... -> {edge_str}")

if __name__ == "__main__":
    main()