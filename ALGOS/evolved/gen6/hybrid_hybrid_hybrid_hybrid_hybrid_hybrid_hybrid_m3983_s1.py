# DARWIN HAMMER — match 3983, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_model__hybrid_hybrid_hybrid_m1206_s6.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2132_s1.py (gen5)
# born: 2026-05-29T23:52:58Z

"""Hybrid Algorithm combining GPU‑VRAM SSIM/MinHash similarity (Parent A) with
Sheaf‑guided pheromone Bayesian updates and Fisher information (Parent B).

Mathematical bridge:
- Parent A provides two similarity measures between the current GPU‑memory
  signal *x* and a model signature *y*: SSIM(x,y) and a MinHash‑based Jaccard
  estimate Ĵ(x,y).
- Parent B supplies a *Sheaf* whose node sections σₙ ∈ [0,1] are used as raw
  weights for the two similarity components:
      αₙ = σₙ ,   βₙ = 1‑σₙ .
- The entropy H of the pheromone distribution p over the model pool modulates
  these weights exactly as in Parent A:
      αₙ′ = αₙ·(1‑H/Hₘₐₓ) ,   βₙ′ = βₙ·(H/Hₘₐₓ) .
- The hybrid similarity for model *n* is
      Σₙ = αₙ′·SSIM(x,yₙ) + βₙ′·Ĵ(x,yₙ) .
- Σₙ is then treated as a likelihood in a Bayesian update of the pheromone
  vector.  The Fisher information I(p) = diag(1/p) (up to a constant) is used
  to sharpen the posterior, yielding a unified decision rule for model‑pool
  management.
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Helper utilities (shared)
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _rel(path: pathlib.Path | str) -> str:
    """Return a path relative to the repository root (best‑effort)."""
    try:
        root = pathlib.Path(__file__).resolve().parents[2]
        return str(pathlib.Path(path).resolve().relative_to(root))
    except Exception:
        return str(path)

# ----------------------------------------------------------------------
# Parent A – GPU‑VRAM signal & similarity
# ----------------------------------------------------------------------
def generate_gpu_memory_signal(
    length: int = 256,
    base_mb: int = 4096,
    variance_mb: int = 512,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a 1‑D GPU‑memory usage signal as a noisy sinusoid."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * math.pi, length)
    signal = base_mb + variance_mb * np.sin(t) + rng.normal(0, variance_mb * 0.1, length)
    return signal.astype(np.float32)


def compute_ssim(x: np.ndarray, y: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """Simplified SSIM for 1‑D signals."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    cov = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * cov + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)


def minhash_signature(data: np.ndarray, num_perm: int = 128, seed: int | None = None) -> np.ndarray:
    """Very lightweight MinHash: treat each element as an integer and keep minima."""
    rng = np.random.default_rng(seed)
    max_hash = 2 ** 32 - 1
    signatures = np.full(num_perm, max_hash, dtype=np.uint32)
    for i in range(num_perm):
        a = rng.integers(1, max_hash, dtype=np.uint64)
        b = rng.integers(0, max_hash, dtype=np.uint64)
        hashed = (a * data.astype(np.uint64) + b) % max_hash
        signatures[i] = np.min(hashed)
    return signatures


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    assert sig_a.shape == sig_b.shape
    return float(np.mean(sig_a == sig_b))


def shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Entropy H(p) = -∑ p log p."""
    p = p / (p.sum() + eps)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


# ----------------------------------------------------------------------
# Parent B – Sheaf, pheromones, Bayesian/Fisher mechanics
# ----------------------------------------------------------------------
class Sheaf:
    """Simple sheaf storing sections (weights) on nodes and linear restrictions on edges."""
    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)
        self.edges = list(edge_list)
        self._restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[int, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[int, int], src_map: List[float], dst_map: List[float]):
        u, v = edge
        self._restrictions[(u, v)] = (np.array(src_map, dtype=float), np.array(dst_map, dtype=float))

    def set_section(self, node: int, value: List[float]):
        self._sections[node] = np.array(value, dtype=float)

    def get_section(self, node: int) -> np.ndarray:
        return self._sections.get(node, np.array([0.5]))  # default 0.5 if missing


def fisher_information(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Diagonal Fisher information approximation I_i = 1 / p_i."""
    p = np.clip(p, eps, 1.0)
    return 1.0 / p


def bayesian_pheromone_update(
    prior: np.ndarray,
    likelihood: np.ndarray,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Posterior ∝ prior * (likelihood)^temperature, then normalized."""
    unnorm = prior * np.power(likelihood + eps, temperature)
    total = unnorm.sum()
    if total == 0:
        return np.full_like(prior, 1.0 / prior.size)
    return unnorm / total


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_weights_from_sheaf(sheaf: Sheaf, node: int) -> Tuple[float, float]:
    """
    Derive raw α, β from the sheaf section at ``node``.
    The section is assumed to be a single scalar in [0,1];
    α = σ , β = 1‑σ .
    """
    sigma = float(sheaf.get_section(node).mean())
    sigma = np.clip(sigma, 0.0, 1.0)
    return sigma, 1.0 - sigma


def hybrid_similarity(
    signal: np.ndarray,
    model_sig: np.ndarray,
    sheaf: Sheaf,
    node: int,
    pheromone: np.ndarray,
) -> float:
    """
    Compute the hybrid similarity Σ = α′·SSIM + β′·Ĵ,
    where α′,β′ are entropy‑modulated weights derived from the sheaf.
    """
    # 1. Base SSIM and Jaccard
    ssim_val = compute_ssim(signal, model_sig)
    jaccard_val = estimate_jaccard(
        minhash_signature(signal.astype(np.uint32)),
        minhash_signature(model_sig.astype(np.uint32)),
    )

    # 2. Raw weights from sheaf
    alpha, beta = hybrid_weights_from_sheaf(sheaf, node)

    # 3. Entropy modulation
    H = shannon_entropy(pheromone)
    Hmax = math.log(len(pheromone) + 1e-12)
    if Hmax == 0:
        Hmax = 1.0
    alpha_mod = alpha * (1.0 - H / Hmax)
    beta_mod = beta * (H / Hmax)

    # 4. Combine
    sigma = alpha_mod * ssim_val + beta_mod * jaccard_val
    return float(sigma)


def update_model_pool(
    signal: np.ndarray,
    model_signatures: List[np.ndarray],
    sheaf: Sheaf,
    pheromone: np.ndarray,
    temperature: float = 0.7,
) -> np.ndarray:
    """
    Perform a full hybrid step:
    1. Compute Σₙ for each model n.
    2. Use Σ as likelihood to update the pheromone vector via Bayesian rule.
    3. Sharpen posterior with Fisher information.
    Returns the updated pheromone distribution.
    """
    n_models = len(model_signatures)
    assert pheromone.shape == (n_models,)

    # Compute hybrid similarities (likelihoods)
    likelihoods = np.empty(n_models, dtype=float)
    for i, sig in enumerate(model_signatures):
        # Use node index i as the sheaf node identifier
        likelihoods[i] = hybrid_similarity(signal, sig, sheaf, node=i, pheromone=pheromone)

    # Normalize likelihoods to avoid extreme scaling
    if likelihoods.max() > 0:
        likelihoods = likelihoods / likelihoods.max()

    # Bayesian update
    posterior = bayesian_pheromone_update(pheromone, likelihoods, temperature=temperature)

    # Fisher information sharpening
    I = fisher_information(posterior)
    posterior = posterior * (1.0 + 0.1 * I)  # small boost proportional to I
    posterior /= posterior.sum()  # renormalize

    return posterior


# ----------------------------------------------------------------------
# Demonstration / Smoke test
# ----------------------------------------------------------------------
def _demo():
    # 1. Generate a synthetic GPU‑memory signal
    signal = generate_gpu_memory_signal(seed=42)

    # 2. Create a pool of dummy model signatures (noisy copies of the signal)
    n_models = 5
    rng = np.random.default_rng(123)
    model_signatures = [
        signal + rng.normal(0, 200, size=signal.shape).astype(np.float32)
        for _ in range(n_models)
    ]

    # 3. Initialise a sheaf with a section per model node.
    #    Here we simply assign a linearly increasing σ in [0.2,0.8].
    node_dims = {i: 1 for i in range(n_models)}
    edge_list: List[Tuple[int, int]] = []  # not used in this demo
    sheaf = Sheaf(node_dims, edge_list)
    for i in range(n_models):
        sigma = 0.2 + 0.6 * i / (n_models - 1)
        sheaf.set_section(i, [sigma])

    # 4. Uniform pheromone initialization
    pheromone = np.full(n_models, 1.0 / n_models, dtype=float)

    # 5. Run a hybrid update step
    updated_pheromone = update_model_pool(
        signal,
        model_signatures,
        sheaf,
        pheromone,
        temperature=0.8,
    )

    # 6. Print results
    print("Initial pheromone :", pheromone)
    print("Updated pheromone :", updated_pheromone)
    print("Sum (should be 1) :", updated_pheromone.sum())

if __name__ == "__main__":
    _demo()