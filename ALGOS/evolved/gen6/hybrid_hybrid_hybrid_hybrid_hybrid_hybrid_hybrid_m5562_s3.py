# DARWIN HAMMER — match 5562, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_percyp_hybrid_hybrid_hybrid_m1870_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s5.py (gen5)
# born: 2026-05-30T00:03:03Z

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Symmetric sphericity index based on the volume‑to‑surface‑area ratio.

    This formulation is invariant to permutations of (length, width, height) and
    yields a value in (0, 1] for positive dimensions.
    """
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    volume = length * width * height
    surface = 2 * (length * width + width * height + height * length)
    return (math.pi ** (1.0 / 3.0)) * (6 * volume) ** (2.0 / 3.0) / surface


def morphology_from_dimensions(
    length: float, width: float, height: float, mass: float
) -> Morphology:
    """Factory for a Morphology instance."""
    return Morphology(length=length, width=width, height=height, mass=mass)


# ----------------------------------------------------------------------
# Parent B – Stylometry & TTT utilities
# ----------------------------------------------------------------------


FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent wasnt werent".split()
    ),
}


def _tokenise(text: str) -> List[str]:
    """Very small tokeniser – split on whitespace and strip punctuation."""
    return [
        t.strip(".,;:!?\"'()[]{}").lower()
        for t in text.split()
        if t.strip(".,;:!?\"'()[]{}")
    ]


def stylometry_vector(text: str) -> np.ndarray:
    """
    Produce a stylometry feature vector f ∈ ℝ^k where each component counts the
    occurrences of a lexical category defined in FUNCTION_CATS.
    """
    tokens = _tokenise(text)
    counts = np.zeros(len(FUNCTION_CATS), dtype=float)
    for idx, cat in enumerate(FUNCTION_CATS):
        counts[idx] = sum(1 for t in tokens if t in FUNCTION_CATS[cat])
    norm = np.linalg.norm(counts)
    return counts / norm if norm > 0 else counts


def ttt_transform(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear TTT transformation y = W·x."""
    if W.shape[1] != x.shape[0]:
        raise ValueError(f"Incompatible shapes: W{W.shape} @ x{x.shape}")
    return W @ x


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Robust similarity measure defined for vectors of any dimension."""
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return (a @ b) / denom if denom != 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid core – deeper and mathematically sound integration
# ----------------------------------------------------------------------


def _laplace_mechanism(value: float, sensitivity: float, epsilon: float, rng: np.random.Generator) -> float:
    """Add Laplace noise calibrated to the L1‑sensitivity of the query."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive for differential privacy")
    scale = sensitivity / epsilon
    noise = rng.laplace(0.0, scale)
    return value + noise


def _bayesian_posterior(
    prior: Dict[int, float],
    y: np.ndarray,
    class_prototypes: Dict[int, np.ndarray],
    sigma_likelihood: float,
) -> Tuple[int, float]:
    """
    Compute posterior probabilities p(c|y) ∝ π_c · 𝒩(y; μ_c, σ²I)
    and return the MAP class together with its posterior probability.
    """
    if not class_prototypes:
        raise ValueError("class_prototypes must be provided for data‑driven inference")
    log_posts = {}
    for c, pi_c in prior.items():
        mu_c = class_prototypes.get(c)
        if mu_c is None:
            raise KeyError(f"Missing prototype for class {c}")
        # Gaussian log‑likelihood (ignoring constant terms)
        diff = y - mu_c
        ll = -0.5 * np.sum(diff ** 2) / (sigma_likelihood ** 2)
        log_posts[c] = math.log(pi_c) + ll
    map_class = max(log_posts, key=log_posts.get)
    # Convert back to probability for reporting (softmax over log_posts)
    max_log = max(log_posts.values())
    exp_shifted = {c: math.exp(v - max_log) for c, v in log_posts.items()}
    total = sum(exp_shifted.values())
    posterior = exp_shifted[map_class] / total if total > 0 else 0.0
    return map_class, posterior


def morphology_weighted_hybrid_cost(
    text: str,
    morphology: Morphology,
    W: np.ndarray,
    prior: Dict[int, float],
    class_prototypes: Dict[int, np.ndarray],
    lam: float,
    epsilon: float,
    sigma_likelihood: float = 1.0,
    *,
    rng_seed: int | None = None,
) -> Tuple[float, float, int, float]:
    """
    Deeply integrated hybrid pipeline.

    Steps
    -----
    1. Extract stylometry vector f.
    2. Compute symmetric sphericity index σ and weight f → f̃ = σ·f.
    3. Apply TTT linear map y = W·f̃.
    4. Infer class via a proper Bayesian posterior using Gaussian likelihoods
       centred at class‑specific prototypes in the transformed space.
    5. Compute Bayesian cost C = -log π̂_c + λ·‖y‖₁ where π̂_c is the posterior
       probability of the MAP class.
    6. Add Laplace noise calibrated to the true L1‑sensitivity of C.
    7. Measure fidelity between the original stylometry space and the
       reconstructed representation x̂ = W⁺·y using cosine similarity.

    Returns
    -------
    (private_cost, fidelity, predicted_class, posterior_probability)
    """
    rng = np.random.default_rng(rng_seed)

    # 1. Stylometry extraction
    f = stylometry_vector(text)                     # shape (k,)

    # 2. Morphology weighting (σ ∈ (0,1])
    sigma = sphericity_index(morphology.length, morphology.width, morphology.height)
    f_weighted = sigma * f

    # 3. TTT transformation
    y = ttt_transform(W, f_weighted)               # shape (m,)

    # 4. Bayesian posterior inference
    pred_class, posterior = _bayesian_posterior(
        prior, y, class_prototypes, sigma_likelihood
    )

    # 5. Bayesian cost (using posterior as data‑aware prior)
    cost = -math.log(posterior) + lam * np.linalg.norm(y, ord=1)

    # 6. Sensitivity analysis:
    #    ΔC ≤ λ·‖W‖₁·Δ‖f‖₁·σ_max .
    #    ‖f‖₁ ≤ √k because f is L2‑normalised and non‑negative.
    k = f.shape[0]
    sigma_max = 1.0  # σ is bounded by 1 for the chosen definition
    l1_f_bound = math.sqrt(k)
    sensitivity = lam * np.linalg.norm(W, ord=1) * l1_f_bound * sigma_max

    private_cost = _laplace_mechanism(cost, sensitivity, epsilon, rng)

    # 7. Fidelity metric – map back to original space and compare with f
    W_pinv = np.linalg.pinv(W)                     # shape (n, m)
    f_recon = W_pinv @ y                           # shape (k,)
    fidelity = cosine_similarity(f, f_recon)

    return private_cost, fidelity, pred_class, posterior


def generate_random_weight_matrix(out_dim: int, in_dim: int, seed: int | None = None) -> np.ndarray:
    """Create a reproducible random matrix W for the TTT transform."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(out_dim, in_dim))


def example_prior(num_classes: int, seed: int | None = None) -> Dict[int, float]:
    """Generate a Dirichlet‑distributed prior over `num_classes`."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(num_classes))
    return {i: float(p) for i, p in enumerate(probs)}


def example_class_prototypes(num_classes: int, out_dim: int, seed: int | None = None) -> Dict[int, np.ndarray]:
    """
    Produce synthetic class prototypes μ_c ∈ ℝ^{out_dim} for the transformed space.
    They are drawn from a standard normal distribution and normalised to unit L2.
    """
    rng = np.random.default_rng(seed)
    prototypes = {}
    for c in range(num_classes):
        vec = rng.standard_normal(out_dim)
        vec /= np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else 1.0
        prototypes[c] = vec
    return prototypes


# ----------------------------------------------------------------------
# Simple smoke test demonstrating end‑to‑end execution
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Synthetic data
    txt = "I think therefore I am. The quick brown fox jumps over the lazy dog."
    morph = morphology_from_dimensions(2.0, 1.5, 1.0, mass=3.2)

    W = generate_random_weight_matrix(out_dim=8, in_dim=len(FUNCTION_CATS), seed=42)
    prior = example_prior(num_classes=3, seed=7)
    prototypes = example_class_prototypes(num_classes=3, out_dim=8, seed=7)

    private_cost, fidelity, pred_class, post = morphology_weighted_hybrid_cost(
        text=txt,
        morphology=morph,
        W=W,
        prior=prior,
        class_prototypes=prototypes,
        lam=0.3,
        epsilon=1.0,
        sigma_likelihood=0.8,
        rng_seed=123,
    )

    print(f"Private cost: {private_cost:.4f}")
    print(f"Fidelity (cosine similarity): {fidelity:.4f}")
    print(f"Predicted class: {pred_class}")
    print(f"Posterior probability of prediction: {post:.4f}")