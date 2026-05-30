# DARWIN HAMMER — match 3438, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_distri_m1385_s3.py (gen4)
# born: 2026-05-29T23:50:15Z

import numpy as np
from typing import Dict, FrozenSet, Tuple, Iterable

# ----------------------------------------------------------------------
# Clifford algebra utilities (Parent A) – corrected and extended
# ----------------------------------------------------------------------
def _canonical_blade(indices: Iterable[int]) -> Tuple[FrozenSet[int], int]:
    """
    Return a sorted, duplicate‑free blade (as frozenset) and the sign produced
    by reordering and cancelling equal indices (e_i*e_i = 1).
    """
    counts: Dict[int, int] = {}
    for i in indices:
        counts[i] = counts.get(i, 0) + 1
    # Cancel even occurrences
    remaining = [i for i, c in counts.items() if c % 2 == 1]
    # The sign is (-1)^{# swaps needed to sort the remaining list}
    sign = 1
    for i in range(len(remaining)):
        for j in range(i + 1, len(remaining)):
            if remaining[i] > remaining[j]:
                sign *= -1
    return frozenset(sorted(remaining)), sign


def _multiply_blades(a: FrozenSet[int], b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two basis blades."""
    combined = list(a) + list(b)
    return _canonical_blade(combined)


def geometric_product(
    a: Dict[FrozenSet[int], float],
    b: Dict[FrozenSet[int], float],
) -> Dict[FrozenSet[int], float]:
    """
    Full Clifford geometric product of two multivectors.
    Returns a dict mapping blades to signed scalar coefficients.
    """
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in a.items():
        for blade_b, coeff_b in b.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
    # Prune near‑zero entries
    return {b: c for b, c in result.items() if abs(c) > 1e-12}


# ----------------------------------------------------------------------
# Linear adaptation utilities (Parent A)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d_out = d_in if d_out is None else d_out
    return rng.standard_normal((d_out, d_in), dtype=float) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    pred = W @ x
    t = x if target is None else target
    resid = pred - t
    return float(resid @ resid)


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    pred = W @ x
    t = x if target is None else target
    resid = pred - t
    return 2.0 * np.outer(resid, x)


# ----------------------------------------------------------------------
# SSIM similarity (Parent A) – kept as a scalar likelihood
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    mu_x, mu_y = np.mean(x), np.mean(y)
    sigma_x, sigma_y = np.std(x), np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return float(num / den)


# ----------------------------------------------------------------------
# Bayesian utilities (Parent B) – robust probability handling
# ----------------------------------------------------------------------
def _positive_normalise(d: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """Force non‑negative entries and L1‑normalise."""
    eps = 1e-12
    pos = {k: max(v, 0.0) + eps for k, v in d.items()}
    total = sum(pos.values())
    return {k: v / total for k, v in pos.items()}


def bayesian_update(
    prior: Dict[FrozenSet[int], float],
    likelihood: Dict[FrozenSet[int], float],
    scalar_factor: float = 1.0,
) -> Dict[FrozenSet[int], float]:
    """
    Proper Bayesian update:
        posterior ∝ prior ⊙ likelihood ⊙ scalar_factor
    where ⊙ denotes element‑wise multiplication (missing blades get zero).
    The result is forced to be a valid probability mass function.
    """
    # Union of support to avoid discarding blades that appear only in likelihood
    all_blades = set(prior) | set(likelihood)
    unnorm: Dict[FrozenSet[int], float] = {}
    for b in all_blades:
        p = prior.get(b, 0.0)
        l = likelihood.get(b, 0.0)
        unnorm[b] = p * l * scalar_factor
    return _positive_normalise(unnorm)


def vector_to_multivector(v: np.ndarray) -> Dict[FrozenSet[int], float]:
    """Encode a real vector as a grade‑1 multivector."""
    return {frozenset({i}): float(c) for i, c in enumerate(v)}


def multivector_to_vector(mv: Dict[FrozenSet[int], float], dim: int) -> np.ndarray:
    """Project a multivector onto its grade‑1 part."""
    vec = np.zeros(dim, dtype=float)
    for blade, coeff in mv.items():
        if len(blade) == 1:
            idx = next(iter(blade))
            if idx < dim:
                vec[idx] = coeff
    return vec


# ----------------------------------------------------------------------
# Deeper hybrid operation – improved mathematical coupling
# ----------------------------------------------------------------------
def _likelihood_from_product(prod: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """
    Convert signed geometric‑product coefficients into a proper likelihood:
    take absolute values (magnitude of contribution) and renormalise.
    """
    abs_vals = {b: abs(c) for b, c in prod.items()}
    return _positive_normalise(abs_vals)


def hybrid_step_improved(
    W: np.ndarray,
    x: np.ndarray,
    prior_mv: Dict[FrozenSet[int], float],
    target: np.ndarray | None = None,
    lr: float = 0.1,
    ssim_beta: float = 1.0,
) -> Tuple[np.ndarray, Dict[FrozenSet[int], float]]:
    """
    One iteration of the enhanced hybrid algorithm.

    1. Linear prediction `p = W @ x`.
    2. Encode `p` as a multivector and compute the geometric product with the prior.
       Its magnitude yields a proper likelihood distribution.
    3. Compute SSIM between prediction and reference (target or x) and turn it
       into an exponentiated scalar factor `sim^β` that modulates the whole update.
    4. Perform a Bayesian update producing a posterior multivector.
    5. Use the *expected* posterior confidence (mean coefficient) to scale the
       Test‑Time Training gradient, and also blend the prior towards the likelihood
       using the same confidence (a natural‑gradient‑like step on the probability
       simplex).
    Returns the updated weight matrix and the new posterior multivector.
    """
    # 1. Linear prediction
    pred = W @ x

    # 2. Likelihood from geometric product
    pred_mv = vector_to_multivector(pred)
    prod_mv = geometric_product(prior_mv, pred_mv)
    likelihood_mv = _likelihood_from_product(prod_mv)

    # 3. SSIM‑derived scalar factor (clamped to [0,1] then exponentiated)
    ref = x if target is None else target
    sim = ssim(pred, ref)
    sim = max(0.0, min(1.0, sim))
    scalar_factor = sim ** ssim_beta

    # 4. Bayesian posterior
    posterior_mv = bayesian_update(prior_mv, likelihood_mv, scalar_factor)

    # 5. Confidence‑driven adaptation
    confidence = float(np.mean(list(posterior_mv.values())))  # in (0,1]

    # Gradient step for W
    grad_W = ttt_grad(W, x, target)
    W_new = W - lr * confidence * grad_W

    # Optional: move the prior a little towards the likelihood using the same confidence
    # (convex combination on the simplex)
    blended_prior = {
        b: (1 - confidence) * prior_mv.get(b, 0.0) + confidence * likelihood_mv.get(b, 0.0)
        for b in set(prior_mv) | set(likelihood_mv)
    }
    prior_mv_updated = _positive_normalise(blended_prior)

    return W_new, posterior_mv


# ----------------------------------------------------------------------
# Example usage (for testing purposes only – can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dim = 5
    W = init_ttt(dim)
    x = np.random.rand(dim)
    prior = vector_to_multivector(np.ones(dim) / dim)  # uniform grade‑1 prior

    for _ in range(10):
        W, posterior = hybrid_step_improved(W, x, prior, lr=0.05, ssim_beta=2.0)
        prior = posterior  # feed posterior back as the next prior
    print("Final weight matrix:\n", W)
    print("Posterior mass (first 5 blades):")
    for b, v in list(posterior.items())[:5]:
        print(b, v)