# DARWIN HAMMER — match 4926, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_bayes_claim_k_m261_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_regret_m2442_s0.py (gen6)
# born: 2026-05-29T23:58:54Z

import numpy as np
import math
from collections import Counter
from typing import List, Dict, Any, Tuple


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Return the value of a normalized Gaussian beam."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Compute the (scalar) Fisher information for a single Gaussian beam.
    The formula reduces to (dI/dθ)² / I where I is the intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def shannon_entropy(counts: Dict[Any, int]) -> float:
    """
    Compute Shannon entropy (base‑2) from a dictionary of counts.
    Returns 0 for empty or single‑element collections.
    """
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def _normalize_weights(weights: List[float]) -> List[float]:
    """Safely normalize a list of non‑negative weights."""
    total = sum(weights)
    if total == 0.0:
        # Uniform distribution when no information is present
        return [1.0 / len(weights) for _ in weights] if weights else []
    return [w / total for w in weights]


def compute_regret_weighted_strategy(
    actions: List[Dict[str, float]],
    fisher_scores: List[float],
) -> List[float]:
    """
    Produce regret‑weighted scores for a set of actions.
    The Fisher scores act as confidence weights; the final regret weight
    is the product of the normalized Fisher weight and the raw action value.
    An entropy‑based damping term is added to reduce over‑confidence
    when the action values are highly peaked.
    """
    if len(actions) != len(fisher_scores):
        raise ValueError("actions and fisher_scores must have the same length")

    action_vals = np.array([a["value"] for a in actions], dtype=float)
    fisher_weights = np.array(_normalize_weights(fisher_scores), dtype=float)

    # Base regret weight (confidence × value)
    base_weights = fisher_weights * action_vals

    # Entropy damping: the more certain the action distribution,
    # the stronger the damping (1 - entropy / log2(N)).
    value_counts = Counter(action_vals.round(decimals=6).tolist())
    entropy = shannon_entropy(value_counts)
    N = len(actions)
    damping = 1.0 - (entropy / math.log2(N)) if N > 1 else 1.0

    return (base_weights * damping).tolist()


def update_hypothesis(
    hypothesis: Dict[str, Any],
    evidence: Dict[str, Any],
    likelihood_ratio: float,
    fisher_score_val: float,
) -> Dict[str, Any]:
    """
    Bayesian update of a hypothesis with an additional evidence item.
    The Fisher score modulates the update strength via a logistic scaling
    factor that keeps the posterior within (0,1) and respects information
    content of the new evidence.
    """
    if likelihood_ratio < 0.0:
        raise ValueError("likelihood_ratio must be non‑negative")

    # Clamp prior to a valid probability interval
    prior = max(0.0, min(1.0, hypothesis.get("posterior", hypothesis.get("prior", 0.5))))

    # Standard Bayesian odds update
    if prior in (0.0, 1.0) or likelihood_ratio == 0.0:
        posterior = 0.0 if prior == 0.0 else 1.0
    else:
        odds = prior / (1.0 - prior)
        new_odds = odds * likelihood_ratio
        posterior = new_odds / (1.0 + new_odds)

    # Entropy of the accumulated evidence IDs (including the new one)
    evidence_ids = hypothesis.get("evidence_ids", []) + [evidence["id"]]
    entropy = shannon_entropy(Counter(evidence_ids))

    # Fisher‑driven scaling: stronger Fisher information yields a larger
    # adjustment, bounded by a sigmoid to avoid extreme jumps.
    fisher_factor = 1.0 / (1.0 + math.exp(-10.0 * (fisher_score_val - 0.5)))  # maps [0,1] → (0,1)

    # Combine all modulations
    modulated = posterior * (1.0 - entropy) * fisher_factor
    modulated = max(0.0, min(1.0, modulated))

    return {
        "id": hypothesis["id"],
        "prior": prior,
        "posterior": modulated,
        "evidence_ids": evidence_ids,
    }


def _blade_sign(indices: Tuple[int, ...]) -> Tuple[List[int], int]:
    """
    Return a sorted list of basis indices together with the sign
    resulting from the required swaps to achieve the ordering.
    Duplicate indices cancel (Grassmann algebra property).
    """
    lst = list(indices)
    sign = 1
    i = 0
    while i < len(lst):
        j = i + 1
        while j < len(lst):
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
                sign *= -1
            elif lst[i] == lst[j]:
                # Cancel duplicate basis vectors
                del lst[j]
                del lst[i]
                sign *= 1
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign


def geometric_product(blade_a: np.ndarray, blade_b: np.ndarray) -> np.ndarray:
    """
    Compute a simple geometric (Clifford) product for binary‑encoded blades.
    The implementation treats each blade as a bitmask stored in an integer
    array; the product is the XOR of the masks combined with the sign from
    the ordering of basis vectors.
    """
    if blade_a.shape != blade_b.shape:
        raise ValueError("blade arrays must have the same shape")
    result = np.empty_like(blade_a, dtype=int)
    for idx, (a, b) in enumerate(zip(blade_a, blade_b)):
        # Convert integer masks to index tuples
        a_idxs = tuple(i for i in range(a.bit_length()) if (a >> i) & 1)
        b_idxs = tuple(i for i in range(b.bit_length()) if (b >> i) & 1)

        merged, sign = _blade_sign(a_idxs + b_idxs)
        mask = 0
        for i in merged:
            mask |= 1 << i
        result[idx] = sign * mask
    return result


if __name__ == "__main__":
    # Demo of the improved hybrid system
    theta, center, width = 0.5, 0.0, 1.0
    fisher_val = fisher_score(theta, center, width)
    print(f"Fisher score (scalar): {fisher_val:.6f}")

    actions = [{"value": 0.2}, {"value": 0.5}, {"value": 0.3}]
    fisher_scores = [0.1, 0.3, 0.6]
    regret_weights = compute_regret_weighted_strategy(actions, fisher_scores)
    print(f"Regret‑weighted scores: {regret_weights}")

    hypothesis = {"id": 0, "prior": 0.5, "posterior": 0.5, "evidence_ids": []}
    evidence = {"id": 1}
    updated = update_hypothesis(hypothesis, evidence, likelihood_ratio=2.0, fisher_score_val=fisher_val)
    print(f"Updated hypothesis: {updated}")

    # Simple geometric product test
    a = np.array([0b001, 0b010, 0b100])  # e1, e2, e3
    b = np.array([0b010, 0b100, 0b001])  # e2, e3, e1
    prod = geometric_product(a, b)
    print(f"Geometric product masks: {prod}")