# DARWIN HAMMER — match 1837, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_fisher_m22_s4.py (gen4)
# parent_b: hybrid_hybrid_ssim_hybrid_h_hybrid_bandit_router_m1109_s2.py (gen5)
# born: 2026-05-29T23:39:07Z

"""Hybrid Pheromone‑Fisher‑SSIM‑Bandit Algorithm.

Parents:
- hybrid_hybrid_hybrid_pherom_hybrid_hybrid_fisher_m22_s4.py (Pheromone tracking,
  entropy, Fisher information, decision‑hygiene)
- hybrid_hybrid_ssim_hybrid_h_hybrid_bandit_router_m1109_s2.py (SSIM similarity,
  Clifford‑algebra Multivector, temperature‑dependent Schoolfield rate,
  contextual multi‑armed bandit)

Mathematical bridge:
1. The pheromone probability vector **p** is analysed with Fisher information
   **I(p)**, providing an information‑theoretic scalar that modulates the
   decision‑hygiene score.
2. The Structural Similarity index **s ∈ [0,1]** between two images and the
   Schoolfield developmental rate **r(T)** (temperature‑dependent) are combined
   into a single scalar weight **w = s·r(T)**.
3. The weight **w** scales the components of a Clifford‑algebra Multivector
   representing the hygiene state and also biases the propensities of a
   contextual multi‑armed bandit.  The final action selection therefore depends
   on (i) pheromone‑derived information (entropy & Fisher), (ii) visual similarity,
   (iii) ambient temperature, and (iv) learned bandit statistics.

The following implementation fuses these concepts into a unified decision
pipeline."""
import math
import random
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Pheromone & Information‑theoretic utilities (Parent A)
# ----------------------------------------------------------------------
def calculate_pheromone_probabilities(
    surface_key: str,
    limit: int = 10,
    db_url: str | None = None,
) -> List[float]:
    """
    Return a normalized list of recent pheromone signal values for *surface_key*.
    In a production setting the values are fetched from a PostgreSQL table;
    for this self‑contained module a random list is generated when *db_url* is
    ``None``.
    """
    if db_url is None:
        # Simulated pheromone values (positive floats)
        raw = [random.expovariate(1.0) for _ in range(limit)]
    else:
        # Placeholder for real DB access – omitted per repository constraints.
        raise NotImplementedError("Database access not implemented in this demo.")
    total = sum(raw)
    if total == 0:
        raise ValueError("Pheromone sum cannot be zero.")
    return [v / total for v in raw]


def entropy(probabilities: Sequence[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    probs = np.array(probabilities, dtype=float)
    probs = probs / probs.sum()
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log(probs)))


def fisher_information(probabilities: Sequence[float], eps: float = 1e-12) -> float:
    """
    A simple scalar Fisher information for a discrete distribution.
    Using the variance of the log‑probabilities as a proxy:
        I = Var[log p_i]
    """
    probs = np.array(probabilities, dtype=float)
    probs = probs / probs.sum()
    logp = np.log(np.clip(probs, eps, None))
    return float(logp.var())


# ----------------------------------------------------------------------
# SSIM & Temperature utilities (Parent B – part A)
# ----------------------------------------------------------------------
def _mean_std(img: np.ndarray) -> Tuple[float, float]:
    """Return mean and standard deviation of an image (flattened)."""
    flat = img.ravel()
    return float(flat.mean()), float(flat.std(ddof=1))


def ssim(img1: np.ndarray, img2: np.ndarray, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """
    Simplified Structural Similarity Index (SSIM) for two grayscale images.
    The implementation follows the classic formula using means, variances and
    covariance.
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions.")
    mu1, sigma1 = _mean_std(img1)
    mu2, sigma2 = _mean_std(img2)
    cov = float(np.mean((img1 - mu1) * (img2 - mu2)))
    numerator = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2)
    return numerator / denominator


def schoolfield_rate(
    T: float,
    E: float = 0.65,
    H: float = 0.0,
    Tl: float = 283.15,
    Th: float = 313.15,
    B: float = 0.0001,
) -> float:
    """
    Schoolfield model for temperature‑dependent developmental rate.
    Parameters are typical biological defaults; they can be overridden.
    """
    k = 8.617333262e-5  # Boltzmann constant (eV·K⁻¹)
    exp_term = math.exp(-E / (k * T))
    low_term = 1.0 + math.exp((H - Tl) / (k * T))
    high_term = 1.0 + math.exp((Th - H) / (k * T))
    return B * exp_term / (low_term * high_term)


# ----------------------------------------------------------------------
# Clifford‑algebra Multivector (Parent B – part B)
# ----------------------------------------------------------------------
class Multivector:
    """Very small Clifford‑algebra multivector.

    Components are stored in a dict mapping a sorted tuple of basis indices
    to a float coefficient.  The empty tuple ``()`` represents the scalar part.
    """

    def __init__(self, components: Dict[Tuple[int, ...], float] | None = None, n: int = 0):
        self.n = int(n)
        self.components: Dict[Tuple[int, ...], float] = {}
        if components:
            for blade, val in components.items():
                if abs(val) > 1e-15:
                    self.components[tuple(sorted(blade))] = float(val)

    def scalar_part(self) -> float:
        return self.components.get((), 0.0)

    def scale(self, factor: float) -> "Multivector":
        """Return a new multivector with all components multiplied by *factor*."""
        return Multivector({blade: coeff * factor for blade, coeff in self.components.items()}, self.n)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coeff in sorted(self.components.items()):
            label = "1" if not blade else "e" + "".join(str(i) for i in blade)
            terms.append(f"{coeff:+.4g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"


# ----------------------------------------------------------------------
# Contextual Multi‑Armed Bandit (Parent B – part C)
# ----------------------------------------------------------------------
class Bandit:
    """
    Simple Thompson‑sampling contextual bandit.
    Each action holds a Beta(α,β) posterior for success probability.
    The context weight *w* biases the sampled propensities.
    """

    def __init__(self, actions: Sequence[str]):
        self.actions = list(actions)
        self.alpha = {a: 1.0 for a in self.actions}
        self.beta = {a: 1.0 for a in self.actions}

    def sample_propensities(self, weight: float) -> Dict[str, float]:
        """Draw a Beta sample for each action and multiply by *weight*."""
        draws = {}
        for a in self.actions:
            draws[a] = random.betavariate(self.alpha[a], self.beta[a]) * weight
        return draws

    def select_action(self, weight: float) -> str:
        """Select the action with the highest weighted sample."""
        prop = self.sample_propensities(weight)
        return max(prop, key=prop.get)

    def update(self, action: str, reward: float) -> None:
        """Update Beta posterior with binary reward (1 = success, 0 = failure)."""
        if reward not in (0, 1):
            raise ValueError("Reward must be 0 or 1 for Bernoulli bandit.")
        self.alpha[action] += reward
        self.beta[action] += 1 - reward


# ----------------------------------------------------------------------
# Hybrid core functions (fusion of both parents)
# ----------------------------------------------------------------------
def compute_hygiene_score(
    pheromone_probs: Sequence[float],
    fisher_weight: float = 1.0,
) -> float:
    """
    Decision‑hygiene score combining entropy and Fisher information.
    Higher entropy → more uncertainty → lower hygiene; higher Fisher → more
    information → higher hygiene.  The scalar *fisher_weight* (derived from the
    SSIM/temperature bridge) modulates the Fisher contribution.
    """
    H = entropy(pheromone_probs)
    I = fisher_information(pheromone_probs)
    # Normalise both terms to [0,1] using simple heuristics
    H_norm = H / math.log(len(pheromone_probs) + 1e-12)
    I_norm = I / (I + 1.0)  # maps to (0,1)
    score = (1.0 - H_norm) + fisher_weight * I_norm
    return float(score)


def fuse_multivector_with_context(
    base_mv: Multivector,
    ssim_val: float,
    temp_rate: float,
) -> Multivector:
    """
    Scale a multivector by the bridge weight w = ssim·rate.
    The scalar part is interpreted as a hygiene bias, while higher‑grade
    components receive the same scaling.
    """
    w = ssim_val * temp_rate
    return base_mv.scale(w)


def hybrid_decision_pipeline(
    surface_key: str,
    limit: int,
    db_url: str | None,
    img_ref: np.ndarray,
    img_cur: np.ndarray,
    temperature_K: float,
    actions: Sequence[str],
) -> Tuple[str, Multivector, float]:
    """
    End‑to‑end hybrid decision:
    1. Gather pheromone probabilities and compute entropy & Fisher.
    2. Compute SSIM between reference and current images.
    3. Evaluate temperature‑dependent rate via the Schoolfield model.
    4. Form bridge weight w = s·r(T) and scale a hygiene multivector.
    5. Combine Fisher‑modulated hygiene score with the multivector scalar
       part to bias a contextual bandit.
    6. Return the selected action, the scaled multivector, and the hygiene score.
    """
    # 1. Pheromones
    probs = calculate_pheromone_probabilities(surface_key, limit, db_url)
    fisher = fisher_information(probs)

    # 2. Visual similarity
    s = ssim(img_ref, img_cur)

    # 3. Temperature context
    r = schoolfield_rate(temperature_K)

    # 4. Multivector scaling
    base_mv = Multivector({(): 1.0, (1,): 0.5, (2,): -0.3})  # example initial state
    scaled_mv = fuse_multivector_with_context(base_mv, s, r)

    # 5. Hygiene score (Fisher weighted by bridge weight)
    hygiene = compute_hygiene_score(probs, fisher_weight=fisher * s * r)

    # 6. Bandit selection using the scalar part of the multivector as context weight
    bandit = Bandit(actions)
    context_weight = scaled_mv.scalar_part() * hygiene
    chosen = bandit.select_action(context_weight)

    return chosen, scaled_mv, hygiene


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy inputs
    surface = "surface-xyz"
    limit = 8
    db = None  # No real DB; random pheromones will be used

    # Create two simple grayscale images (8×8) with slight differences
    rng = np.random.default_rng(42)
    img_a = rng.integers(0, 256, size=(8, 8), dtype=np.uint8)
    img_b = img_a.copy()
    img_b[0, 0] = (img_b[0, 0] + 30) % 256  # introduce a small perturbation

    temperature = 298.15  # ~25 °C in Kelvin
    action_set = ["move_left", "move_right", "stay", "jump"]

    action, mv, hyg = hybrid_decision_pipeline(
        surface_key=surface,
        limit=limit,
        db_url=db,
        img_ref=img_a,
        img_cur=img_b,
        temperature_K=temperature,
        actions=action_set,
    )

    print(f"Chosen action: {action}")
    print(f"Scaled multivector: {mv}")
    print(f"Hygiene score: {hyg:.4f}")