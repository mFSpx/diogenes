# DARWIN HAMMER — match 1109, survivor 2
# gen: 5
# parent_a: hybrid_ssim_hybrid_hybrid_hybrid_m134_s0.py (gen4)
# parent_b: hybrid_bandit_router_poikilotherm_schoolf_m20_s0.py (gen1)
# born: 2026-05-29T23:32:51Z

"""Hybrid SSIM‑Bandit‑Temperature algorithm.

Parent A: ``hybrid_ssim_hybrid_hybrid_hybrid_m134_s0.py`` – provides a
Structural Similarity (SSIM) measure and a Clifford‑algebra based
``Multivector`` representation for decision‑hygiene scores.

Parent B: ``hybrid_bandit_router_poikilotherm_schoolf_m20_s0.py`` – provides a
contextual multi‑armed bandit (actions, updates, policy) whose context is a
temperature‑dependent developmental rate given by the Schoolfield model.

Mathematical bridge:
Both parents employ a scalar weighting that modulates a higher‑dimensional
object.  In the hybrid we use the SSIM value *s* (0…1) as a similarity weight
and the temperature‑dependent developmental rate *r(T)* as a context weight.
The product w = s·r(T) scales the components of the decision‑hygiene
multivector and also biases the bandit propensities.  Thus a single scalar
w links the geometric‑algebra part (Parent A) with the bandit‑routing part
(Parent B)."""

import math, random, sys, pathlib
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Multivector and SSIM utilities
# ----------------------------------------------------------------------
class Multivector:
    """Simple Clifford‑algebra multivector with geometric product defined by
    concatenating basis indices (order‑independent, sign ignored for brevity)."""
    def __init__(self, components: Dict[Tuple[int, ...], float], n: int = 0):
        # Remove near‑zero components
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector({blade: coef for blade, coef in self.components.items()
                            if len(blade) == k}, self.n)

    def scalar_part(self) -> float:
        return self.components.get((), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            label = "1" if not blade else "e" + "".join(str(i) for i in sorted(blade))
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (simplified): combine basis sets and multiply coefficients."""
        result: Dict[Tuple[int, ...], float] = {}
        for blade1, c1 in self.components.items():
            for blade2, c2 in other.components.items():
                # Combine blades, remove duplicates (XOR‑like behavior)
                combined = list(blade1) + list(blade2)
                # Count occurrences and keep those with odd multiplicity
                uniq = []
                for b in combined:
                    if combined.count(b) % 2 == 1:
                        uniq.append(b)
                new_blade = tuple(sorted(set(uniq)))
                result[new_blade] = result.get(new_blade, 0.0) + c1 * c2
        return Multivector(result, self.n)

def ssim(img1: np.ndarray, img2: np.ndarray,
         C1: float = 6.5025, C2: float = 58.5225) -> float:
    """Return the mean SSIM index between two equal‑length 1‑D grayscale samples."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(numerator / denominator)


def decision_hygiene_to_multivector(scores: Dict[str, float],
                                    basis_map: Dict[str, int]) -> Multivector:
    """Map a dict of named hygiene scores onto a multivector using ``basis_map``."""
    comps = {}
    for name, val in scores.items():
        idx = basis_map.get(name)
        if idx is None:
            continue
        comps[(idx,)] = float(val)
    # also store the scalar part as the average of all scores
    if scores:
        comps[()] = sum(scores.values()) / len(scores)
    return Multivector(comps, n=max(basis_map.values()) + 1)


# ----------------------------------------------------------------------
# Parent B – Bandit routing with temperature‑dependent rate (Schoolfield)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def _reward(action_id: str) -> float:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield‐Rollinson temperature‑dependent rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = (params.rho_25 *
                 (temp_k / 298.15) *
                 math.exp((params.delta_h_activation / params.r_cal) *
                          ((1.0 / 298.15) - (1.0 / temp_k))))
    low = math.exp((params.delta_h_low / params.r_cal) *
                   ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) *
                    ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)


# ----------------------------------------------------------------------
# Hybrid core – linking the two parents
# ----------------------------------------------------------------------
def temperature_scaled_multivector(base_mv: Multivector,
                                   temperature_c: float) -> Multivector:
    """Scale every component of ``base_mv`` by the developmental rate at the
    supplied temperature (Celsius).  The rate acts as the bridge scalar."""
    rate = developmental_rate(c_to_k(temperature_c))
    scaled_components = {blade: coef * rate for blade, coef in base_mv.components.items()}
    return Multivector(scaled_components, base_mv.n)


def hybrid_ssim_weighted_action(actions: List[BanditAction],
                                img_ref: np.ndarray,
                                img_candidate: np.ndarray,
                                temperature_c: float) -> BanditAction:
    """
    Compute SSIM between ``img_ref`` and ``img_candidate``,
    combine it with the temperature‑dependent rate to obtain a scalar weight
    ``w = ssim * r(T)``.  The weight multiplies the action's propensity and
    adjusts its confidence bound, returning a new ``BanditAction`` that
    reflects both visual similarity and thermal context.
    """
    s = ssim(img_ref, img_candidate)                     # 0…1
    r = developmental_rate(c_to_k(temperature_c))      # >0
    w = s * r

    # Choose the action with the highest (propensity * w) + confidence term
    best = None
    best_score = -math.inf
    for a in actions:
        adjusted_propensity = a.propensity * w
        score = adjusted_propensity + a.confidence_bound
        if score > best_score:
            best_score = score
            best = a

    if best is None:
        raise RuntimeError("No actions provided")
    # Return a copy with updated propensity (for downstream use)
    return BanditAction(
        action_id=best.action_id,
        propensity=best.propensity * w,
        expected_reward=best.expected_reward,
        confidence_bound=best.confidence_bound,
        algorithm=best.algorithm,
    )


def hybrid_update_and_select(context_id: str,
                             actions: List[BanditAction],
                             img_ref: np.ndarray,
                             img_pool: List[np.ndarray],
                             temperature_c: float) -> Tuple[BanditAction, List[BanditUpdate]]:
    """
    For a given ``context_id``:
    1. Compute SSIM between ``img_ref`` and each candidate image in ``img_pool``.
    2. Build a temporary multivector from fictitious hygiene scores derived from SSIM.
    3. Scale the multivector with temperature (bridge scalar).
    4. Use the scaled multivector's scalar part to bias the bandit selection via
       ``hybrid_ssim_weighted_action``.
    5. Produce a list of ``BanditUpdate`` objects reflecting the chosen action's
       reward (here simulated as the scalar part of the scaled multivector).
    """
    # Step 1 – SSIM list
    ssim_vals = [ssim(img_ref, cand) for cand in img_pool]

    # Step 2 – fabricate hygiene scores (e.g., 'similarity', 'temperature')
    scores = {
        "similarity": float(np.mean(ssim_vals)),
        "temperature": temperature_c,
    }
    basis_map = {"similarity": 1, "temperature": 2}
    base_mv = decision_hygiene_to_multivector(scores, basis_map)

    # Step 3 – temperature scaling
    scaled_mv = temperature_scaled_multivector(base_mv, temperature_c)

    # Step 4 – select action using the first candidate as representative
    # (any candidate works because the weight depends only on the scalar part)
    chosen_action = hybrid_ssim_weighted_action(
        actions, img_ref, img_pool[0], temperature_c
    )

    # Step 5 – simulate a reward proportional to the scalar part of the scaled MV
    reward = scaled_mv.scalar_part()
    update = BanditUpdate(
        context_id=context_id,
        action_id=chosen_action.action_id,
        reward=reward,
        propensity=chosen_action.propensity,
    )
    update_policy([update])
    return chosen_action, [update]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed
    random.seed(0)
    np.random.seed(0)

    # synthetic grayscale patches (length 100)
    ref = np.random.rand(100).astype(np.float32)
    candidates = [np.random.rand(100).astype(np.float32) for _ in range(3)]

    # initialise a tiny bandit policy
    actions = [
        BanditAction(action_id="A", propensity=1.0, expected_reward=0.0, confidence_bound=0.5),
        BanditAction(action_id="B", propensity=0.8, expected_reward=0.0, confidence_bound=0.3),
        BanditAction(action_id="C", propensity=0.6, expected_reward=0.0, confidence_bound=0.2),
    ]

    reset_policy()
    chosen, updates = hybrid_update_and_select(
        context_id="test_ctx",
        actions=actions,
        img_ref=ref,
        img_pool=candidates,
        temperature_c=22.0,
    )
    print("Chosen action:", chosen)
    print("Policy after update:", _POLICY)