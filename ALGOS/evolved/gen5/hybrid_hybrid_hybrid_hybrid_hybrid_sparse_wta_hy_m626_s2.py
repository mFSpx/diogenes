# DARWIN HAMMER — match 626, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_regret_m236_s1.py (gen4)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py (gen2)
# born: 2026-05-29T23:30:13Z

"""Hybrid Algorithm integrating SSIM-based similarity (Parent A) with
Sparse Winner‑Take‑All expansion and differential‑privacy‑aware regret matching (Parent B).

Mathematical bridge:
1. A raw input vector `v` is sparsely projected to a high‑dimensional space
   `e = Expand(v)` using the hash‑based expansion from Parent B.
2. The prototype vector is projected with the same hash function producing
   `e_p = Expand(prototype)`.  The Structural Similarity Index (SSIM) between
   `e` and `e_p` (Parent A) yields a similarity score `s ∈ [‑1, 1]`.
3. The similarity score is interpreted as a utility for a regret‑matching
   process.  Regret updates are perturbed with Laplace noise whose scale is
   proportional to a privacy risk term `risk = uqis / total_records`
   (Parent B).  This yields a privacy‑aware action selection that respects both
   the similarity‑driven objective and differential‑privacy constraints.

The module provides three core hybrid operations:
* `hybrid_expand_ssim` – performs sparse expansion and computes SSIM.
* `add_laplace_noise` – adds calibrated Laplace noise for DP.
* `regret_match_step` – updates regrets with noisy utilities and selects an
  action according to the resulting mixed strategy.
"""

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared constants and utilities
# ----------------------------------------------------------------------
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)


def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index between two equal‑length vectors."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Parent B – Sparse Winner‑Take‑All utilities
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash‑based sparse expansion of `values` into a vector of length `m`."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):  # three hash probes per entry
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    """Binary mask with 1 at the indices of the top‑k values."""
    k = max(0, min(k, len(values)))
    winners = {
        i for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]


def hamming(a: List[int], b: List[int]) -> int:
    """Hamming distance between two binary vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must be the same length")
    return sum(ai != bi for ai, bi in zip(a, b))


# ----------------------------------------------------------------------
# Differential privacy helper
# ----------------------------------------------------------------------
def add_laplace_noise(value: float, scale: float) -> float:
    """Return `value` perturbed with Laplace noise of the given scale."""
    if scale <= 0:
        return value  # no noise needed
    noise = np.random.laplace(loc=0.0, scale=scale)
    return float(value + noise)


def compute_risk(unique_quasi_identifiers: int, total_records: int) -> float:
    """Simple risk estimator used as a scale factor for privacy noise."""
    if total_records <= 0:
        raise ValueError("total_records must be positive")
    return float(unique_quasi_identifiers) / float(total_records)


# ----------------------------------------------------------------------
# Hybrid core operations
# ----------------------------------------------------------------------
def hybrid_expand_ssim(
    payload: List[float],
    m: int = 128,
    salt: str = "",
) -> float:
    """
    1️⃣ Expand the raw `payload` and the prototype vector into the same
    high‑dimensional space using the sparse hash projection.
    2️⃣ Compute SSIM between the two expanded vectors.

    Returns a similarity score in the range [‑1, 1].
    """
    expanded_payload = expand(payload, m, salt)
    expanded_proto = expand(PROTOTYPE_VECTOR.tolist(), m, salt)
    return compute_ssim(expanded_payload, expanded_proto)


class RegretMatcher:
    """
    Regret‑matching engine with Laplace‑perturbed updates.

    Attributes
    ----------
    actions : List[str]
        Identifiers of available actions.
    regrets : Dict[str, float]
        Cumulative (noisy) regrets for each action.
    epsilon : float
        Base privacy budget (larger → less noise).
    """

    def __init__(self, actions: List[str], epsilon: float = 1.0):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.actions = actions
        self.epsilon = epsilon
        self.regrets: Dict[str, float] = {a: 0.0 for a in actions}
        self._last_utilities: Dict[str, float] = {a: 0.0 for a in actions}

    def _positive_regrets(self) -> List[float]:
        return [max(r, 0.0) for r in self.regrets.values()]

    def get_strategy(self) -> Dict[str, float]:
        """Mixed strategy proportional to positive cumulative regrets."""
        pos = self._positive_regrets()
        total = sum(pos)
        if total == 0.0:
            # uniform distribution if no positive regret
            prob = 1.0 / len(self.actions)
            return {a: prob for a in self.actions}
        return {a: r / total for a, r in zip(self.actions, pos)}

    def sample_action(self) -> str:
        """Draw an action according to the current mixed strategy."""
        strategy = self.get_strategy()
        rnd = random.random()
        cumulative = 0.0
        for a, p in strategy.items():
            cumulative += p
            if rnd <= cumulative:
                return a
        # fallback (numerical safety)
        return self.actions[-1]

    def update_regrets(
        self,
        chosen_action: str,
        utility: float,
        risk_scale: float,
    ) -> None:
        """
        Perform a regret‑matching update.

        Parameters
        ----------
        chosen_action : str
            Action taken in the current round.
        utility : float
            Observed utility (here derived from SSIM).
        risk_scale : float
            Scale factor for Laplace noise; larger risk → more noise.
        """
        # Record utility for all actions (here we assume the same utility for
        # every action; in a real setting each action would have its own payoff)
        for a in self.actions:
            self._last_utilities[a] = utility

        avg_utility = sum(self._last_utilities.values()) / len(self.actions)

        for a in self.actions:
            regret = self._last_utilities[a] - avg_utility
            # Apply Laplace noise calibrated to risk and privacy budget
            scale = risk_scale / self.epsilon
            noisy_regret = add_laplace_noise(regret, scale)
            self.regrets[a] += noisy_regret


def regret_match_step(
    payload: List[float],
    matcher: RegretMatcher,
    uqis: int,
    total_records: int,
    m: int = 128,
    salt: str = "",
) -> Tuple[str, float]:
    """
    Execute one hybrid iteration:

    1. Compute similarity‑derived utility via `hybrid_expand_ssim`.
    2. Derive a privacy risk scale from `uqis / total_records`.
    3. Update the regret matcher with Laplace‑perturbed regrets.
    4. Sample and return the selected action together with the raw utility.

    Returns
    -------
    chosen_action : str
        Action selected after the update.
    utility : float
        Raw (non‑noised) SSIM utility used in this step.
    """
    utility = hybrid_expand_ssim(payload, m=m, salt=salt)
    risk = compute_risk(uqis, total_records)
    chosen = matcher.sample_action()
    matcher.update_regrets(chosen_action=chosen, utility=utility, risk_scale=risk)
    # Resample after the update to reflect the new strategy
    post_update_action = matcher.sample_action()
    return post_update_action, utility


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample payloads (simulated packet payloads)
    sample_payloads = [
        [0.12, 0.48, 0.33, 0.71],
        [0.25, 0.55, 0.10, 0.60, 0.05],
        [0.19, 0.49, 0.31],
    ]

    actions = ["ALLOW", "DENY", "THROTTLE"]
    matcher = RegretMatcher(actions=actions, epsilon=0.8)

    # Simulated privacy parameters
    uqis = 3          # number of unique quasi‑identifiers
    total_records = 1000

    for idx, payload in enumerate(sample_payloads, 1):
        action, util = regret_match_step(
            payload=payload,
            matcher=matcher,
            uqis=uqis,
            total_records=total_records,
            m=256,
            salt="session42",
        )
        print(
            f"Round {idx}: utility={util:.4f}, selected_action={action}, "
            f"strategy={matcher.get_strategy()}"
        )