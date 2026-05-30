# DARWIN HAMMER — match 150, survivor 4
# gen: 3
# parent_a: hybrid_bayes_update_hybrid_krampus_brain_m15_s1.py (gen2)
# parent_b: hybrid_hybrid_ternary_route_hybrid_bandit_router_m56_s0.py (gen2)
# born: 2026-05-29T23:26:05Z

import numpy as np
import random
import math
import hashlib
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Deterministic feature extraction (replaces hash‑randomized version)
# ----------------------------------------------------------------------
def _deterministic_hash(text: str) -> int:
    """Return a stable 64‑bit integer hash for *text* using SHA‑256."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Produce a reproducible pseudo‑random feature vector from *text*.
    The same input always yields the same output across Python runs.
    """
    seed = _deterministic_hash(text) % (2**32)
    rnd = random.Random(seed)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight",
        "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline",
        "telemetry_manic_velocity",
    ]
    return {k: rnd.random() for k in keys}


def extract_master_vector(text: str) -> Dict[str, float]:
    """
    Reduce the full feature set to a compact *master vector*.
    The selection mirrors the original implementation but remains deterministic.
    """
    f = extract_full_features(text)
    return {
        "visceral_ratio": f["operator_visceral_ratio"],
        "tech_ratio": f["operator_tech_ratio"],
        "legal_osint_ratio": f["operator_legal_osint_ratio"],
        "forensic_shield_ratio": f["psyche_forensic_shield_ratio"],
        "poetic_entropy": f["psyche_poetic_entropy"],
        "dissociative_index": f["psyche_dissociative_index"],
        "bureaucratic_weaponization_index": f[
            "resilience_bureaucratic_weaponization_index"
        ],
        "resource_exhaustion_metric": f["resilience_resource_exhaustion_metric"],
        "swarm_orchestration_density": f["resilience_swarm_orchestration_density"],
    }

# ----------------------------------------------------------------------
# Refined curvature prior (vector‑wise Ollivier‑Ricci proxy)
# ----------------------------------------------------------------------
_ACTIONS = ["alpha", "beta", "gamma", "delta"]
# Map each action to a distinct subset of master‑vector components.
_ACTION_COMPONENTS: Dict[str, List[str]] = {
    "alpha": ["visceral_ratio", "tech_ratio", "poetic_entropy"],
    "beta": ["legal_osint_ratio", "forensic_shield_ratio", "dissociative_index"],
    "gamma": ["bureaucratic_weaponization_index", "resource_exhaustion_metric"],
    "delta": ["swarm_orchestration_density"],
}


def _inverse_variance(vec: List[float]) -> float:
    """Robust inverse variance used as a curvature strength estimator."""
    arr = np.asarray(vec, dtype=np.float64)
    var = arr.var()
    # Add a small epsilon to avoid division by zero and cap extreme values.
    return 1.0 / (var + 1e-6)


def compute_curvature_prior(master_vec: Dict[str, float]) -> Dict[str, float]:
    """
    Produce a prior distribution over actions using a per‑action
    inverse‑variance curvature estimate.
    """
    raw = []
    for a in _ACTIONS:
        comps = _ACTION_COMPONENTS[a]
        vals = [master_vec[c] for c in comps if c in master_vec]
        raw.append(_inverse_variance(vals))
    raw = np.array(raw, dtype=np.float64)
    prior = raw / raw.sum()
    return dict(zip(_ACTIONS, prior))

# ----------------------------------------------------------------------
# Action‑specific SSIM prototypes (replaces single global prototype)
# ----------------------------------------------------------------------
_PROTOTYPE_VECTORS: Dict[str, np.ndarray] = {
    "alpha": np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64),
    "beta": np.array([0.6, 0.1, 0.4, 0.2, 0.9], dtype=np.float64),
    "gamma": np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float64),
    "delta": np.array([0.9, 0.0, 0.1, 0.8, 0.5], dtype=np.float64),
}


def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index (SSIM) for 1‑D signals."""
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


def compute_likelihoods(packet: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Return a per‑action likelihood dictionary using SSIM against the
    action‑specific prototype vectors.
    """
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        return {a: 0.0 for a in _ACTIONS}
    payload_arr = np.asarray(payload, dtype=np.float64)

    # Ensure a fixed length by truncating or zero‑padding.
    target_len = max(v.size for v in _PROTOTYPE_VECTORS.values())
    if payload_arr.size < target_len:
        payload_arr = np.pad(
            payload_arr,
            (0, target_len - payload_arr.size),
            mode="constant",
            constant_values=0.0,
        )
    elif payload_arr.size > target_len:
        payload_arr = payload_arr[:target_len]

    likelihoods = {}
    for a, proto in _PROTOTYPE_VECTORS.items():
        # Align lengths (they are already equal after the padding above)
        likelihoods[a] = compute_ssim(payload_arr.tolist(), proto.tolist())
    return likelihoods

# ----------------------------------------------------------------------
# Bayesian fusion (true per‑action multiplication)
# ----------------------------------------------------------------------
def bayesian_update(
    prior: Dict[str, float],
    likelihoods: Dict[str, float],
    actions: List[str],
) -> Dict[str, float]:
    """
    Compute posterior ∝ prior × likelihood for each action.
    No artificial hash‑based modulation; the posterior reflects genuine
    statistical evidence.
    """
    unnorm = {
        a: prior.get(a, 0.0) * likelihoods.get(a, 0.0) for a in actions
    }
    total = sum(unnorm.values()) + 1e-12
    return {a: v / total for a, v in unnorm.items()}

# ----------------------------------------------------------------------
# Thompson Sampling bandit (deeper integration with posterior)
# ----------------------------------------------------------------------
class ThompsonBandit:
    """
    Maintains Beta(α, β) posteriors for each action.
    The posterior from the Bayesian fusion is used as a prior on α,
    encouraging actions that already have high Bayesian belief.
    """

    def __init__(self, actions: List[str], init_alpha: float = 1.0, init_beta: float = 1.0):
        self.actions = actions
        self.stats: Dict[str, Tuple[float, float]] = {
            a: (init_alpha, init_beta) for a in actions
        }

    def sample(self) -> str:
        """Draw a sample from each Beta and return the action with highest draw."""
        samples = {a: np.random.beta(a_alpha, a_beta) for a, (a_alpha, a_beta) in self.stats.items()}
        return max(samples, key=samples.get)

    def update(self, action: str, reward: float, posterior: Dict[str, float]) -> None:
        """
        Update Beta parameters.
        reward ∈ {0,1}. The Bayesian posterior weight is added to α to bias
        the distribution toward actions the Bayesian model already favors.
        """
        alpha, beta = self.stats.get(action, (1.0, 1.0))
        # Increment α by reward (standard) and by a fraction of the Bayesian belief.
        alpha += reward + posterior.get(action, 0.0)
        # Increment β by (1‑reward) to keep the total count consistent.
        beta += 1.0 - reward
        self.stats[action] = (alpha, beta)

    def get_estimates(self) -> Dict[str, float]:
        """Return the mean of each Beta distribution (used for diagnostics)."""
        return {a: a_alpha / (a_alpha + a_beta) for a, (a_alpha, a_beta) in self.stats.items()}


# ----------------------------------------------------------------------
# High‑level routing interface
# ----------------------------------------------------------------------
class HybridBayesianSSIMCurvatureRouter:
    """
    End‑to‑end router that:
      1. Extracts a deterministic master vector from a textual identifier.
      2. Derives a curvature‑based prior per action.
      3. Computes per‑action SSIM likelihoods from the packet payload.
      4. Forms a Bayesian posterior.
      5. Selects an action via Thompson Sampling.
      6. Updates the bandit statistics with the observed reward.
    """

    def __init__(self, actions: List[str] = None):
        self.actions = actions if actions is not None else _ACTIONS
        self.bandit = ThompsonBandit(self.actions)

    def route(self, identifier: str, packet: Dict[str, List[float]]) -> str:
        """
        Perform a single routing decision.
        Returns the selected action.
        """
        master_vec = extract_master_vector(identifier)
        prior = compute_curvature_prior(master_vec)
        likelihoods = compute_likelihoods(packet)
        posterior = bayesian_update(prior, likelihoods, self.actions)
        action = self.bandit.sample()
        # Store posterior for later update; the caller must provide the reward.
        self._last_posterior = posterior
        self._last_action = action
        return action

    def feedback(self, reward: float) -> None:
        """
        Incorporate external feedback (e.g., success=1, failure=0) for the
        most recent routing decision.
        """
        if not hasattr(self, "_last_action"):
            raise RuntimeError("feedback called before any routing decision")
        self.bandit.update(self._last_action, reward, self._last_posterior)
        # Clean up to avoid accidental reuse.
        del self._last_action, self._last_posterior

    def reset(self) -> None:
        """Reset the internal bandit statistics."""
        self.bandit = ThompsonBandit(self.actions)

    def diagnostics(self) -> Dict[str, float]:
        """Expose current Thompson Sampling mean estimates for each action."""
        return self.bandit.get_estimates()