# DARWIN HAMMER — match 2563, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_ternary_lens__m953_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_liquid_m39_s0.py (gen3)
# born: 2026-05-29T23:43:01Z

"""Hybrid Algorithm Combining:
- Parent A: `hybrid_hybrid_hybrid_ternar_hybrid_ternary_lens__m953_s0.py`
  (bandit‑policy update, SSIM similarity evaluation, decreasing‑rate pruning)
- Parent B: `hybrid_hybrid_hybrid_decisi_hybrid_hybrid_liquid_m39_s0.py`
  (regex‑based feature extraction, weighted scoring, Liquid‑Time‑Constant (LTC) recurrent
   cell with diffusion forcing driven by MinHash‑like similarity)

Mathematical Bridge
-------------------
The bridge is a *feature‑driven similarity term* that feeds both the bandit
policy and the LTC dynamics:

1. Text input → binary feature vector **f** (regex matches).
2. A weighted inner‑product **s = f·w** (where **w** are learned action weights)
   is used as a proxy for the SSIM similarity between the raw input and the
   router’s output.
3. The same scalar **s** scales the diffusion forcing in the LTC update:
   `dx = -λ·x + D·s·(1‑x)` where `λ` is a decay constant and `D` follows the
   decreasing‑rate pruning schedule `D(t) = D₀ / (1 + β·t)`.
4. The bandit reward for an action `a` is updated with the similarity
   `s`, thus coupling the decision‑making and the continuous LTC dynamics
   into a single unified system.

The following module implements this fused mathematics with three core
functions:
`extract_features`, `ltc_update`, and `hybrid_step`.  A simple smoke test
demonstrates end‑to‑end execution."""


import sys
import math
import random
import json
from pathlib import Path
import re
import numpy as np

# ----------------------------------------------------------------------
# Global structures shared by the bandit policy (Parent A) and the hybrid.
# ----------------------------------------------------------------------
_POLICY: dict[str, list[float]] = {}  # action_id → [cumulative_reward, count]

_CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}

# ----------------------------------------------------------------------
# Regex feature set – taken from Parent B.
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)

_FEATURE_REGEXES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
}

# ----------------------------------------------------------------------
# Helper utilities (mirroring Parent A)
# ----------------------------------------------------------------------
def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]


def update_policy(updates: list[dict]) -> None:
    """Apply a list of updates where each dict contains:
    {'action_id': str, 'reward': float}
    """
    for u in updates:
        stats = _POLICY.setdefault(u["action_id"], [0.0, 0.0])
        stats[0] += float(u.get("reward", 0.0))
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Core mathematical building blocks
# ----------------------------------------------------------------------
def extract_features(text: str) -> np.ndarray:
    """
    Convert raw text into a binary feature vector using the regex set from
    Parent B.

    Returns
    -------
    np.ndarray shape (F,) where F = number of feature regexes, dtype=float.
    """
    feats = [1.0 if regex.search(text) else 0.0 for regex in _FEATURE_REGEXES.values()]
    return np.asarray(feats, dtype=float)


def ssim(u: np.ndarray, v: np.ndarray) -> float:
    """
    A lightweight Structural Similarity (SSIM) approximation for 1‑D vectors.
    Uses mean, variance and covariance as in the classic formulation.

    Parameters
    ----------
    u, v : np.ndarray
        Input vectors of identical shape.

    Returns
    -------
    float
        SSIM score in [0, 1].
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_u = u.mean()
    mu_v = v.mean()
    sigma_u = u.var()
    sigma_v = v.var()
    sigma_uv = np.mean((u - mu_u) * (v - mu_v))

    numerator = (2 * mu_u * mu_v + C1) * (2 * sigma_uv + C2)
    denominator = (mu_u ** 2 + mu_v ** 2 + C1) * (sigma_u + sigma_v + C2)
    return float(numerator / denominator) if denominator != 0 else 0.0


def decreasing_prune_rate(initial: float, step: int, beta: float = 0.01) -> float:
    """
    Implements the decreasing‑rate pruning schedule from Parent A.

    Parameters
    ----------
    initial : float
        Initial pruning coefficient (e.g., diffusion constant).
    step : int
        Current iteration step.
    beta : float, optional
        Decay factor (default 0.01).

    Returns
    -------
    float
        Pruned coefficient for the current step.
    """
    return initial / (1.0 + beta * step)


def ltc_update(
    state: np.ndarray,
    features: np.ndarray,
    weights: np.ndarray,
    dt: float,
    step: int,
    lambda_decay: float = 0.5,
    diffusion_initial: float = 1.0,
) -> np.ndarray:
    """
    Liquid‑Time‑Constant (LTC) recurrent cell update that incorporates the
    similarity term derived from the feature vector.

    The diffusion term D(t) follows the decreasing‑rate schedule of Parent A.
    The similarity scalar s is the SSIM between the weighted feature vector
    and the current state (treated as a vector of the same length).

    Parameters
    ----------
    state : np.ndarray
        Current hidden state (shape (H,)).
    features : np.ndarray
        Binary feature vector (shape (F,)).
    weights : np.ndarray
        Action‑specific weight vector (shape (F,)). Must match feature size.
    dt : float
        Time step.
    step : int
        Global iteration counter (used for pruning schedule).
    lambda_decay : float
        Linear decay constant.
    diffusion_initial : float
        Base diffusion coefficient before pruning.

    Returns
    -------
    np.ndarray
        Updated hidden state.
    """
    # Ensure compatible shapes
    if features.shape != weights.shape:
        raise ValueError("features and weights must have identical shape")
    # Project features into the same dimensionality as state via a simple linear map
    proj = np.tanh(features * weights)  # shape (F,)
    # If needed, broadcast to state size
    if proj.shape != state.shape:
        # simple repeat/truncate to match state length
        repeat_factor = math.ceil(state.size / proj.size)
        proj = np.tile(proj, repeat_factor)[: state.size]

    # Similarity term (acts like MinHash‑derived similarity)
    s = ssim(state, proj)

    # Diffusion coefficient with decreasing schedule
    D = decreasing_prune_rate(diffusion_initial, step)

    # LTC differential equation: dx = -λ·x + D·s·(1‑x)
    dx = -lambda_decay * state + D * s * (1.0 - state)
    return state + dt * dx


def hybrid_step(
    text: str,
    state: np.ndarray,
    action_id: str,
    dt: float = 0.1,
    step: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Perform one hybrid iteration:
    1. Extract regex features from `text`.
    2. Compute a similarity score via SSIM between the feature vector and
       the current state.
    3. Update the bandit policy for `action_id` using the similarity as reward.
    4. Propagate the LTC state using the similarity‑scaled diffusion term.
    5. Return the new state and a diagnostics dictionary.

    Parameters
    ----------
    text : str
        Input text.
    state : np.ndarray
        Current LTC hidden state.
    action_id : str
        Identifier for the bandit action to be rewarded/penalised.
    dt : float, optional
        Time step for the LTC update.
    step : int, optional
        Global iteration count (affects pruning schedule).

    Returns
    -------
    new_state : np.ndarray
        Updated LTC hidden state.
    info : dict
        Diagnostic information (features, similarity, reward, policy stats).
    """
    # 1. Feature extraction
    feats = extract_features(text)  # shape (F,)

    # 2. Build a weight vector for the current action (use stored policy mean as weight)
    # If the action is unseen, initialise a uniform weight vector.
    if action_id not in _POLICY:
        weight_vec = np.ones_like(feats) * 0.5
    else:
        # Use the average reward as a scalar multiplier for each feature
        avg_reward = _reward(action_id)
        weight_vec = np.full_like(feats, avg_reward)

    # 3. Compute similarity (SSIM) between state and projected features
    # Ensure state and projected features share shape
    proj_feat = np.tanh(feats * weight_vec)
    if proj_feat.shape != state.shape:
        # Broadcast/truncate to match
        repeat_factor = math.ceil(state.size / proj_feat.size)
        proj_feat = np.tile(proj_feat, repeat_factor)[: state.size]
    sim = ssim(state, proj_feat)

    # 4. Update bandit policy: treat similarity as reward
    update_policy([{"action_id": action_id, "reward": sim}])

    # 5. LTC state propagation
    new_state = ltc_update(
        state=state,
        features=feats,
        weights=weight_vec,
        dt=dt,
        step=step,
    )

    # 6. Diagnostics
    info = {
        "features": feats.tolist(),
        "weights": weight_vec.tolist(),
        "similarity": sim,
        "reward": sim,
        "policy_mean_reward": _reward(action_id),
        "policy_count": _count(action_id),
        "state_before": state.tolist(),
        "state_after": new_state.tolist(),
    }
    return new_state, info


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a random hidden state (size 5 for demonstration)
    rng = np.random.default_rng(42)
    hidden_state = rng.random(5)

    # Example text that matches some regexes
    example_text = (
        "The audit confirmed the evidence and we have a plan to "
        "schedule the next steps after a short delay."
    )

    # Run a few hybrid iterations
    for i in range(5):
        hidden_state, diagnostics = hybrid_step(
            text=example_text,
            state=hidden_state,
            action_id="audit_decision",
            dt=0.1,
            step=i,
        )
        print(f"Step {i} | similarity={diagnostics['similarity']:.4f} | "
              f"policy_reward={diagnostics['policy_mean_reward']:.4f}")

    # Ensure the policy dict is populated
    assert "audit_decision" in _POLICY
    print("Smoke test completed successfully.")