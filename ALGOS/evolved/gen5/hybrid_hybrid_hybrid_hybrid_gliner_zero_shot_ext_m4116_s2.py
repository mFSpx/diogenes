# DARWIN HAMMER — match 4116, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m449_s2.py (gen4)
# parent_b: gliner_zero_shot_extractor.py (gen0)
# born: 2026-05-29T23:53:45Z

"""Hybrid Thompson‑Bandit + GLiNER‑style Zero‑Shot Extraction

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m449_s2.py (Thompson‑sampling bandit with
  Ollivier‑Ricci curvature prior shift)
- gliner_zero_shot_extractor.py (Zero‑shot label extraction returning character spans)

Mathematical bridge:
A curvature vector κ∈ℝᴸ is derived from the distribution of extracted label spans.
Each component κ_i (for label i) is a normalized measure of geometric “density’’ of
that label in the input text.  The bandit maintains Beta posteriors
(α_i,β_i) for the Bernoulli reward of each label‑action.  The curvature
modulates the prior by a linear shift

    α_i ← α_i + η·κ_î ,   κ̂ = κ /‖κ‖₁ ,

where η>0 is a tunable coupling coefficient.  Thus geometric information extracted
from raw text directly biases exploration‑exploitation in the Thompson sampler.

The module below implements this hybrid system with three core functions:
    1. literal_extraction – zero‑shot span extraction (fallback when GLiNER is absent)
    2. compute_curvature   – builds κ from the span statistics
    3. select_action       – samples Thompson‑bandit posteriors after curvature‑shift
"""

import json
import math
import random
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility helpers (shared by both parents)
# ----------------------------------------------------------------------
def now_iso() -> str:
    """Current UTC timestamp in ISO‑8601 with trailing “Z”."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    """SHA‑256 hash of a UTF‑8 string."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# ----------------------------------------------------------------------
# GLiNER‑style zero‑shot extraction (literal fallback)
# ----------------------------------------------------------------------
DEFAULT_LABELS = [
    "Operator",
    "Rainmaker",
    "Paladin / God-Mode",
    "Psyche / State-Collapse",
    "Forensic Shield",
    "Infinite Sink",
    "Anchor Weight",
    "Server Wipe",
    "API Rate Limiting",
    "Environment Migration",
    "Cruelty Protocols",
    "Master’s Eye",
    "Chrono-Ledger",
    "KRAMPUSCHEWING",
    "KORPUS",
    "DIOGENES",
    "FairyFuse",
    "Job Fair Allocator",
    "Darwinian Surfaces",
    "Command Envelope Protocol",
]


@dataclass(frozen=True)
class Span:
    """Exact character‑offset extraction result."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback_no_gliner"


def parse_labels(raw: str | None) -> List[str]:
    """Parse a JSON file, a comma‑separated string or fall back to defaults."""
    if not raw:
        return list(DEFAULT_LABELS)
    p = Path(raw)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            labels = data.get("required_exact_labels") or data.get("labels") or []
        else:
            labels = data
        return [str(x).strip() for x in labels if str(x).strip()]
    # treat as CSV list
    return [part.strip() for part in raw.split(",") if part.strip()]


def literal_extraction(text: str, labels: List[str]) -> List[Span]:
    """
    Very simple zero‑shot extractor: for each label perform a case‑insensitive
    literal search and emit a Span for every match.
    """
    spans: List[Span] = []
    for label in labels:
        # Escape regex meta‑characters in the label
        pattern = re.escape(label)
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            span = Span(
                start=match.start(),
                end=match.end(),
                text=match.group(),
                label=label,
                score=1.0,  # exact literal match confidence
                backend="literal_fallback_no_gliner",
            )
            spans.append(span)
    return spans


# ----------------------------------------------------------------------
# Thompson‑sampling Bernoulli bandit (parent A core)
# ----------------------------------------------------------------------
@dataclass
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "thompson_sampling"


@dataclass
class BanditUpdate:
    """Observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float = 1.0


class ThompsonBandit:
    """
    Minimal Thompson‑sampling Bernoulli bandit.
    For each action i we keep Beta(α_i, β_i).  Sampling a theta_i from this
    posterior yields the action with maximal theta.
    """

    def __init__(self, action_ids: List[str], init_alpha: float = 1.0, init_beta: float = 1.0):
        self.actions = action_ids
        self.alpha = {a: float(init_alpha) for a in action_ids}
        self.beta = {a: float(init_beta) for a in action_ids}

    def sample_theta(self) -> Dict[str, float]:
        """Draw a sample from each Beta posterior."""
        return {a: np.random.beta(self.alpha[a], self.beta[a]) for a in self.actions}

    def select_action(self) -> BanditAction:
        """Select the action with the highest sampled theta."""
        theta = self.sample_theta()
        best = max(theta, key=theta.get)
        # Propensity is the probability of drawing this action under uniform random policy
        propensity = 1.0 / len(self.actions)
        expected = self.alpha[best] / (self.alpha[best] + self.beta[best])
        # 95% Wilson confidence bound for a Bernoulli variable
        n = self.alpha[best] + self.beta[best] - 2  # subtract prior mass
        if n <= 0:
            bound = 0.0
        else:
            phat = expected
            z = 1.96
            bound = (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)
        return BanditAction(
            action_id=best,
            propensity=propensity,
            expected_reward=expected,
            confidence_bound=bound,
        )

    def update(self, upd: BanditUpdate) -> None:
        """Bayesian update of the Beta posterior for the selected action."""
        if upd.reward not in (0, 1):
            raise ValueError("Reward must be binary for Bernoulli bandit")
        self.alpha[upd.action_id] += upd.reward
        self.beta[upd.action_id] += 1 - upd.reward


# ----------------------------------------------------------------------
# Curvature computation (proxy for Ollivier‑Ricci)
# ----------------------------------------------------------------------
def compute_curvature(spans: List[Span], label_space: List[str]) -> np.ndarray:
    """
    Build a simple curvature proxy κ from span frequencies.

    For each label ℓ we count occurrences c_ℓ, then define
        κ_ℓ = sqrt(c_ℓ + ε)
    where ε>0 avoids zero.  The vector is finally L1‑normalized so that
    Σ_i κ_i = 1, yielding κ̂ used in the prior shift.
    """
    epsilon = 1e-6
    counts = np.array([0.0] * len(label_space))
    label_to_idx = {lab: idx for idx, lab in enumerate(label_space)}
    for sp in spans:
        if sp.label in label_to_idx:
            counts[label_to_idx[sp.label]] += 1.0
    curvature = np.sqrt(counts + epsilon)
    total = curvature.sum()
    if total == 0:
        return np.full_like(curvature, 1.0 / len(curvature))
    return curvature / total


# ----------------------------------------------------------------------
# Hybrid operation functions
# ----------------------------------------------------------------------
def incorporate_curvature(
    bandit: ThompsonBandit,
    curvature_vec: np.ndarray,
    eta: float = 0.5,
) -> None:
    """
    Apply the curvature‑derived prior shift to the bandit's α parameters.

    α_i ← α_i + η·κ̂_i   for each action i.
    """
    if len(curvature_vec) != len(bandit.actions):
        raise ValueError("Curvature vector length must match number of bandit actions")
    for idx, action in enumerate(bandit.actions):
        bandit.alpha[action] += eta * curvature_vec[idx]


def hybrid_update_from_spans(
    bandit: ThompsonBandit,
    spans: List[Span],
    label_space: List[str],
    reward_map: Dict[str, float] | None = None,
    eta: float = 0.5,
) -> None:
    """
    End‑to‑end update:
        1. Compute curvature from spans.
        2. Shift bandit priors.
        3. Optionally feed a synthetic reward (e.g. 1 if a span exists for the selected action).
    """
    curvature = compute_curvature(spans, label_space)
    incorporate_curvature(bandit, curvature, eta=eta)

    # If a reward map is supplied, use it; otherwise generate a dummy reward
    if reward_map is None:
        reward_map = {lbl: 1.0 if any(s.label == lbl for s in spans) else 0.0 for lbl in label_space}

    # Perform a single Thompson selection and update with the synthetic reward
    action = bandit.select_action()
    reward = float(reward_map.get(action.action_id, 0.0))
    upd = BanditUpdate(
        context_id=sha256_text(now_iso()),
        action_id=action.action_id,
        reward=reward,
    )
    bandit.update(upd)


def select_action_with_curvature(
    text: str,
    labels: List[str],
    bandit: ThompsonBandit,
    eta: float = 0.5,
) -> BanditAction:
    """
    High‑level helper that extracts spans, updates the bandit with curvature,
    and returns the selected action.
    """
    spans = literal_extraction(text, labels)
    hybrid_update_from_spans(bandit, spans, labels, eta=eta)
    return bandit.select_action()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The Operator initiated a Server Wipe while the Rainmaker monitored the API Rate Limiting. "
        "Later, the Paladin / God-Mode activated the Infinite Sink."
    )
    label_list = parse_labels(None)  # use defaults
    # Extracted spans (literal fallback)
    extracted = literal_extraction(sample_text, label_list)
    print(f"Extracted {len(extracted)} spans:")
    for sp in extracted[:5]:
        print(asdict(sp))

    # Initialise bandit with the same label set as actions
    bandit = ThompsonBandit(action_ids=label_list, init_alpha=1.0, init_beta=1.0)

    # Perform a hybrid selection
    chosen = select_action_with_curvature(sample_text, label_list, bandit, eta=0.7)
    print("\nChosen action after curvature‑augmented Thompson sampling:")
    print(asdict(chosen))

    # Show updated α parameters for inspection
    print("\nAlpha parameters after update (first 10):")
    for lbl in label_list[:10]:
        print(f"{lbl}: α={bandit.alpha[lbl]:.3f}, β={bandit.beta[lbl]:.3f}")