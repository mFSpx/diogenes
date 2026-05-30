# DARWIN HAMMER — match 4116, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m449_s2.py (gen4)
# parent_b: gliner_zero_shot_extractor.py (gen0)
# born: 2026-05-29T23:53:45Z

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

    def select_action(self, curvature_vec: np.ndarray = None, eta: float = 0.5) -> BanditAction:
        """Select the action with the highest sampled theta."""
        if curvature_vec is not None:
            alpha_shifted = {a: self.alpha[a] + eta * curvature_vec[i] for i, a in enumerate(self.actions)}
            theta = {a: np.random.beta(alpha_shifted[a], self.beta[a]) for a in self.actions}
        else:
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
def hybrid_select_action(
    bandit: ThompsonBandit,
    spans: List[Span],
    label_space: List[str],
    eta: float = 0.5,
) -> BanditAction:
    curvature_vec = compute_curvature(spans, label_space)
    return bandit.select_action(curvature_vec, eta)


def hybrid_update(
    bandit: ThompsonBandit,
    context_id: str,
    action_id: str,
    reward: float,
) -> None:
    bandit.update(BanditUpdate(context_id, action_id, reward))


# Example usage
if __name__ == "__main__":
    labels = parse_labels(None)
    bandit = ThompsonBandit(labels)
    text = "Some example text."
    spans = literal_extraction(text, labels)
    action = hybrid_select_action(bandit, spans, labels)
    print(action)