# DARWIN HAMMER — match 651, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s0.py (gen4)
# parent_b: hybrid_krampus_chrono_infotaxis_m67_s0.py (gen1)
# born: 2026-05-29T23:30:16Z

"""Hybrid Date‑Certainty Bandit Algorithm
=======================================

This module fuses the two parent algorithms:

* **Parent A** – *hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s0.py* –  
  provides the ``CertaintyFlag`` dataclass that encodes epistemic certainty
  (confidence expressed in basis‑points) together with an authority label.

* **Parent B** – *hybrid_krampus_chrono_infotaxis_m67_s0.py* –  
  supplies a date‑parsing routine that returns an entropy measure for the
  parsing ambiguity.

**Mathematical bridge**  
Both parents treat *uncertainty* as a first‑class quantity.  In the
parent‑A world the uncertainty is expressed as a confidence value
``c ∈ [0,1]`` (stored as basis‑points).  In the parent‑B world the
uncertainty of a parsed date is quantified by Shannon entropy
``H = -∑ p_i log p_i`` of the candidate‑date distribution.  The hybrid
algorithm maps the entropy ``H`` to a confidence ``c`` (higher entropy →
lower confidence) and builds a ``CertaintyFlag``.  The resulting certainty
drives a lightweight multi‑armed bandit that selects an action
``a∈A`` by maximizing an Upper‑Confidence‑Bound (UCB) term
``Q(a) + β·c·√(log t / n_a)`` where ``Q(a)`` is the estimated reward,
``t`` the total number of selections, ``n_a`` the count for action ``a``,
and ``β`` a scaling constant.

The three core functions below demonstrate this integration:

* ``parse_date_with_entropy`` – parses a date string and returns the best
  candidate together with its entropy.
* ``certainty_from_entropy`` – converts entropy into a ``CertaintyFlag``.
* ``ucb_select_action`` – picks an action using the confidence derived
  from the parsed date.

The module is self‑contained and uses only the standard library,
``numpy``, ``math`` and ``random`` as required.
"""

import datetime as dt
import re
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A: CertaintyFlag definition
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable representation of epistemic certainty.

    Attributes
    ----------
    label : str
        One of ``EPISTEMIC_FLAGS`` describing the qualitative certainty.
    confidence_bps : int
        Confidence expressed in basis‑points (0 .. 10 000 → 0 % .. 100 %).
    authority_class : str
        Optional authority identifier (e.g. “date_parser”).
    rationale : str
        Human‑readable explanation of the confidence.
    evidence_refs : Tuple[str, ...]
        References to supporting evidence (e.g. matched regex names).
    generated_at : str
        ISO‑8601 timestamp of creation.
    """
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = field(default_factory=lambda: dt.datetime.utcnow().isoformat())

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }


# ----------------------------------------------------------------------
# Parent‑B: Date parsing and entropy utilities
# ----------------------------------------------------------------------
CONTENT_DATE_PATTERNS = [
    ("frontmatter_date", re.compile(
        r"(?im)^\s*(?:date|created|created_at|created at|generated|timestamp|time|filed|updated|modified)\s*[:=]\s*[\"']?((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)")),
    ("iso_inline", re.compile(
        r"\b((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)\b")),
    ("compact_yyyymmdd", re.compile(
        r"\b((?:20|19)\d{2})(\d{2})(\d{2})(?:[T_-]?(\d{2})(\d{2})(\d{2})?)?Z?\b")),
]

MONTH_NAME_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+([0-3]?\d)(?:st|nd|rd|th)?,?\s+((?:20|19)\d{2})\b",
    re.I,
)


def _safe_log(p: float, eps: float = 1e-12) -> float:
    """Log with protection against zero."""
    return math.log(p + eps)


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a discrete distribution.

    Parameters
    ----------
    probabilities : list[float]
        Unnormalised probability masses.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        Entropy in nats.
    """
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    probs = [p / total for p in probabilities]
    return -sum(p * _safe_log(p, eps) for p in probs)


def _match_patterns(date_str: str) -> List[Tuple[str, dt.datetime]]:
    """Return a list of (pattern_name, parsed_datetime) candidates."""
    candidates = []

    for name, regex in CONTENT_DATE_PATTERNS:
        for m in regex.finditer(date_str):
            try:
                # Build ISO‑like string from groups, then parse.
                if name == "compact_yyyymmdd":
                    y, mo, d, hh, mm, ss = m.groups()
                    iso = f"{y}-{mo}-{d}"
                    if hh:
                        iso += f" {hh}:{mm}:{ss}"
                    dt_obj = dt.datetime.fromisoformat(iso.replace(" ", "T"))
                else:
                    iso = m.group(1)
                    # Normalise separators to hyphens and 'T' for time.
                    iso = iso.replace("_", "-").replace("/", "-")
                    iso = iso.replace(" ", "T")
                    dt_obj = dt.datetime.fromisoformat(iso.rstrip("Z"))
                candidates.append((name, dt_obj))
            except Exception:
                continue

    # Month‑name format
    for m in MONTH_NAME_RE.finditer(date_str):
        month_str, day, year = m.groups()
        month = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct":10, "nov":11, "dec":12
        }[month_str[:3].lower()]
        try:
            dt_obj = dt.datetime(int(year), month, int(day))
            candidates.append(("month_name", dt_obj))
        except Exception:
            continue

    return candidates


def parse_date_with_entropy(date_str: str) -> Tuple[Optional[dt.datetime], float, List[str]]:
    """Parse ``date_str`` and return the most likely datetime, its entropy,
    and the list of pattern names that contributed to the distribution.

    The function treats each successful regex match as a hypothesis with equal
    prior weight.  Entropy is computed over these hypotheses.

    Returns
    -------
    parsed : datetime | None
        The datetime with the highest hypothesis weight (ties broken arbitrarily).
    H : float
        Shannon entropy of the hypothesis distribution (0 if a single hypothesis).
    evidence : list[str]
        Names of the patterns that produced hypotheses.
    """
    candidates = _match_patterns(date_str)
    if not candidates:
        return None, 0.0, []

    # Uniform weights → entropy depends only on count.
    weights = np.ones(len(candidates), dtype=float)
    H = entropy(weights.tolist())

    # Choose the first candidate as the representative parsed date.
    # In a more sophisticated system one could rank by pattern reliability.
    _, best_dt = candidates[0]
    evidence = [name for name, _ in candidates]
    return best_dt, H, evidence


# ----------------------------------------------------------------------
# Hybrid bridge: entropy → certainty flag
# ----------------------------------------------------------------------
def certainty_from_entropy(entropy_val: float, max_entropy: float = 10.0) -> CertaintyFlag:
    """Map an entropy value to a ``CertaintyFlag``.

    Parameters
    ----------
    entropy_val : float
        Entropy (nats) computed from the date‑parsing hypotheses.
    max_entropy : float
        Entropy value that corresponds to 0 % confidence.  Larger values are clipped.

    Returns
    -------
    CertaintyFlag
        Qualitative label and confidence derived from the entropy.
    """
    # Clip entropy to the interval [0, max_entropy].
    e = max(0.0, min(entropy_val, max_entropy))
    # Linear mapping: 0 entropy → 100 % confidence, max_entropy → 0 % confidence.
    confidence = 1.0 - e / max_entropy
    confidence_bps = int(round(confidence * 10_000))

    # Determine qualitative label.
    if confidence_bps >= 9_000:
        label = "FACT"
    elif confidence_bps >= 7_000:
        label = "PROBABLE"
    elif confidence_bps >= 4_000:
        label = "POSSIBLE"
    elif confidence_bps >= 1_000:
        label = "SURE_MAYBE"
    else:
        label = "BULLSHIT"

    rationale = f"Entropy {entropy_val:.3f} (max {max_entropy}) mapped to confidence {confidence:.2%}"
    return CertaintyFlag(
        label=label,
        confidence_bps=confidence_bps,
        authority_class="date_entropy_bridge",
        rationale=rationale,
        evidence_refs=(),
    )


# ----------------------------------------------------------------------
# Simple UCB bandit that uses the certainty confidence as the exploration
# scaling factor.
# ----------------------------------------------------------------------
class UCBBandit:
    """A lightweight Upper‑Confidence‑Bound bandit.

    The bandit maintains empirical average rewards ``Q`` and selection counts
    ``N`` for each action.  The exploration term is scaled by the confidence
    extracted from a ``CertaintyFlag``.
    """
    def __init__(self, actions: List[str], beta: float = 1.0):
        self.actions = actions
        self.beta = beta
        self.Q = {a: 0.0 for a in actions}
        self.N = {a: 0 for a in actions}
        self.t = 0

    def select(self, certainty: CertaintyFlag) -> str:
        """Select an action using UCB with confidence‑aware exploration."""
        self.t += 1
        c = certainty.confidence_bps / 10_000.0  # convert to [0,1]

        # If any action has never been tried, choose it first (standard UCB init).
        for a in self.actions:
            if self.N[a] == 0:
                return a

        ucb_values = {}
        for a in self.actions:
            avg = self.Q[a]
            explore = self.beta * c * math.sqrt(math.log(self.t) / self.N[a])
            ucb_values[a] = avg + explore

        # Return action with maximal UCB value.
        return max(ucb_values, key=ucb_values.get)

    def update(self, action: str, reward: float) -> None:
        """Incorporate observed reward for ``action``."""
        self.N[action] += 1
        n = self.N[action]
        old_avg = self.Q[action]
        # Incremental mean update.
        self.Q[action] = old_avg + (reward - old_avg) / n


# ----------------------------------------------------------------------
# Public hybrid API
# ----------------------------------------------------------------------
def hybrid_parse_and_select(date_str: str, actions: List[str]) -> Tuple[Optional[dt.datetime], str, CertaintyFlag]:
    """Parse a date string, compute certainty, and select an action.

    Parameters
    ----------
    date_str : str
        Input string possibly containing a date.
    actions : list[str]
        Candidate actions for the bandit.

    Returns
    -------
    parsed_date : datetime | None
        The datetime extracted from ``date_str`` (if any).
    chosen_action : str
        Action selected by the confidence‑aware UCB.
    certainty : CertaintyFlag
        Certainty derived from the parsing entropy.
    """
    parsed, H, evidence = parse_date_with_entropy(date_str)
    certainty = certainty_from_entropy(H)

    # Attach evidence to the flag for traceability.
    object.__setattr__(certainty, "evidence_refs", tuple(evidence))

    bandit = UCBBandit(actions)
    action = bandit.select(certainty)

    # In a real system we would now execute the action and observe a reward.
    # Here we simulate a dummy reward proportional to confidence.
    dummy_reward = certainty.confidence_bps / 10_000.0
    bandit.update(action, dummy_reward)

    return parsed, action, certainty


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_dates = [
        "Created: 2023-07-15T14:23:00Z",
        "20211205 08:30",
        "July 4th, 2021",
        "No date here",
    ]
    actions = ["store", "notify", "ignore"]

    for s in sample_dates:
        dt_obj, act, flag = hybrid_parse_and_select(s, actions)
        print(f"Input: {s!r}")
        print(f"  Parsed date   : {dt_obj}")
        print(f"  Entropy label : {flag.label} ({flag.confidence_bps/100:.2f}%)")
        print(f"  Chosen action : {act}")
        print("-" * 40)