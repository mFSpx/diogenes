# DARWIN HAMMER — match 651, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s0.py (gen4)
# parent_b: hybrid_krampus_chrono_infotaxis_m67_s0.py (gen1)
# born: 2026-05-29T23:30:16Z

"""Hybrid Date‑Certainty Bandit Algorithm
====================================

This module fuses **Parent A** (epistemic certainty & bandit selection) with **Parent B**
(date parsing & entropy quantification).  

The mathematical bridge is the *duality between certainty and entropy*:

*   In Parent A a `CertaintyFlag` stores a confidence value `confidence_bps ∈ [0,10000]`.
*   In Parent B the uncertainty of a parsed date is measured by Shannon entropy
    `H = -∑ p_i log₂ p_i`.

We map the normalized entropy `H / H_max` to a certainty score  


confidence = (1 – H / H_max) * 10000


where `H_max = log₂(N)` for `N` equally‑likely parsing hypotheses.  
The resulting `CertaintyFlag` is then fed to a lightweight multi‑armed bandit
(UCB1) that chooses the *best date interpretation* among competing candidates.

The three public functions demonstrate the hybrid workflow:

1. ``parse_date_with_certainty`` – parses a string, computes entropy and builds a
   `CertaintyFlag`.
2. ``bandit_select`` – given a list of `CertaintyFlag`s, selects an arm using UCB1.
3. ``best_date_action`` – end‑to‑end routine that parses many date strings,
   creates certainty flags and returns the most promising interpretation.

Only the Python standard library and ``numpy`` are used.
"""

import math
import random
import sys
from pathlib import Path
import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Set, Callable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Epistemic certainty structures
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable container for epistemic certainty."""
    label: str
    confidence_bps: int                # basis points, 0 … 10 000
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

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
# Parent B – Date parsing + entropy utilities
# ----------------------------------------------------------------------
CONTENT_DATE_PATTERNS = [
    ("frontmatter_date", re.compile(
        r"(?im)^\s*(?:date|created|created_at|created at|generated|timestamp|time|filed|updated|modified)\s*[:=]\s*[\"']?"
        r"((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?"
        r"(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)"
    )),
    ("iso_inline", re.compile(
        r"\b((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?"
        r"(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)\b"
    )),
    ("compact_yyyymmdd", re.compile(
        r"\b((?:20|19)\d{2})(\d{2})(\d{2})(?:[T_-]?(\d{2})(\d{2})(\d{2})?)?Z?\b"
    )),
]

MONTH_NAME_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+([0-3]?\d)(?:st|nd|rd|th)?,?\s+((?:20|19)\d{2})\b",
    re.I,
)

def _shannon_entropy(probs: List[float]) -> float:
    """Return Shannon entropy (bits) for a probability mass function."""
    total = sum(probs)
    if total <= 0:
        raise ValueError("probability mass must be positive")
    probs = [p / total for p in probs]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def calculate_date_entropy(num_hypotheses: int) -> float:
    """Maximum entropy for a uniform distribution over `num_hypotheses`."""
    if num_hypotheses <= 0:
        return 0.0
    return math.log2(num_hypotheses)

# ----------------------------------------------------------------------
# Hybrid core – mapping entropy → certainty & bandit decision
# ----------------------------------------------------------------------
def _confidence_from_entropy(entropy_val: float, max_entropy: float) -> int:
    """Map entropy to confidence basis points."""
    if max_entropy == 0:
        return 10_000
    ratio = max(0.0, min(1.0, 1.0 - entropy_val / max_entropy))
    return int(round(ratio * 10_000))

def _label_from_confidence(confidence_bps: int) -> str:
    """Assign an epistemic flag based on confidence thresholds."""
    if confidence_bps >= 9_000:
        return "FACT"
    if confidence_bps >= 7_000:
        return "PROBABLE"
    if confidence_bps >= 4_000:
        return "POSSIBLE"
    if confidence_bps >= 1_000:
        return "SURE_MAYBE"
    return "BULLSHIT"

def parse_date_with_certainty(date_str: str,
                              authority: str = "date_parser",
                              rationale_prefix: str = "parsed via hybrid") -> Tuple[Optional[dt.datetime], CertaintyFlag]:
    """
    Parse ``date_str`` using the patterns from Parent B,
    compute the entropy of the parsing hypotheses,
    and return a ``CertaintyFlag`` that encodes the resulting confidence.
    """
    matches: List[Tuple[str, dt.datetime]] = []

    # Try each regex; on success, build a datetime object.
    for name, pattern in CONTENT_DATE_PATTERNS:
        m = pattern.search(date_str)
        if not m:
            continue
        try:
            if name == "compact_yyyymmdd":
                year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                hour = int(m.group(4) or 0)
                minute = int(m.group(5) or 0)
                second = int(m.group(6) or 0)
                dt_obj = dt.datetime(year, month, day, hour, minute, second, tzinfo=dt.timezone.utc)
            else:
                # ISO‑like strings – let fromisoformat handle most cases
                iso = m.group(1)
                # Normalise separators
                iso = iso.replace('_', '-').replace(' ', 'T')
                dt_obj = dt.datetime.fromisoformat(iso.rstrip('Z'))
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
            matches.append((name, dt_obj))
        except Exception:
            continue

    # Month‑name fallback
    if not matches:
        m = MONTH_NAME_RE.search(date_str)
        if m:
            month_name, day, year = m.group(1), int(m.group(2)), int(m.group(3))
            month = dt.datetime.strptime(month_name[:3], "%b").month
            dt_obj = dt.datetime(year, month, day, tzinfo=dt.timezone.utc)
            matches.append(("month_name", dt_obj))

    # Entropy computation
    num_hyp = len(matches) if matches else 1
    max_entropy = calculate_date_entropy(num_hyp)
    # Uniform hypothesis probabilities → entropy = max_entropy
    entropy_val = max_entropy if matches else 0.0

    confidence = _confidence_from_entropy(entropy_val, max_entropy)
    label = _label_from_confidence(confidence)

    rationale = f"{rationale_prefix}: {date_str!r}, patterns={num_hyp}"
    flag = CertaintyFlag(
        label=label,
        confidence_bps=confidence,
        authority_class=authority,
        rationale=rationale,
        evidence_refs=tuple(p for p, _ in matches),
        generated_at=dt.datetime.utcnow().isoformat() + "Z",
    )

    # Return the *most* specific datetime (first match) or None.
    parsed_dt = matches[0][1] if matches else None
    return parsed_dt, flag

# ----------------------------------------------------------------------
# Simple UCB1 bandit using certainty as reward
# ----------------------------------------------------------------------
class UCB1Bandit:
    """Multi‑armed bandit with Upper Confidence Bound (UCB1)."""

    def __init__(self) -> None:
        self.counts: List[int] = []          # pulls per arm
        self.values: List[float] = []        # average reward per arm

    def select_arm(self) -> int:
        """Return index of arm to pull next."""
        total_counts = sum(self.counts)
        if total_counts == 0:
            return random.randrange(len(self.counts))
        ucb_values = [
            self.values[i] + math.sqrt(2 * math.log(total_counts) / self.counts[i])
            for i in range(len(self.counts))
        ]
        return int(np.argmax(ucb_values))

    def update(self, chosen_arm: int, reward: float) -> None:
        """Incorporate observed reward for ``chosen_arm``."""
        if chosen_arm >= len(self.counts):
            # extend structures lazily
            needed = chosen_arm - len(self.counts) + 1
            self.counts.extend([0] * needed)
            self.values.extend([0.0] * needed)

        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # incremental average
        new_value = ((n - 1) * value + reward) / n
        self.values[chosen_arm] = new_value

def bandit_select(flags: List[CertaintyFlag],
                 bandit: Optional[UCB1Bandit] = None) -> int:
    """
    Given a list of ``CertaintyFlag`` objects, use a UCB1 bandit to select
    the most promising index.  The reward for an arm is the flag's
    ``confidence_bps`` (scaled to [0,1]).
    """
    if not flags:
        raise ValueError("empty flag list")

    if bandit is None:
        bandit = UCB1Bandit()
        # initialise one arm per flag
        bandit.counts = [0] * len(flags)
        bandit.values = [0.0] * len(flags)

    # Choose arm
    arm = bandit.select_arm()
    # Observe reward (scaled)
    reward = flags[arm].confidence_bps / 10_000.0
    bandit.update(arm, reward)
    return arm

# ----------------------------------------------------------------------
# End‑to‑end hybrid operation
# ----------------------------------------------------------------------
def best_date_action(date_strings: List[str]) -> Tuple[Optional[dt.datetime], CertaintyFlag]:
    """
    Parse each string, produce a ``CertaintyFlag`` and run a bandit over the
    resulting candidates.  Returns the datetime and flag of the selected arm.
    """
    parsed: List[Tuple[Optional[dt.datetime], CertaintyFlag]] = [
        parse_date_with_certainty(s) for s in date_strings
    ]
    flags = [flag for _, flag in parsed]
    chosen_idx = bandit_select(flags)
    return parsed[chosen_idx]  # (datetime|None, CertaintyFlag)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_dates = [
        "Created: 2023-07-15T14:23:00Z",
        "20230716 120000",
        "July 17, 2023",
        "2023/07/18",
        "random text without date",
    ]

    dt_obj, flag = best_date_action(sample_dates)
    print("Chosen datetime :", dt_obj)
    print("Certainty flag  :", flag)
    print("Flag dict       :", flag.as_dict())
    sys.exit(0)