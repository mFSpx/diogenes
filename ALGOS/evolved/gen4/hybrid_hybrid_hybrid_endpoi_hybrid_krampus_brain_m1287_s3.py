# DARWIN HAMMER — match 1287, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_hybrid_decisi_m189_s0.py (gen3)
# parent_b: hybrid_krampus_brainmap_hybrid_pheromone_inf_m37_s1.py (gen2)
# born: 2026-05-29T23:36:32Z

import re
import uuid
import math
import numpy as np
from collections import Counter
from datetime import datetime, timezone
from typing import List, Dict, Tuple


# ----------------------------------------------------------------------
# Feature detection regular expressions (kept for possible future extensions)
# ----------------------------------------------------------------------
_EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
_PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
_DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)
_SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|"
    r"advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
_BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|"
    r"protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
_OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe)\b",
    re.I,
)

_FEATURE_REGEXES: List[Tuple[str, re.Pattern]] = [
    ("evidence", _EVIDENCE_RE),
    ("planning", _PLANNING_RE),
    ("delay", _DELAY_RE),
    ("support", _SUPPORT_RE),
    ("boundary", _BOUNDARY_RE),
    ("outcome", _OUTCOME_RE),
]


# ----------------------------------------------------------------------
# Pheromone infrastructure
# ----------------------------------------------------------------------
class PheromoneEntry:
    """A single pheromone signal with exponential decay."""

    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = float(signal_value)
        self.half_life_seconds = int(half_life_seconds)
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay factor since the last decay event."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        """Apply decay in‑place and update the timestamp."""
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)

    def normalized_value(self) -> float:
        """Return the current (decayed) signal value, clipped to non‑negative."""
        self.apply_decay()
        return max(self.signal_value, 0.0)


class PheromoneStore:
    """Simple in‑memory singleton store for demo purposes."""

    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def clear(cls) -> None:
        cls._entries.clear()


# ----------------------------------------------------------------------
# Entropy utilities
# ----------------------------------------------------------------------
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Return a list of lower‑cased word tokens."""
    return [t.lower() for t in _WORD_RE.findall(text)]


def calculate_shannon_entropy(text: str) -> float:
    """
    Compute Shannon entropy of a text based on word frequencies.
    Returns 0.0 for empty or whitespace‑only strings.
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    total = len(tokens)
    entropy = -sum((count / total) * math.log2(count / total) for count in freq.values())
    return entropy


def calculate_pheromone_entropy(pheromone_entries: List[PheromoneEntry]) -> float:
    """
    Compute entropy of pheromone signals, weighting each kind by its
    decayed signal value (interpreted as a probability mass).
    Returns 0.0 for an empty list.
    """
    if not pheromone_entries:
        return 0.0

    # Gather decayed values per kind
    mass_per_kind: Dict[str, float] = {}
    total_mass = 0.0
    for entry in pheromone_entries:
        mass = entry.normalized_value()
        if mass <= 0:
            continue
        mass_per_kind[entry.signal_kind] = mass_per_kind.get(entry.signal_kind, 0.0) + mass
        total_mass += mass

    if total_mass == 0.0:
        return 0.0

    entropy = -sum(
        (mass / total_mass) * math.log2(mass / total_mass) for mass in mass_per_kind.values()
    )
    return entropy


def _categorize_token(token: str) -> str:
    """Map a token to a high‑level feature category, or 'other'."""
    for name, regex in _FEATURE_REGEXES:
        if regex.search(token):
            return name
    return "other"


def calculate_joint_entropy(
    text: str, pheromone_entries: List[PheromoneEntry]
) -> float:
    """
    Compute joint entropy H(Text, Pheromone) by constructing a joint
    distribution over (feature_category, signal_kind). The probability
    mass for each pair is proportional to the product of:
      * frequency of the feature category in the text,
      * decayed signal value of the pheromone kind.
    This creates a deeper mathematical coupling between the two systems.
    """
    tokens = _tokenize(text)
    if not tokens and not pheromone_entries:
        return 0.0

    # Feature category frequencies
    cat_counts = Counter(_categorize_token(t) for t in tokens)
    total_cats = sum(cat_counts.values()) or 1  # avoid division by zero

    # Decayed pheromone masses per kind
    mass_per_kind: Dict[str, float] = {}
    total_mass = 0.0
    for entry in pheromone_entries:
        mass = entry.normalized_value()
        if mass <= 0:
            continue
        mass_per_kind[entry.signal_kind] = mass_per_kind.get(entry.signal_kind, 0.0) + mass
        total_mass += mass
    total_mass = total_mass or 1.0  # avoid division by zero

    # Joint probability table
    joint_prob: Dict[Tuple[str, str], float] = {}
    for cat, cat_cnt in cat_counts.items():
        p_cat = cat_cnt / total_cats
        for kind, kind_mass in mass_per_kind.items():
            p_kind = kind_mass / total_mass
            joint_prob[(cat, kind)] = p_cat * p_kind

    # Joint entropy
    joint_entropy = -sum(p * math.log2(p) for p in joint_prob.values() if p > 0)
    return joint_entropy


def hybrid_decision(text: str, pheromone_entries: List[PheromoneEntry]) -> float:
    """
    Produce a decision score that reflects the information gain obtained
    by observing pheromone signals given the textual context.

    Decision = (H_text + H_pheromone - H_joint) / (H_text + H_pheromone + ε)

    The numerator is the mutual information I(Text;Pheromone), i.e. the
    reduction in uncertainty about the pheromone distribution once the
    text is known. The denominator normalises the score to [0, 1].
    """
    ε = 1e-12  # numerical stability

    h_text = calculate_shannon_entropy(text)
    h_pher = calculate_pheromone_entropy(pheromone_entries)
    h_joint = calculate_joint_entropy(text, pheromone_entries)

    mutual_info = h_text + h_pher - h_joint
    normaliser = h_text + h_pher + ε

    decision_score = mutual_info / normaliser
    # Clip to [0, 1] for downstream consumers
    return max(0.0, min(1.0, decision_score))


# ----------------------------------------------------------------------
# Demonstration entry point
# ----------------------------------------------------------------------
def _demo() -> None:
    sample_text = (
        "The evidence confirms the plan, but we need to wait for the final outcome. "
        "Support from the team is essential."
    )
    # Create a heterogeneous set of pheromone entries
    kinds = ["evidence", "plan", "delay", "support", "boundary"]
    entries = [
        PheromoneEntry(
            surface_key="demo_surface",
            signal_kind=random.choice(kinds),
            signal_value=random.uniform(0.5, 2.0),
            half_life_seconds=3600,
        )
        for _ in range(25)
    ]

    # Populate store (optional, just to show usage)
    for e in entries:
        PheromoneStore.add(e)

    score = hybrid_decision(sample_text, entries)
    print(f"Hybrid decision score (0 = no information gain, 1 = max gain): {score:.4f}")


if __name__ == "__main__":
    _demo()