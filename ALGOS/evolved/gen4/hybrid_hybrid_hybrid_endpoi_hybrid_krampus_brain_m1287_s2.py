# DARWIN HAMMER — match 1287, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_hybrid_decisi_m189_s0.py (gen3)
# parent_b: hybrid_krampus_brainmap_hybrid_pheromone_inf_m37_s1.py (gen2)
# born: 2026-05-29T23:36:32Z

import math
import numpy as np
import re
import uuid
import datetime
import collections

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

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
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe)\b",
    re.I,
)

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.datetime.now(datetime.timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.datetime.now(datetime.timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Return the multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.datetime.now(datetime.timezone.utc)


class PheromoneStore:
    """Singleton-like in-memory store for demo purposes."""
    _entries: dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> list[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]


def calculate_shannon_entropy(frequencies: collections.Counter) -> float:
    """Calculate the Shannon Entropy of a given frequency distribution."""
    entropy = 0.0
    for count in frequencies.values():
        prob = count / sum(frequencies.values())
        if prob > 0:
            entropy -= prob * math.log2(prob)
    
    return entropy


def calculate_text_entropy(text: str) -> float:
    """Calculate the Shannon Entropy of a given text."""
    words = text.split()
    freq = collections.Counter(words)
    return calculate_shannon_entropy(freq)


def calculate_pheromone_entropy(pheromone_entries: list[PheromoneEntry]) -> float:
    """Calculate the entropy of a list of pheromone entries."""
    signals = [e.signal_kind for e in pheromone_entries]
    freq = collections.Counter(signals)
    return calculate_shannon_entropy(freq)


def hybrid_decision(text: str, pheromone_entries: list[PheromoneEntry]) -> float:
    """Make a decision based on the Shannon Entropy of the text and the pheromone entries."""
    text_entropy = calculate_text_entropy(text)
    pheromone_entropy = calculate_pheromone_entropy(pheromone_entries)
    
    # Use a weighted sum to combine the two entropies
    decision = 0.7 * text_entropy + 0.3 * pheromone_entropy
    
    return decision


def main():
    text = "This is a test text."
    pheromone_entries = [PheromoneEntry("surface_key", "signal_kind", 1.0, 3600) for _ in range(10)]
    
    decision = hybrid_decision(text, pheromone_entries)
    print("Decision:", decision)


if __name__ == "__main__":
    main()