# DARWIN HAMMER — match 3865, survivor 2
# gen: 5
# parent_a: hard_truth_math.py (gen0)
# parent_b: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_fisher_m432_s1.py (gen4)
# born: 2026-05-29T23:52:07Z

# fusion_math.py
"""
Parent A: hard_truth_math (stylometry features/classifier helpers, ISO/vector parsing)
Parent B: hybrid_hybrid_gliner_hybrid_krampus_brain_m30_s2 (deterministic span matcher, entropy/pheromone decay) &
  hybrid_hybrid_fisher_locali_hybrid_ternary_lens__m71_s1 (Gaussian beam, Fisher information, regex-driven ternary dimension & SSIM)

Mathematical Bridge:
The bridge is the notion of *information gain*.  Parent A models it via stylometry features (e.g. pronoun usage) and ISO/vectors, while Parent B quantifies it analytically with Fisher information of a Gaussian beam.  This module maps each stylometry feature vector to a Gaussian beam whose intensity is the feature score.
The Fisher information of that beam becomes the raw pheromone signal.
A ternary dimension vector derived from regex matches on the text modulates the pheromone half-life, while the SSIM between vectorised text provides a similarity measure that can be used to reinforce or attenuate pheromone updates.
"""
import math
import random
import sys
import pathlib
import uuid
import re
from datetime import datetime, timezone
import numpy as np

# ----------------------------------------------------------------------
# Stylometry features/classifier helpers (adapted from Parent A)
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

PERSONA_PATTERNS: dict[str, re.Pattern[str]] = {
    "ponyboy": re.compile(r"\bponyboy\b", re.I),
    "northern_strike": re.compile(r"\bnorthern\.?strike\b", re.I),
    "lucidota": re.compile(r"\blucidota\b|\bluci\b", re.I),
    "indy_reads": re.compile(r"\bindy[_ -]?reads\b|\bindy\b", re.I),
    "zachary": re.compile(r"\bzachary\b|\bzach\b", re.I),
    "operator": re.compile(r"\boperator\b", re.I),
}

PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {cat: sum(cnt[w] for w in vocab) / total for cat, vocab in FUNCTION_CATS.items()}

def lsm_score(a: dict[str, float], b: dict[str, float]) -> tuple[float, dict[str, float]]:
    detail: dict[str, float] = {}
    vals: list[float] = []
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
        vals.append(score)
    return np.mean(vals), detail

# ----------------------------------------------------------------------
# Deterministic span matcher, entropy/pheromone decay (adapted from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """Deterministic span produced by the label matcher."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"

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
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

def pheromone_update(entry: PheromoneEntry, signal_value: float) -> PheromoneEntry:
    decay_factor = math.exp(-1 * (entry.age_seconds() / entry.half_life_seconds))
    new_signal_value = entry.signal_value * decay_factor + signal_value * (1 - decay_factor)
    return PheromoneEntry(entry.surface_key, entry.signal_kind, new_signal_value, entry.half_life_seconds)

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def stylometry_to_pheromone(text: str) -> PheromoneEntry:
    lsm_vector_ = lsm_vector(text)
    pheromone_signal = lsm_score(lsm_vector_, lsm_vector_("default_text"))[0]
    return PheromoneEntry("text", "stylometry", pheromone_signal, 3600)

def gaussian_beam(text: str, pheromone_entry: PheromoneEntry) -> np.ndarray:
    beam_intensity = pheromone_entry.signal_value
    beam_width = 2 * math.sqrt(2 * math.log(2)) * math.sqrt(beam_intensity)
    return np.exp(-(np.linspace(-beam_width, beam_width, 100) ** 2) / 2)

def pheromone_from_gaussian(text: str, pheromone_entry: PheromoneEntry) -> PheromoneEntry:
    gaussian_beam_ = gaussian_beam(text, pheromone_entry)
    pheromone_signal = np.mean(gaussian_beam_)
    return PheromoneEntry("text", "gaussian", pheromone_signal, 3600)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    text = "Hello, world!"
    pheromone_entry = stylometry_to_pheromone(text)
    pheromone_entry = pheromone_update(pheromone_entry, 0.5)
    gaussian_beam_ = gaussian_beam(text, pheromone_entry)
    pheromone_entry = pheromone_from_gaussian(text, pheromone_entry)
    print(pheromone_entry.signal_value)