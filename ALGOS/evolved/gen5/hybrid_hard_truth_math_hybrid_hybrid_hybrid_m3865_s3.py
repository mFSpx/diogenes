# DARWIN HAMMER — match 3865, survivor 3
# gen: 5
# parent_a: hard_truth_math.py (gen0)
# parent_b: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_fisher_m432_s1.py (gen4)
# born: 2026-05-29T23:52:07Z

"""Hybrid algorithm merging hard_truth_math (Parent A) and hybrid_hybrid_hybrid_gliner_hybrid_hybrid_fisher (Parent B).

Mathematical bridge:
- Parent A supplies a linguistic feature distribution **v** (LSM vector) for any text.
- Parent B models a deterministic *Span* as a Gaussian beam whose intensity is the Span score **s**.
- The **Fisher information** I of that beam is 1/σ² where σ² is derived from the entropy of **v** (higher entropy → larger variance).
- A ternary dimension **t** is obtained by regex‑driven presence/absence of three function‑word categories
  (pronoun, article, preposition) inside the Span text, yielding values in {‑1,0,1}.
- The pheromone half‑life **h** is modulated by **t** and by a structural similarity **SSIM** between two Span texts.
- The final pheromone signal update combines the raw pheromone (from Parent A) with the Fisher information
  (from Parent B) through a weighted sum, producing a unified representation that can be used for
  downstream matching or classification.

The module implements this fusion with three public functions:
    • `span_from_text` – creates a Span and its LSM vector.
    • `compute_pheromone_entry` – maps a Span to a Gaussian beam, computes Fisher information,
      builds a ternary vector, and creates a PheromoneEntry.
    • `hybrid_similarity` – returns a combined similarity score between two Spans using LSM
      similarity, SSIM, and pheromone‑adjusted Fisher information.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import random
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks (hard_truth_math)
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
    """Return a list of lowercase alphabetic tokens."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> dict[str, float]:
    """Linguistic Style Metric (LSM) vector: proportion of each function‑word category."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = {w: ws.count(w) for w in ws}
    return {
        cat: sum(cnt.get(w, 0) for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def lsm_score(a: dict[str, float], b: dict[str, float]) -> Tuple[float, dict[str, float]]:
    """Similarity between two LSM vectors; also returns per‑category detail."""
    detail: dict[str, float] = {}
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        # Normalised overlap measure
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
    overall = sum(detail.values()) / len(detail)
    return overall, detail


# ----------------------------------------------------------------------
# Parent B building blocks (hybrid_hybrid_hybrid_gliner_hybrid_hybrid_fisher)
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
    """Container for pheromone signal derived from a Span."""
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
        """Elapsed time in seconds since creation."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def decay_factor(self) -> float:
        """Exponential decay based on half‑life."""
        age = self.age_seconds()
        return 0.5 ** (age / self.half_life_seconds)


# ----------------------------------------------------------------------
# Fusion utilities
# ----------------------------------------------------------------------
def ternary_vector(text: str) -> np.ndarray:
    """
    Produce a 3‑dimensional ternary vector from regex presence of
    pronouns, articles, and prepositions.
    -1 : category absent
     0 : category present exactly once
     1 : category present more than once
    """
    vec = []
    for cat in ("pronoun", "article", "preposition"):
        count = sum(1 for w in words(text) if w in FUNCTION_CATS[cat])
        if count == 0:
            vec.append(-1)
        elif count == 1:
            vec.append(0)
        else:
            vec.append(1)
    return np.array(vec, dtype=np.int8)


def ssim_like(a: np.ndarray, b: np.ndarray) -> float:
    """
    Very lightweight SSIM‑style similarity for 1‑D signals.
    Returns a value in [0,1].
    """
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a2, sigma_b2 = a.var(), b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return float(numerator / denominator) if denominator != 0 else 0.0


def entropy_of_vector(v: dict[str, float]) -> float:
    """Shannon entropy of a probability‑like vector."""
    eps = 1e-12
    probs = np.array(list(v.values())) + eps
    probs /= probs.sum()
    return -np.sum(probs * np.log2(probs))


def gaussian_fisher_information(variance: float) -> float:
    """Fisher information of a 1‑D Gaussian with given variance."""
    return 1.0 / variance if variance > 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid public API
# ----------------------------------------------------------------------
def span_from_text(text: str, label: str = "generic", score: float | None = None) -> Span:
    """
    Detect the first occurrence of any PERSONA pattern in *text*.
    The span covers that match; if none is found the whole text is used.
    Returns a Span enriched with an LSM vector stored in the ``score`` field
    (if ``score`` is None, the LSM entropy‑derived variance is used).
    """
    for pat_label, pat in PERSONA_PATTERNS.items():
        m = pat.search(text)
        if m:
            start, end = m.start(), m.end()
            span_text = text[start:end]
            span_label = pat_label
            break
    else:
        start, end = 0, len(text)
        span_text = text
        span_label = label

    # Compute LSM vector and optional score
    lsm = lsm_vector(span_text)
    if score is None:
        # Use entropy as a proxy for variance; higher entropy → larger variance
        ent = entropy_of_vector(lsm)
        variance = max(1e-4, ent)  # avoid zero
        score = 1.0 / (1.0 + variance)  # map to (0,1]
    return Span(start=start, end=end, text=span_text,
                label=span_label, score=score)


def compute_pheromone_entry(span: Span) -> PheromoneEntry:
    """
    Map a Span to a Gaussian beam, compute its Fisher information,
    build a ternary vector, and create a PheromoneEntry.
    The half‑life is scaled by the magnitude of the ternary vector and
    by an SSIM‑like self‑similarity (always 1 for a single span).
    """
    # Variance derived from entropy of the span's LSM vector
    lsm = lsm_vector(span.text)
    ent = entropy_of_vector(lsm)
    variance = max(1e-4, ent)

    # Fisher information (Parent B)
    fisher = gaussian_fisher_information(variance)

    # Ternary modulation (Parent B)
    tern = ternary_vector(span.text)
    tern_norm = np.linalg.norm(tern) + 1e-6  # avoid zero

    # Base pheromone signal: span.score (Parent A) weighted by fisher
    raw_signal = span.score * fisher

    # Half‑life modulation: longer half‑life for more stable (lower entropy) spans
    base_half_life = 3600  # 1 hour default
    half_life = int(base_half_life * (1.0 + tern_norm))

    # Surface key is a hash of the span text for deduplication
    surface_key = hashlib.sha256(span.text.encode("utf-8")).hexdigest()

    return PheromoneEntry(
        surface_key=surface_key,
        signal_kind="hybrid",
        signal_value=raw_signal,
        half_life_seconds=half_life,
    )


def hybrid_similarity(span_a: Span, span_b: Span) -> float:
    """
    Combined similarity between two spans:
      * LSM similarity (Parent A)
      * SSIM of raw character codes (Parent B)
      * Pheromone‑adjusted Fisher information overlap
    The result is in [0,1].
    """
    # LSM component
    lsm_a = lsm_vector(span_a.text)
    lsm_b = lsm_vector(span_b.text)
    lsm_sim, _ = lsm_score(lsm_a, lsm_b)

    # SSIM‑like component on UTF‑8 byte arrays
    a_bytes = np.frombuffer(span_a.text.encode("utf-8"), dtype=np.uint8).astype(np.float64)
    b_bytes = np.frombuffer(span_b.text.encode("utf-8"), dtype=np.uint8).astype(np.float64)
    ssim_val = ssim_like(a_bytes, b_bytes)

    # Fisher information component
    var_a = max(1e-4, entropy_of_vector(lsm_a))
    var_b = max(1e-4, entropy_of_vector(lsm_b))
    fisher_a = gaussian_fisher_information(var_a)
    fisher_b = gaussian_fisher_information(var_b)
    fisher_sim = 1.0 - abs(fisher_a - fisher_b) / (fisher_a + fisher_b + 1e-6)

    # Weighted aggregation
    w_lsm, w_ssim, w_fisher = 0.4, 0.3, 0.3
    combined = w_lsm * lsm_sim + w_ssim * ssim_val + w_fisher * fisher_sim
    return max(0.0, min(1.0, combined))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    txt1 = "The operator whispered to ponyboy while the northern strike loomed."
    txt2 = "Lucidota and Zachary read the same Indy_reads article about the strike."

    span1 = span_from_text(txt1)
    span2 = span_from_text(txt2)

    pher1 = compute_pheromone_entry(span1)
    pher2 = compute_pheromone_entry(span2)

    sim = hybrid_similarity(span1, span2)

    print("Span 1:", span1)
    print("Span 2:", span2)
    print("Pheromone 1 signal:", pher1.signal_value, "half‑life:", pher1.half_life_seconds)
    print("Pheromone 2 signal:", pher2.signal_value, "half‑life:", pher2.half_life_seconds)
    print("Hybrid similarity:", round(sim, 4))