# DARWIN HAMMER — match 273, survivor 2
# gen: 4
# parent_a: hybrid_krampus_brainmap_hybrid_pheromone_inf_m37_s1.py (gen2)
# parent_b: hybrid_indy_learning_vector_hybrid_hybrid_geomet_m113_s1.py (gen3)
# born: 2026-05-29T23:28:02Z

"""Hybrid Krampus‑Brainmap / Indy‑Learning Vector Algorithm

This module fuses the two parent algorithms:

* **krampus_brainmap + hybrid_pheromone_infotaxis** – provides a high‑dimensional
  vector representation (the “brain map”) and an infotaxis decision process that
  uses entropy and information‑gain on pheromone signals.

* **indy_learning_vector + hybrid_hybrid_geometric_pro** – supplies a deterministic
  tokenisation / chunking pipeline that builds sparse term‑frequency vectors from
  free‑form text using an ontology of GO‑like terms.

**Mathematical bridge**

The bridge is the shared vector space.  The Indy‑learning pipeline yields a
term‑frequency vector **v** ∈ ℝⁿ (n = number of ontology terms).  The pheromone
store holds a signal value s_i for each term i, forming a discrete probability
distribution p_i = s_i / Σ_j s_j.  Entropy H(p) = – Σ_i p_i log p_i quantifies the
uncertainty of the pheromone field.  Adding information from a new text chunk
updates the signal values: s_i ← s_i + α·v_i.  The resulting change in entropy
ΔH = H_before – H_after is the *information gain* that drives the infotaxis
policy.  Thus the hybrid algorithm continuously maps textual vectors onto the
pheromone field, steering it toward lower entropy (more certain) configurations.

The implementation below provides three core functions demonstrating this
integration:
1. `build_term_vector` – tokenises text and builds the ontology‑based vector.
2. `entropy` – computes Shannon entropy of the current pheromone distribution.
3. `infotaxis_update` – injects the vector into the pheromone store and returns
   the information gain.

A simple smoke test runs the pipeline on a sample paragraph."""

import json
import math
import random
import re
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Ontology / token utilities (from parent B)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[0]
WORD_RE = re.compile(r"\S+")
DEFAULT_TERMS = (
    "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "ACTION", "EVENT", "TIME", "EVIDENCE",
    "CLAIM", "HYPOTHESIS", "SIGNAL", "PATTERN", "TOOL", "ALGORITHM", "BOOK",
    "SOURCE", "LEAD", "LOCATION", "LAW", "RULE",
)


def load_ontology_terms(root: Path = ROOT) -> List[str]:
    """Load ontology terms from OFFICIAL_ONTOLOGY.json; fall back to defaults."""
    p = root / "OFFICIAL_ONTOLOGY.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        terms = data.get("active_terms") or []
        return [str(t).upper() for t in terms if str(t).strip()]
    except Exception:
        return list(DEFAULT_TERMS)


ONTOLOGY_TERMS = load_ontology_terms()


def tokenize(text: str) -> List[Dict[str, Any]]:
    """Return a list of token dicts with start/end character offsets."""
    return [
        {"token": m.group(0), "start": m.start(), "end": m.end()}
        for m in WORD_RE.finditer(text)
    ]


def build_term_vector(text: str) -> np.ndarray:
    """
    Produce a dense term‑frequency vector aligned with ONTOLOGY_TERMS.

    The vector is L2‑normalized; zero‑vectors are left as‑is.
    """
    tokens = tokenize(text)
    term_counts = Counter()
    for tok in tokens:
        t = tok["token"].upper()
        if t in ONTOLOGY_TERMS:
            term_counts[t] += 1
    vec = np.zeros(len(ONTOLOGY_TERMS), dtype=float)
    for idx, term in enumerate(ONTOLOGY_TERMS):
        vec[idx] = term_counts.get(term, 0)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ----------------------------------------------------------------------
# Pheromone infrastructure (from parent A)
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: int = 3600,
    ):
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
        """Multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """In‑memory singleton store for pheromone entries."""

    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_all(cls) -> None:
        for e in cls._entries.values():
            e.apply_decay()


# ----------------------------------------------------------------------
# Entropy / Infotaxis core (mathematical bridge)
# ----------------------------------------------------------------------
def entropy(signal_values: np.ndarray) -> float:
    """Shannon entropy of a non‑negative signal vector."""
    total = signal_values.sum()
    if total == 0:
        return 0.0
    probs = signal_values / total
    # Guard against log(0) by adding a tiny epsilon
    eps = np.finfo(float).eps
    return -np.sum(probs * np.log(probs + eps))


def infotaxis_update(
    term_vector: np.ndarray,
    surface_key: str,
    learning_rate: float = 0.1,
    half_life_seconds: int = 3600,
) -> float:
    """
    Inject a term vector into the pheromone field for *surface_key*.

    Returns the information gain (entropy reduction) achieved by this update.
    """
    # Gather current entries for the surface
    entries = PheromoneStore.get_by_surface(surface_key)
    # Map signal_kind (ontology term) -> entry
    kind_to_entry = {e.signal_kind: e for e in entries}

    # Build current signal array aligned with ontology order
    current_signals = np.array(
        [
            kind_to_entry.get(term, PheromoneEntry(surface_key, term, 0.0, half_life_seconds)).signal_value
            for term in ONTOLOGY_TERMS
        ],
        dtype=float,
    )

    H_before = entropy(current_signals)

    # Apply learning: s_i ← s_i + α·v_i
    updated_signals = current_signals + learning_rate * term_vector

    # Write back updated values (create missing entries)
    for idx, term in enumerate(ONTOLOGY_TERMS):
        if term in kind_to_entry:
            kind_to_entry[term].signal_value = float(updated_signals[idx])
        else:
            new_entry = PheromoneEntry(
                surface_key=surface_key,
                signal_kind=term,
                signal_value=float(updated_signals[idx]),
                half_life_seconds=half_life_seconds,
            )
            PheromoneStore.add(new_entry)

    H_after = entropy(updated_signals)
    info_gain = H_before - H_after
    return info_gain


# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def process_text_chunk(text: str, surface_key: str) -> Dict[str, Any]:
    """
    Full hybrid pipeline for a single text chunk:

    1. Build the ontology‑based term vector.
    2. Update the pheromone store via infotaxis.
    3. Return a summary dict with vector norm, entropy change, and
       top‑3 pheromone signals.
    """
    vec = build_term_vector(text)
    gain = infotaxis_update(vec, surface_key)

    # Refresh decay before reading distribution
    PheromoneStore.decay_all()
    entries = PheromoneStore.get_by_surface(surface_key)
    signals = np.array([e.signal_value for e in entries], dtype=float)
    current_entropy = entropy(signals)

    # Sort entries by signal strength
    top_entries = sorted(entries, key=lambda e: e.signal_value, reverse=True)[:3]
    top_signals = [
        {"term": e.signal_kind, "value": e.signal_value} for e in top_entries
    ]

    return {
        "vector_norm": float(np.linalg.norm(vec)),
        "information_gain": gain,
        "current_entropy": current_entropy,
        "top_signals": top_signals,
    }


def batch_process(texts: List[str], surface_key: str) -> List[Dict[str, Any]]:
    """
    Apply `process_text_chunk` to an iterable of texts, returning the list of
    summaries.  This demonstrates the hybrid algorithm operating over a batch.
    """
    results = []
    for txt in texts:
        results.append(process_text_chunk(txt, surface_key))
    return results


def summarize_surface(surface_key: str) -> Dict[str, Any]:
    """
    Produce a concise summary of the pheromone state for *surface_key*:
    total signal mass, entropy, and the dominant term.
    """
    entries = PheromoneStore.get_by_surface(surface_key)
    if not entries:
        return {"total_signal": 0.0, "entropy": 0.0, "dominant_term": None}
    signals = np.array([e.signal_value for e in entries], dtype=float)
    total = float(signals.sum())
    ent = entropy(signals)
    dominant = max(entries, key=lambda e: e.signal_value)
    return {
        "total_signal": total,
        "entropy": ent,
        "dominant_term": {"term": dominant.signal_kind, "value": dominant.signal_value},
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_paragraphs = [
        "The algorithm predicts an EVENT based on observed SIGNAL patterns.",
        "Evidence suggests that the ENTITY has ATTRIBUTE X and ACTION Y.",
        "A new TOOL was introduced to improve the PROCESS of DATA analysis.",
    ]
    surface = "demo_surface"

    print("Running hybrid batch processing...")
    batch_results = batch_process(sample_paragraphs, surface)
    for i, res in enumerate(batch_results, 1):
        print(f"\nChunk {i}:")
        print(f"  Vector norm          : {res['vector_norm']:.4f}")
        print(f"  Information gain    : {res['information_gain']:.6f}")
        print(f"  Current entropy     : {res['current_entropy']:.6f}")
        print(f"  Top signals         : {res['top_signals']}")

    final_summary = summarize_surface(surface)
    print("\nFinal surface summary:")
    print(f"  Total signal mass   : {final_summary['total_signal']:.4f}")
    print(f"  Entropy             : {final_summary['entropy']:.6f}")
    print(f"  Dominant term       : {final_summary['dominant_term']}")