# DARWIN HAMMER — match 720, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_gliner_zero_s_hybrid_krampus_brain_m30_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_sketch_epistemic_certainty_m2_s3.py (gen3)
# born: 2026-05-29T23:30:34Z

"""
Hybrid Span‑Sheaf‑Pheromone Algorithm
====================================

Parents
-------
* **hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s4** – produces
  `Span` objects that label substrings of text with a confidence `score`.
* **hybrid_krampus_brainmap_hybrid_pheromone_inf_m37_s1** – maintains a
  `PheromoneStore` where each entry decays over time and provides a
  scalar `signal_value` that influences decisions.
* **hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s0** – treats each
  datum as a section of a sheaf and evaluates a coboundary discrepancy.
* **epistemic_certainty.py** – attaches a `CertaintyFlag` (scalar confidence
  in `[0,1]`) to any datum.

Mathematical Bridge
-------------------
A `Span` is interpreted as a *sheaf section* `s(v) = score` attached to a
vertex `v`.  Each vertex also carries a `CertaintyFlag` whose scalar
confidence `c(v) = confidence_bps / 10000`.  Edges between consecutive
spans inherit a *pheromone* factor `p_{uv}` from the `PheromoneStore`.  

For an edge `(u, v)` the coboundary discrepancy is


δ_{uv} = p_{uv} · (s(u) – s(v))


and the edge weight is the geometric mean of endpoint certainties


w_{uv} = sqrt( c(u) * c(v) )


The global hybrid inconsistency metric is the confidence‑weighted ℓ₂‑norm


Γ = sqrt( Σ_{(u,v)} w_{uv} · δ_{uv}² )


The algorithm therefore fuses:
* span generation (text → sections),
* pheromone decay (dynamic edge scaling),
* epistemic certainty (confidence weighting),
* sheaf coboundary computation (global inconsistency).

The functions below implement this fused pipeline.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

# ----------------------------------------------------------------------
# Span definition (from parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


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

    def __init__(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
        self.uuid = str(pathlib.uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay since the last decay timestamp."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """In‑memory singleton‑like store for demo purposes."""

    _instance = None

    def __new__(cls) -> "PheromoneStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._entries: Dict[Tuple[str, str], PheromoneEntry] = {}
        return cls._instance

    def add(self, surface_key: str, signal_kind: str, value: float, half_life_seconds: int = 60) -> None:
        key = (surface_key, signal_kind)
        if key in self._entries:
            entry = self._entries[key]
            entry.signal_value += value
            entry.apply_decay()
        else:
            self._entries[key] = PheromoneEntry(surface_key, signal_kind, value, half_life_seconds)

    def get(self, surface_key: str, signal_kind: str) -> float:
        entry = self._entries.get((surface_key, signal_kind))
        if entry is None:
            return 0.0
        entry.apply_decay()
        return entry.signal_value

    def decay_all(self) -> None:
        for entry in self._entries.values():
            entry.apply_decay()

    def dump(self) -> Dict[Tuple[str, str], float]:
        return {(k[0], k[1]): v.signal_value for k, v in self._entries.items()}


# ----------------------------------------------------------------------
# Certainty infrastructure (from parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def confidence(self) -> float:
        """Return confidence as a scalar in [0,1]."""
        return int(self.confidence_bps) / 10000.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=confidence_bps,
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(evidence_refs),
    )


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def generate_spans(text: str) -> List[Span]:
    """
    Very simple mock‑up that splits `text` into word‑level spans,
    assigns a random label and a random confidence score.
    """
    words = text.split()
    spans: List[Span] = []
    pos = 0
    for w in words:
        start = pos
        end = start + len(w)
        label = random.choice(EPISTEMIC_FLAGS)
        score = random.uniform(0.0, 1.0)  # raw model confidence
        spans.append(Span(start, end, w, label, score))
        pos = end + 1  # skip the space
    return spans


def populate_pheromones(spans: List[Span], store: PheromoneStore) -> None:
    """
    For every adjacent pair of spans create a pheromone entry whose
    initial value is the product of their scores (simulating reinforcement).
    The surface_key is a deterministic identifier for the edge.
    """
    for i in range(len(spans) - 1):
        u, v = spans[i], spans[i + 1]
        surface_key = f"{u.start}-{v.end}"
        signal_kind = "edge_strength"
        value = u.score * v.score
        store.add(surface_key, signal_kind, value, half_life_seconds=120)


def hybrid_inconsistency_metric(
    spans: List[Span],
    store: PheromoneStore,
    certainty_map: Dict[str, CertaintyFlag],
) -> float:
    """
    Compute Γ = sqrt( Σ_{edges} w_uv * (p_uv * (s_u - s_v))² )
    where:
        s_u, s_v          – span scores (sheaf sections)
        p_uv              – pheromone signal for the edge (decayed)
        w_uv = sqrt( c(u) * c(v) )  – geometric mean of endpoint certainties
    """
    if len(spans) < 2:
        return 0.0

    total = 0.0
    for i in range(len(spans) - 1):
        u, v = spans[i], spans[i + 1]

        # sheaf sections
        s_u, s_v = u.score, v.score

        # pheromone factor
        surface_key = f"{u.start}-{v.end}"
        p_uv = store.get(surface_key, "edge_strength")

        # endpoint certainties
        c_u = certainty_map[u.label].confidence()
        c_v = certainty_map[v.label].confidence()
        w_uv = math.sqrt(c_u * c_v)

        delta = p_uv * (s_u - s_v)
        total += w_uv * (delta ** 2)

    return math.sqrt(total)


def build_certainty_map(spans: List[Span]) -> Dict[str, CertaintyFlag]:
    """
    Create a `CertaintyFlag` for each distinct label appearing in `spans`.
    The confidence is derived from the label's position in `EPISTEMIC_FLAGS`
    (higher index → lower confidence) for deterministic testing.
    """
    label_to_index = {lbl: idx for idx, lbl in enumerate(EPISTEMIC_FLAGS)}
    cmap: Dict[str, CertaintyFlag] = {}
    for lbl in {s.label for s in spans}:
        idx = label_to_index[lbl]
        # Map index linearly to 0..10000 bps (e.g., FACT=10000, BULLSHIT=0)
        confidence_bps = 10000 - int( (len(EPISTEMIC_FLAGS) - 1 - idx) * (10000 / (len(EPISTEMIC_FLAGS) - 1)) )
        cmap[lbl] = certainty(
            lbl,
            confidence_bps=confidence_bps,
            authority_class="demo",
            rationale="derived from label ordering",
        )
    return cmap


def decay_and_report(store: PheromoneStore) -> None:
    """
    Apply decay to all pheromone entries and print a concise summary.
    """
    store.decay_all()
    dump = store.dump()
    print("Pheromone state after decay:")
    for (surf, kind), val in dump.items():
        print(f"  [{surf}:{kind}] = {val:.5f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "The quick brown fox jumps over the lazy dog"
    spans = generate_spans(sample_text)
    store = PheromoneStore()
    populate_pheromones(spans, store)
    cert_map = build_certainty_map(spans)

    metric_before = hybrid_inconsistency_metric(spans, store, cert_map)
    print(f"Hybrid inconsistency metric (initial): {metric_before:.6f}")

    # Simulate time passing
    import time
    time.sleep(1.2)  # short pause to give a non‑zero age

    decay_and_report(store)

    metric_after = hybrid_inconsistency_metric(spans, store, cert_map)
    print(f"Hybrid inconsistency metric (after decay): {metric_after:.6f}")