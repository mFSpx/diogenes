# DARWIN HAMMER — match 4320, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_krampus_stick_hybrid_hybrid_hard_t_m15_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s0.py (gen4)
# born: 2026-05-29T23:54:56Z

"""Hybrid Algorithm combining Pheromone Diffusion (Parent A) with Epistemic Certainty Bandit (Parent B).

Mathematical Bridge:
Both parents manipulate a scalar measure that evolves over time:
- Parent A uses a *decay factor*  δ = 0.5^{age/half_life} to attenuate pheromone signal values.
- Parent B uses a *confidence* c ∈ [0, 1] (stored as basis‑points) to weight labeling‑function results.

We identify δ as a continuous analogue of epistemic confidence decay.  
The hybrid therefore treats the pheromone signal value as a *raw confidence* and
applies the decay factor to obtain an *effective confidence* that drives the
bandit‑style action selection.  Additionally, the Shannon entropy of the
distribution of pheromone signals provides a global information‑entropy term
that can modulate the certainty flags (high entropy → lower overall confidence).

The core operations are:
    effective_confidence = confidence_bps * δ
    global_entropy = - Σ p_i log p_i  (p_i ∝ signal_value_i)
These equations are fused in the functions below.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable
import numpy as np

# ---------- Parent A components (pheromone) ----------
MAX_COMPONENT_TOKENS = 500

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(np.random.randint(0, 1_000_000))
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = float(signal_value)
        self.half_life_seconds = int(half_life_seconds)
        self.created_at = pathlib.Path.cwd()
        self.last_decay = pathlib.Path.cwd()

    def age_seconds(self) -> float:
        # In a real system this would be a timestamp diff; we simulate.
        return np.random.uniform(0, 100)

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = pathlib.Path.cwd()


class PheromoneStore:
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def all_entries(cls) -> List[PheromoneEntry]:
        return list(cls._entries.values())

# ---------- Parent B components (epistemic certainty & bandit) ----------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # basis points, 0..10_000 representing 0..1
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

# ---------- Hybrid Operations ----------
def pheromone_entropy(store: PheromoneStore) -> float:
    """Compute Shannon entropy of the normalized pheromone signal values."""
    values = np.array([e.signal_value for e in store.all_entries()], dtype=float)
    if values.size == 0:
        return 0.0
    total = values.sum()
    if total == 0.0:
        return 0.0
    probs = values / total
    # Avoid log(0) by masking zero probabilities
    mask = probs > 0
    entropy = -np.sum(probs[mask] * np.log2(probs[mask]))
    return float(entropy)


def blend_confidence(flag: CertaintyFlag, entry: PheromoneEntry) -> CertaintyFlag:
    """
    Fuse a CertaintyFlag with a PheromoneEntry.

    The pheromone decay factor δ acts as a multiplicative attenuator on the
    original confidence (expressed in basis points).  The resulting confidence
    is capped at 10 000 bps.
    """
    entry.apply_decay()
    delta = entry.decay_factor()
    new_conf = int(min(10_000, flag.confidence_bps * delta))
    # Preserve all other fields; update generated_at to reflect fusion time.
    return CertaintyFlag(
        label=flag.label,
        confidence_bps=new_conf,
        authority_class=flag.authority_class,
        rationale=flag.rationale + f" | fused with pheromone {entry.uuid}",
        evidence_refs=flag.evidence_refs,
        generated_at=str(pathlib.Path.cwd()),
    )


def bandit_select(flags: Iterable[CertaintyFlag]) -> CertaintyFlag:
    """
    Simple multi‑armed bandit selection: choose a flag with probability
    proportional to its confidence (basis points).  If all confidences are zero,
    select uniformly at random.
    """
    flags = list(flags)
    total = sum(f.confidence_bps for f in flags)
    if total == 0:
        return random.choice(flags)
    # Build cumulative distribution
    cumulative = []
    running = 0
    for f in flags:
        running += f.confidence_bps
        cumulative.append((running, f))
    r = random.uniform(0, total)
    for threshold, flag in cumulative:
        if r <= threshold:
            return flag
    return flags[-1]  # fallback


def update_pheromone_from_outcome(entry: PheromoneEntry, reward: float) -> None:
    """
    Adjust the pheromone signal value based on an external reward signal.
    Positive reward amplifies the signal, negative reward attenuates it.
    The update respects the half‑life decay model by scaling the value.
    """
    # Clamp reward to a reasonable range to avoid explosion.
    reward = max(-1.0, min(1.0, reward))
    entry.signal_value = max(0.0, entry.signal_value * (1.0 + reward))


# ---------- Smoke Test ----------
if __name__ == "__main__":
    # Create pheromone entries
    p1 = PheromoneEntry(surface_key="topic_A", signal_kind="info", signal_value=5.0, half_life_seconds=30)
    p2 = PheromoneEntry(surface_key="topic_B", signal_kind="info", signal_value=3.0, half_life_seconds=45)
    PheromoneStore.add(p1)
    PheromoneStore.add(p2)

    # Compute entropy (demonstrates hybrid matrix‑like operation)
    ent = pheromone_entropy(PheromoneStore)
    print(f"Pheromone entropy: {ent:.4f} bits")

    # Create epistemic certainty flags
    cf1 = CertaintyFlag(label="FACT", confidence_bps=8000, authority_class="expert", rationale="initial")
    cf2 = CertaintyFlag(label="PROBABLE", confidence_bps=5000, authority_class="crowd", rationale="survey")

    # Fuse each flag with a pheromone entry
    fused1 = blend_confidence(cf1, p1)
    fused2 = blend_confidence(cf2, p2)

    # Bandit selection based on fused confidences
    chosen = bandit_select([fused1, fused2])
    print(f"Bandit selected label: {chosen.label} with confidence {chosen.confidence_bps} bps")

    # Simulate outcome reward and update the underlying pheromone
    reward = 0.2 if chosen.label == "FACT" else -0.1
    update_pheromone_from_outcome(p1 if chosen is fused1 else p2, reward)

    # Verify that signal value changed
    updated_entry = p1 if chosen is fused1 else p2
    print(f"Updated pheromone {updated_entry.uuid} signal value: {updated_entry.signal_value:.4f}")