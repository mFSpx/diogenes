# DARWIN HAMMER — match 3667, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_privacy_hybri_m2252_s3.py (gen6)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m520_s0.py (gen5)
# born: 2026-05-29T23:51:07Z

"""Hybrid Algorithm Fusion of:
- hybrid_hybrid_hybrid_gliner_hybrid_privacy_hybri_m2252_s3 (spans, entropy, pheromones)
- hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m520_s0 (Physarum conductance, flux, weekday weights)

Mathematical Bridge
------------------
The bridge is built on the observation that **entropy** of a labeled‑text span
captures the amount of information contained in a semantic observation.
In a Physarum‑inspired transport network the edge conductance `g` determines
flux `Φ = g/ℓ·(p_a‑p_b)`.  By scaling `g` with a factor derived from the span
entropy we inject semantic information directly into the physical adaptation
law.  Conversely, pheromone signals generated from the network are weighted by
a *weekday weight vector* that reflects epistemic certainty flags, closing the
loop between the textual and the topological subsystems.

The core update for an edge `e` at iteration `t` becomes:

    Φ_e   = flux(g_e, ℓ_e, p_a, p_b)
    g_e'  = update_conductance(g_e, Φ_e) * (1 + α·H_spans) * ω_f

where `H_spans` is the Shannon entropy of the current span set,
`α` is a tunable coupling constant, and `ω_f` is the epistemic‑flag weight
derived from the weekday vector.  Pheromone entries are then decayed or
amplified using the same factor, ensuring a mathematically unified hybrid
system.
"""

import math
import random
import sys
import pathlib
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
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
        self.created_at = datetime.now(timezone.utc)
        self.last_decay = self.created_at


# ----------------------------------------------------------------------
# Physarum network primitives (from Parent B)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE",
                                   "BULLSHIT", "SURE_MAYBE")


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """Create a normalized weight vector that varies with day‑of‑week."""
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       gain: float = 0.1, decay: float = 0.01,
                       dt: float = 1.0) -> float:
    """Update conductance based on flux magnitude."""
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_span_entropy(spans: List[Span]) -> float:
    """
    Shannon entropy of the label distribution weighted by span scores.
    H = - Σ p_i log2 p_i  where p_i ∝ Σ_{spans with label i} score.
    """
    if not spans:
        return 0.0
    label_weights: Dict[str, float] = {}
    total = 0.0
    for s in spans:
        label_weights[s.label] = label_weights.get(s.label, 0.0) + max(0.0, s.score)
        total += max(0.0, s.score)
    if total == 0.0:
        return 0.0
    probs = np.array([w / total for w in label_weights.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return float(entropy)


def epistemic_flag_weights(flags: Sequence[str],
                           groups: Sequence[str],
                           dow: int) -> np.ndarray:
    """
    Map epistemic flags to a weight vector using the weekday weighting scheme.
    The index of a flag in `EPISTEMIC_FLAGS` determines the group it modulates.
    """
    base_vec = weekday_weight_vector(groups, dow)
    weight_vec = np.copy(base_vec)
    for i, flag in enumerate(flags):
        if flag not in EPISTEMIC_FLAGS:
            continue
        idx = EPISTEMIC_FLAGS.index(flag) % len(groups)
        # Boost the corresponding group proportionally to flag confidence
        boost = 0.05 * (i + 1)  # simple monotonic boost
        weight_vec[idx] += boost
    # Renormalize
    weight_vec /= weight_vec.sum()
    return weight_vec


def hybrid_step(edges: List[Dict[str, Any]],
                pressures: Dict[int, float],
                pheromones: List[PheromoneEntry],
                spans: List[Span],
                flags: List[str],
                date: datetime,
                alpha: float = 0.3) -> Tuple[List[Dict[str, Any]], List[PheromoneEntry]]:
    """
    Perform a single hybrid iteration:
      1. Compute span entropy H.
      2. Derive epistemic weight vector ω from flags and weekday.
      3. For each edge, compute flux, update conductance, then scale by
         (1 + α·H)·ω_f where ω_f is the weight associated with the edge's group.
      4. Decay/boost pheromone entries using the same scaling factor.
    Returns updated edges and pheromones.
    """
    # 1. Entropy from textual side
    H = compute_span_entropy(spans)

    # 2. Epistemic weekday weighting
    dow = date.weekday()  # Monday=0 … Sunday=6
    ω = epistemic_flag_weights(flags, GROUPS, dow)

    # 3. Edge update
    updated_edges = []
    for e in edges:
        g = e["conductance"]
        ℓ = e["length"]
        a, b = e["nodes"]
        Φ = flux(g, ℓ, pressures[a], pressures[b])
        g_new = update_conductance(g, Φ)

        # Determine group index for this edge (simple hash)
        group_idx = hash(e["surface_key"]) % len(GROUPS)
        scale = (1.0 + alpha * H) * ω[group_idx]
        g_scaled = g_new * scale

        e_updated = {
            "surface_key": e["surface_key"],
            "conductance": g_scaled,
            "length": ℓ,
            "nodes": (a, b)
        }
        updated_edges.append(e_updated)

    # 4. Pheromone update
    updated_pheromones = []
    for p in pheromones:
        # decay based on half‑life, then apply same scaling factor as its surface key
        elapsed = (datetime.now(timezone.utc) - p.created_at).total_seconds()
        decay_factor = 0.5 ** (elapsed / p.half_life_seconds) if p.half_life_seconds > 0 else 1.0
        base_val = p.signal_value * decay_factor

        group_idx = hash(p.surface_key) % len(GROUPS)
        scale = (1.0 + alpha * H) * ω[group_idx]
        new_val = base_val * scale

        p.signal_value = new_val
        p.last_decay = datetime.now(timezone.utc)
        updated_pheromones.append(p)

    return updated_edges, updated_pheromones


def run_hybrid_simulation(num_steps: int = 5) -> None:
    """Convenient driver that builds a toy network, generates spans & pheromones,
    and runs `hybrid_step` repeatedly, printing summary statistics."""
    # --- toy network ----------------------------------------------------
    edges = [
        {"surface_key": "edge_a", "conductance": 1.0, "length": 1.2, "nodes": (0, 1)},
        {"surface_key": "edge_b", "conductance": 0.8, "length": 0.9, "nodes": (1, 2)},
        {"surface_key": "edge_c", "conductance": 1.2, "length": 1.5, "nodes": (0, 2)},
    ]
    pressures = {0: 1.0, 1: 0.5, 2: 0.0}

    # --- synthetic spans ------------------------------------------------
    spans = [
        Span(0, 5, "The cat", "ANIMAL", 0.9),
        Span(6, 12, "jumps", "ACTION", 0.8),
        Span(13, 18, "high", "ADVERB", 0.4),
    ]

    # --- pheromones -----------------------------------------------------
    pheromones = [
        PheromoneEntry("edge_a", "traffic", 1.0, half_life_seconds=60),
        PheromoneEntry("edge_b", "traffic", 0.7, half_life_seconds=45),
        PheromoneEntry("edge_c", "traffic", 0.5, half_life_seconds=30),
    ]

    # --- epistemic flags ------------------------------------------------
    flags = ["FACT", "PROBABLE", "POSSIBLE"]

    today = datetime.now(timezone.utc)

    for step in range(num_steps):
        edges, pheromones = hybrid_step(
            edges=edges,
            pressures=pressures,
            pheromones=pheromones,
            spans=spans,
            flags=flags,
            date=today,
            alpha=0.25
        )
        avg_g = np.mean([e["conductance"] for e in edges])
        avg_p = np.mean([p.signal_value for p in pheromones])
        print(f"Step {step+1:02d} | Avg Conductance: {avg_g:.4f} | Avg Pheromone: {avg_p:.4f}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_hybrid_simulation()