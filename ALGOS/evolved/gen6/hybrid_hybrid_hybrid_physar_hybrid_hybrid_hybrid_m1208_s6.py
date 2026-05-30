# DARWIN HAMMER — match 1208, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m520_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_hybrid_m795_s2.py (gen4)
# born: 2026-05-29T23:34:39Z

"""Hybrid Physarum‑Epistemic Decision Network
Integrates:

* Parent A: Physarum‑inspired flux and conductance dynamics.
* Parent B: Week‑day weight vectors, epistemic flag weighting and regex‑based evidence extraction.

Mathematical bridge:
We treat each *group* (e.g. a language model) as a node in a Physarum network.
The node “pressure” is an evidence‑derived scalar (higher pressure = stronger
support). Edge length is a geometric distance derived from the Morphology of the
two groups. Conductance of each edge evolves with the absolute flux
(Parent A) but the flux itself is multiplied by the epistemic weight of the
evidence that generated the pressures (Parent B). The weekday‑dependent
weight vector modulates the final selection probabilities, providing the
temporal bias from Parent A’s weekday routine.

The core hybrid step therefore consists of:
1. Extract evidence from text → epistemic flag → scalar weight wₑ.
2. Build a pressure vector P (evidence scores per group) and a length matrix L.
3. Compute flux matrix Q using Physarum flux formula scaled by wₑ.
4. Update conductance matrix C via the ODE step.
5. Combine updated conductances with the weekday weight vector to obtain a
   probability distribution over groups.

The following module implements this pipeline."""
from __future__ import annotations

import math
import random
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A primitives (physarum flux & conductance dynamics)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 0.1,
                       gain: float = 1.0,
                       decay: float = 0.01,
                       eps: float = 1e-12) -> float:
    """Conductance ODE step based on absolute flux."""
    delta = dt * (gain * abs(q) - decay * conductance)
    new_c = max(0.0, conductance + delta)
    return new_c


# ----------------------------------------------------------------------
# Parent B primitives (weekday weight vector & epistemic flag handling)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
_EPISTEMIC_WEIGHT: Dict[str, float] = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.2,
    "SURE_MAYBE": 0.6,
}


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """Return a normalized weight vector that varies sinusoidally with day‑of‑week."""
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec


# ----------------------------------------------------------------------
# Regex‑based evidence extraction (Parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PROBABLE_RE = re.compile(r"\b(?:likely|probably|high\s*probability)\b", re.I)
POSSIBLE_RE = re.compile(r"\b(?:maybe|possible|could\s*be)\b", re.I)
BULLSHIT_RE = re.compile(r"\b(?:fake|hoax|myth|unverified|rumor)\b", re.I)
SURE_MAYBE_RE = re.compile(r"\b(?:sure|definitely|certain)\b", re.I)


def epistemic_flag_from_text(text: str) -> str:
    """Return the strongest epistemic flag found in *text*."""
    if EVIDENCE_RE.search(text):
        return "FACT"
    if PROBABLE_RE.search(text):
        return "PROBABLE"
    if POSSIBLE_RE.search(text):
        return "POSSIBLE"
    if BULLSHIT_RE.search(text):
        return "BULLSHIT"
    if SURE_MAYBE_RE.search(text):
        return "SURE_MAYBE"
    return "POSSIBLE"  # default fallback


def epistemic_weight(flag: str) -> float:
    """Map an epistemic flag to its scalar trust weight."""
    return _EPistemic_WEIGHT.get(flag, 0.5)


# ----------------------------------------------------------------------
# Data structures for the hybrid network
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """Descriptor for a model (parent B)."""
    name: str
    ram_mb: int
    tier: str


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


def morphological_distance(m1: Morphology, m2: Morphology) -> float:
    """Euclidean distance in the 4‑D morphology space."""
    vec1 = np.array([m1.length, m1.width, m1.height, m1.mass])
    vec2 = np.array([m2.length, m2.width, m2.height, m2.mass])
    return float(np.linalg.norm(vec1 - vec2) + 1e-12)  # avoid zero length


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def build_length_matrix(morphologies: Dict[str, Morphology]) -> np.ndarray:
    """Construct a symmetric matrix L where L[i,j] is the distance between models i and j."""
    names = list(morphologies.keys())
    n = len(names)
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = morphological_distance(morphologies[names[i]], morphologies[names[j]])
            L[i, j] = L[j, i] = d
    return L


def compute_flux_matrix(conductances: np.ndarray,
                        lengths: np.ndarray,
                        pressures: np.ndarray,
                        epistemic_scale: float) -> np.ndarray:
    """
    Vectorised flux computation.
    Q_ij = epistemic_scale * flux(C_ij, L_ij, P_i, P_j)
    """
    n = pressures.shape[0]
    P_i = pressures.reshape((n, 1))
    P_j = pressures.reshape((1, n))
    raw_flux = conductances / np.maximum(lengths, 1e-12) * (P_i - P_j)
    return epistemic_scale * raw_flux


def hybrid_step(models: List[ModelTier],
                morphologies: Dict[str, Morphology],
                conductance_matrix: np.ndarray,
                pressures: np.ndarray,
                text: str,
                dow: int,
                dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a single hybrid update:
    1. Extract epistemic flag from *text* and obtain scaling weight.
    2. Compute flux matrix Q using physarum dynamics scaled by epistemic weight.
    3. Update each edge conductance via the ODE step.
    4. Produce a probability distribution over models using weekday weights
       and the updated conductances.
    Returns (new_conductance_matrix, selection_probabilities).
    """
    # 1. Epistemic scaling
    flag = epistemic_flag_from_text(text)
    w_epistemic = epistemic_weight(flag)

    # 2. Flux matrix
    length_mat = build_length_matrix(morphologies)
    Q = compute_flux_matrix(conductance_matrix, length_mat, pressures, w_epistemic)

    # 3. Conductance update (element‑wise, ignoring diagonal)
    n = conductance_matrix.shape[0]
    new_C = conductance_matrix.copy()
    for i in range(n):
        for j in range(i + 1, n):
            new_val = update_conductance(conductance_matrix[i, j], Q[i, j], dt=dt)
            new_C[i, j] = new_C[j, i] = new_val

    # 4. Selection probabilities
    #   - Use the sum of incident conductances as a node strength.
    node_strength = new_C.sum(axis=1)
    #   - Combine with weekday weight vector.
    dow_weights = weekday_weight_vector([m.name for m in models], dow)
    raw = node_strength * dow_weights
    probs = raw / raw.sum() if raw.sum() > 0 else np.ones_like(raw) / len(raw)
    return new_C, probs


def initialize_hybrid(models: List[ModelTier],
                      morphologies: Dict[str, Morphology]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialise conductance matrix (small positive values) and pressure vector.
    Pressures are set to the epistemic weight of a neutral 'POSSIBLE' flag.
    """
    n = len(models)
    C0 = np.full((n, n), 0.05, dtype=float)  # low uniform conductance
    np.fill_diagonal(C0, 0.0)
    # Neutral pressure based on default epistemic weight
    default_pressure = epistemic_weight("POSSIBLE")
    P0 = np.full(n, default_pressure, dtype=float)
    return C0, P0


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define four toy models
    model_list = [
        ModelTier(name="codex", ram_mb=2048, tier="A"),
        ModelTier(name="groq", ram_mb=4096, tier="B"),
        ModelTier(name="cohere", ram_mb=1024, tier="A"),
        ModelTier(name="local_models", ram_mb=8192, tier="C"),
    ]

    # Simple synthetic morphologies
    morpho_dict = {
        "codex": Morphology(1.0, 0.5, 0.3, 0.2),
        "groq": Morphology(1.2, 0.6, 0.35, 0.25),
        "cohere": Morphology(0.9, 0.45, 0.28, 0.18),
        "local_models": Morphology(1.5, 0.8, 0.5, 0.4),
    }

    # Initialise network
    C, P = initialize_hybrid(model_list, morpho_dict)

    # Example input text
    sample_text = (
        "The latest benchmark provides verified evidence that the model "
        "outperforms previous versions. However, some sources claim it is "
        "still a possible improvement."
    )

    # Run a few hybrid steps
    dow_today = datetime.now(timezone.utc).weekday()  # 0 = Monday
    for step in range(3):
        C, probs = hybrid_step(
            models=model_list,
            morphologies=morpho_dict,
            conductance_matrix=C,
            pressures=P,
            text=sample_text,
            dow=dow_today,
            dt=0.1,
        )
        print(f"Step {step+1} probabilities:", {m.name: round(p, 3) for m, p in zip(model_list, probs)})
    sys.exit(0)