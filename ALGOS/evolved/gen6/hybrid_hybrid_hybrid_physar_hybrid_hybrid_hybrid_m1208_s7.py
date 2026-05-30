# DARWIN HAMMER — match 1208, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m520_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_hybrid_m795_s2.py (gen4)
# born: 2026-05-29T23:34:39Z

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
    return _EPISTEMIC_WEIGHT.get(flag, 0.5)


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
    raw_flux = np.where(np.eye(n) == 1, 0, conductances / np.maximum(lengths, 1e-12) * (P_i - P_j))
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
            q = Q[i, j]
            new_val = update_conductance(conductance_matrix[i, j], q, dt)
            new_C[i, j] = new_val
            new_C[j, i] = new_val  # symmetric update

    # 4. Produce probability distribution
    weekday_weights = weekday_weight_vector([m.name for m in models], dow)
    selection_probabilities = new_C @ weekday_weights
    selection_probabilities /= selection_probabilities.sum()
    return new_C, selection_probabilities