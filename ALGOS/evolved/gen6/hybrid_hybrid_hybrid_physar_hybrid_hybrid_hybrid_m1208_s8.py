# DARWIN HAMMER — match 1208, survivor 8
# gen: 6
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m520_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_hybrid_m795_s2.py (gen4)
# born: 2026-05-29T23:34:39Z

import math
import random
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Any

import numpy as np

GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
_EPISTEMIC_WEIGHT: Dict[str, float] = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.2,
    "SURE_MAYBE": 0.6,
}

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def morphological_distance(m1: Morphology, m2: Morphology) -> float:
    vec1 = np.array([m1.length, m1.width, m1.height, m1.mass])
    vec2 = np.array([m2.length, m2.width, m2.height, m2.mass])
    return float(np.linalg.norm(vec1 - vec2) + 1e-12)

def build_length_matrix(morphologies: Dict[str, Morphology]) -> np.ndarray:
    names = list(morphologies.keys())
    n = len(names)
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = morphological_distance(morphologies[names[i]], morphologies[names[j]])
            L[i, j] = L[j, i] = d
    return L

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float, dt: float = 0.1, gain: float = 1.0, decay: float = 0.01, eps: float = 1e-12) -> float:
    delta = dt * (gain * abs(q) - decay * conductance)
    new_c = max(0.0, conductance + delta)
    return new_c

EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PROBABLE_RE = re.compile(r"\b(?:likely|probably|high\s*probability)\b", re.I)
POSSIBLE_RE = re.compile(r"\b(?:maybe|possible|could\s*be)\b", re.I)
BULLSHIT_RE = re.compile(r"\b(?:fake|hoax|myth|unverified|rumor)\b", re.I)
SURE_MAYBE_RE = re.compile(r"\b(?:sure|definitely|certain)\b", re.I)

def epistemic_flag_from_text(text: str) -> str:
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
    return "POSSIBLE"

def epistemic_weight(flag: str) -> float:
    return _EPISTEMIC_WEIGHT.get(flag, 0.5)

def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec

def compute_flux_matrix(conductances: np.ndarray, lengths: np.ndarray, pressures: np.ndarray, epistemic_scale: float) -> np.ndarray:
    n = pressures.shape[0]
    P_i = pressures.reshape((n, 1))
    P_j = pressures.reshape((1, n))
    raw_flux = conductances / np.maximum(lengths, 1e-12) * (P_i - P_j)
    return epistemic_scale * raw_flux

def hybrid_step(models: List[ModelTier], morphologies: Dict[str, Morphology], conductance_matrix: np.ndarray, pressures: np.ndarray, text: str, dow: int, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    flag = epistemic_flag_from_text(text)
    w_epistemic = epistemic_weight(flag)
    length_mat = build_length_matrix(morphologies)
    Q = compute_flux_matrix(conductance_matrix, length_mat, pressures, w_epistemic)
    n = conductance_matrix.shape[0]
    new_C = conductance_matrix.copy()
    for i in range(n):
        for j in range(i + 1, n):
            new_val = update_conductance(conductance_matrix[i, j], Q[i, j], dt=dt)
            new_C[i, j] = new_val
            new_C[j, i] = new_val
    weight_vec = weekday_weight_vector([model.name for model in models], dow)
    selection_probabilities = np.zeros(n)
    for i in range(n):
        for j in range(n):
            selection_probabilities[i] += new_C[i, j] * weight_vec[j]
    selection_probabilities /= selection_probabilities.sum()
    return new_C, selection_probabilities