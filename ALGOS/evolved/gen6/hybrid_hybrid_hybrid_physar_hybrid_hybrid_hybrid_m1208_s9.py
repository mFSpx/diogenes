# DARWIN HAMMER — match 1208, survivor 9
# gen: 6
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m520_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_hybrid_m795_s2.py (gen4)
# born: 2026-05-29T23:34:39Z

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A primitives (Physarum flux & conductance dynamics)
# ----------------------------------------------------------------------
def physarum_flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float) -> float:
    """Flux on a single edge according to the Physarum model."""
    if edge_length <= 0.0:
        raise ValueError("edge_length must be positive")
    return conductance / edge_length * (pressure_a - pressure_b)


def update_conductance_matrix(
    C: np.ndarray,
    Q: np.ndarray,
    dt: float = 0.1,
    gain: float = 1.0,
    decay: float = 0.01,
) -> np.ndarray:
    """
    Vectorised ODE step for the conductance matrix.
    C_{ij} ← max(0, C_{ij} + dt·(gain·|Q_{ij}| – decay·C_{ij}))
    Diagonal entries are forced to zero (no self‑loops).
    """
    delta = dt * (gain * np.abs(Q) - decay * C)
    new_C = np.maximum(0.0, C + delta)
    np.fill_diagonal(new_C, 0.0)
    return new_C


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
    """
    Sinusoidally varying weight vector over the week.
    Normalised to sum to 1.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    return raw / raw.sum()


# ----------------------------------------------------------------------
# Regex‑based evidence extraction (Parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PROBABLE_RE = re.compile(r"\b(?:likely|probably|high\s*probability)\b", re.I)
POSSIBLE_RE = re.compile(r"\b(?:maybe|possible|could\s*be)\b", re.I)
BULLSHIT_RE = re.compile(r"\b(?:fake|hoax|myth|unverified|rumor)\b", re.I)
SURE_MAYBE_RE = re.compile(r"\b(?:sure|definitely|certain)\b", re.I)


def epistemic_flag_from_text(text: str) -> str:
    """
    Return the strongest epistemic flag found in *text*.
    Order respects decreasing confidence.
    """
    if EVIDENCE_RE.search(text):
        return "FACT"
    if PROBABLE_RE.search(text):
        return "PROBABLE"
    if SURE_MAYBE_RE.search(text):
        return "SURE_MAYBE"
    if POSSIBLE_RE.search(text):
        return "POSSIBLE"
    if BULLSHIT_RE.search(text):
        return "BULLSHIT"
    return "POSSIBLE"  # fallback


def epistemic_weight(flag: str) -> float:
    """Map an epistemic flag to its scalar trust weight."""
    return _EPISTEMIC_WEIGHT.get(flag, 0.5)


# ----------------------------------------------------------------------
# Morphology utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a logical entity."""
    length: float
    width: float
    height: float
    mass: float


def morphological_distance(m1: Morphology, m2: Morphology) -> float:
    """Euclidean distance in the 4‑D morphology space, with epsilon guard."""
    v1 = np.array([m1.length, m1.width, m1.height, m1.mass])
    v2 = np.array([m2.length, m2.width, m2.height, m2.mass])
    return float(np.linalg.norm(v1 - v2) + 1e-12)


def build_length_matrix(morphologies: Dict[str, Morphology]) -> np.ndarray:
    """Symmetric matrix L where L[i,j] = morphological distance between i and j."""
    names = list(morphologies.keys())
    n = len(names)
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = morphological_distance(morphologies[names[i]], morphologies[names[j]])
            L[i, j] = L[j, i] = d
    # Add a tiny epsilon on the diagonal to avoid division‑by‑zero in vectorised flux.
    np.fill_diagonal(L, 1e-12)
    return L


# ----------------------------------------------------------------------
# Hybrid network encapsulation
# ----------------------------------------------------------------------
class HybridPhysarumEpistemicNetwork:
    """
    A deeper integration of Physarum conductance dynamics with epistemic
    weighting and weekday‑dependent biases.
    """

    def __init__(
        self,
        models: List[str],
        morphologies: Dict[str, Morphology],
        init_conductance: float = 0.1,
        dt: float = 0.1,
        gain: float = 1.0,
        decay: float = 0.01,
    ):
        if set(models) != set(morphologies.keys()):
            raise ValueError("Model names must match keys of morphologies")
        self.models = models
        self.morphologies = morphologies
        self.n = len(models)
        self.dt = dt
        self.gain = gain
        self.decay = decay

        # Initialise conductance matrix (symmetric, zero diagonal)
        self.C = np.full((self.n, self.n), init_conductance, dtype=float)
        np.fill_diagonal(self.C, 0.0)

        # Pre‑compute length matrix (static)
        self.L = build_length_matrix(morphologies)

        # Mapping from model name to index for fast lookup
        self._idx = {name: i for i, name in enumerate(models)}

    def _pressures_from_text(self, text: str) -> np.ndarray:
        """
        Derive a pressure vector from textual evidence.
        For demonstration we map the epistemic flag to a uniform pressure
        scaled by the flag weight; a more sophisticated implementation could
        parse per‑model evidence.
        """
        flag = epistemic_flag_from_text(text)
        w = epistemic_weight(flag)
        # Uniform base pressure, modulated by epistemic confidence
        base = np.ones(self.n, dtype=float) * w
        return base

    def step(self, text: str, dow: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute one hybrid update.

        Returns
        -------
        new_conductance : np.ndarray (n×n)
            Updated conductance matrix.
        selection_probs : np.ndarray (n,)
            Probability distribution over models after applying weekday bias.
        """
        # 1️⃣ Epistemic scaling (already embedded in pressure creation)
        pressures = self._pressures_from_text(text)

        # 2️⃣ Apply weekday bias directly to pressures (deeper integration)
        weekday_vec = weekday_weight_vector(self.models, dow)
        pressures = pressures * weekday_vec

        # 3️⃣ Vectorised flux computation
        P_i = pressures[:, None]          # shape (n,1)
        P_j = pressures[None, :]          # shape (1,n)
        raw_flux = self.C / np.maximum(self.L, 1e-12) * (P_i - P_j)
        Q = raw_flux  # epistemic scaling already in pressures

        # 4️⃣ Conductance ODE update
        self.C = update_conductance_matrix(self.C, Q, dt=self.dt, gain=self.gain, decay=self.decay)

        # 5️⃣ Selection probabilities: combine conductance strength with weekday bias
        #   We interpret the sum of incident conductances as a “attractiveness” score.
        attractiveness = self.C.sum(axis=1)
        # Mix with weekday vector (α controls mixing; here 0.5 for equal influence)
        alpha = 0.5
        mixed = alpha * attractiveness + (1 - alpha) * weekday_vec
        selection_probs = mixed / mixed.sum()
        return self.C.copy(), selection_probs

    # ------------------------------------------------------------------
    # Utility methods for external inspection
    # ------------------------------------------------------------------
    def get_conductance_matrix(self) -> np.ndarray:
        return self.C.copy()

    def get_length_matrix(self) -> np.ndarray:
        return self.L.copy()

    def model_index(self, name: str) -> int:
        return self._idx[name]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(models={self.models}, dt={self.dt}, "
            f"gain={self.gain}, decay={self.decay})"
        )