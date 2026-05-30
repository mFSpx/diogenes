# DARWIN HAMMER — match 1208, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m520_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_hybrid_m795_s2.py (gen4)
# born: 2026-05-29T23:34:39Z

"""Hybrid Physarum‑Epistemic‑Morphology Network
================================================

This module fuses the two parent algorithms:

* **Parent A** – Physarum‑inspired flux and conductance dynamics.
* **Parent B** – Week‑day weight vectors, epistemic flag handling and
  morphological descriptors.

Mathematical bridge
------------------
Both parents manipulate *resource allocation*:

* In the Physarum model the conductance `c_ij` of an edge evolves according
  to the absolute flux `|q_ij|` (Eq. 1).
* In the epistemic model the trustworthiness of a piece of evidence is
  encoded by a scalar weight `w_e` (derived from regex matches) and a
  day‑of‑week vector `w_d` that distributes trust over a set of groups.

The hybrid algorithm treats each edge of a Physarum network as a
*morphological entity* (length, width, height, mass).  The edge length
`ℓ_ij` is computed from the morphology, while the gain term of the
conductance ODE is modulated by the product of the epistemic weight and
the appropriate weekday weight.  Thus the update rule becomes


c_ij(t+Δt) = max(0,
    c_ij(t) + Δt·(γ·w_e·w_d[g]·|q_ij| - λ·c_ij(t)))


where `γ` is the base gain, `λ` the decay, `g` the group index and
`w_d[g]` the weekday‑dependent scalar for that group.  This single equation
captures the dynamics of both parents.

The module provides three core functions that demonstrate this hybrid
operation.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import re

# ----------------------------------------------------------------------
# Parent B – epistemic handling
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

# Regexes from Parent B
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|"
    r"criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)

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

def epistemic_weight_from_text(text: str) -> float:
    """Map a free‑form text to an epistemic scalar using regex matches."""
    # Prioritize evidence; fallback to planning; otherwise low trust.
    if EVIDENCE_RE.search(text):
        flag = "FACT"
    elif PLANNING_RE.search(text):
        flag = "PROBABLE"
    elif DELAY_RE.search(text):
        flag = "POSSIBLE"
    else:
        flag = "BULLSHIT"
    return _EPISTEMIC_WEIGHT[flag]

# ----------------------------------------------------------------------
# Parent A – physarum primitives (flux & conductance dynamics)
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
# Hybrid data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float

    def effective_length(self) -> float:
        """Derive a scalar length used by the physarum flux."""
        # Use Euclidean norm of spatial dimensions as a proxy.
        return math.sqrt(self.length ** 2 + self.width ** 2 + self.height ** 2) + 1e-9

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_flux(conductance: float,
                morphology: Morphology,
                pressure_a: float,
                pressure_b: float) -> float:
    """Flux that incorporates morphological edge length."""
    edge_len = morphology.effective_length()
    return flux(conductance, edge_len, pressure_a, pressure_b)

def hybrid_update_conductance(conductance: float,
                              q: float,
                              dt: float,
                              group_idx: int,
                              dow: int,
                              evidence_text: str,
                              base_gain: float = 1.0,
                              decay: float = 0.01) -> float:
    """
    Conductance update where the gain term is modulated by:
      * epistemic trust derived from `evidence_text`
      * weekday weight for the selected group
    """
    # Epistemic scalar
    w_e = epistemic_weight_from_text(evidence_text)

    # Weekday weight vector for all groups, then pick the entry for the group.
    w_vec = weekday_weight_vector(GROUPS, dow)
    w_d = w_vec[group_idx] if 0 <= group_idx < len(w_vec) else 1.0 / len(GROUPS)

    # Effective gain
    gain = base_gain * w_e * w_d

    return update_conductance(conductance, q, dt=dt, gain=gain, decay=decay)

def build_conductance_matrix(morphologies: List[Morphology],
                             pressures: List[float],
                             dt: float,
                             dow: int,
                             evidence_texts: List[str]) -> np.ndarray:
    """
    Construct a symmetric conductance matrix for a fully connected network
    of `n` nodes, each node possessing a morphology.  The update for each
    edge (i, j) uses the hybrid update rule.
    """
    n = len(morphologies)
    if n < 2:
        raise ValueError("At least two morphologies are required")
    if len(pressures) != n:
        raise ValueError("Pressures length must match morphologies")
    if len(evidence_texts) != n:
        raise ValueError("Evidence texts length must match morphologies")

    # Initialise conductances to a small positive value.
    C = np.full((n, n), 0.1, dtype=float)

    # Weekday weight vector (same for all edges, group index chosen cyclically)
    w_vec = weekday_weight_vector(GROUPS, dow)

    for i in range(n):
        for j in range(i + 1, n):
            # Compute flux based on current conductance.
            q = hybrid_flux(C[i, j], morphologies[i], pressures[i], pressures[j])

            # Choose a group index (simple deterministic scheme).
            group_idx = (i + j) % len(GROUPS)

            # Update conductance using hybrid rule.
            new_c = hybrid_update_conductance(
                conductance=C[i, j],
                q=q,
                dt=dt,
                group_idx=group_idx,
                dow=dow,
                evidence_text=evidence_texts[i] + " " + evidence_texts[j],
                base_gain=1.0,
                decay=0.01,
            )
            C[i, j] = C[j, i] = new_c
    return C

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic example with three nodes.
    random.seed(0)

    morphs = [
        Morphology(length=1.0, width=0.5, height=0.2, mass=2.0),
        Morphology(length=0.8, width=0.4, height=0.3, mass=1.5),
        Morphology(length=1.2, width=0.6, height=0.25, mass=2.5),
    ]

    pressures = [1.0, 0.5, -0.2]  # arbitrary pressure values
    dow = datetime.now(timezone.utc).weekday()  # 0 = Monday

    evidence = [
        "verified source shows strong correlation",
        "plan includes multiple steps",
        "waiting for further data",
    ]

    C = build_conductance_matrix(morphs, pressures, dt=0.1, dow=dow, evidence_texts=evidence)

    print("Conductance matrix after one update step:")
    print(C)
    # Verify symmetry and non‑negativity
    assert np.allclose(C, C.T), "Matrix not symmetric"
    assert np.all(C >= 0), "Negative conductance encountered"
    print("Smoke test passed.")