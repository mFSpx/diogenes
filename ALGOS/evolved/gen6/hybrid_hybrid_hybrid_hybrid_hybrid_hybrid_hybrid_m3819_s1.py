# DARWIN HAMMER — match 3819, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m2091_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2578_s0.py (gen5)
# born: 2026-05-29T23:51:47Z

"""Hybrid Module: fusion of
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m2091_s1.py
- Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2578_s0.py

Mathematical Bridge
-------------------
Parent A represents a weight matrix **W** as a **Multivector** in a Clifford algebra
and updates it with a **geometric product** while a liquid‑time constant τ evolves.
Parent B provides a **Gini coefficient** G that quantifies inequality of a scalar
distribution and a **Fisher information matrix** **F** derived from spatial data.
The hybrid algorithm multiplies the multivector **W** by the scalar G (weighting the
Clifford components by the inequality measure) and then forms a **rotor**
R = W ⊗ F (geometric product) that drives the update of τ.  The rotor thus
encodes both the algebraic structure of Parent A and the statistical‑geometric
information of Parent B, yielding a unified dynamics for temporal‑motif
analysis, min‑hash compression and liquid‑time adaptation.

The module implements three public functions that showcase this fusion:
1. `hybrid_gini_fisher_rotor` – builds the weighted rotor from entity scores.
2. `hybrid_temporal_motif_minhash` – extracts temporal motifs and compresses them
   with a simple MinHash scheme, returning a multivector representation.
3. `update_liquid_time_constant` – evolves the liquid‑time constant τ using the
   rotor via geometric product.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Tuple, Dict, Iterable, Any
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Clifford algebra utilities (multivector & geometric product)
# ----------------------------------------------------------------------
def _blade_sign(indices: Iterable[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list.
    Repeated indices cancel (e² = 0)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)
        elif lst[i] == lst[i + 1]:
            # cancel duplicate basis vector
            lst.pop(i)
            lst.pop(i)  # second element shifts to i
            n -= 2
            i = max(i - 1, 0)
        else:
            i += 1
    return lst, sign

def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Geometric product of two basis blades (ignoring metric, Euclidean)."""
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign

class Multivector:
    """Element of Cl(n,0) represented as a sparse dict {blade: coeff}."""
    def __init__(self, components: Dict[frozenset, float] = None, n: int = 3):
        self.n = int(n)
        self.components: Dict[frozenset, float] = {}
        if components:
            for blade, val in components.items():
                if abs(val) > 1e-12:
                    self.components[frozenset(blade)] = float(val)

    def __add__(self, other: 'Multivector') -> 'Multivector':
        res = Multivector(self.components.copy(), self.n)
        for b, v in other.components.items():
            res.components[b] = res.components.get(b, 0.0) + v
            if abs(res.components[b]) < 1e-12:
                del res.components[b]
        return res

    def __rmul__(self, scalar: float) -> 'Multivector':
        return Multivector({b: scalar * v for b, v in self.components.items()}, self.n)

    __mul__ = __rmul__

    def geometric_product(self, other: 'Multivector') -> 'Multivector':
        """Full geometric product (Clifford) between two multivectors."""
        result: Dict[frozenset, float] = {}
        for b1, v1 in self.components.items():
            for b2, v2 in other.components.items():
                blade, sign = _multiply_blades(b1, b2)
                coeff = v1 * v2 * sign
                result[blade] = result.get(blade, 0.0) + coeff
        # prune near‑zero entries
        result = {b: c for b, c in result.items() if abs(c) > 1e-12}
        return Multivector(result, self.n)

    def norm(self) -> float:
        """Euclidean norm of the multivector (treating blades as independent axes)."""
        return math.sqrt(sum(v * v for v in self.components.values()))

    def __repr__(self) -> str:
        if not self.components:
            return "0"
        terms = []
        for blade, coeff in sorted(self.components.items(), key=lambda x: (len(x[0]), list(x[0]))):
            if not blade:
                term = f"{coeff:.3g}"
            else:
                indices = ''.join(str(i) for i in sorted(blade))
                term = f"{coeff:.3g}e{indices}"
            terms.append(term)
        return " + ".join(terms)

# ----------------------------------------------------------------------
# Parent B – statistical / spatial utilities
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Return Gini coefficient for a 1‑D iterable of non‑negative numbers."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        raise ValueError("Gini is undefined for negative values")
    arr_sorted = np.sort(arr)
    n = arr.shape[0]
    cumulative = np.cumsum(arr_sorted)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)

def fisher_information_matrix(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Simple Fisher information for a 2‑D Gaussian estimated from lat/lon."""
    if latitudes.size != longitudes.size or latitudes.size == 0:
        raise ValueError("Latitude and longitude arrays must be same non‑zero length")
    # Estimate covariance matrix Σ
    coords = np.stack([latitudes, longitudes], axis=1)
    cov = np.cov(coords, rowvar=False)
    # Fisher information for Gaussian = Σ⁻¹
    try:
        fisher = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        fisher = np.linalg.pinv(cov)
    return fisher

# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def _minhash_signature(tokens: Iterable[str], num_perm: int = 64) -> List[int]:
    """Very lightweight MinHash: for each permutation (simulated by a different seed)
    keep the minimum hash value of the token set."""
    sig = []
    for seed in range(num_perm):
        min_hash = None
        for t in tokens:
            h = hash((t, seed))
            if (min_hash is None) or (h < min_hash):
                min_hash = h
        sig.append(min_hash if min_hash is not None else 0)
    return sig

def _signature_to_multivector(sig: List[int], n: int = 3) -> Multivector:
    """Map a list of integer hashes to a multivector.
    Each hash contributes to a blade determined by its low‑order bits."""
    comps: Dict[frozenset, float] = {}
    for idx, val in enumerate(sig):
        # Use three lowest bits to pick a basis blade (e1, e2, e3, e12, e13, e23, e123)
        bits = idx % (2 ** n - 1) + 1  # avoid empty set
        blade = frozenset(i + 1 for i in range(n) if bits & (1 << i))
        comps[blade] = comps.get(blade, 0.0) + float(val & 0xFFFFFFFF)
    return Multivector(comps, n)

# ----------------------------------------------------------------------
# Public hybrid functions
# ----------------------------------------------------------------------
def hybrid_gini_fisher_rotor(entities: List[Any]) -> Multivector:
    """
    Build a rotor R = (G * W) ⊗ F
    where
        G = Gini coefficient of entity scores,
        W = multivector constructed from MinHash of entity signatures,
        F = Fisher information matrix (as a multivector).

    The rotor encodes the weighted statistical geometry of the dataset.
    """
    # 1. Gini of scores
    scores = [float(getattr(e, "score", 0.0)) for e in entities]
    G = gini_coefficient(scores) if scores else 0.0

    # 2. MinHash over concatenated signatures
    all_tokens = []
    for e in entities:
        sig = getattr(e, "address_signature", "")
        if sig:
            all_tokens.extend(sig.split())
    if not all_tokens:
        all_tokens = ["empty"]
    minhash = _minhash_signature(all_tokens, num_perm=32)
    W = _signature_to_multivector(minhash, n=3)

    # 3. Weight W by G (scalar multiplication)
    WG = G * W

    # 4. Fisher information from spatial coordinates
    lats = np.array([float(getattr(e, "lat", 0.0)) for e in entities])
    lons = np.array([float(getattr(e, "lon", 0.0)) for e in entities])
    if lats.size == 0:
        # degenerate case – identity matrix as placeholder
        F_mat = np.eye(2)
    else:
        F_mat = fisher_information_matrix(lats, lons)

    # 5. Convert Fisher matrix to a multivector (e1↔lat, e2↔lon, bivector for covariance)
    comps = {
        frozenset({1}): float(F_mat[0, 0]),
        frozenset({2}): float(F_mat[1, 1]),
        frozenset({1, 2}): float(F_mat[0, 1] + F_mat[1, 0]) / 2.0,
    }
    F = Multivector(comps, n=3)

    # 6. Rotor via geometric product
    rotor = WG.geometric_product(F)
    return rotor

def hybrid_temporal_motif_minhash(events: List[Tuple[datetime, str]]) -> Multivector:
    """
    Detect simple temporal motifs (sequences of categories within a sliding window)
    and encode them as a multivector using MinHash.

    Parameters
    ----------
    events : list of (timestamp, category) tuples, assumed sorted by timestamp.

    Returns
    -------
    Multivector representing the compressed motif set.
    """
    if not events:
        return Multivector({}, n=3)

    # Parameters for motif detection
    window = timedelta(hours=2)
    motifs: List[str] = []

    # Sliding window to collect sequences
    start_idx = 0
    for i, (ts_i, cat_i) in enumerate(events):
        # expand window start until within `window`
        while ts_i - events[start_idx][0] > window:
            start_idx += 1
        # collect categories in current window
        seq = [events[j][1] for j in range(start_idx, i + 1)]
        if len(seq) >= 3:
            motif = "-".join(seq)
            motifs.append(motif)

    # MinHash over motif strings
    minhash = _minhash_signature(motifs, num_perm=24)
    mv = _signature_to_multivector(minhash, n=3)
    return mv

def update_liquid_time_constant(tau: float, rotor: Multivector, dt: float = 0.01) -> float:
    """
    Evolve the liquid‑time constant τ using the rotor.
    The update rule is:
        τ_{new} = τ + dt * sign(⟨R, e1⟩) * exp(-|R|)

    where ⟨R, e1⟩ extracts the coefficient of the basis vector e1 from the rotor.
    """
    e1 = frozenset({1})
    coeff_e1 = rotor.components.get(e1, 0.0)
    sign = 1.0 if coeff_e1 >= 0 else -1.0
    delta = dt * sign * math.exp(-rotor.norm())
    return tau + delta

# ----------------------------------------------------------------------
# Simple data class for entities (mirrors Parent B's Entity)
# ----------------------------------------------------------------------
class Entity:
    __slots__ = ("id", "lat", "lon", "category", "score", "address_signature")
    def __init__(self, id: str, lat: float, lon: float, category: str,
                 score: float = 0.0, address_signature: str = ""):
        self.id = id
        self.lat = float(lat)
        self.lon = float(lon)
        self.category = category
        self.score = float(score)
        self.address_signature = address_signature

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset
    ents = [
        Entity("A", 40.7128, -74.0060, "type1", score=0.9, address_signature="123 main st"),
        Entity("B", 34.0522, -118.2437, "type2", score=0.3, address_signature="456 oak ave"),
        Entity("C", 51.5074, -0.1278, "type1", score=0.6, address_signature="789 pine rd"),
    ]

    # Build rotor
    rotor = hybrid_gini_fisher_rotor(ents)
    print("Rotor (multivector):", rotor)

    # Temporal motif test
    now = datetime.utcnow()
    evts = [
        (now - timedelta(hours=3), "login"),
        (now - timedelta(hours=2, minutes=30), "view"),
        (now - timedelta(hours=2), "click"),
        (now - timedelta(hours=1, minutes=45), "click"),
        (now - timedelta(hours=1, minutes=15), "logout"),
    ]
    mv_motifs = hybrid_temporal_motif_minhash(evts)
    print("Motif multivector:", mv_motifs)

    # Update liquid time constant
    tau = 1.0
    tau_new = update_liquid_time_constant(tau, rotor, dt=0.05)
    print(f"Liquid time constant updated: {tau:.4f} -> {tau_new:.4f}")