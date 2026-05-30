# DARWIN HAMMER — match 1442, survivor 3
# gen: 3
# parent_a: hybrid_geometric_product_hybrid_model_vram_sc_m22_s2.py (gen2)
# parent_b: hybrid_percyphon_hybrid_endpoint_circ_m45_s0.py (gen2)
# born: 2026-05-29T23:36:33Z

import numpy as np
from dataclasses import dataclass
import math
import hashlib
from typing import Tuple, Dict

# ----------------------------------------------------------------------
# Clifford algebra (Cl(3,0)) utilities
# ----------------------------------------------------------------------
# Multivector layout (8 components):
# 0: scalar (1)
# 1: e1
# 2: e2
# 3: e3
# 4: e12
# 5: e13
# 6: e23
# 7: e123
#
# Blade is encoded as a bitmask:
#   e1 -> 0b001, e2 -> 0b010, e3 -> 0b100
#   e12 -> 0b011, e13 -> 0b101, e23 -> 0b110, e123 -> 0b111
# The index in the array is given by the following map:
BLADE_TO_INDEX = {
    0b000: 0,
    0b001: 1,
    0b010: 2,
    0b100: 3,
    0b011: 4,
    0b101: 5,
    0b110: 6,
    0b111: 7,
}
INDEX_TO_BLADE = {v: k for k, v in BLADE_TO_INDEX.items()}

def _grade(blade_mask: int) -> int:
    """Number of set bits = grade of the blade."""
    return bin(blade_mask).count("1")

def _sign_of_permutation(a: int, b: int) -> int:
    """
    Compute the sign resulting from swapping basis vectors when
    concatenating blades a and b (both as bit masks).
    """
    # Count swaps needed to reorder the concatenated list of basis indices
    # into increasing order.
    a_bits = [i for i in range(3) if a & (1 << i)]
    b_bits = [i for i in range(3) if b & (1 << i)]
    combined = a_bits + b_bits
    swaps = 0
    for i in range(len(combined)):
        for j in range(i + 1, len(combined)):
            if combined[i] > combined[j]:
                swaps += 1
    return -1 if swaps % 2 else 1

def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Geometric product of two multivectors a and b in Cl(3,0).
    Both a and b are length‑8 arrays ordered as described above.
    """
    result = np.zeros(8, dtype=np.float64)
    for i in range(8):
        if a[i] == 0:
            continue
        mask_i = INDEX_TO_BLADE[i]
        for j in range(8):
            if b[j] == 0:
                continue
            mask_j = INDEX_TO_BLADE[j]
            mask_res = mask_i ^ mask_j  # XOR gives symmetric difference
            sign = _sign_of_permutation(mask_i, mask_j)
            # If the same basis vector appears twice, it squares to +1 (Euclidean)
            # Hence we need to multiply by (-1)^{|intersection|} but in 3‑D Euclidean
            # the square is +1, so only the permutation sign matters.
            result_idx = BLADE_TO_INDEX[mask_res]
            result[result_idx] += sign * a[i] * b[j]
    return result

def reverse(mv: np.ndarray) -> np.ndarray:
    """Reverse (reversion) of a multivector."""
    rev = np.copy(mv)
    for idx in range(8):
        g = _grade(INDEX_TO_BLADE[idx])
        if (g * (g - 1) // 2) % 2:
            rev[idx] = -rev[idx]
    return rev

def rotor_from_bivector(biv: np.ndarray) -> np.ndarray:
    """
    Construct a rotor R = exp(-B/2) where B is a bivector (grade‑2 part).
    For small bivectors we approximate with the first two terms of the series:
        R ≈ 1 - B/2
    This keeps the implementation lightweight while still providing a proper
    even multivector.
    """
    if biv.shape != (8,):
        raise ValueError("Bivector must be an 8‑component multivector")
    # Ensure biv only contains grade‑2 components
    for idx in range(8):
        if _grade(INDEX_TO_BLADE[idx]) != 2 and biv[idx] != 0:
            raise ValueError("Input contains non‑bivector components")
    R = np.zeros(8, dtype=np.float64)
    R[0] = 1.0                     # scalar part
    R[4:] = -0.5 * biv[4:]          # bivector part scaled by -1/2
    return R

def apply_rotor(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Apply rotor R to a vector v (grade‑1 multivector).
    Returns the rotated vector as a grade‑1 multivector.
    """
    if R.shape != (8,) or v.shape != (8,):
        raise ValueError("Both rotor and vector must be 8‑component multivectors")
    # v_rot = R * v * reverse(R)
    temp = geometric_product(R, v)
    v_rot = geometric_product(temp, reverse(R))
    # Zero‑out non‑vector components (should be only grades 0,2,3)
    vec = np.zeros(8, dtype=np.float64)
    for idx in (1, 2, 3):
        vec[idx] = v_rot[idx]
    return vec

def vector_to_mv(vec: np.ndarray) -> np.ndarray:
    """Convert a 3‑component Euclidean vector to a multivector."""
    if vec.shape != (3,):
        raise ValueError("Input must be a length‑3 vector")
    mv = np.zeros(8, dtype=np.float64)
    mv[1:4] = vec
    return mv

def mv_to_vector(mv: np.ndarray) -> np.ndarray:
    """Extract the vector part (e1,e2,e3) from a multivector."""
    return mv[1:4].copy()

# ----------------------------------------------------------------------
# Morphology utilities
# ----------------------------------------------------------------------
@dataclass
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def _positive_dimensions(*dims: float) -> None:
    if any(d <= 0 for d in dims):
        raise ValueError("All morphological dimensions must be strictly positive")

def sphericity_index(length: float, width: float, height: float) -> float:
    _positive_dimensions(length, width, height)
    # Ratio of the geometric mean to the longest edge (more stable than length alone)
    gm = (length * width * height) ** (1.0 / 3.0)
    longest = max(length, width, height)
    return gm / longest

def flatness_index(length: float, width: float, height: float) -> float:
    _positive_dimensions(length, width, height)
    # Ratio of the average of the two larger dimensions to the smallest
    dims = sorted([length, width, height])
    return (dims[1] + dims[2]) / (2.0 * dims[0])

# ----------------------------------------------------------------------
# VRAM‑like scheduler (simple memoisation)
# ----------------------------------------------------------------------
class VramCache:
    """
    Very lightweight “VRAM” cache that stores results of expensive
    operations keyed by a hash of the inputs.  It mimics a scheduler by
    limiting the cache size.
    """
    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self.store: Dict[int, np.ndarray] = {}

    def _hash(self, *arrays: np.ndarray) -> int:
        h = hashlib.sha256()
        for arr in arrays:
            h.update(arr.tobytes())
        return int.from_bytes(h.digest()[:8], "little")

    def get(self, *arrays: np.ndarray):
        key = self._hash(*arrays)
        return self.store.get(key, None)

    def set(self, value: np.ndarray, *arrays: np.ndarray):
        key = self._hash(*arrays)
        if len(self.store) >= self.max_entries:
            # Evict a random entry (deterministic for reproducibility)
            evict_key = next(iter(self.store))
            del self.store[evict_key]
        self.store[key] = value

# ----------------------------------------------------------------------
# Core hybrid algorithm
# ----------------------------------------------------------------------
def _build_morphology_matrix(morph: Morphology) -> np.ndarray:
    """
    Construct a 3×3 scaling matrix derived from morphological indices.
    The matrix is symmetric positive‑definite and encodes anisotropic
    scaling based on sphericity, flatness and mass.
    """
    s = sphericity_index(morph.length, morph.width, morph.height)
    f = flatness_index(morph.length, morph.width, morph.height)
    m = morph.mass
    # Base isotropic scaling
    base = np.eye(3) * (s * f) ** 0.5
    # Introduce anisotropy proportional to the normalized dimensions
    dims = np.array([morph.length, morph.width, morph.height])
    norm = dims / np.linalg.norm(dims)
    anisotropy = np.outer(norm, norm) * (m / (m + 1.0))
    return base + anisotropy

def _build_bivector_from_morphology(morph: Morphology) -> np.ndarray:
    """
    Create a bivector whose components are modulated by morphology.
    The bivector is placed in the appropriate slots of an 8‑component
    multivector (indices 4,5,6).
    """
    s = sphericity_index(morph.length, morph.width, morph.height)
    f = flatness_index(morph.length, morph.width, morph.height)
    # Simple heuristic: larger flatness emphasizes e12, larger sphericity emphasizes e23
    biv = np.zeros(8, dtype=np.float64)
    biv[4] = f * 0.3          # e12
    biv[5] = s * 0.2          # e13
    biv[6] = (s + f) * 0.1    # e23
    return biv

def ttt_ga_forward(W: np.ndarray,
                   R: np.ndarray,
                   x_seq: np.ndarray,
                   eta_w: float,
                   eta_r: float,
                   cache: VramCache) -> np.ndarray:
    """
    Perform a single forward pass of the TTT‑GA layer.
    - W : (d, d) weight matrix
    - R : rotor (8‑component multivector)
    - x_seq : (d,) input vector
    - eta_w, eta_r : learning‑rate‑like scalars for optional online update
    - cache : VramCache instance for memoisation
    Returns the transformed vector (d,).
    """
    # Check cache first
    cached = cache.get(W, R, x_seq)
    if cached is not None:
        return cached.copy()

    # Linear transformation
    lin_out = W @ x_seq

    # Convert to multivector, rotate, then back to vector
    mv_vec = vector_to_mv(lin_out)
    rotated_mv = apply_rotor(R, mv_vec)
    out_vec = mv_to_vector(rotated_mv)

    # Optional online update (simple gradient‑free adaptation)
    W += eta_w * np.outer(out_vec, x_seq)
    # Update rotor by nudging its bivector part towards the current output
    biv = _build_bivector_from_morphology(
        Morphology(length=out_vec[0] if out_vec[0] > 0 else 1.0,
                   width=out_vec[1] if out_vec[1] > 0 else 1.0,
                   height=out_vec[2] if out_vec[2] > 0 else 1.0,
                   mass=1.0)
    )
    R_new = rotor_from_bivector(biv)
    R[:] = (1 - eta_r) * R + eta_r * R_new  # simple interpolation

    cache.set(out_vec, W, R, x_seq)
    return out_vec

def hybrid_ttt_ga_vram(x_seq: np.ndarray,
                       W: np.ndarray,
                       R: np.ndarray,
                       eta_w: float,
                       eta_r: float,
                       morphology: Morphology,
                       cache: VramCache) -> np.ndarray:
    """
    Deeply integrated hybrid operation:
    1. Derive an anisotropic scaling matrix from morphology and apply it to W.
    2. Build a morphology‑driven bivector, convert it to a rotor and blend with
       the supplied rotor.
    3. Run the TTT‑GA forward pass with VRAM caching.
    """
    # 1️⃣ Morphology‑aware weight scaling
    M = _build_morphology_matrix(morphology)          # 3×3 matrix
    if W.shape[0] != 3 or W.shape[1] != 3:
        raise ValueError("Current implementation expects 3‑dimensional data")
    adjusted_W = M @ W @ M.T                           # similarity transform

    # 2️⃣ Morphology‑driven rotor blending
    morph_biv = _build_bivector_from_morphology(morphology)
    morph_rotor = rotor_from_bivector(morph_biv)
    # Blend rotors via linear interpolation in the even subspace
    blended_R = (1.0 - eta_r) * R + eta_r * morph_rotor
    blended_R /= np.linalg.norm(blended_R)  # renormalise to keep unit rotor

    # 3️⃣ Forward pass with caching
    output = ttt_ga_forward(adjusted_W, blended_R, x_seq, eta_w, eta_r, cache)
    return output

def generate_procedural_entity(morphology: Morphology, slot_index: int) -> dict:
    """
    Produce a richer procedural entity whose attributes are directly
    derived from morphological indices and the current rotor state.
    """
    s = sphericity_index(morphology.length, morphology.width, morphology.height)
    f = flatness_index(morphology.length, morphology.width, morphology.height)

    # Combine indices non‑linearly for a more varied offset
    ternary_offset = int((s ** 2 + f ** 2) * (slot_index + 1) * 7) % 1_000_000

    persona_choices = [
        "ledger", "runner", "witness",
        "archivist", "carrier", "scribe",
        "oracle", "watcher", "guardian"
    ]
    persona = persona_choices[ternary_offset % len(persona_choices)]

    entity = {
        "slot_index": slot_index,
        "name": f"Entity-{slot_index}",
        "alias": f"Alias-{slot_index}",
        "persona": persona,
        "uuid": f"{ternary_offset:06x}-{slot_index:04d}",
        "ternary_offset": ternary_offset,
        "sphericity": round(s, 4),
        "flatness": round(f, 4)
    }
    return entity

# ----------------------------------------------------------------------
# Demo / self‑test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Fixed 3‑dimensional setting for clarity
    morph = Morphology(length=1.2, width=0.8, height=1.5, mass=2.3)

    # Random seed for reproducibility
    rng = np.random.default_rng(42)
    x_seq = rng.random(3)
    W = rng.random((3, 3))
    # Initialise rotor as identity (scalar = 1)
    R = np.zeros(8, dtype=np.float64)
    R[0] = 1.0

    eta_w = 0.02
    eta_r = 0.05
    cache = VramCache(max_entries=128)

    out = hybrid_ttt_ga_vram(x_seq, W, R, eta_w, eta_r, morph, cache)
    ent = generate_procedural_entity(morph, slot_index=7)

    print("Hybrid output vector:", out)
    print("Procedural entity:", ent)