# DARWIN HAMMER — match 2766, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m1404_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1862_s2.py (gen5)
# born: 2026-05-29T23:45:43Z

"""Hybrid Caputo‑MinHash‑Weekday Rotor (HCMWR) algorithm.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m1404_s2.py (HCGSI)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1862_s2.py (Hybrid MinHash‑Weekday)

Mathematical bridge:
The Caputo fractional derivative provides a power‑law memory kernel.  By
interpreting the kernel weight  t⁻ᵅ  as an angular factor we build a
2‑D rotor (geometric‑algebra rotation) whose angle is proportional to the
mean of a MinHash signature.  The same MinHash signature modulates the
amplitude of the weekday‑dependent weight vector used in the second parent.
Thus a single scalar (the MinHash‑derived mean) couples the fractional‑
memory, the geometric rotation and the weekday weighting, yielding a unified
update rule for a state vector."""
import math
import random
import sys
from pathlib import Path
from datetime import date
import hashlib
import numpy as np

# ----------------------------------------------------------------------
# Helper functions (parent A)
# ----------------------------------------------------------------------
def lanczos_gamma(z: float) -> float:
    """Lanczos approximation of Gamma(z) for z > 0."""
    if z < 0.5:
        return math.gamma(1 - z) * math.gamma(z) / math.sin(math.pi * z)
    g = 7
    z = z + g + 0.5
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    term = 1.0
    for c in p:
        term *= (z + c) / (z - c)
    return math.sqrt(2 * math.pi) * (z ** (z + 0.5)) * math.exp(-z) * term


def caputo_derivative(f: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    """
    Approximate the Caputo fractional derivative of order ``alpha`` for a
    discrete signal ``f`` sampled at times ``t`` (both 1‑D arrays of equal
    length).  The implementation follows the definition

        Dᵅ f(t) ≈ Σ (f[i+1]‑f[i]) * (t[i+1]‑t[i])^{-α}

    and returns a vector of the same length as ``f`` (the last element is
    zero because the finite‑difference stencil is one element shorter).
    """
    if len(f) != len(t):
        raise ValueError("f and t must have the same length")
    dt = np.diff(t)
    df = np.diff(f)
    # weight = dt^{-α}
    weight = dt ** (-alpha)
    integral = np.dot(df, weight)
    # Broadcast the scalar integral to the size of f for simplicity
    result = np.empty_like(f, dtype=float)
    result[:-1] = integral
    result[-1] = 0.0
    return result


# ----------------------------------------------------------------------
# MinHash utilities (parent B)
# ----------------------------------------------------------------------
def _minhash_token(token: str, seed: int) -> int:
    """Hash a token with a given seed and return a 64‑bit integer."""
    h = hashlib.blake2b(digest_size=8)
    h.update(seed.to_bytes(4, "little") + token.encode("utf-8"))
    return int.from_bytes(h.digest(), "little")


def minhash_signature(text: str, num_perm: int = 8) -> list[int]:
    """
    Compute a simple MinHash signature for *text* using ``num_perm`` independent
    hash permutations.  The signature is a list of the minimum hash value for
    each permutation over all whitespace‑separated tokens.
    """
    tokens = text.split()
    signature = []
    for seed in range(num_perm):
        mins = min(_minhash_token(tok, seed) for tok in tokens) if tokens else 0
        signature.append(mins)
    return signature


# ----------------------------------------------------------------------
# Weekday weight vector (parent B)
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: tuple[str, ...], dow: int, minhash_sig: list[int]) -> np.ndarray:
    """
    Produce a normalized weight vector for *groups* based on the weekday index
    ``dow`` (0 = Sunday … 6 = Saturday) and modulate its amplitude using the
    *minhash_sig*.

    The base shape is a sinusoid over the group index; the amplitude is
    scaled by the mean of the MinHash signature.
    """
    n = len(groups)
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    mean_sig = np.mean(minhash_sig) if minhash_sig else 0.0
    # Scale factor maps mean_sig ∈ [0, 2**64) → roughly [0.1, 1.0]
    amplitude = 0.1 + 0.9 * (mean_sig / (2 ** 64 - 1))
    weight_vec = amplitude * np.sin(base_angles + phase)
    # Normalize to L2‑norm = 1 (avoid division by zero)
    norm = np.linalg.norm(weight_vec)
    return weight_vec / norm if norm > 0 else weight_vec


# ----------------------------------------------------------------------
# Geometric‑Algebra rotor derived from Caputo kernel (bridge)
# ----------------------------------------------------------------------
def rotor_from_caputo(alpha: float, minhash_sig: list[int]) -> np.ndarray:
    """
    Build a 2‑D rotation matrix (rotor) whose angle θ is proportional to the
    fractional order ``alpha`` and to the mean of the MinHash signature.
    """
    mean_sig = np.mean(minhash_sig) if minhash_sig else 0.0
    # Map mean_sig to a factor in [0, 1]
    factor = mean_sig / (2 ** 64 - 1)
    theta = alpha * math.pi * factor  # angle in radians
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def apply_rotor(vec: np.ndarray, rotor: np.ndarray) -> np.ndarray:
    """
    Apply a 2‑D rotor to a vector.  If ``vec`` has length >2, the rotor is
    applied independently to each consecutive pair (i.e. a block‑diagonal
    action).  The function returns a vector of the same shape.
    """
    if vec.ndim != 1:
        raise ValueError("vec must be a 1‑D array")
    out = np.empty_like(vec, dtype=float)
    for i in range(0, len(vec) - 1, 2):
        block = vec[i : i + 2]
        out[i : i + 2] = rotor @ block
    if len(vec) % 2 == 1:
        out[-1] = vec[-1]  # leave the last element untouched
    return out


# ----------------------------------------------------------------------
# Hybrid operation (core)
# ----------------------------------------------------------------------
def hybrid_state_update(
    state: np.ndarray,
    t: np.ndarray,
    signal: np.ndarray,
    groups: tuple[str, ...],
    today: tuple[int, int, int],
    alpha: float,
    num_minhash_perm: int = 8,
) -> np.ndarray:
    """
    Perform one hybrid update step.

    1. Compute a Caputo fractional derivative of ``signal``.
    2. Build a MinHash signature from the string representation of ``signal``.
    3. Generate a weekday‑dependent weight vector for ``groups``.
    4. Construct a rotor from ``alpha`` and the MinHash signature.
    5. Rotate the derivative, modulate it with the weight vector, and add
       the result to the current ``state``.

    Parameters
    ----------
    state : np.ndarray
        Current state vector (length = len(groups) or a multiple thereof).
    t : np.ndarray
        Monotonic time stamps corresponding to ``signal``.
    signal : np.ndarray
        Raw input signal whose fractional derivative is required.
    groups : tuple[str, ...]
        Logical groups used for the weekday weight vector.
    today : tuple[int, int, int]
        (year, month, day) for weekday calculation.
    alpha : float
        Order of the Caputo derivative (0 < alpha ≤ 1).
    num_minhash_perm : int
        Number of permutations for the MinHash signature.

    Returns
    -------
    np.ndarray
        Updated state vector.
    """
    # 1. Fractional derivative (scalar for simplicity)
    deriv = caputo_derivative(signal, t, alpha)

    # 2. MinHash signature of the signal (as text)
    sig_text = " ".join(map(str, signal.tolist()))
    mh_sig = minhash_signature(sig_text, num_perm=num_minhash_perm)

    # 3. Weekday weight vector
    dow = doomsday(*today)
    w_vec = weekday_weight_vector(groups, dow, mh_sig)

    # 4. Rotor from Caputo + MinHash
    rotor = rotor_from_caputo(alpha, mh_sig)

    # 5. Rotate derivative (need same length as state)
    # Pad or truncate derivative to match state length
    der_padded = np.resize(deriv, state.shape)
    der_rotated = apply_rotor(der_padded, rotor)

    # Modulate with weight vector (broadcast if needed)
    if w_vec.shape != der_rotated.shape:
        # broadcast to match the longer length
        w_broadcast = np.resize(w_vec, der_rotated.shape)
    else:
        w_broadcast = w_vec

    update = w_broadcast * der_rotated
    new_state = state + update
    return new_state


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (date(year, month, day).weekday() + 1) % 7


def _pct(value: float) -> float:
    """Round a float to six decimal places (utility from parent B)."""
    return round(float(value), 6)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Dummy time series
    t = np.linspace(0, 10, 50)
    signal = np.sin(t) + 0.1 * np.random.randn(t.size)

    # State vector aligned with groups
    GROUPS = ("codex", "groq", "cohere", "local_models")
    state = np.zeros(len(GROUPS))

    # Today's date
    today = (2026, 5, 29)

    # Fractional order
    alpha = 0.7

    new_state = hybrid_state_update(
        state=state,
        t=t,
        signal=signal,
        groups=GROUPS,
        today=today,
        alpha=alpha,
        num_minhash_perm=8,
    )

    print("Old state:", state)
    print("New state:", new_state)
    print("State delta (rounded):", [_pct(v) for v in new_state - state])