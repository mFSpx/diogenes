# DARWIN HAMMER — match 1735, survivor 1
# gen: 4
# parent_a: hybrid_ternary_lens_router_hybrid_hybrid_liquid_m124_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s2.py (gen3)
# born: 2026-05-29T23:38:33Z

"""
Hybrid MinHash‑Ternary‑Clifford Fusion

Parents:
- hybrid_ternary_lens_router_hybrid_hybrid_liquid_m124_s0.py (MinHash signatures, ternary vectors,
  diffusion noise schedule)
- hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s2.py (Clifford algebra multivectors,
  geometric product, decision‑tree weighting, Shannon entropy)

Mathematical Bridge:
The ternary vector generated from a command is interpreted as a grade‑1 multivector
(e₀, e₁, …). A MinHash similarity score between the command’s token set and a
reference token set yields a scalar weight w∈[0,1]. The weighted multivector
w·M is then combined with other weighted multivectors using the geometric product.
A diffusion noise schedule perturbs the token set before hashing, allowing the
fusion to explore stochastic similarity spaces. Entropy of the resulting weighted
multivectors quantifies decision‑making uncertainty.
"""

import hashlib
import json
import math
import random
import sys
import pathlib
from datetime import datetime, timezone
import numpy as np

# ---------- Parent A components ------------------------------------------------

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')


def signature(tokens: list[str], k: int = 128) -> list[int]:
    """MinHash signature of a token list."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: list[int], sig_b: list[int]) -> float:
    """Jaccard‑like similarity based on MinHash equality."""
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def ternary_vector(raw_command: str,
                   normalized_intent: str,
                   context: dict[str, str],
                   dims: int = 12) -> list[int]:
    """Deterministic ternary vector derived from command+intent+context."""
    digest = hashlib.sha256(
        (raw_command + "\0" + normalized_intent + "\0" + json.dumps(context, sort_keys=True))
        .encode()
    ).digest()
    return [((digest[i] % 3) - 1) for i in range(dims)]


def diffusion_noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """Noise schedule ᾱ_t used in diffusion models."""
    if schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        alpha_bars = np.clip(alpha_bars, a_min=1e-8, a_max=1.0)
        return alpha_bars
    raise ValueError(f"Unsupported schedule: {schedule}")


def corrupt_tokens(tokens: list[str], T: int, schedule: str = "cosine") -> list[str]:
    """Stochastically corrupt a token list according to a diffusion schedule."""
    alpha_bars = diffusion_noise_schedule(T, schedule)
    # probability of keeping a token at step t is sqrt(alpha_bar_t)
    keep_prob = math.sqrt(alpha_bars[-1])
    return [t for t in tokens if random.random() < keep_prob]


# ---------- Parent B components ------------------------------------------------

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # e_i * e_i = 1 → remove both and stop processing this pair
                lst.pop(j)
                lst.pop(j)  # second element shifts into position j
                n -= 2
                i -= 1  # stay on same outer iteration
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]):
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Clifford algebra element in Cl(n,0) stored as a dict {blade: coeff}."""

    def __init__(self, blades: dict[frozenset[int], float] | None = None):
        self.blades: dict[frozenset[int], float] = blades if blades is not None else {}

    @staticmethod
    def from_ternary(vec: list[int]) -> "Multivector":
        """Map a ternary vector to a grade‑1 multivector (e_i with coeff = value)."""
        blades = {}
        for idx, val in enumerate(vec):
            if val != 0:
                blades[frozenset({idx})] = float(val)
        return Multivector(blades)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self.blades.copy())
        for blade, coeff in other.blades.items():
            result.blades[blade] = result.blades.get(blade, 0.0) + coeff
        # prune near‑zero entries
        result.blades = {b: c for b, c in result.blades.items() if abs(c) > 1e-12}
        return result

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result = {}
        for blade_a, coeff_a in self.blades.items():
            for blade_b, coeff_b in other.blades.items():
                new_blade, sign = _multiply_blades(blade_a, blade_b)
                new_coeff = coeff_a * coeff_b * sign
                result[new_blade] = result.get(new_blade, 0.0) + new_coeff
        # prune
        result = {b: c for b, c in result.items() if abs(c) > 1e-12}
        return Multivector(result)

    def scale(self, scalar: float) -> "Multivector":
        """Scalar multiplication."""
        return Multivector({b: c * scalar for b, c in self.blades.items()})

    def coeffs(self) -> list[float]:
        """Return list of absolute coefficients for entropy calculations."""
        return [abs(c) for c in self.blades.values()]

    def __repr__(self) -> str:
        terms = [f"{c:.3g}*e{sorted(list(b))}" if b else f"{c:.3g}" for b, c in self.blades.items()]
        return " + ".join(terms) if terms else "0"


def shannon_entropy(probs: list[float]) -> float:
    """Standard Shannon entropy (base e)."""
    eps = 1e-12
    return -sum(p * math.log(p + eps) for p in probs if p > 0)


def entropy_of_multivector(mv: Multivector) -> float:
    """Entropy of the normalized absolute coefficient distribution."""
    coeffs = mv.coeffs()
    total = sum(coeffs)
    if total == 0:
        return 0.0
    probs = [c / total for c in coeffs]
    return shannon_entropy(probs)


# ---------- Hybrid functions ---------------------------------------------------

def hybrid_weighted_multivector(raw_command: str,
                                normalized_intent: str,
                                context: dict[str, str],
                                tokens: list[str],
                                reference_tokens: list[str],
                                k: int = 64,
                                dims: int = 8,
                                diffusion_steps: int = 10) -> Multivector:
    """
    Build a weighted multivector from command data.

    1. Corrupt the token list with a diffusion schedule.
    2. Compute MinHash signatures for the corrupted and reference token sets.
    3. Derive a similarity weight w ∈ [0,1].
    4. Convert the ternary vector of the command to a grade‑1 multivector M.
    5. Return w·M.
    """
    # 1. Diffusion‑based corruption
    corrupted = corrupt_tokens(tokens, diffusion_steps)

    # 2. MinHash signatures
    sig_a = signature(corrupted, k)
    sig_b = signature(reference_tokens, k)

    # 3. Similarity weight
    w = similarity(sig_a, sig_b)

    # 4. Ternary → multivector
    tv = ternary_vector(raw_command, normalized_intent, context, dims)
    M = Multivector.from_ternary(tv)

    # 5. Weight
    return M.scale(w)


def combine_weighted_multivectors(mvs: list[Multivector]) -> Multivector:
    """
    Combine a list of weighted multivectors using successive geometric products.
    The order is preserved; the result encodes interactions between all inputs.
    """
    if not mvs:
        return Multivector()
    result = mvs[0]
    for mv in mvs[1:]:
        result = result * mv
    return result


def hybrid_entropy_analysis(command_sets: list[dict]) -> float:
    """
    For each command dictionary (containing raw_command, normalized_intent,
    context, tokens, reference_tokens) compute a weighted multivector, combine
    them with geometric products, and return the Shannon entropy of the final
    multivector.
    """
    weighted = [
        hybrid_weighted_multivector(
            raw_command=cmd["raw_command"],
            normalized_intent=cmd["normalized_intent"],
            context=cmd["context"],
            tokens=cmd["tokens"],
            reference_tokens=cmd["reference_tokens"],
        )
        for cmd in command_sets
    ]
    combined = combine_weighted_multivectors(weighted)
    return entropy_of_multivector(combined)


# ---------- Smoke test ---------------------------------------------------------

if __name__ == "__main__":
    # Minimal reproducible example
    random.seed(42)

    # Example command groups
    cmd1 = {
        "raw_command": "move north",
        "normalized_intent": "navigate",
        "context": {"location": "room1"},
        "tokens": ["move", "north", "player"],
        "reference_tokens": ["go", "north", "character"],
    }

    cmd2 = {
        "raw_command": "pick up key",
        "normalized_intent": "acquire",
        "context": {"location": "room2"},
        "tokens": ["pick", "up", "key", "player"],
        "reference_tokens": ["grab", "key", "item"],
    }

    cmd3 = {
        "raw_command": "open door",
        "normalized_intent": "interact",
        "context": {"location": "room3"},
        "tokens": ["open", "door", "player"],
        "reference_tokens": ["unlock", "door", "object"],
    }

    commands = [cmd1, cmd2, cmd3]

    entropy = hybrid_entropy_analysis(commands)
    print(f"Hybrid entropy of combined multivector: {entropy:.6f}")

    # Demonstrate individual weighted multivector
    wm = hybrid_weighted_multivector(**cmd1)
    print(f"Weighted multivector for cmd1: {wm}")
    print(f"Entropy of cmd1 multivector: {entropy_of_multivector(wm):.6f}")