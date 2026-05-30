# DARWIN HAMMER — match 5798, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_fracti_m2251_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hybrid_m1197_s3.py (gen5)
# born: 2026-05-30T00:04:55Z

import numpy as np
import math
import random
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Tuple, List, Sequence


# ----------------------------------------------------------------------
# Constants & Helper Types
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = (
    "FACT",
    "PROBABLE",
    "POSSIBLE",
    "BULLSHIT",
    "SURE_MAYBE",
)


# ----------------------------------------------------------------------
# Core Data Structure
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CertaintyFlag:
    """
    Immutable container for a single epistemic certainty flag.
    """
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")

        # confidence expressed in basis points (0..10 000)
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10 000")

        # Auto‑populate timestamp if omitted
        if not self.generated_at:
            object.__setattr__(self, "generated_at", "2024-01-01T00:00:00Z")

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain‑dict representation suitable for JSON serialisation."""
        return asdict(self)


# ----------------------------------------------------------------------
# Factory Helper
# ----------------------------------------------------------------------
def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    """
    Convenience wrapper that validates arguments and builds a :class:`CertaintyFlag`.
    """
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


# ----------------------------------------------------------------------
# Mathematical Primitives
# ----------------------------------------------------------------------
def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1‑D arrays."""
    return np.linalg.norm(a - b)


def rbf_similarity_matrix(
    features: Sequence[np.ndarray],
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Vectorised computation of the Gaussian RBF similarity matrix.

    Parameters
    ----------
    features : sequence of 1‑D ``np.ndarray``
        Input vectors (shape ``(n, d)`` after conversion).
    epsilon : float, default 1.0
        Bandwidth parameter of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Symmetric similarity matrix ``S`` where ``S[i, j] = exp(-epsilon²‖x_i‑x_j‖²)``.
    """
    X = np.asarray(features, dtype=float)  # (n, d)
    if X.ndim != 2:
        raise ValueError("features must be a 2‑D collection of vectors")

    # Compute squared Euclidean distances via the identity
    # ||a‑b||² = ||a||² + ||b||² – 2·a·b
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)          # (n, 1)
    dists_sq = sq_norms + sq_norms.T - 2.0 * X @ X.T           # (n, n)
    np.clip(dists_sq, 0, None, out=dists_sq)                  # numerical safety

    return np.exp(-epsilon ** 2 * dists_sq)


def tropical_relu(matrix: np.ndarray) -> np.ndarray:
    """
    Tropical “max‑plus” projection using the ReLU (max(x, 0)) element‑wise.
    """
    return np.maximum(matrix, 0.0)


def gini_coefficient(values: Iterable[float]) -> float:
    """
    Gini coefficient for a non‑negative 1‑D iterable.

    Returns 0 for empty input or when the sum is zero.
    """
    xs = np.asarray(list(values), dtype=float)
    if xs.size == 0 or xs.sum() == 0:
        return 0.0
    if np.any(xs < 0):
        raise ValueError("values must be non‑negative")
    xs_sorted = np.sort(xs)
    n = xs_sorted.size
    cum = np.cumsum(xs_sorted)
    return float((n + 1 - 2 * np.sum(cum) / xs_sorted.sum()) / n)


# ----------------------------------------------------------------------
# LSH‑style MinHash (binary sketch)
# ----------------------------------------------------------------------
def _random_hyperplanes(d: int, n_planes: int, rng: random.Random) -> np.ndarray:
    """
    Generate ``n_planes`` random hyperplanes in ``d`` dimensions.
    Each hyperplane is a unit vector drawn from a standard normal distribution.
    """
    rng_state = np.random.RandomState(rng.randint(0, 2 ** 31 - 1))
    planes = rng_state.normal(size=(n_planes, d))
    norms = np.linalg.norm(planes, axis=1, keepdims=True)
    planes /= np.where(norms == 0, 1, norms)  # avoid division by zero
    return planes


def minhash_signature(
    features: Sequence[np.ndarray],
    n_planes: int = 64,
    rng: random.Random | None = None,
) -> np.ndarray:
    """
    Compute a binary MinHash‑like signature using random hyperplane LSH.

    The signature for a vector ``x`` is a bit‑string where each bit is
    ``1`` if ``x`` lies on the positive side of the corresponding hyperplane,
    otherwise ``0``.  This is a cheap, differentiable surrogate for classical
    MinHash and works directly on continuous features.

    Parameters
    ----------
    features : sequence of 1‑D ``np.ndarray``
        Input vectors.
    n_planes : int, default 64
        Number of random hyperplanes (signature length).
    rng : ``random.Random`` or ``None``
        Random source; if ``None`` a fresh ``random.Random`` is created.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(len(features), n_planes)`` containing 0/1 values.
    """
    if rng is None:
        rng = random.Random()
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError("features must be a 2‑D collection of vectors")
    planes = _random_hyperplanes(X.shape[1], n_planes, rng)          # (n_planes, d)
    # Projection → sign → 0/1
    proj = X @ planes.T                                               # (n, n_planes)
    return (proj >= 0).astype(np.uint8)


# ----------------------------------------------------------------------
# Fractional Power Binding
# ----------------------------------------------------------------------
def fractional_power_binding(
    matrix: np.ndarray,
    power: float = 0.5,
) -> np.ndarray:
    """
    Apply element‑wise fractional exponentiation followed by a tropical ReLU.

    This operation deepens the integration between the RBF similarity space
    and the tropical algebraic world used elsewhere in the system.
    """
    if power <= 0:
        raise ValueError("power must be positive")
    # Element‑wise power (preserves symmetry and non‑negativity)
    powered = np.power(matrix, power, out=np.empty_like(matrix))
    # Tropical ReLU ensures the result stays in the max‑plus semiring
    return tropical_relu(powered)


# ----------------------------------------------------------------------
# Core Hybrid Operation
# ----------------------------------------------------------------------
def hybrid_operation(
    features: List[np.ndarray],
    labels: List[str],
    confidence_bps: List[int],
    authority_class: List[str],
    rationale: List[str],
    *,
    epsilon: float = 1.0,
    minhash_planes: int = 64,
    binding_power: float = 0.5,
    rng: random.Random | None = None,
) -> List[CertaintyFlag]:
    """
    Perform the full hybrid pipeline:

    1. Compute the Gaussian RBF similarity matrix.
    2. Strengthen it with fractional‑power binding (deep integration).
    3. Derive a compact MinHash‑style sketch from the original features.
    4. Fuse the sketch with the bound similarity to obtain an evidence reference
       for each label.

    The resulting :class:`CertaintyFlag` objects contain a deterministic
    evidence reference that reflects both the continuous similarity structure
    and the discrete LSH sketch.
    """
    if not (len(features) == len(labels) == len(confidence_bps) == len(authority_class) == len(rationale)):
        raise ValueError("All input lists must have the same length")

    # 1. RBF similarity
    sim_matrix = rbf_similarity_matrix(features, epsilon=epsilon)

    # 2. Fractional power binding + tropical ReLU
    bound_matrix = fractional_power_binding(sim_matrix, power=binding_power)

    # 3. MinHash sketch (binary signature)
    sketch = minhash_signature(features, n_planes=minhash_planes, rng=rng)  # (n, n_planes)

    # 4. Fuse: for each row we compute a simple hash of the bound row
    #    concatenated with the sketch bits.  This yields a short, reproducible
    #    string that can be stored as evidence.
    evidence_refs: List[str] = []
    for i in range(len(labels)):
        # Convert bound row to a deterministic 64‑bit integer via xor‑folding
        row_bytes = bound_matrix[i].tobytes()
        row_hash = np.frombuffer(row_bytes, dtype=np.uint64).sum()  # simple reduction

        # Convert sketch bits to an integer
        sketch_int = int("".join(map(str, sketch[i].tolist())), 2)

        # Combine both using a prime‑based mixing function
        combined = (row_hash * 31 + sketch_int * 17) & ((1 << 64) - 1)
        evidence_refs.append(f"0x{combined:016x}")

    # Build CertaintyFlag objects
    flags: List[CertaintyFlag] = []
    for i, label in enumerate(labels):
        flags.append(
            certainty(
                label=label,
                confidence_bps=int(confidence_bps[i]),
                authority_class=authority_class[i],
                rationale=rationale[i],
                evidence_refs=(evidence_refs[i],),
            )
        )
    return flags


# ----------------------------------------------------------------------
# Gini of Certainty Flags
# ----------------------------------------------------------------------
def compute_gini_coefficient_of_certainty_flags(
    certainty_flags: List[CertaintyFlag],
) -> float:
    """
    Compute the Gini coefficient of the ``confidence_bps`` values across a list
    of :class:`CertaintyFlag` objects.
    """
    confidence_vals = [flag.confidence_bps for flag in certainty_flags]
    return gini_coefficient(confidence_vals)


# ----------------------------------------------------------------------
# Demonstration / CLI
# ----------------------------------------------------------------------
def _demo() -> None:
    """Simple demonstration that runs when the module is executed as a script."""
    rng = random.Random(42)

    features = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
    ]
    labels = ["FACT", "PROBABLE", "POSSIBLE"]
    confidence_bps = [1000, 2000, 3000]
    authority_class = ["high", "medium", "low"]
    rationale = ["good", "fair", "poor"]

    flags = hybrid_operation(
        features,
        labels,
        confidence_bps,
        authority_class,
        rationale,
        epsilon=0.8,
        minhash_planes=128,
        binding_power=0.6,
        rng=rng,
    )
    gini_val = compute_gini_coefficient_of_certainty_flags(flags)

    print("Certainty Flags:")
    for f in flags:
        print(f.as_dict())
    print("\nGini Coefficient of confidence_bps:", gini_val)


if __name__ == "__main__":
    _demo()