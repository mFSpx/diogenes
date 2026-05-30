# DARWIN HAMMER — match 3297, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m1641_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2588_s1.py (gen6)
# born: 2026-05-29T23:49:06Z

"""Hybrid Multivector‑Hoeffding‑Entropy Algorithm
================================================

This module fuses the mathematical cores of the two parent algorithms:

* **Parent A** – *Hybrid Hoeffding‑XGBoost‑Regret MinHash Analysis with Leader‑Tree Election*.
  It provides a probabilistic acceptance rule based on an energy change `ΔE`,
  a temperature term, and an entropy contribution derived from a MinHash/Jaccard
  similarity.  It also uses the classic Hoeffding bound to decide when a
  statistical split is reliable.

* **Parent B** – *Multivector‑based Feature Extraction with Certainty Flags*.
  It introduces a geometric‑algebra `Multivector` representation for arbitrary
  feature vectors and a `CertaintyFlag` data‑class that records the epistemic
  status of a decision.

**Mathematical bridge**

Both parents manipulate a *scalar energy* that is a function of the data.
In the hybrid we:

1. Encode a raw feature vector `x ∈ ℝ^d` as a **multivector** `M(x)`.
2. Compute a similarity `s = Jaccard(M(x), M(r))` (approximated by MinHash) 
   between the current multivector and a reference multivector `r`.
3. Transform the similarity into an **entropy term**  
   `H(s) = -s·log(s) - (1-s)·log(1-s)`.
4. Combine `H(s)` with the energy change `ΔE` in the **acceptance probability**  

   `p_acc = 1               if ΔE < 0`  
   `p_acc = exp( -(ΔE + λ·H(s)) / T )   otherwise`

   where `λ` weights the entropy contribution and `T` is a temperature.
5. The Hoeffding bound `ε = sqrt( (1/(2n))·ln(1/δ) )` (with `n` samples and
   confidence `δ`) is *tightened* by the same entropy term, producing a
   **regularised bound** `ε' = ε / (1 + λ·H(s))`.

The hybrid algorithm therefore decides whether to *accept* a new state
(and optionally split a Hoeffding tree node) using a single scalar that
encodes geometric similarity, information‑theoretic regularisation, and
statistical confidence.

The public API consists of three representative functions:

* `multivector_from_features`
* `acceptance_probability`
* `hybrid_decision`

which together demonstrate the fused mathematics.  A small smoke test
exercises the pipeline without external dependencies.  
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Tuple, Sequence, Mapping, Hashable, List

import numpy as np

# ----------------------------------------------------------------------
# Parent B – geometric algebra utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Multivector:
    """Simple multivector storing a list of scalar components.

    The class supports addition, scalar multiplication and an inner product
    that is used as a similarity proxy.
    """
    components: Tuple[float, ...]

    def __add__(self, other: "Multivector") -> "Multivector":
        if len(self.components) != len(other.components):
            raise ValueError("Multivectors must have the same dimension")
        return Multivector(tuple(a + b for a, b in zip(self.components, other.components)))

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector(tuple(scalar * c for c in self.components))

    __mul__ = __rmul__

    def inner(self, other: "Multivector") -> float:
        """Standard Euclidean inner product."""
        return sum(a * b for a, b in zip(self.components, other.components))

    def norm(self) -> float:
        return math.sqrt(self.inner(self))

    def to_numpy(self) -> np.ndarray:
        return np.array(self.components, dtype=float)


def multivector_from_features(features: Sequence[float]) -> Multivector:
    """Convert an arbitrary feature sequence into a Multivector.

    The conversion is identity – the feature values become the scalar
    components of the multivector.  For robustness the input is cast to
    ``float`` and padded/truncated to a fixed dimension (here 8) so that
    all multivectors are comparable.
    """
    dim = 8
    arr = np.asarray(features, dtype=float)
    if arr.size < dim:
        arr = np.pad(arr, (0, dim - arr.size), constant_values=0.0)
    elif arr.size > dim:
        arr = arr[:dim]
    return Multivector(tuple(arr.tolist()))


# ----------------------------------------------------------------------
# Parent A – entropy, acceptance and Hoeffding bound
# ----------------------------------------------------------------------


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    x_arr = np.asarray(x, dtype=float)
    pos_mask = x_arr >= 0
    neg_mask = ~pos_mask
    out = np.empty_like(x_arr, dtype=float)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x_arr[pos_mask]))
    exp_x = np.exp(x_arr[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


def jaccard_similarity(a: Multivector, b: Multivector) -> float:
    """Jaccard‑style similarity based on binary support of components.

    Each component is treated as present if its absolute value exceeds a
    small threshold.  The similarity is |A∩B| / |A∪B|.
    """
    thresh = 1e-6
    set_a = {i for i, v in enumerate(a.components) if abs(v) > thresh}
    set_b = {i for i, v in enumerate(b.components) if abs(v) > thresh}
    if not set_a and not set_b:
        return 1.0  # both empty → identical
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union


def entropy_term(similarity: float) -> float:
    """Binary entropy of a similarity value in [0,1].

    Returns 0 for extremes (0 or 1) and the maximal value 0.693… for 0.5.
    """
    if similarity <= 0.0 or similarity >= 1.0:
        return 0.0
    return -similarity * math.log(similarity) - (1.0 - similarity) * math.log(1.0 - similarity)


def acceptance_probability(delta_e: float, temperature: float, entropy: float, lam: float = 1.0) -> float:
    """Hybrid acceptance rule.

    If the energy change is favourable (negative) we accept deterministically.
    Otherwise we exponentiate the *regularised* energy change
    ``delta_e + λ·entropy`` scaled by the temperature.
    """
    if delta_e < 0.0:
        return 1.0
    exponent = -(delta_e + lam * entropy) / max(temperature, 1e-9)
    return math.exp(exponent)


def hoeffding_bound(n: int, delta: float) -> float:
    """Classic Hoeffding bound ε = sqrt( (1/(2n)) * ln(1/δ) )."""
    if n <= 0:
        raise ValueError("sample count n must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt(math.log(1.0 / delta) / (2 * n))


def regularised_hoeffding(epsilon: float, entropy: float, lam: float = 1.0) -> float:
    """Tighten the bound using the entropy term."""
    if entropy <= 0.0:
        return epsilon
    return epsilon / (1.0 + lam * entropy)


# ----------------------------------------------------------------------
# Epistemic flag (Parent B)
# ----------------------------------------------------------------------


EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", date.today().isoformat())

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=confidence_bps,
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(evidence_refs),
    )


# ----------------------------------------------------------------------
# Hybrid core – three demonstrative functions
# ----------------------------------------------------------------------


def hybrid_energy_change(
    current: Multivector,
    candidate: Multivector,
    weight: float = 1.0,
) -> float:
    """Compute a scalar energy difference between two states.

    Energy is defined as the squared Euclidean distance scaled by ``weight``.
    """
    diff = np.array(candidate.components) - np.array(current.components)
    return weight * float(np.dot(diff, diff))


def hybrid_decision(
    current: Multivector,
    candidate: Multivector,
    reference: Multivector,
    *,
    temperature: float = 1.0,
    lam: float = 1.0,
    hoeffding_n: int = 100,
    hoeffding_delta: float = 0.05,
) -> Tuple[bool, CertaintyFlag]:
    """Perform a single hybrid step.

    Returns a tuple ``(accept, flag)`` where ``accept`` indicates whether the
    candidate state is adopted and ``flag`` encodes the epistemic confidence.
    """
    # 1. Energy change
    delta_e = hybrid_energy_change(current, candidate)

    # 2. Similarity & entropy
    sim = jaccard_similarity(candidate, reference)
    ent = entropy_term(sim)

    # 3. Acceptance probability
    p_acc = acceptance_probability(delta_e, temperature, ent, lam)

    # 4. Hoeffding regularisation (used only to set confidence)
    eps = hoeffding_bound(hoeffding_n, hoeffding_delta)
    eps_reg = regularised_hoeffding(eps, ent, lam)

    # Confidence is higher when the regularised bound is tighter
    confidence = max(0, min(10000, int((1.0 - eps_reg) * 10000)))

    accept = random.random() < p_acc
    flag_label = "FACT" if accept else "POSSIBLE"
    flag = certainty(
        flag_label,
        confidence_bps=confidence,
        authority_class="HybridEngine",
        rationale=(
            f"ΔE={delta_e:.4f}, sim={sim:.3f}, ent={ent:.3f}, "
            f"p_acc={p_acc:.3f}, ε'={eps_reg:.5f}"
        ),
        evidence_refs=("multivector", "hoeffding"),
    )
    return accept, flag


def hybrid_batch_process(
    data: Sequence[Sequence[float]],
    reference_vec: Sequence[float],
    *,
    temperature: float = 1.0,
    lam: float = 1.0,
    hoeffding_n: int = 100,
    hoeffding_delta: float = 0.05,
) -> List[Tuple[int, CertaintyFlag]]:
    """Process a batch of feature vectors, returning decisions for each.

    The first element of each result tuple is ``1`` for acceptance, ``0`` otherwise.
    """
    reference = multivector_from_features(reference_vec)
    current = reference  # initialise current state as the reference
    results: List[Tuple[int, CertaintyFlag]] = []

    for vec in data:
        candidate = multivector_from_features(vec)
        accept, flag = hybrid_decision(
            current,
            candidate,
            reference,
            temperature=temperature,
            lam=lam,
            hoeffding_n=hoeffding_n,
            hoeffding_delta=hoeffding_delta,
        )
        results.append((int(accept), flag))
        if accept:
            current = candidate  # move the state forward
    return results


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Generate a tiny synthetic dataset
    random.seed(42)
    np.random.seed(42)

    reference = [0.5, 0.2, 0.1, 0.0, 0.3]
    batch = [
        [0.51, 0.19, 0.11, 0.0, 0.29],
        [0.6, 0.1, 0.0, 0.0, 0.2],
        [0.4, 0.25, 0.15, 0.0, 0.35],
        [0.5, 0.2, 0.1, 0.0, 0.3],  # identical to reference
    ]

    decisions = hybrid_batch_process(
        batch,
        reference,
        temperature=0.8,
        lam=0.7,
        hoeffding_n=200,
        hoeffding_delta=0.01,
    )

    for i, (accept, flag) in enumerate(decisions):
        print(f"Sample {i}: accept={bool(accept)}, flag={flag.label}, confidence={flag.confidence_bps}")
    sys.exit(0)