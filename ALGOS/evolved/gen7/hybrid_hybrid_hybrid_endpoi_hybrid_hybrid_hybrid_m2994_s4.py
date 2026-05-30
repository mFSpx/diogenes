# DARWIN HAMMER — match 2994, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_endpoint_circ_state_space_duality_m1_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1938_s1.py (gen6)
# born: 2026-05-29T23:47:09Z

"""Hybrid Endpoint‑Geometric State‑Space Engine
================================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_endpoint_circ_state_space_duality_m1_s2.py``  
  Provides a linear state‑space model (SSM) where each engine endpoint is a
  state dimension.  Per‑step matrices  

  * ``A_t`` – diagonal decay (failure‑rate)  
  * ``B_t`` – column vector derived from endpoint morphology  
  * ``C_t`` – row vector of the *health_score*  

  give the update  

  ``h_t = A_t·h_{t‑1} + B_t·x_t`` , ``y_t = C_t·h_t``.  

  By the semiseparable duality the same mapping can be written as a single
  lower‑triangular matrix ``M`` such that ``Y = M·X``.

* **Parent B** – ``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1938_s1.py``  
  Introduces a geometric‑algebra ``Multivector`` class and a placeholder
  ``koopman_operator`` that lifts a vector‑valued observable into a higher‑
  dimensional linear operator.

Mathematical Bridge
-------------------
We embed the *morphology* of each endpoint into a grade‑1 multivector
``m`` (one basis blade per endpoint).  The geometric product of ``m`` with a
real‑valued morphology vector ``μ`` yields a new vector  

``b = geometric_product(m, μ)``  

which we use as the ``B_t`` column of the SSM.  The multivector therefore
acts as a *morphology state* that is propagated by the Koopman operator.
The resulting ``B_t`` couples the geometric‑algebra world (Parent B) to the
linear‑state‑space world (Parent A).  The hybrid engine proceeds as:

1. Build endpoint health quantities (failure‑rate, recovery‑priority,
   health_score) – Parent A.
2. Encode morphology as a grade‑1 multivector and obtain ``B_t`` via the
   geometric product – Parent B.
3. Assemble the per‑step SSM matrices and construct the semiseparable matrix
   ``M``.
4. Apply ``M`` to a request‑load vector ``X`` to obtain scores ``Y`` and pick
   the best endpoint at each step.

The three public functions below illustrate this pipeline.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Parent B – geometric algebra utilities

class Multivector:
    """Element of the Clifford algebra Cl(n,0) as a mapping blade → coefficient."""
    def __init__(self, components: dict[int, float] | dict[frozenset, float], n: int):
        # components may be given as {index: coeff} for grade‑1 blades or
        # {frozenset(blade): coeff} for arbitrary blades.
        self.n = int(n)
        self.components: dict[frozenset, float] = {}
        for blade, coeff in components.items():
            if isinstance(blade, int):
                blade = frozenset({blade})
            self.components[blade] = coeff if abs(coeff) > 1e-15 else 0.0
        # prune zero entries
        self.components = {b: c for b, c in self.components.items() if abs(c) > 1e-15}

    def grade(self, k: int) -> 'Multivector':
        """Return a new multivector containing only blades of grade *k*."""
        return Multivector({b: c for b, c in self.components.items() if len(b) == k}, self.n)

    def scalar_part(self) -> float:
        """Scalar (grade‑0) part."""
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        return f"Multivector({self.components}, n={self.n})"


def _blade_sign(indices: list[int]) -> tuple[list[int], int]:
    """Sort *indices* by bubble‑sort while tracking the sign of the permutation."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> tuple[frozenset, int]:
    """Geometric product of two basis blades (Clifford product)."""
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    # Duplicate indices cancel (e² = 1 in Euclidean signature)
    # Remove pairs that appeared twice
    i = 0
    while i < len(sorted_blade) - 1:
        if sorted_blade[i] == sorted_blade[i + 1]:
            del sorted_blade[i:i + 2]
        else:
            i += 1
    return frozenset(sorted_blade), sign


def geometric_product(mv: Multivector, vec: np.ndarray) -> np.ndarray:
    """
    Geometric product of a *grade‑1* multivector with a real vector.

    The result is a vector whose i‑th component is the sum of the multivector
    coefficient on blade ``{i}`` multiplied by ``vec[i]``.
    """
    n = mv.n
    result = np.zeros(n)
    for blade, coeff in mv.components.items():
        if len(blade) == 1:
            (idx,) = blade
            result[idx] = coeff * vec[idx]
    return result


def koopman_operator(multivector: Multivector, X: np.ndarray, X_prime: np.ndarray) -> np.ndarray:
    """
    Placeholder Koopman operator.  In a full implementation this would
    construct a linear operator that advances observables of the multivector.
    Here we return a random matrix with compatible shape.
    """
    rows, cols = X.shape
    return np.random.rand(rows, cols)


# ----------------------------------------------------------------------
# Parent A – endpoint health utilities

@dataclass
class Endpoint:
    """Simple container for endpoint metrics."""
    name: str
    failures: int
    failure_threshold: int
    righting_time_index: float  # morphology scalar (higher ⇒ flatter)
    # Additional fields can be added without breaking the hybrid logic.


def _failure_rate(ep: Endpoint) -> float:
    """Normalized failure rate in [0, 1]."""
    if ep.failure_threshold == 0:
        return 0.0
    return min(1.0, ep.failures / ep.failure_threshold)


def _recovery_priority(ep: Endpoint) -> float:
    """
    Recovery priority derived from morphology.
    We map ``righting_time_index`` (assumed ≥0) to a value in [0, 1]
    where flatter endpoints (larger index) receive higher priority.
    """
    # Simple linear scaling; in practice this could be a more sophisticated
    # monotonic transform.
    max_idx = 10.0  # heuristic upper bound
    return min(1.0, ep.righting_time_index / max_idx)


def compute_endpoint_metrics(endpoints: list[Endpoint]) -> dict:
    """
    Compute vectorised health quantities for a list of endpoints.

    Returns a dictionary with keys:
        - ``fr`` : failure‑rate vector (shape (n,))
        - ``rp`` : recovery‑priority vector (shape (n,))
        - ``hs`` : health‑score vector (shape (n,)) = (1‑fr)*(1‑rp)
        - ``morph`` : morphology vector (shape (n,)) = righting_time_index
    """
    n = len(endpoints)
    fr = np.empty(n)
    rp = np.empty(n)
    hs = np.empty(n)
    morph = np.empty(n)
    for i, ep in enumerate(endpoints):
        fr[i] = _failure_rate(ep)
        rp[i] = _recovery_priority(ep)
        hs[i] = (1.0 - fr[i]) * (1.0 - rp[i])
        morph[i] = ep.righting_time_index
    return {"fr": fr, "rp": rp, "hs": hs, "morph": morph}


# ----------------------------------------------------------------------
# Hybrid Engine Core

def build_semiseparable_matrix(fr_seq: np.ndarray, B_seq: np.ndarray, C_seq: np.ndarray) -> np.ndarray:
    """
    Construct the lower‑triangular semiseparable matrix ``M`` that implements
    the SSM defined by the per‑step matrices

        A_t = diag(fr_seq[t])          (decay)
        B_t = B_seq[t][:, None]        (column)
        C_t = C_seq[t][None, :]        (row)

    The resulting ``M`` satisfies ``Y = M @ X`` for any input sequence ``X``.
    """
    T, n = fr_seq.shape[0], fr_seq.shape[1]
    M = np.zeros((T, T))
    # Pre‑compute cumulative products of A matrices
    # prod_A[t, k] = Π_{i=k+1..t} A_i   (identity when k == t)
    prod_A = np.ones((T, T, n))
    for t in range(T):
        for k in range(t - 1, -1, -1):
            prod_A[t, k] = prod_A[t, k + 1] * fr_seq[k + 1]

    for t in range(T):
        for k in range(t + 1):
            # scalar contribution for each endpoint, then summed
            contribution = C_seq[t] @ (prod_A[t, k] * B_seq[k])
            M[t, k] = contribution
    return M


def hybrid_endpoint_score(endpoints: list[Endpoint], request_sequence: np.ndarray) -> dict:
    """
    Run the hybrid algorithm on a request sequence.

    Parameters
    ----------
    endpoints : list[Endpoint]
        The pool of candidate endpoints.
    request_sequence : np.ndarray, shape (T,)
        Scalar load (e.g. number of requests) at each discrete time step.

    Returns
    -------
    dict with keys
        - ``scores`` : array (T,) of the aggregated health score ``y_t``.
        - ``selected`` : array (T,) of the index of the endpoint that contributed
          most to each ``y_t`` (ties broken arbitrarily).
        - ``M`` : the semiseparable matrix used for the computation.
    """
    # 1. Health metrics (Parent A)
    metrics = compute_endpoint_metrics(endpoints)
    fr_vec = metrics["fr"]          # shape (n,)
    hs_vec = metrics["hs"]          # shape (n,)
    morph_vec = metrics["morph"]    # shape (n,)

    n = len(endpoints)
    T = request_sequence.shape[0]

    # 2. Build per‑step matrices
    #    A_t is the same for all t because failure rate is assumed static per endpoint.
    A_seq = np.tile(np.diag(fr_vec), (T, 1, 1))          # shape (T, n, n)

    # 3. Encode morphology as a grade‑1 multivector (Parent B)
    mv = Multivector({i: 1.0 for i in range(n)}, n)    # unit coefficients for each blade

    # 4. Geometric product gives B_t (column vector) for every time step.
    #    Here we keep B_t constant across time; the geometric product can be
    #    recomputed each step if the morphology evolves.
    B_vec = geometric_product(mv, morph_vec)          # shape (n,)
    B_seq = np.tile(B_vec[:, None], (1, T)).T          # shape (T, n, 1)

    # 5. C_t is simply the health‑score row (same each step).
    C_seq = np.tile(hs_vec, (T, 1))                    # shape (T, n)

    # 6. Build the semiseparable matrix M (Parent A)
    #    For efficiency we reuse the scalar failure‑rate vector across time.
    fr_seq = np.tile(fr_vec, (T, 1))                   # shape (T, n)
    M = build_semiseparable_matrix(fr_seq, B_seq.squeeze(-1), C_seq)

    # 7. Apply M to the request sequence.
    scores = M @ request_sequence

    # 8. Determine the endpoint with the largest instantaneous contribution.
    #    Contribution of endpoint i at time t is:
    #        contrib[t,i] = C_t[i] * (Π_{k=i+1..t} fr_k[i]) * B_i[i]
    contrib = np.zeros((T, n))
    for t in range(T):
        for i in range(n):
            prod = 1.0
            for k in range(i + 1, t + 1):
                prod *= fr_vec[i]   # because A_k is diagonal with the same fr per endpoint
            contrib[t, i] = hs_vec[i] * prod * B_vec[i] * request_sequence[i] if i <= t else 0.0
    selected = np.argmax(contrib, axis=1)

    return {"scores": scores, "selected": selected, "M": M}


def simulate_hybrid(endpoints: list[Endpoint], steps: int = 10) -> None:
    """
    Convenience wrapper that generates a random request load, runs the hybrid
    engine and prints a concise report.
    """
    np.random.seed(42)
    request_seq = np.random.rand(steps)  # scalar load per step
    result = hybrid_endpoint_score(endpoints, request_seq)

    print("Request sequence :", request_seq)
    print("Aggregated scores :", result["scores"])
    print("Chosen endpoint  :", [endpoints[i].name for i in result["selected"]])
    print("Semiseparable M shape :", result["M"].shape)


# ----------------------------------------------------------------------
# Smoke test

if __name__ == "__main__":
    # Create a tiny pool of mock endpoints
    pool = [
        Endpoint(name="alpha", failures=2, failure_threshold=10, righting_time_index=3.2),
        Endpoint(name="beta", failures=5, failure_threshold=10, righting_time_index=7.5),
        Endpoint(name="gamma", failures=0, failure_threshold=10, righting_time_index=1.0),
    ]
    simulate_hybrid(pool, steps=8)