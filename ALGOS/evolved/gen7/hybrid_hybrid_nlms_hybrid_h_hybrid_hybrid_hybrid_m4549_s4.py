# DARWIN HAMMER — match 4549, survivor 4
# gen: 7
# parent_a: hybrid_nlms_hybrid_hybrid_worksh_m167_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2456_s0.py (gen6)
# born: 2026-05-29T23:56:34Z

import numpy as np
from collections import defaultdict
from typing import Dict, FrozenSet, Tuple, Iterable

# ----------------------------------------------------------------------
# Clifford algebra utilities (grade‑0 and grade‑1 only, sufficient for a
# vector‑valued weight representation).  The implementation is fully
# deterministic and works for any Euclidean metric.
# ----------------------------------------------------------------------
def _canonicalize(indices: Iterable[int]) -> Tuple[FrozenSet[int], int]:
    """
    Return a sorted frozenset of indices together with the sign (+1 / -1)
    that results from swapping adjacent basis vectors into canonical order.
    Duplicate indices cancel because e_i ^ e_i = 0.
    """
    lst = list(indices)
    sign = 1
    i = 0
    while i < len(lst):
        # bubble the current element to its correct position
        j = i
        while j > 0 and lst[j - 1] > lst[j]:
            lst[j - 1], lst[j] = lst[j], lst[j - 1]
            sign = -sign
            j -= 1
        # cancel duplicates
        if j > 0 and lst[j - 1] == lst[j]:
            # remove both copies
            del lst[j - 1 : j + 1]
            sign = -sign  # swapping identical indices introduces a sign flip
            i = max(i - 1, 0)  # re‑check the new neighbour
        else:
            i += 1
    return frozenset(lst), sign


def geometric_product(a: Dict[FrozenSet[int], float],
                      b: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """
    Compute the full Clifford (geometric) product of two multivectors.
    The multivectors are represented as dictionaries mapping a frozenset of
    basis indices to a scalar coefficient.  The empty frozenset denotes the
    scalar (grade‑0) part.
    """
    result = defaultdict(float)
    for blade_a, coeff_a in a.items():
        for blade_b, coeff_b in b.items():
            combined = list(blade_a) + list(blade_b)
            blade, sign = _canonicalize(combined)
            result[blade] += sign * coeff_a * coeff_b
    # prune zero entries
    return {k: v for k, v in result.items() if abs(v) > 1e-12}


def scalar(s: float) -> Dict[FrozenSet[int], float]:
    """Convenient constructor for a scalar multivector."""
    return {frozenset(): s}


def vector(v: np.ndarray) -> Dict[FrozenSet[int], float]:
    """
    Convert a 1‑D numpy array into a grade‑1 multivector.
    Basis e_i is represented by the frozenset {i}.
    """
    return {frozenset({i}): float(val) for i, val in enumerate(v)}


def multivector_to_array(mv: Dict[FrozenSet[int], float],
                         dim: int) -> np.ndarray:
    """
    Extract the grade‑1 part of a multivector as a numpy array of length `dim`.
    Scalars are ignored; missing basis vectors are treated as zero.
    """
    arr = np.zeros(dim, dtype=float)
    for blade, coeff in mv.items():
        if len(blade) == 1:
            i = next(iter(blade))
            arr[i] = coeff
    return arr


# ----------------------------------------------------------------------
# RBF activation expressed as a multivector
# ----------------------------------------------------------------------
def rbf_activation_mv(x: np.ndarray,
                      centers: np.ndarray,
                      sigma: float) -> Dict[FrozenSet[int], float]:
    """
    Compute a Gaussian RBF activation for each centre and embed the result
    into a multivector whose grade‑1 coefficients are the activations.
    """
    dists = np.linalg.norm(centers - x, axis=1)
    act = np.exp(- (dists ** 2) / (2 * sigma ** 2))
    # treat each centre as a separate basis direction
    return {frozenset({i}): float(val) for i, val in enumerate(act)}


# ----------------------------------------------------------------------
# Adaptive weight update (deterministic LMS variant) with Clifford coupling
# ----------------------------------------------------------------------
def lms_update(weights: np.ndarray,
               x: np.ndarray,
               target: float,
               mu: float = 0.5,
               eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    """
    Classic normalized LMS update.
    Returns the updated weight vector and the instantaneous error.
    """
    y = np.dot(weights, x)
    error = target - y
    norm_sq = np.dot(x, x) + eps
    new_weights = weights + mu * error * x / norm_sq
    return new_weights, error


def hybrid_update(weights: np.ndarray,
                  x: np.ndarray,
                  target: float,
                  mu: float = 0.5,
                  eps: float = 1e-9,
                  centers: np.ndarray = None,
                  sigma: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Perform one hybrid adaptation step:
      1. LMS weight update (Parent A).
      2. Build an RBF‑based multivector from the input (Parent B).
      3. Couple the updated weight vector with the activation multivector
         via the geometric product, thereby rotating / scaling the weight
         in the Clifford space.
    """
    # ----- 1. LMS step -------------------------------------------------
    w_lms, error = lms_update(weights, x, target, mu, eps)

    # ----- 2. Build activation multivector -----------------------------
    if centers is not None:
        act_mv = rbf_activation_mv(x, centers, sigma)          # grade‑1 only
    else:
        # If no centres are supplied we fall back to the identity multivector
        act_mv = scalar(1.0)

    # ----- 3. Clifford coupling -----------------------------------------
    # Convert the LMS‑updated weight vector to a multivector (grade‑1)
    w_mv = vector(w_lms)

    # Geometric product mixes scalar and vector parts; the result is a new
    # multivector.  We keep only the vector part as the new weight vector.
    prod_mv = geometric_product(w_mv, act_mv)

    # Extract the vector part back to a numpy array
    new_weights = multivector_to_array(prod_mv, dim=weights.shape[0])

    # If the activation was scalar (no centres) the product reduces to w_mv,
    # so new_weights equals w_lms – preserving the pure LMS behaviour.
    return new_weights, error


def hybrid_predict(weights: np.ndarray,
                   x: np.ndarray,
                   centers: np.ndarray = None,
                   sigma: float = 1.0) -> float:
    """
    Predict using the hybrid model.  The same Clifford coupling used in
    training is applied to the raw dot‑product, ensuring consistency.
    """
    y = np.dot(weights, x)

    if centers is not None:
        act_mv = rbf_activation_mv(x, centers, sigma)
        w_mv = vector(weights)
        prod_mv = geometric_product(w_mv, act_mv)
        # Use the scalar part of the product as the final output; this
        # corresponds to the inner product of the transformed weight with
        # the input direction.
        scalar_part = sum(coeff for blade, coeff in prod_mv.items()
                          if len(blade) == 0)
        # If no scalar part exists we fall back to the plain dot product.
        y = scalar_part if scalar_part != 0.0 else y
    return y


def hybrid_train(initial_weights: np.ndarray,
                 X: np.ndarray,
                 y_target: np.ndarray,
                 epochs: int = 1,
                 mu: float = 0.5,
                 eps: float = 1e-9,
                 centers: np.ndarray = None,
                 sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train on a dataset (X, y_target) using the hybrid update.
    Returns the final weight vector and an array of epoch‑wise mean‑square errors.
    """
    weights = initial_weights.copy()
    errors = np.zeros(epochs, dtype=float)

    for epoch in range(epochs):
        epoch_err = 0.0
        for xi, ti in zip(X, y_target):
            weights, err = hybrid_update(weights, xi, ti,
                                         mu=mu, eps=eps,
                                         centers=centers, sigma=sigma)
            epoch_err += err ** 2
        errors[epoch] = epoch_err / X.shape[0]
    return weights, errors


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic problem: learn to map a 5‑dimensional constant input
    # to the scalar value 2.0.
    dim = 5
    w0 = np.ones(dim)
    x0 = np.ones(dim)
    target = 2.0

    # One centre per input dimension – this gives a full‑rank activation.
    centres = np.eye(dim) * 2.0  # spread the centres for illustration
    sigma_val = 0.8

    # Train for a few epochs
    final_w, epoch_err = hybrid_train(w0,
                                      X=np.tile(x0, (100, 1)),
                                      y_target=np.full(100, target),
                                      epochs=20,
                                      mu=0.3,
                                      centers=centres,
                                      sigma=sigma_val)

    print("Learned weights :", final_w)
    print("Final epoch MSE :", epoch_err[-1])
    # Verify prediction
    pred = hybrid_predict(final_w, x0, centers=centres, sigma=sigma_val)
    print("Prediction on training sample :", pred)