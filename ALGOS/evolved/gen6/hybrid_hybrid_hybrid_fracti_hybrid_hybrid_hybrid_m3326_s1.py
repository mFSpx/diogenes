# DARWIN HAMMER — match 3326, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_fractional_hd_hybrid_hybrid_nlms_o_m1127_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2315_s2.py (gen5)
# born: 2026-05-29T23:49:16Z

"""
Hybrid Adaptive Filter & Geometric Allocation

This module fuses the two parent algorithms:

* **Parent A** – provides high‑dimensional random vectors (hv), FFT‑based binding/unbinding,
  and an NLMS adaptive filter (nlms_predict / nlms_update).

* **Parent B** – defines a multivector together with LTC parameters and a day‑of‑week
  allocation scheme that originally relied on a geometric product.

**Mathematical bridge**

The geometric product in Parent B is replaced by the *binding* operation from Parent A,
which is an element‑wise multiplication in the Fourier domain (convolution).  Binding
produces a high‑dimensional representation of the interaction between a day‑of‑week
one‑hot vector and the random hyper‑vector `hv`.  This bound vector is then fed to the
NLMS filter for prediction and simultaneously used to update the multivector/LTC
parameters, thereby unifying the adaptive‑filter dynamics with the geometric‑allocation
logic.
"""

import math
import random
import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def random_hv(d: int = 10000, kind: str = "real", seed=None):
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"kind must be 'complex', 'bipolar', or 'real'; got {kind!r}")


def bind(X, Y):
    """FFT‑based binding (convolution) of two vectors."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))


def unbind(Z, Y):
    """Unbinding using the conjugate of Y in the frequency domain."""
    Z = np.asarray(Z)
    Y = np.asarray(Y)
    FY = np.fft.fft(Y)
    mag = np.abs(FY)
    inv_FY = np.conj(FY) / (mag ** 2 + 1e-30)
    return np.fft.ifft(np.fft.fft(Z) * inv_FY)


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, float]:
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    y = nlms_predict(weights, x)
    error = target - y
    power = float(x @ x) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error


# ----------------------------------------------------------------------
# Parent‑B building blocks (trimmed / re‑implemented)
# ----------------------------------------------------------------------
GROU = 5   # number of groups
DIM = 7    # dimensionality of day‑of‑week one‑hot vector
MAX_INDEX = 10.0  # unused placeholder, kept for compatibility


def init_hybrid_ltc_gp(dim: int, num_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Initialize multivector and LTC parameters with uniform random values."""
    multivector = np.random.rand(dim, num_groups)
    ltc_params = np.random.rand(dim, num_groups)
    return multivector, ltc_params


# ----------------------------------------------------------------------
# Fusion core
# ----------------------------------------------------------------------
def init_hybrid_system(
    dim: int = DIM,
    num_groups: int = GROU,
    hv_kind: str = "real",
    seed: int | None = None,
) -> dict:
    """
    Initialise the whole hybrid system.

    Returns a dictionary containing:
        - weights      : NLMS filter weights (size = dim)
        - hv          : high‑dimensional random vector (size = dim)
        - multivector : geometric‑product matrix (dim × num_groups)
        - ltc_params  : LTC modulation matrix (dim × num_groups)
        - mu          : learning rate used for all updates
    """
    rng = np.random.default_rng(seed)
    hv = random_hv(d=dim, kind=hv_kind, seed=seed)
    multivector, ltc_params = init_hybrid_ltc_gp(dim, num_groups)
    weights = np.zeros(dim, dtype=np.complex128)  # allow complex after binding
    mu = 0.5
    return {
        "weights": weights,
        "hv": hv,
        "multivector": multivector,
        "ltc_params": ltc_params,
        "mu": mu,
        "rng": rng,
    }


def hybrid_predict_step(state: dict, day_one_hot: np.ndarray, target: float) -> float:
    """
    Perform a single prediction‑and‑update step.

    * Binding `hv` with the day‑of‑week one‑hot vector yields a high‑dimensional
      feature representation (the surrogate geometric product).
    * The NLMS filter predicts the target from this feature.
    * The same error drives updates of the NLMS weights **and** the multivector/LTC
      matrices, linking the two parent topologies.
    Returns the prediction error.
    """
    hv = state["hv"]
    mu = state["mu"]
    # 1) Encode the day vector via binding (FFT‑based geometric product)
    phi = bind(hv, day_one_hot)  # complex vector of length dim
    # 2) Predict using NLMS
    pred = nlms_predict(state["weights"], phi)
    # 3) NLMS weight update
    new_weights, error = nlms_update(state["weights"], phi, target, mu=mu)
    state["weights"] = new_weights
    # 4) Update multivector & LTC parameters using the same error signal
    #    Simple Hebbian‑like rule: ΔM = mu * error * outer(phi, 1)
    delta_mv = mu * error * np.real(phi)[:, np.newaxis]  # shape (dim,1)
    state["multivector"] += delta_mv  # broadcast over groups
    #    LTC parameters are scaled proportionally to the magnitude of the error
    state["ltc_params"] *= (1.0 + mu * np.tanh(error))
    return error


def compute_entropy(errors: np.ndarray) -> float:
    """
    Shannon entropy of the normalized absolute errors.
    """
    abs_err = np.abs(errors) + 1e-12
    prob = abs_err / abs_err.sum()
    return -float(np.sum(prob * np.log(prob + 1e-12)))


def adjust_ltc_by_entropy(state: dict, errors: np.ndarray, target_entropy: float = 1.0):
    """
    Modulate LTC parameters based on the entropy of recent errors.
    If entropy is below the target, increase diversity by enlarging LTC values,
    otherwise shrink them.
    """
    entropy = compute_entropy(errors)
    mu = state["mu"]
    factor = 1.0 + mu * (entropy - target_entropy)
    state["ltc_params"] *= factor


def allocate_resources(
    dates: list[date],
    state: dict,
) -> np.ndarray:
    """
    For each date, compute a per‑group allocation vector.

    Allocation = (multivector_row ⊙ ltc_params_row) ⊙ bind(hv, day_one_hot)

    where ⊙ denotes element‑wise multiplication.
    The result is a (len(dates), num_groups) array.
    """
    dim = state["multivector"].shape[0]
    num_groups = state["multivector"].shape[1]
    allocations = np.zeros((len(dates), num_groups), dtype=np.complex128)

    for idx, cur_date in enumerate(dates):
        dow = cur_date.weekday()  # 0 = Monday … 6 = Sunday
        # one‑hot vector for the day of week (length = dim)
        day_vec = np.zeros(dim)
        day_vec[dow % dim] = 1.0
        # bound feature
        phi = bind(state["hv"], day_vec)  # shape (dim,)
        # geometric‑product‑like combination
        base = state["multivector"][dow % dim, :] * state["ltc_params"][dow % dim, :]
        # final allocation (broadcast phi over groups)
        allocations[idx] = base * phi.real  # keep real part for readability
    return allocations.real  # return real‑valued allocations


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise system
    system = init_hybrid_system(dim=DIM, num_groups=GROU, seed=42)

    # Generate a week of dates
    start = date.today()
    dates = [start + timedelta(days=i) for i in range(7)]

    # Simulated targets (e.g., some daily metric)
    rng = np.random.default_rng(123)
    targets = rng.normal(loc=0.0, scale=1.0, size=7)

    # Run hybrid steps and collect errors
    errors = []
    for cur_date, tgt in zip(dates, targets):
        # day one‑hot vector
        dow = cur_date.weekday()
        day_one_hot = np.zeros(DIM)
        day_one_hot[dow % DIM] = 1.0
        err = hybrid_predict_step(system, day_one_hot, tgt)
        errors.append(err)

    errors = np.array(errors)
    # Adjust LTC parameters based on error entropy
    adjust_ltc_by_entropy(system, errors, target_entropy=1.0)

    # Compute allocations for the same dates
    allocs = allocate_resources(dates, system)

    # Simple sanity prints
    print("Prediction errors:", errors)
    print("Entropy‑adjusted LTC params (sample):", system["ltc_params"][:2, :2])
    print("Allocations (first three dates):\n", allocs[:3])
    sys.exit(0)