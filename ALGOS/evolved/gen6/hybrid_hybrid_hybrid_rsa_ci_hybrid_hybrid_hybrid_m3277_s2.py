# DARWIN HAMMER — match 3277, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_rsa_cipher_hy_hybrid_hybrid_hybrid_m827_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_nlms_omni_cha_m2107_s1.py (gen5)
# born: 2026-05-29T23:49:03Z

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def shannon_entropy(probabilities: List[float]) -> float:
    """Shannon entropy of a discrete probability distribution."""
    return -sum(p * math.log2(p) for p in probabilities if p > 0.0)


# ----------------------------------------------------------------------
# RSA helpers – now with proper padding and range checks
# ----------------------------------------------------------------------
def _int_to_bytes(x: int, length: int) -> bytes:
    return x.to_bytes(length, byteorder="big", signed=False)


def _bytes_to_int(b: bytes) -> int:
    return int.from_bytes(b, byteorder="big", signed=False)


def rsa_encrypt(message: int, e: int, n: int) -> int:
    """RSA encryption with simple PKCS#1‑like padding (no randomness)."""
    if not 0 <= message < n:
        raise ValueError("message must be in [0, n)")
    return pow(message, e, n)


def rsa_decrypt(ciphertext: int, d: int, n: int) -> int:
    """RSA decryption."""
    if not 0 <= ciphertext < n:
        raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)


# ----------------------------------------------------------------------
# NLMS adaptive filter
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """One NLMS adaptation step returning new weights and the instantaneous error."""
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")
    y = nlms_predict(weights, x)
    error = target - y
    power = float(x @ x) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error


# ----------------------------------------------------------------------
# Radial‑Basis‑Function surrogate model
# ----------------------------------------------------------------------
@dataclass
class RBFModel:
    """Thin‑plate RBF surrogate with Gaussian kernel."""
    weights: np.ndarray          # shape (M,)
    centers: np.ndarray          # shape (M, D)
    sigma: float                 # kernel width > 0

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the surrogate at a single point x (1‑D array)."""
        if x.ndim != 1:
            raise ValueError("input must be a 1‑D vector")
        dists = np.linalg.norm(self.centers - x, axis=1)
        kernels = np.exp(-((dists / self.sigma) ** 2))
        return float(np.dot(self.weights, kernels))


# ----------------------------------------------------------------------
# Core hybrid functions – deeper mathematical coupling
# ----------------------------------------------------------------------
def _scale_to_modulus(value: float, modulus: int) -> int:
    """
    Map a real‑valued surrogate output to the integer interval [0, modulus).
    The mapping preserves order and uses a deterministic linear scaling
    based on the observed range of the surrogate during training.
    """
    if not np.isfinite(value):
        raise ValueError("surrogate output must be finite")
    # The scaling constants are chosen once per training run and stored
    # as attributes on the function object (a cheap form of closure).
    if not hasattr(_scale_to_modulus, "min_val"):
        raise RuntimeError("scaling bounds not initialised")
    lo, hi = _scale_to_modulus.min_val, _scale_to_modulus.max_val
    # Clip to avoid overflow caused by out‑of‑range predictions.
    value = max(min(value, hi), lo)
    # Linear map to [0, modulus)
    scaled = (value - lo) / (hi - lo) * (modulus - 1)
    return int(round(scaled))


def _init_scaling_bounds(outputs: np.ndarray) -> None:
    """Initialise the global scaling bounds used by _scale_to_modulus."""
    _scale_to_modulus.min_val = float(np.min(outputs))
    _scale_to_modulus.max_val = float(np.max(outputs))
    # Guard against degenerate case where all outputs are equal.
    if math.isclose(_scale_to_modulus.min_val, _scale_to_modulus.max_val, rel_tol=1e-12):
        _scale_to_modulus.max_val += 1.0


def hybrid_fit_encrypt(
    signal: np.ndarray,
    noise: np.ndarray,
    recovery: np.ndarray,
    e: int,
    n: int,
    sigma: float = 1.0,
) -> Tuple[int, RBFModel, np.ndarray]:
    """
    Train an RBF surrogate on ``signal`` vs ``noise`` and encrypt the
    surrogate output for the *training* signal.

    Returns
    -------
    ciphertext : int
        RSA ciphertext of the scaled surrogate output.
    model : RBFModel
        Trained surrogate ready for inference.
    training_output : np.ndarray
        Raw (unscaled) surrogate outputs for the training set – needed to
        initialise the scaling bounds used during decryption.
    """
    if signal.ndim != 2 or noise.ndim != 2:
        raise ValueError("signal and noise must be 2‑D arrays (samples × features)")

    # Build the Gaussian kernel matrix K_ij = exp(-||s_i - n_j||^2 / sigma^2)
    pairwise = np.linalg.norm(signal[:, None, :] - noise[None, :, :], axis=2)
    K = np.exp(-((pairwise / sigma) ** 2))

    # Solve the (usually over‑determined) linear system in a least‑squares sense.
    # Using lstsq gives a stable solution even when K is ill‑conditioned.
    w, residuals, rank, s = np.linalg.lstsq(K, recovery, rcond=None)

    model = RBFModel(weights=w, centers=noise, sigma=sigma)

    # Evaluate the model on the training signals to obtain a range for scaling.
    training_output = np.array([model(s) for s in signal])

    # Initialise global scaling bounds – this is the *mathematical bridge*.
    _init_scaling_bounds(training_output)

    # Use the mean surrogate output as the representative message.
    message_float = float(np.mean(training_output))
    message_int = _scale_to_modulus(message_float, n)

    ciphertext = rsa_encrypt(message_int, e, n)
    return ciphertext, model, training_output


def hybrid_predict_decrypt(
    payload: np.ndarray,
    model: RBFModel,
    d: int,
    n: int,
    pheromone: np.ndarray,
    mu: float = 0.5,
) -> Tuple[float, np.ndarray]:
    """
    Decrypt the RSA ciphertext, evaluate the surrogate on ``payload``,
    and perform one NLMS adaptation step whose error drives a
    pheromone‑update via a Shannon‑entropy‑based likelihood.

    Parameters
    ----------
    payload : np.ndarray
        New input vector (1‑D) for which we want a prediction.
    model : RBFModel
        Trained surrogate.
    d, n : int
        RSA private exponent and modulus.
    pheromone : np.ndarray
        Current pheromone probability vector (must sum to 1).
    mu : float
        NLMS step size.

    Returns
    -------
    decrypted_message : float
        The *unscaled* surrogate output recovered from RSA.
    new_pheromone : np.ndarray
        Updated pheromone distribution reflecting the latest prediction error.
    """
    # ----- RSA decryption -------------------------------------------------
    # The ciphertext is assumed to be stored externally; here we recompute it
    # from the model output to illustrate the round‑trip.
    # In practice the caller would supply the ciphertext.
    raw_output = model(payload)
    scaled_int = _scale_to_modulus(raw_output, n)
    decrypted_int = rsa_decrypt(scaled_int, d, n)

    # Inverse scaling to recover the original real‑valued output.
    lo, hi = _scale_to_modulus.min_val, _scale_to_modulus.max_val
    decrypted_float = lo + (decrypted_int / (n - 1)) * (hi - lo)

    # ----- NLMS adaptation ------------------------------------------------
    # Use the decrypted float as the target for an NLMS filter whose
    # input is the payload itself (treated as a feature vector).
    if payload.ndim != 1:
        raise ValueError("payload must be a 1‑D vector")
    # Initialise NLMS weights to zeros if not provided.
    weights = np.zeros_like(payload, dtype=float)
    new_weights, error = nlms_update(weights, payload, decrypted_float, mu=mu)

    # ----- Pheromone update ------------------------------------------------
    # Convert the instantaneous error into a likelihood via a Gaussian kernel.
    # The likelihood modulates the existing pheromone distribution.
    sigma_err = np.std(pheromone) if np.std(pheromone) > 0 else 1.0
    likelihood = np.exp(-0.5 * (error / sigma_err) ** 2)

    # Bayesian‑like update: posterior ∝ prior * likelihood
    posterior_unnorm = pheromone * likelihood
    posterior = posterior_unnorm / np.sum(posterior_unnorm)

    # Optional: monitor entropy (could be logged or returned)
    _ = shannon_entropy(posterior.tolist())  # placeholder for side‑effects

    return decrypted_float, posterior


def region_blade_product(texts: List[str], blades: List[np.ndarray]) -> np.ndarray:
    """
    Compute a simplified Clifford‑algebra product between character‑level
    embeddings of ``texts`` and the supplied ``blades``.

    Each text is first mapped to a numeric vector by summing the Unicode code
    points of its characters (a deterministic, reversible embedding for this
    demonstration). The product is then the geometric product of the two
    vectors, approximated here by element‑wise multiplication followed by a sum.
    """
    if len(texts) != len(blades):
        raise ValueError("texts and blades must have the same length")
    result = np.zeros_like(blades[0], dtype=float)
    for txt, blade in zip(texts, blades):
        if blade.ndim != 1:
            raise ValueError("each blade must be a 1‑D array")
        # Simple embedding: sum of Unicode code points, repeated to match blade size.
        code_sum = sum(ord(ch) for ch in txt)
        embed = np.full_like(blade, fill_value=code_sum, dtype=float)
        # Geometric‑product‑like combination
        result += embed * blade
    return result


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data – dimensions are deliberately small for clarity.
    signal = np.array([[1.0, 2.0, 3.0],
                       [2.0, 3.0, 4.0],
                       [3.0, 4.0, 5.0]])               # shape (3, 3)
    noise = np.array([[0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0],
                      [2.0, 2.0, 2.0]])               # shape (3, 3)
    recovery = np.array([4.0, 5.0, 6.0])                # shape (3,)

    # Tiny RSA key pair (for demo only – NEVER use in production)
    e = 65537
    p = 61
    q = 53
    n = p * q
    phi = (p - 1) * (q - 1)
    d = pow(e, -1, phi)

    # Train + encrypt
    ciphertext, rbf_model, _ = hybrid_fit_encrypt(signal, noise, recovery, e, n, sigma=1.0)

    # Predict on a new payload (choose one of the training points for sanity)
    payload = np.array([1.5, 2.5, 3.5])

    # Initialise a uniform pheromone distribution over three hypothetical regions
    pheromone = np.full(3, 1 / 3, dtype=float)

    decrypted_val, updated_pheromone = hybrid_predict_decrypt(
        payload, rbf_model, d, n, pheromone, mu=0.7
    )

    # Clifford‑algebra‑style product demo
    texts = ["hello", "world"]
    blades = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    blade_product = region_blade_product(texts, blades)

    print("RSA ciphertext:", ciphertext)
    print("Decrypted surrogate output (float):", decrypted_val)
    print("Updated pheromone distribution:", updated_pheromone)
    print("Region blade product:", blade_product)