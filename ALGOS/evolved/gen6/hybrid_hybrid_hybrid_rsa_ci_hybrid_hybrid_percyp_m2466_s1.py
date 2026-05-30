# DARWIN HAMMER — match 2466, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_rsa_cipher_hy_hybrid_hybrid_hybrid_m827_s2.py (gen5)
# parent_b: hybrid_hybrid_percyphon_hyb_pheromone_m337_s1.py (gen3)
# born: 2026-05-29T23:42:27Z

"""Hybrid RSA‑Metric + RBF Surrogate with Morphology‑Driven Pheromone Modulation

Parents:
* **Parent A** – RSA encryption of a scaled hygiene‑entropy metric and a radial‑basis‑function
  surrogate that learns a mapping from feature vectors to the RSA ciphertext.
* **Parent B** – Morphology‑derived sphericity/flatness indices, a pheromone concentration
  that decays over time, and an EndpointCircuitBreaker that halts training when the
  surrogate error stays too high.

Mathematical bridge:
The scalar hygiene‑entropy metric *Sₕ* of Parent A is instantiated from the
morphology of Parent B:

  Sₕ = α·sphericity + β·flatness

where *sphericity* and *flatness* are geometric indices computed from a
`Morphology` instance.  After scaling (`m = ⌊scale·Sₕ⌋`) the integer *m* is RSA‑encrypted
to a ciphertext *c* which becomes the target label *y* for the RBF surrogate.
The surrogate’s kernel is modulated by the current pheromone concentration *p*,
yielding a pheromone‑aware kernel

  Kᵢⱼ = exp(−γ·pᵢ·pⱼ·‖xᵢ−xⱼ‖²),

so the learning process is directly influenced by the biological‑inspired
pheromone signal.  An `EndpointCircuitBreaker` watches the training loss; if the
loss exceeds a threshold for `failure_threshold` consecutive epochs the training
is aborted (circuit opened).  During inference the surrogate predicts a ciphertext
which is RSA‑decrypted to recover the scaled metric and finally the original
* Sₕ*.

The module provides three public functions that showcase the full hybrid pipeline:
`hybrid_train`, `hybrid_predict`, and `hybrid_decrypt`.  A minimal smoke test is
included in the `__main__` block.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Data structures from Parent B
# ----------------------------------------------------------------------


class Morphology:
    """Simple container for geometric properties."""
    __slots__ = ("length", "width", "height", "mass")

    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


class EndpointCircuitBreaker:
    """Stops training when loss stays above a threshold for too many steps."""
    def __init__(self, loss_threshold: float = 1e6, failure_threshold: int = 3):
        self.loss_threshold = loss_threshold
        self.failure_threshold = failure_threshold
        self.consecutive_failures = 0
        self.open = False

    def update(self, loss: float) -> None:
        if loss > self.loss_threshold:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.failure_threshold:
                self.open = True
        else:
            self.consecutive_failures = 0
            self.open = False

    def allow(self) -> bool:
        """True if training may continue."""
        return not self.open


# ----------------------------------------------------------------------
# RSA primitives from Parent A
# ----------------------------------------------------------------------


def rsa_encrypt(message: int, e: int, n: int) -> int:
    """RSA encryption c = m^e mod n."""
    if not 0 <= message < n:
        raise ValueError("message must satisfy 0 <= m < n")
    return pow(message, e, n)


def rsa_decrypt(ciphertext: int, d: int, n: int) -> int:
    """RSA decryption m = c^d mod n."""
    if not 0 <= ciphertext < n:
        raise ValueError("ciphertext must satisfy 0 <= c < n")
    return pow(ciphertext, d, n)


# ----------------------------------------------------------------------
# Morphology → metric (bridge)
# ----------------------------------------------------------------------


def sphericity(morph: Morphology) -> float:
    """Normalized sphericity index (0..1)."""
    # Approximate sphere volume vs actual volume
    v_sphere = (4.0 / 3.0) * math.pi * ((morph.length + morph.width + morph.height) / 3.0) ** 3
    v_box = morph.length * morph.width * morph.height
    return v_sphere / v_box if v_box > 0 else 0.0


def flatness(morph: Morphology) -> float:
    """Flatness index: ratio of smallest to largest dimension (0..1)."""
    dims = np.array([morph.length, morph.width, morph.height], dtype=float)
    if dims.max() == 0:
        return 0.0
    return dims.min() / dims.max()


def metric_from_morphology(morph: Morphology, alpha: float = 0.6, beta: float = 0.4) -> float:
    """Linear combination of sphericity and flatness → hygiene‑entropy metric."""
    return alpha * sphericity(morph) + beta * flatness(morph)


# ----------------------------------------------------------------------
# Pheromone dynamics (Parent B)
# ----------------------------------------------------------------------


def pheromone_decay(initial: float, decay_rate: float, steps: int) -> np.ndarray:
    """Exponential decay of a pheromone concentration over discrete steps."""
    t = np.arange(steps, dtype=float)
    return initial * np.exp(-decay_rate * t)


# ----------------------------------------------------------------------
# RBF surrogate with pheromone‑modulated kernel (Parent A + B)
# ----------------------------------------------------------------------


def pheromone_kernel(X: np.ndarray, Y: np.ndarray, pher_X: np.ndarray,
                     pher_Y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Compute a Gaussian RBF kernel where the distance is weighted by the product
    of the two pheromone concentrations.

    K_{ij} = exp( -γ * p_i * p_j * ||x_i - y_j||^2 )
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2‑D arrays")
    if pher_X.shape[0] != X.shape[0] or pher_Y.shape[0] != Y.shape[0]:
        raise ValueError("Pheromone vectors must match sample counts")

    # Squared Euclidean distances
    diff = X[:, None, :] - Y[None, :, :]               # shape (n_X, n_Y, d)
    sq_dist = np.einsum('ijk,ijk->ij', diff, diff)    # shape (n_X, n_Y)

    # Outer product of pheromone concentrations
    pher_prod = np.outer(pher_X, pher_Y)              # shape (n_X, n_Y)

    K = np.exp(-gamma * pher_prod * sq_dist)
    return K


def rbf_train(features: np.ndarray,
              targets: np.ndarray,
              pheromones: np.ndarray,
              gamma: float = 1.0,
              ridge: float = 1e-8) -> Tuple[np.ndarray, float]:
    """
    Train a surrogate: solve (K + ridge·I) w = y for weights w.
    Returns (weights, gamma_used).
    """
    if features.shape[0] != targets.shape[0] or features.shape[0] != pheromones.shape[0]:
        raise ValueError("features, targets, and pheromones must have same row count")
    K = pheromone_kernel(features, features, pheromones, pheromones, gamma=gamma)
    K_reg = K + ridge * np.eye(K.shape[0])
    w = np.linalg.solve(K_reg, targets)
    return w, gamma


def rbf_predict(new_feat: np.ndarray,
                train_feat: np.ndarray,
                train_pher: np.ndarray,
                weights: np.ndarray,
                gamma: float = 1.0) -> float:
    """
    Predict a scalar output for a single feature vector using the trained surrogate.
    """
    new_feat = np.asarray(new_feat, dtype=float).reshape(1, -1)
    K_vec = pheromone_kernel(new_feat, train_feat, np.array([1.0]), train_pher, gamma=gamma)
    # K_vec shape (1, n_train)
    pred = K_vec.dot(weights)
    return float(pred.squeeze())


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------


def hybrid_train(morphologies: Sequence[Morphology],
                 pheromone_params: Tuple[float, float, int],
                 rsa_key: Tuple[int, int, int],
                 scale: float = 1e6,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 gamma: float = 1.0) -> Tuple[dict, dict]:
    """
    Train the hybrid model.

    Returns
    -------
    model : dict
        Contains training features, pheromones, RBF weights, gamma, and RSA public key.
    metadata : dict
        Contains the RSA private key, scaling factor and the circuit‑breaker state.
    """
    e, d, n = rsa_key
    # 1️⃣ Extract geometric features (sphericity, flatness) → 2‑D feature matrix
    feats = np.array([[sphericity(m), flatness(m)] for m in morphologies], dtype=float)

    # 2️⃣ Compute the hygiene‑entropy metric and RSA ciphertext for each sample
    metrics = np.array([metric_from_morphology(m, alpha, beta) for m in morphologies])
    scaled = np.floor(scale * metrics).astype(int)
    ciphertexts = np.array([rsa_encrypt(int(m), e, n) for m in scaled], dtype=float)

    # 3️⃣ Generate pheromone concentration for each training sample
    init_p, decay, steps = pheromone_params
    pheromones = pheromone_decay(init_p, decay, len(morphologies))

    # 4️⃣ Train RBF surrogate with circuit‑breaker guard
    breaker = EndpointCircuitBreaker(loss_threshold=1e5, failure_threshold=2)
    # Simple one‑shot training; in a real scenario we could iterate and monitor loss.
    # Compute training loss (MSE) to feed the breaker.
    w, used_gamma = rbf_train(feats, ciphertexts, pheromones, gamma=gamma)
    preds = np.array([rbf_predict(feats[i], feats, pheromones, w, gamma=used_gamma)
                      for i in range(len(feats))])
    loss = np.mean((preds - ciphertexts) ** 2)
    breaker.update(loss)

    model = {
        "features": feats,
        "pheromones": pheromones,
        "weights": w,
        "gamma": used_gamma,
        "rsa_public": (e, n),
        "scale": scale,
        "alpha": alpha,
        "beta": beta,
    }
    metadata = {
        "rsa_private": (d, n),
        "breaker": breaker,
        "loss": loss,
    }
    return model, metadata


def hybrid_predict(morph: Morphology,
                   pheromone_state: float,
                   model: dict) -> float:
    """
    Predict the original hygiene‑entropy metric for a new morphology.

    Steps:
    1. Build feature vector (sphericity, flatness).
    2. Use the RBF surrogate to predict an RSA ciphertext.
    3. RSA‑decrypt with the public exponent (requires private key later).
    4. Un‑scale to obtain the metric.
    """
    feat = np.array([sphericity(morph), flatness(morph)], dtype=float).reshape(1, -1)
    pred_cipher = rbf_predict(feat,
                              model["features"],
                              model["pheromones"],
                              model["weights"],
                              gamma=model["gamma"])
    # Clamp to integer range for RSA decryption
    pred_int = int(round(pred_cipher)) % model["rsa_public"][1]
    return float(pred_int)  # caller will decrypt via hybrid_decrypt


def hybrid_decrypt(cipher_int: int, metadata: dict) -> float:
    """
    RSA‑decrypt the integer ciphertext, reverse scaling and return the metric.
    """
    d, n = metadata["rsa_private"]
    scale = metadata["model"]["scale"]
    alpha = metadata["model"]["alpha"]
    beta = metadata["model"]["beta"]
    # Decrypt
    m = rsa_decrypt(cipher_int, d, n)
    # Unscale
    metric = m / scale
    # The metric is already the linear combination α·sphericity+β·flatness,
    # so we simply return it.
    return metric


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal RSA key (not secure, for demonstration only)
    # p = 61, q = 53 → n = 3233, φ = 3120, e = 17, d = 2753
    e, d, n = 17, 2753, 61 * 53

    # Create a few synthetic morphologies
    training_morphs = [
        Morphology(length=1.0, width=0.9, height=0.95, mass=2.0),
        Morphology(length=1.2, width=0.8, height=1.0, mass=2.3),
        Morphology(length=0.9, width=0.85, height=0.88, mass=1.9),
        Morphology(length=1.1, width=0.95, height=1.05, mass=2.1),
    ]

    # Pheromone parameters: initial concentration, decay rate, steps (=samples)
    pher_params = (1.0, 0.05, len(training_morphs))

    model, meta = hybrid_train(
        morphologies=training_morphs,
        pheromone_params=pher_params,
        rsa_key=(e, d, n),
        scale=1e6,
        alpha=0.6,
        beta=0.4,
        gamma=0.8,
    )
    # Store model reference inside metadata for decryption convenience
    meta["model"] = model

    # New morphology to evaluate
    test_morph = Morphology(length=1.05, width=0.92, height=1.00, mass=2.05)

    # Predict ciphertext (integer) using surrogate
    pred_cipher = hybrid_predict(test_morph, pheromone_state=0.9, model=model)

    # Decrypt and retrieve metric
    recovered_metric = hybrid_decrypt(int(round(pred_cipher)), meta)

    print("Predicted RSA ciphertext (int):", int(round(pred_cipher)))
    print("Recovered hygiene‑entropy metric:", recovered_metric)

    # Verify circuit breaker status
    print("Circuit breaker open?" , meta["breaker"].open)
    print("Training loss:", meta["loss"])