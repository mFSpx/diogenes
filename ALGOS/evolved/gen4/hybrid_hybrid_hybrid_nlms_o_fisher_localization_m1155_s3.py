# DARWIN HAMMER — match 1155, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_liquid_m141_s1.py (gen3)
# parent_b: fisher_localization.py (gen0)
# born: 2026-05-29T23:33:06Z

"""Hybrid NLMS‑LTC‑Fisher Fusion

This module combines the two parent algorithms:

* **Parent A** – a Normalized Least‑Mean‑Squares (NLMS) adaptive filter whose
  weights **w** scale a deterministic diffusion schedule **α̅**.  The filter
  receives a feature vector derived from a MinHash signature of the current
  token set.

* **Parent B** – a Fisher‑information based scoring of a Gaussian beam,
  providing a data‑dependent confidence measure **F(θ)** for an angular
  parameter **θ**.

**Mathematical bridge** – the angle **θ** is obtained from the first component
of the MinHash signature (mapped to the interval \([0,2π]\)).  The Fisher
score **F(θ)** is then used as a multiplicative factor for the NLMS error
signal, effectively weighting the weight update by the information content of
the current token distribution.  Consequently the diffusion schedule becomes
both learnable (via NLMS) and information‑aware (via Fisher weighting).

The core hybrid operations are:

1. `nlms_update` – classic NLMS weight adaptation, now scaled by the Fisher
   score.
2. `noise_schedule` – deterministic diffusion schedule (cosine).
3. `hybrid_predict` – prediction using the scaled schedule and Fisher‑weighted
   NLMS output.
4. `hybrid_train` – one‑pass training loop that ties the components together.
"""

import sys
import math
import random
import hashlib
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Parent B – Fisher‑information utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam at angle `theta`."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Parent A – MinHash signature utilities
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on a seed and a token."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: list[str], k: int = 128) -> list[int]:
    """MinHash signature of a token set with `k` hash functions."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: list[int], sig_b: list[int]) -> float:
    """Jaccard‑like similarity based on exact MinHash collisions."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ----------------------------------------------------------------------
# Core NLMS utilities (Parent A)
# ----------------------------------------------------------------------
def nlms_update(w: np.ndarray,
                x: np.ndarray,
                error: float,
                mu: float = 0.01,
                eps: float = 1e-8) -> np.ndarray:
    """
    Normalized Least‑Mean‑Squares weight update.

    The classic NLMS rule is
        w ← w + μ * e * x / (‖x‖² + ε)

    In the hybrid we multiply the error `e` by a Fisher‑information weight
    `F(θ)` before applying the update (the caller handles that scaling).
    """
    norm_sq = np.dot(x, x) + eps
    return w + (mu * error / norm_sq) * x


def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """
    Deterministic diffusion schedule α̅ₜ for t = 0 … T‑1.

    * cosine – the schedule used in many diffusion models.
    * linear  – simple linear decay.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    t = np.arange(T, dtype=np.float64)
    if schedule == "cosine":
        return np.cos(t / T * math.pi / 2) ** 2
    elif schedule == "linear":
        return 1.0 - t / (T - 1)
    else:
        raise ValueError(f"unknown schedule '{schedule}'")


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def token_angle(sig: list[int]) -> float:
    """
    Map the first component of a MinHash signature to an angle θ ∈ [0, 2π].

    The raw 64‑bit integer is normalised by MAX64.
    """
    if not sig:
        raise ValueError("signature must not be empty")
    normalized = sig[0] / MAX64  # ∈ [0,1]
    return normalized * 2.0 * math.pi


def hybrid_predict(w: np.ndarray,
                   x: np.ndarray,
                   schedule_vec: np.ndarray,
                   theta: float,
                   center: float = 0.0,
                   width: float = 1.0) -> float:
    """
    Hybrid prediction.

    1. Compute the NLMS linear model output y = w·x.
    2. Scale the diffusion schedule by w element‑wise → s = α̅ ⊙ w.
    3. Take the mean of the scaled schedule as a scalar schedule factor σ.
    4. Combine the model output with the schedule factor and weight by the
       Fisher information F(θ) to obtain the final prediction.
    """
    if w.shape != x.shape:
        raise ValueError("w and x must have the same shape")
    if schedule_vec.shape != w.shape:
        raise ValueError("schedule_vec must match weight dimension")

    model_out = float(np.dot(w, x))
    schedule_scaled = schedule_vec * w
    sigma = float(np.mean(schedule_scaled))

    fisher = fisher_score(theta, center, width)
    return model_out * sigma * fisher


def hybrid_train(token_stream: list[list[str]],
                 T: int,
                 mu: float = 0.01,
                 schedule_type: str = "cosine",
                 center: float = 0.0,
                 width: float = 1.0) -> dict:
    """
    One‑pass training over a sequence of token sets.

    Returns a dictionary with the final weight vector, the diffusion schedule,
    and the list of predictions made during training.
    """
    # Initialise diffusion schedule (shared across all steps)
    schedule_vec = noise_schedule(T, schedule_type)

    # Weight dimension equals schedule length
    w = np.zeros(T, dtype=np.float64)

    predictions = []
    prev_sig = None

    for tokens in token_stream:
        # 1️⃣ MinHash signature → feature vector
        sig = signature(tokens, k=T)
        x = np.array(sig, dtype=np.float64) / MAX64  # normalised to [0,1]

        # 2️⃣ Angle from signature (first component)
        theta = token_angle(sig)

        # 3️⃣ Fisher information weight
        fisher = fisher_score(theta, center, width)

        # 4️⃣ Hybrid prediction
        pred = hybrid_predict(w, x, schedule_vec, theta, center, width)
        predictions.append(pred)

        # 5️⃣ Desired target – for demonstration we use the similarity to the
        #    previous signature (or 0 for the first step)
        target = 0.0 if prev_sig is None else similarity(sig, prev_sig)

        # 6️⃣ Error signal, weighted by Fisher information
        error = fisher * (target - pred)

        # 7️⃣ NLMS weight update (the error already includes the Fisher factor)
        w = nlms_update(w, x, error, mu)

        prev_sig = sig

    return {
        "weights": w,
        "schedule": schedule_vec,
        "predictions": predictions,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a synthetic token stream: each step contains a few random words.
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    random.seed(42)
    token_stream = []
    for _ in range(20):
        step_tokens = random.sample(vocab, k=random.randint(2, 5))
        token_stream.append(step_tokens)

    result = hybrid_train(
        token_stream=token_stream,
        T=64,
        mu=0.05,
        schedule_type="cosine",
        center=math.pi,   # centre of the Gaussian beam
        width=0.8,
    )

    # Simple sanity checks – they should not raise.
    assert isinstance(result, dict)
    assert result["weights"].shape == (64,)
    assert len(result["predictions"]) == len(token_stream)

    # Print a brief summary
    print("Final weight norm :", np.linalg.norm(result["weights"]))
    print("First 5 predictions:", result["predictions"][:5])
    print("Training completed without errors.")