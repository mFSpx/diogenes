# DARWIN HAMMER — match 4014, survivor 3
# gen: 5
# parent_a: diffusion_forcing.py (gen0)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m900_s2.py (gen4)
# born: 2026-05-29T23:53:10Z

import numpy as np
import math
from typing import List, Tuple

_EPS = 1e-12  # numerical guard for log/zero divisions


def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Vectorised Shannon entropy for a batch of probability vectors.

    Parameters
    ----------
    probs : np.ndarray
        Shape (..., K) where the last dimension sums to 1.

    Returns
    -------
    np.ndarray
        Entropy values with shape probs.shape[:-1].
    """
    safe_probs = np.clip(probs, _EPS, 1.0)
    return -np.sum(safe_probs * np.log2(safe_probs), axis=-1)


def diffusion_forcing_noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """
    Produce the cumulative diffusion schedule ᾱₜ for t = 0 … T.

    Parameters
    ----------
    T : int
        Number of diffusion steps.
    schedule : str, optional
        Currently only ``"cosine"`` is supported.

    Returns
    -------
    np.ndarray
        Array of shape (T + 1,) with values in (0, 1].
    """
    if schedule != "cosine":
        raise ValueError(f"Unsupported schedule: {schedule}")

    s = 0.008
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
    alpha_bars = f / f[0]
    return np.clip(alpha_bars, 1e-8, 1.0)


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    x = x / max(temperature, _EPS)
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x)


def hybrid_noise_schedule(
    noise_levels: np.ndarray,
    schedule: str = "cosine",
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Fuse diffusion forcing with a per‑step Shannon‑entropy weighting derived
    from the supplied ``noise_levels``.

    The entropy acts as a confidence weight: high entropy (uncertain) steps are
    down‑scaled, low entropy (certain) steps retain more of the original diffusion
    magnitude.

    Parameters
    ----------
    noise_levels : np.ndarray
        1‑D array of raw noise magnitudes for each diffusion step.
    schedule : str, optional
        Diffusion schedule to use (currently only ``"cosine"``).
    temperature : float, optional
        Temperature for the softmax that turns ``noise_levels`` into a
        probability distribution per step.

    Returns
    -------
    np.ndarray
        Hybrid schedule of shape (len(noise_levels),) ready for downstream use.
    """
    if noise_levels.ndim != 1:
        raise ValueError("noise_levels must be a 1‑D array")
    T = len(noise_levels)
    if T == 0:
        return np.array([])

    # Base diffusion schedule (exclude the terminal ᾱ_T which is not needed for per‑step weighting)
    alpha_bars = diffusion_forcing_noise_schedule(T, schedule)[:-1]

    # Convert raw noise levels into a probability distribution via softmax.
    probs = _softmax(noise_levels, temperature)

    # Compute per‑step entropy (scalar because we have a single distribution over steps).
    # To obtain a per‑step weight we broadcast the scalar entropy across all steps.
    entropy = shannon_entropy(probs[np.newaxis, :])[0]  # scalar

    # Transform entropy into a multiplicative weight in (0, 1].
    # Using exp(-entropy) ensures that higher entropy → smaller weight.
    weight = math.exp(-entropy)

    # Apply the weight element‑wise to the diffusion schedule.
    hybrid = weight * alpha_bars
    return hybrid


def compute_phash(values: np.ndarray) -> int:
    """
    Compute a 64‑bit perceptual hash from a sequence of float values.

    The median of the sequence is used as the threshold; each of the first
    64 values contributes a bit (1 if ≥ median, 0 otherwise). If fewer than 64
    values are provided the sequence is padded with zeros.

    Parameters
    ----------
    values : np.ndarray
        1‑D array of signal values.

    Returns
    -------
    int
        64‑bit integer hash.
    """
    if values.size == 0:
        return 0

    median = np.median(values)
    padded = np.pad(values, (0, max(0, 64 - values.size)), constant_values=0.0)[:64]

    bits = 0
    for v in padded:
        bits = (bits << 1) | int(v >= median)
    return bits


def hybrid_signal_recording(noise_levels: np.ndarray) -> Tuple[int, float]:
    """
    Produce a perceptual hash and the mean of the hybrid schedule derived from
    ``noise_levels``.

    Parameters
    ----------
    noise_levels : np.ndarray
        1‑D array of raw noise magnitudes.

    Returns
    -------
    Tuple[int, float]
        (perceptual hash, mean of hybrid schedule)
    """
    hybrid_schedule = hybrid_noise_schedule(noise_levels)
    phash = compute_phash(hybrid_schedule)
    mean_val = float(np.mean(hybrid_schedule)) if hybrid_schedule.size else 0.0
    return phash, mean_val


if __name__ == "__main__":
    T = 100
    # Simulate a realistic noise profile (e.g., Gaussian‑like magnitudes)
    rng = np.random.default_rng(seed=42)
    noise_levels = rng.normal(loc=0.5, scale=0.15, size=T)
    noise_levels = np.clip(noise_levels, 0.0, 1.0)

    phash, mean_signal = hybrid_signal_recording(noise_levels)
    print(f"Perceptual hash: {phash:#018x}, Mean signal value: {mean_signal:.6f}")