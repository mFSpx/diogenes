# DARWIN HAMMER — match 4014, survivor 2
# gen: 5
# parent_a: diffusion_forcing.py (gen0)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m900_s2.py (gen4)
# born: 2026-05-29T23:53:10Z

import numpy as np
from typing import List, Tuple
import math

def shannon_entropy(values: List[float]) -> float:
    """Compute Shannon entropy from a list of probabilities."""
    entropy = 0.0
    for p in values:
        if p > 0:
            entropy -= p * math.log(p, 2)
    return entropy

def diffusion_forcing_noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """Return the cumulative noise schedule alpha_bar, shape (T+1,)."""
    if schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        # Clip to ensure numerical stability — never exactly zero.
        alpha_bars = np.clip(alpha_bars, 1e-8, 1.0)
        return alpha_bars
    else:
        raise ValueError("Unsupported schedule")

def hybrid_noise_schedule(T: int) -> np.ndarray:
    """Generate a hybrid noise schedule by applying Shannon entropy to the diffusion forcing noise schedule."""
    alpha_bars = diffusion_forcing_noise_schedule(T)
    probabilities = alpha_bars / np.sum(alpha_bars)
    entropy = shannon_entropy(probabilities.tolist())
    signal_values = np.exp(-entropy) * alpha_bars
    return signal_values

def compute_phash(values: np.ndarray) -> int:
    """Compute a perceptual hash from an array of signal values."""
    if len(values) == 0:
        return 0
    avg = np.mean(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hybrid_signal_recording(noise_levels: np.ndarray) -> Tuple[int, float]:
    """Record signals and compute a perceptual hash using the hybrid noise schedule."""
    signal_values = hybrid_noise_schedule(len(noise_levels))
    phash = compute_phash(signal_values)
    return phash, np.mean(signal_values)

def joint_entropy(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute joint entropy between two probability distributions."""
    joint_probabilities = np.outer(p1, p2)
    joint_probabilities = joint_probabilities.flatten()
    entropy = 0.0
    for p in joint_probabilities:
        if p > 0:
            entropy -= p * math.log(p, 2)
    return entropy

def conditional_entropy(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute conditional entropy between two probability distributions."""
    joint_entropy_value = joint_entropy(p1, p2)
    entropy_p2 = shannon_entropy(p2.tolist())
    return joint_entropy_value - entropy_p2

def improved_hybrid_noise_schedule(T: int) -> np.ndarray:
    """Generate an improved hybrid noise schedule by incorporating conditional entropy."""
    alpha_bars = diffusion_forcing_noise_schedule(T)
    probabilities = alpha_bars / np.sum(alpha_bars)
    entropy = shannon_entropy(probabilities.tolist())
    conditional_entropies = []
    for i in range(len(probabilities)):
        p1 = np.array([probabilities[i]])
        p2 = np.array([probabilities[j] for j in range(len(probabilities)) if j != i])
        conditional_entropies.append(conditional_entropy(p1, p2))
    signal_values = np.exp(-np.array(conditional_entropies)) * alpha_bars
    return signal_values

def improved_hybrid_signal_recording(noise_levels: np.ndarray) -> Tuple[int, float]:
    """Record signals and compute a perceptual hash using the improved hybrid noise schedule."""
    signal_values = improved_hybrid_noise_schedule(len(noise_levels))
    phash = compute_phash(signal_values)
    return phash, np.mean(signal_values)

if __name__ == "__main__":
    T = 100
    noise_levels = np.random.uniform(0, 1, T)
    phash, mean_signal = improved_hybrid_signal_recording(noise_levels)
    print(f"Perceptual hash: {phash}, Mean signal value: {mean_signal:.4f}")