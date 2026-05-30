# DARWIN HAMMER — match 4014, survivor 1
# gen: 5
# parent_a: diffusion_forcing.py (gen0)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m900_s2.py (gen4)
# born: 2026-05-29T23:53:10Z

import numpy as np
import math
from typing import List, Tuple

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
    entropy = shannon_entropy(probabilities.tolist())  # Ensure list conversion
    signal_values = np.exp(-entropy * np.ones_like(alpha_bars))  # Broadcast entropy
    return signal_values * alpha_bars

def compute_phash(values: List[float]) -> int:
    """Compute a perceptual hash from a list of signal values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    min_len = min(len(values), 64)  # Prevent index out of range
    for v in values[:min_len]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hybrid_signal_recording(T: int) -> Tuple[int, float]:
    """Record signals and compute a perceptual hash using the hybrid noise schedule."""
    alpha_bars = diffusion_forcing_noise_schedule(T)
    signal_values = hybrid_noise_schedule(T)
    phash = compute_phash(signal_values.tolist())  # Ensure list conversion
    return phash, np.mean(signal_values)

if __name__ == "__main__":
    T = 100
    phash, mean_signal = hybrid_signal_recording(T)
    print(f"Perceptual hash: {phash}, Mean signal value: {mean_signal:.4f}")