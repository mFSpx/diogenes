# DARWIN HAMMER — match 20, survivor 1
# gen: 3
# parent_a: fisher_localization.py (gen0)
# parent_b: hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s0.py (gen2)
# born: 2026-05-29T23:26:12Z

"""
This module fuses the Fisher-information scoring for off-axis sensing from 'fisher_localization.py'
with the lead-lag transform and feature extraction from 'hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s0.py'.
The mathematical bridge between the two structures is based on representing the Fisher score as a function 
that can be approximated using the extracted features and then applying the lead-lag transform to the 
resulting path.

The Fisher score is used to compute the derivative of the Gaussian beam, which is then used as the input 
to the lead-lag transform. The lead-lag transform is used to interleave the lead and lag channels for 
causality encoding, which is then used to compute the hybrid path signature.
"""

import numpy as np
import math
import random
import sys
import pathlib

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def extract_features(theta: float, center: float, width: float) -> np.ndarray:
    features = np.array([
        fisher_score(theta, center, width),
        gaussian_beam(theta, center, width),
        (theta - center) / width
    ])
    return features

def hybrid_path_signature(theta_values: np.ndarray, center: float, width: float) -> np.ndarray:
    features = np.array([extract_features(theta, center, width) for theta in theta_values])
    return lead_lag_transform(features)

def best_angle(candidates: np.ndarray, center: float, width: float) -> float:
    scores = np.array([fisher_score(theta, center, width) for theta in candidates])
    return candidates[np.argmax(scores)]

if __name__ == "__main__":
    theta_values = np.linspace(-10, 10, 100)
    center = 0
    width = 1
    hybrid_signature = hybrid_path_signature(theta_values, center, width)
    print(hybrid_signature.shape)

    candidates = np.linspace(-10, 10, 10)
    best_theta = best_angle(candidates, center, width)
    print(best_theta)