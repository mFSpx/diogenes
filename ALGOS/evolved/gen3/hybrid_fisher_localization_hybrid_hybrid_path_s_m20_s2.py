# DARWIN HAMMER — match 20, survivor 2
# gen: 3
# parent_a: fisher_localization.py (gen0)
# parent_b: hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s0.py (gen2)
# born: 2026-05-29T23:26:12Z

"""
This module implements a hybrid mathematical algorithm that fuses the Fisher-information scoring 
from 'fisher_localization.py' with the lead-lag transform and feature extraction from 
'hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s0.py'. The mathematical bridge between 
the two structures is based on representing the Fisher-information score as a feature that can 
be extracted and used to compute the lead-lag transform.

The Fisher-information scoring is used to compute a score for a given angle, which is then used 
as a feature to compute the lead-lag transform. The lead-lag transform is used to interleave 
the lead and lag channels for causality encoding.
"""

import numpy as np
import math

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

def extract_fisher_feature(angle: float, center: float, width: float) -> float:
    return fisher_score(angle, center, width)

def hybrid_fusion(candidates: list[float], center: float, width: float) -> np.ndarray:
    angles = np.array(candidates)
    fisher_features = np.array([extract_fisher_feature(angle, center, width) for angle in angles])
    lead_lag_path = np.column_stack((angles, fisher_features))
    return lead_lag_transform(lead_lag_path)

def best_angle(candidates: list[float], center: float, width: float) -> float:
    if not candidates:
        raise ValueError('candidates required')
    return max(candidates, key=lambda t: (fisher_score(t, center, width), -abs(t-center)))

if __name__ == "__main__":
    candidates = [1.0, 2.0, 3.0]
    center = 2.0
    width = 1.0
    hybrid_path = hybrid_fusion(candidates, center, width)
    print(hybrid_path)
    best_candidate = best_angle(candidates, center, width)
    print(best_candidate)