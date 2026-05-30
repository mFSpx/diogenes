# DARWIN HAMMER — match 2576, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_cockpi_hybrid_hybrid_fisher_m583_s0.py (gen3)
# parent_b: hybrid_hybrid_tropical_maxp_hybrid_hybrid_geomet_m927_s0.py (gen5)
# born: 2026-05-29T23:43:05Z

import math
import numpy as np

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def trust_weighted_velocity(x0: float, x1: float, trust: float) -> float:
    return trust * (x1 - x0)

def jeap_energy(candidate: float, prev_candidate: float, fisher: float) -> float:
    predictor = np.array([prev_candidate + fisher])
    encoder = np.array([candidate])
    return np.sum((encoder - predictor) ** 2)

def t_add(x, y):
    return np.maximum(x, y)

def t_mul(x, y):
    return np.add(x, y)

def t_matmul(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def t_polyval(coeffs, x):
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
    return np.max(terms, axis=0)

def clifford_geometric_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(np.sum((a - b) ** 2))

def hybrid_trust_tropical_velocity(x0, x1, trust):
    v = np.asarray(x1, dtype=float) - np.asarray(x0, dtype=float)
    v_trust = trust * v
    v_trop = v + trust  # Improved tropical scaling
    return t_add(v_trust, v_trop)

def hybrid_tropical_metric_distance(p, q, trust=1.0):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    d_euc = np.linalg.norm(p - q)
    d_cliff = clifford_geometric_distance(p, q)
    d_trop = t_add(d_euc, d_cliff)
    # Improved distance modulation using logarithmic trust
    d_final = d_trop * np.log(1 + trust)
    return d_final

def hybrid_fisher_tropical_flow_energy(candidate, prev_candidate, theta, trust=1.0):
    F = fisher_score(theta)
    coeffs = np.array([0.0, F, -np.inf])
    poly_val = t_polyval(coeffs, candidate)
    # Improved tropical scaling with non-linear trust effect
    poly_trust = t_mul(poly_val, np.log(1 + trust))
    energy = jeap_energy(candidate, prev_candidate, float(poly_trust))
    return energy

def hybrid_voronoi_tropical_flow(points, num_partitions, trust=1.0):
    n = len(points)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i, j] = hybrid_tropical_metric_distance(points[i], points[j], trust)
            dist_matrix[j, i] = dist_matrix[i, j]
    # Improved Voronoi partition using k-means-like initialization
    centroids = np.random.choice(n, size=num_partitions, replace=False)
    partitions = [[] for _ in range(num_partitions)]
    for i in range(n):
        nearest_centroid = np.argmin(dist_matrix[i, centroids])
        partitions[nearest_centroid].append(i)
    return partitions