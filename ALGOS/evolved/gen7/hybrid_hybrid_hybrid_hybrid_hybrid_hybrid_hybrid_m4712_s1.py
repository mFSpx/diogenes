# DARWIN HAMMER — match 4712, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_fisher_m982_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m1823_s1.py (gen6)
# born: 2026-05-29T23:57:41Z

import numpy as np
import math

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    sx = np.std(x)
    sy = np.std(y)
    sxy = np.mean((x - mx) * (y - my))
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))

def shannon_entropy(values: list[float]) -> float:
    probabilities = [v / sum(values) for v in values]
    return -sum([p * math.log(p, 2) for p in probabilities if p != 0])

def hybrid_rbf_surrogate(node_features: list[list[float]], 
                         query_feature: list[float], 
                         epsilon: float = 1.0, 
                         center: float = 0.0, 
                         width: float = 1.0) -> float:
    surrogate_values = [gaussian(euclidean(node_feature, query_feature), epsilon) 
                        for node_feature in node_features]
    fisher_scores = [fisher_score(surrogate_value, center, width) 
                     for surrogate_value in surrogate_values]
    return sum([fisher_score * surrogate_value for fisher_score, surrogate_value 
                in zip(fisher_scores, surrogate_values)])

def hybrid_ssim_fisher(node_features: list[list[float]], 
                        query_feature: list[float], 
                        epsilon: float = 1.0, 
                        center: float = 0.0, 
                        width: float = 1.0) -> float:
    surrogate_values = np.array([gaussian(euclidean(node_feature, query_feature), epsilon) 
                                  for node_feature in node_features])
    fisher_scores = np.array([fisher_score(surrogate_value, center, width) 
                               for surrogate_value in surrogate_values])
    predicted_values = surrogate_values
    actual_values = np.array([shannon_entropy(surrogate_values.tolist()) 
                              for _ in surrogate_values])
    ssim_score = ssim(predicted_values, actual_values)
    return ssim_score * np.sum(fisher_scores * surrogate_values)

def improved_hybrid_ssim_fisher(node_features: list[list[float]], 
                                query_feature: list[float], 
                                epsilon: float = 1.0, 
                                center: float = 0.0, 
                                width: float = 1.0) -> float:
    surrogate_values = np.array([gaussian(euclidean(node_feature, query_feature), epsilon) 
                                  for node_feature in node_features])
    fisher_scores = np.array([fisher_score(surrogate_value, center, width) 
                               for surrogate_value in surrogate_values])
    predicted_values = surrogate_values
    actual_values = np.array([shannon_entropy(surrogate_values.tolist()) 
                              for _ in surrogate_values])
    ssim_score = ssim(predicted_values, actual_values)
    kl_divergence = np.sum(surrogate_values * np.log(surrogate_values / actual_values))
    return ssim_score * np.sum(fisher_scores * surrogate_values) / (1 + kl_divergence)

if __name__ == "__main__":
    node_features = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    query_feature = [2.0, 3.0, 4.0]
    epsilon = 1.0
    center = 0.0
    width = 1.0
    print(hybrid_rbf_surrogate(node_features, query_feature, epsilon, center, width))
    print(hybrid_ssim_fisher(node_features, query_feature, epsilon, center, width))
    print(improved_hybrid_ssim_fisher(node_features, query_feature, epsilon, center, width))