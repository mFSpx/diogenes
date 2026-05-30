# DARWIN HAMMER — match 4679, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1225_s3.py (gen6)
# parent_b: hybrid_hybrid_xgboost_objec_hybrid_hybrid_hybrid_m1262_s2.py (gen6)
# born: 2026-05-29T23:57:24Z

import math
import numpy as np

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-margin))

def binary_logistic_grad_hess(y_true: np.ndarray, margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h

def optimal_leaf_weight(gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0) -> float:
    return -float(gradient_sum) / (float(hessian_sum) + float(reg_lambda))

def tropical_max_plus_algebra(C: np.ndarray) -> float:
    def t_add(a: float, b: float) -> float:
        return max(a, b)

    def t_mul(a: float, b: float) -> float:
        return a + b

    product = C
    result = float('-inf')
    for i in range(len(C)):
        result = t_add(result, C[i])
    return result

def hybrid_similarities(morphology1: Morphology, morphology2: Morphology) -> tuple[float, float, float]:
    length_similarity = 1 - abs(morphology1.length - morphology2.length) / max(morphology1.length, morphology2.length)
    width_similarity = 1 - abs(morphology1.width - morphology2.width) / max(morphology1.width, morphology2.width)
    height_similarity = 1 - abs(morphology1.height - morphology2.height) / max(morphology1.height, morphology2.height)
    return length_similarity, width_similarity, height_similarity

def hybrid_morphological_features(morphology: Morphology) -> np.ndarray:
    length_feature = morphology.length / (morphology.width + morphology.height)
    width_feature = morphology.width / (morphology.length + morphology.height)
    height_feature = morphology.height / (morphology.length + morphology.width)
    return np.array([length_feature, width_feature, height_feature])

def hybrid_tropical_max_plus_algebra(features: np.ndarray) -> float:
    return tropical_max_plus_algebra(features)

def improved_hybrid_tropical_max_plus_algebra(features: np.ndarray) -> float:
    weights = np.array([0.4, 0.3, 0.3])  # assign weights to features
    return np.sum(features * weights)

def improved_hybrid_similarities(morphology1: Morphology, morphology2: Morphology) -> tuple[float, float, float]:
    length_similarity = 1 - abs(morphology1.length - morphology2.length) / max(morphology1.length, morphology2.length)
    width_similarity = 1 - abs(morphology1.width - morphology2.width) / max(morphology1.width, morphology2.width)
    height_similarity = 1 - abs(morphology1.height - morphology2.height) / max(morphology1.height, morphology2.height)
    return length_similarity, width_similarity, height_similarity

def improved_hybrid_morphological_features(morphology: Morphology) -> np.ndarray:
    length_feature = morphology.length / (morphology.width + morphology.height)
    width_feature = morphology.width / (morphology.length + morphology.height)
    height_feature = morphology.height / (morphology.length + morphology.width)
    volume_feature = morphology.length * morphology.width * morphology.height
    return np.array([length_feature, width_feature, height_feature, volume_feature])

if __name__ == "__main__":
    morphology1 = Morphology(1.0, 2.0, 3.0, 4.0)
    morphology2 = Morphology(1.5, 2.5, 3.5, 4.5)
    similarities = improved_hybrid_similarities(morphology1, morphology2)
    print("Similarities:", similarities)
    features = improved_hybrid_morphological_features(morphology1)
    print("Features:", features)
    result = improved_hybrid_tropical_max_plus_algebra(features)
    print("Improved Tropical Max-Plus Algebra Result:", result)