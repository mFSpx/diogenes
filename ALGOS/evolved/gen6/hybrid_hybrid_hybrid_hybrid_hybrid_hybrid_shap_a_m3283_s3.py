# DARWIN HAMMER — match 3283, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_pherom_m96_s1.py (gen4)
# parent_b: hybrid_hybrid_shap_attribut_dense_associative_me_m2066_s3.py (gen5)
# born: 2026-05-29T23:49:07Z

import sys
import math
import random
from pathlib import Path
import numpy as np

def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return tuple(lst), sign


def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    res, sign = _blade_sign(combined)
    return frozenset(res), sign


class Multivector:
    def __init__(self, components: dict, n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def to_vector(self) -> np.ndarray:
        vec = np.zeros(self.n)
        for blade, coeff in self.components.items():
            if len(blade) == 1:
                idx = next(iter(blade))
                vec[idx] = coeff
        return vec

    @staticmethod
    def from_vector(vec: np.ndarray) -> "Multivector":
        n = vec.shape[0]
        comps = {frozenset({i}): float(v) for i, v in enumerate(vec) if abs(v) > 1e-15}
        return Multivector(comps, n)

    def __add__(self, other):
        comps = self.components.copy()
        for b, v in other.components.items():
            comps[b] = comps.get(b, 0.0) + v
        return Multivector(comps, self.n)

    def __rmul__(self, scalar: float):
        comps = {b: scalar * v for b, v in self.components.items()}
        return Multivector(comps, self.n)

    __mul__ = __rmul__


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    return (
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / math.factorial(feature_count)
    )


def approximate_shap_values(graph, model, feature_count):
    shap = {}
    for node, neighbours in graph.items():
        own = model.get(node, 0.0)
        neigh_sum = sum(model.get(nb, 0.0) for nb in neighbours)
        shap[node] = (own + neigh_sum) / (1 + len(neighbours) + 1e-12)
    return shap


def estimate_koopman_operator(snapshots):
    if len(snapshots) < 2:
        raise ValueError("At least two snapshots required")
    X = np.column_stack([s.to_vector() for s in snapshots[:-1]])   
    Y = np.column_stack([s.to_vector() for s in snapshots[1:]])    
    K, residuals, rank, s = np.linalg.lstsq(X.T, Y.T, rcond=None)  
    K = K.T  
    return K


def apply_koopman(K, mv):
    vec = mv.to_vector()
    pred = K @ vec
    return Multivector.from_vector(pred)


def build_memory_matrix(graph, shap):
    nodes = sorted(graph.keys())
    n = max(nodes) + 1  
    M = np.zeros((len(nodes), n))
    for row, node in enumerate(nodes):
        M[row, node] = shap.get(node, 0.0)
        for nb in graph[node]:
            M[row, nb] = shap.get(nb, 0.0)
    return M


def hopfield_attention(M, query, beta=1.0):
    logits = beta * (M @ query)  
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / exp_logits.sum()
    return probs


def shannon_entropy(p):
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    return -np.sum(p_safe * np.log(p_safe))


def shap_to_multivector(shap, dim):
    comps = {frozenset({i}): float(shap.get(i, 0.0)) for i in range(dim)}
    return Multivector(comps, dim)


def hybrid_predict(graph, model, feature_count, K, M, beta=1.0):
    shap = approximate_shap_values(graph, model, feature_count)
    S = shap_to_multivector(shap, len(graph))
    Ŝ = apply_koopman(K, S)
    query = Ŝ.to_vector()
    attention = hopfield_attention(M, query, beta)
    entropy = shannon_entropy(attention)
    return attention, entropy


def improved_hybrid_predict(graph, model, feature_count, K, M, beta=1.0, alpha=0.1):
    shap = approximate_shap_values(graph, model, feature_count)
    S = shap_to_multivector(shap, len(graph))
    Ŝ = apply_koopman(K, S)
    query = Ŝ.to_vector()
    attention = hopfield_attention(M, query, beta)
    entropy = shannon_entropy(attention)
    # Introduce a feedback loop to adjust the Koopman operator
    adjusted_K = K + alpha * np.outer(query, query)
    adjusted_Ŝ = apply_koopman(adjusted_K, S)
    adjusted_attention = hopfield_attention(M, adjusted_Ŝ.to_vector(), beta)
    adjusted_entropy = shannon_entropy(adjusted_attention)
    return attention, entropy, adjusted_attention, adjusted_entropy


# Test the improved hybrid predict function
if __name__ == "__main__":
    graph = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    model = {0: 1.0, 1: 2.0, 2: 3.0}
    feature_count = 3
    K = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]])
    M = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    attention, entropy, adjusted_attention, adjusted_entropy = improved_hybrid_predict(graph, model, feature_count, K, M)
    print("Attention:", attention)
    print("Entropy:", entropy)
    print("Adjusted Attention:", adjusted_attention)
    print("Adjusted Entropy:", adjusted_entropy)