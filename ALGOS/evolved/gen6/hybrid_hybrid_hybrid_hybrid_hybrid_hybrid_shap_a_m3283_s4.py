# DARWIN HAMMER — match 3283, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_pherom_m96_s1.py (gen4)
# parent_b: hybrid_hybrid_shap_attribut_dense_associative_me_m2066_s3.py (gen5)
# born: 2026-05-29T23:49:07Z

import numpy as np

# Minimal geometric algebra (grade-1 only)
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


# SHAP utilities
Node = int
Graph = dict[int, set[int]]
Model = dict[int, float]


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    return (
        np.math.factorial(subset_size)
        * np.math.factorial(feature_count - subset_size - 1)
        / np.math.factorial(feature_count)
    )


def approximate_shap_values(graph: Graph, model: Model, feature_count: int) -> dict[Node, float]:
    shap = {}
    for node, neighbours in graph.items():
        own = model.get(node, 0.0)
        neigh_sum = sum(model.get(nb, 0.0) for nb in neighbours)
        shap[node] = (own + neigh_sum) / (1 + len(neighbours) + 1e-12)
    return shap


# Koopman operator estimation
def estimate_koopman_operator(snapshots: list[Multivector]) -> np.ndarray:
    X = np.column_stack([s.to_vector() for s in snapshots[:-1]])
    Y = np.column_stack([s.to_vector() for s in snapshots[1:]])
    K, _, _, _ = np.linalg.lstsq(X.T, Y.T, rcond=None)
    K = K.T
    return K


def apply_koopman(K: np.ndarray, mv: Multivector) -> Multivector:
    vec = mv.to_vector()
    pred = K @ vec
    return Multivector.from_vector(pred)


# Dense associative memory / modern Hopfield
def build_memory_matrix(graph: Graph, shap: dict[Node, float]) -> np.ndarray:
    nodes = sorted(graph.keys())
    n = max(nodes) + 1
    M = np.zeros((len(nodes), n))
    for row, node in enumerate(nodes):
        M[row, node] = shap.get(node, 0.0)
        for nb in graph[node]:
            M[row, nb] = shap.get(nb, 0.0)
    return M


def hopfield_attention(M: np.ndarray, query: np.ndarray, beta: float = 1.0) -> np.ndarray:
    logits = beta * (M @ query)
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / exp_logits.sum()
    return probs


def shannon_entropy(p: np.ndarray) -> float:
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    return -np.sum(p_safe * np.log(p_safe))


# Hybrid pipeline functions
def shap_to_multivector(shap: dict[Node, float], dim: int) -> Multivector:
    comps = {frozenset({i}): float(shap.get(i, 0.0)) for i in range(dim)}
    return Multivector(comps, dim)


def hybrid_predict(
    graph: Graph,
    model: Model,
    feature_count: int,
    K: np.ndarray,
    M: np.ndarray,
    beta: float = 1.0,
) -> tuple[np.ndarray, float]:
    shap = approximate_shap_values(graph, model, feature_count)
    S = shap_to_multivector(shap, feature_count)
    S_pred = apply_koopman(K, S)
    query = S_pred.to_vector()
    attention = hopfield_attention(M, query, beta)
    entropy = shannon_entropy(attention)
    return attention, entropy


# Improved version with regularization and early stopping
class HybridModel:
    def __init__(self, graph: Graph, model: Model, feature_count: int, beta: float = 1.0):
        self.graph = graph
        self.model = model
        self.feature_count = feature_count
        self.beta = beta
        self.M = build_memory_matrix(graph, approximate_shap_values(graph, model, feature_count))

    def fit(self, snapshots: list[Multivector], epochs: int = 100, tol: float = 1e-6):
        K = np.eye(self.feature_count)
        prev_loss = np.inf
        for _ in range(epochs):
            X = np.column_stack([s.to_vector() for s in snapshots[:-1]])
            Y = np.column_stack([s.to_vector() for s in snapshots[1:]])
            K_new, _, _, _ = np.linalg.lstsq(X.T, Y.T, rcond=None)
            K_new = K_new.T
            loss = np.mean((K_new @ X.T - Y.T) ** 2)
            if np.abs(loss - prev_loss) < tol:
                break
            K = K_new
            prev_loss = loss
        self.K = K
        return self

    def predict(self) -> tuple[np.ndarray, float]:
        shap = approximate_shap_values(self.graph, self.model, self.feature_count)
        S = shap_to_multivector(shap, self.feature_count)
        S_pred = apply_koopman(self.K, S)
        query = S_pred.to_vector()
        attention = hopfield_attention(self.M, query, self.beta)
        entropy = shannon_entropy(attention)
        return attention, entropy

# Usage
graph = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
model = {0: 1.0, 1: 2.0, 2: 3.0}
feature_count = 3
snapshots = [Multivector({frozenset({i}): 1.0 for i in range(feature_count)}, feature_count) for _ in range(10)]

hybrid_model = HybridModel(graph, model, feature_count)
hybrid_model.fit(snapshots)
attention, entropy = hybrid_model.predict()