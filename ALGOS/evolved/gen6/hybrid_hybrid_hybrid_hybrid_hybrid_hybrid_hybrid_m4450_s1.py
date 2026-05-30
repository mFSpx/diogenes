# DARWIN HAMMER — match 4450, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m938_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m1386_s0.py (gen5)
# born: 2026-05-29T23:55:55Z

import math
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def health_score(self) -> float:
        if self.open:
            return 0.0
        h = 1.0 - (self.failures / self.failure_threshold)
        return max(0.0, min(1.0, h))

def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def pheromone_weighted_graph(A: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    if A.shape != Phi.shape:
        raise ValueError("Adjacency and pheromone matrices must share shape.")
    return np.multiply(A, Phi)

def init_ttt(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)

def morphology_priority(morph: Morphology) -> float:
    max_dim_sum = 100.0  
    dim_sum = morph.length + morph.width + morph.height
    p = dim_sum / max_dim_sum
    return max(0.0, min(1.0, p))

def dynamic_weight(health: float, priority: float) -> float:
    return health * priority

def shap_kernel(M: int, subset_size: int) -> float:
    if subset_size < 0 or subset_size > M:
        raise ValueError("Invalid subset size.")
    numerator = math.factorial(subset_size) * math.factorial(M - subset_size - 1)
    denominator = math.factorial(M)
    return numerator / denominator

def hybrid_forward(
    cb: EndpointCircuitBreaker,
    morph: Morphology,
    A: np.ndarray,
    Phi: np.ndarray,
    x: np.ndarray,
) -> Tuple[float, np.ndarray]:
    h = cb.health_score()
    p = morphology_priority(morph)
    gamma = dynamic_weight(h, p)

    W = pheromone_weighted_graph(A, Phi)          
    W_hat = gamma * W                               

    N = W_hat.shape[0]
    x_vec = x.reshape(-1)                           
    if x_vec.shape[0] != N:
        raise ValueError("Input vector length must match graph size.")
    T = init_ttt(d_in=N, d_out=N, seed=42)

    pred = T @ x_vec
    loss = float((pred - x_vec) @ (pred - x_vec))
    grad = 2.0 * np.outer(pred - x_vec, x_vec)
    return loss, grad

def improved_hybrid_forward(
    cb: EndpointCircuitBreaker,
    morph: Morphology,
    A: np.ndarray,
    Phi: np.ndarray,
    x: np.ndarray,
) -> Tuple[float, np.ndarray]:
    h = cb.health_score()
    p = 1 / (1 + math.exp(-morphology_priority(morph))) # sigmoid mapping
    gamma = dynamic_weight(h, p)

    W = pheromone_weighted_graph(A, Phi)          
    W_hat = gamma * W                               

    N = W_hat.shape[0]
    x_vec = x.reshape(-1)                           
    if x_vec.shape[0] != N:
        raise ValueError("Input vector length must match graph size.")
    T = init_ttt(d_in=N, d_out=N, seed=42)

    pred = T @ x_vec
    loss = float((pred - x_vec) @ (pred - x_vec))
    grad = 2.0 * np.outer(pred - x_vec, x_vec)
    return loss, grad

def deeply_integrated_hybrid_forward(
    cb: EndpointCircuitBreaker,
    morph: Morphology,
    A: np.ndarray,
    Phi: np.ndarray,
    x: np.ndarray,
) -> Tuple[float, np.ndarray]:
    h = cb.health_score()
    p = 1 / (1 + math.exp(-morphology_priority(morph))) 
    gamma = dynamic_weight(h, p)

    W = pheromone_weighted_graph(A, Phi)          
    W_hat = gamma * W                               

    N = W_hat.shape[0]
    x_vec = x.reshape(-1)                           
    if x_vec.shape[0] != N:
        raise ValueError("Input vector length must match graph size.")
    T = init_ttt(d_in=N, d_out=N, seed=42)

    # Inject SHAP kernel weighting into TTT loss
    shap_weights = np.array([shap_kernel(N, i) for i in range(N)])
    weighted_loss = float((pred - x_vec) @ np.diag(shap_weights) @ (pred - x_vec))
    grad = 2.0 * np.outer(pred - x_vec, x_vec) * shap_weights[:, None]
    return weighted_loss, grad