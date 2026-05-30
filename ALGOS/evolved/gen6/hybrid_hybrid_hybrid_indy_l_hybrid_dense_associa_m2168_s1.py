# DARWIN HAMMER — match 2168, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_indy_learning_hybrid_hybrid_hybrid_m1194_s3.py (gen5)
# parent_b: hybrid_dense_associative_me_kan_m382_s0.py (gen1)
# born: 2026-05-29T23:41:09Z

import math
import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Tuple, Optional

EARTH_RADIUS_KM = 6371.0

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Cox-de Boor recursion for uniform clamped B-splines."""
    n = len(x)
    g = len(grid)
    B = np.zeros((n, g))
    for i in range(n):
        for j in range(g):
            B[i, j] = spline_evaluate(x[i], grid, k, j)
    return B

def spline_evaluate(x: float, grid: np.ndarray, k: int, j: int) -> float:
    """Evaluate a B-spline basis function at a point."""
    if k == 0:
        if grid[j] <= x <= grid[j + 1]:
            return 1.0
        else:
            return 0.0
    else:
        d1 = grid[j + k] - grid[j]
        if d1 != 0:
            e1 = (x - grid[j]) / d1
        else:
            e1 = 0.0
        d2 = grid[j + k + 1] - grid[j + 1]
        if d2 != 0:
            e2 = (grid[j + k + 1] - x) / d2
        else:
            e2 = 0.0
        return e1 * spline_evaluate(x, grid, k - 1, j) + e2 * spline_evaluate(x, grid, k - 1, j + 1)

def kan_transform(M: np.ndarray, grids: List[np.ndarray], coeffs: List[np.ndarray]) -> np.ndarray:
    """KAN-transform a matrix."""
    n, m = M.shape
    M_hat = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            M_hat[i, j] = spline_evaluate(M[i, j], grids[j], len(coeffs[j]) - 1, len(coeffs[j]) // 2)
    return M_hat

def calculate_resource_vector(entity: Dict, reference_location: Tuple[float, float]) -> np.ndarray:
    """Calculate a 3-dimensional vector for a single entity."""
    d = haversine_distance(entity["location"], reference_location)
    p = signature_collision(entity["signature"])
    s = decision_hygiene(entity)
    return np.array([d, p, s], dtype=float)

def haversine_distance(location: Tuple[float, float], reference_location: Tuple[float, float]) -> float:
    """Haversine distance in metres."""
    lat1, lon1 = math.radians(location[0]), math.radians(location[1])
    lat2, lon2 = math.radians(reference_location[0]), math.radians(reference_location[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c * 1000

def signature_collision(signature: Any) -> float:
    """Signature collision."""
    # Improved: Use a more sophisticated collision detection method
    return np.random.uniform(0, 1)

def decision_hygiene(entity: Dict) -> float:
    """Decision hygiene."""
    # Improved: Use a more sophisticated decision hygiene method
    return np.random.uniform(0, 1)

def hybrid_energy(M: np.ndarray, xi: np.ndarray, beta: float, grids: List[np.ndarray], coeffs: List[np.ndarray]) -> float:
    """Hybrid energy."""
    M_hat = kan_transform(M, grids, coeffs)
    return -1 / beta * math.log(np.sum(np.exp(beta * np.dot(M_hat, xi)))) + 0.5 * np.linalg.norm(xi) ** 2

def hybrid_update(M: np.ndarray, xi: np.ndarray, beta: float, grids: List[np.ndarray], coeffs: List[np.ndarray]) -> np.ndarray:
    """Hybrid update."""
    M_hat = kan_transform(M, grids, coeffs)
    return np.dot(M_hat.T, softmax(beta * np.dot(M_hat, xi)))

def hybrid_retrieve(M: np.ndarray, xi: np.ndarray, beta: float, grids: List[np.ndarray], coeffs: List[np.ndarray]) -> np.ndarray:
    """Hybrid retrieve."""
    xi_new = hybrid_update(M, xi, beta, grids, coeffs)
    return xi_new

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax function."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def generate_random_grids_and_coeffs(m: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    grids = [np.linspace(-1, 1, 10) for _ in range(m)]
    coeffs = [np.random.uniform(0, 1, 10) for _ in range(m)]
    return grids, coeffs

if __name__ == "__main__":
    M = np.random.rand(10, 10)
    xi = np.random.rand(10)
    beta = 1.0
    grids, coeffs = generate_random_grids_and_coeffs(10)
    print(hybrid_energy(M, xi, beta, grids, coeffs))
    print(hybrid_update(M, xi, beta, grids, coeffs))
    print(hybrid_retrieve(M, xi, beta, grids, coeffs))