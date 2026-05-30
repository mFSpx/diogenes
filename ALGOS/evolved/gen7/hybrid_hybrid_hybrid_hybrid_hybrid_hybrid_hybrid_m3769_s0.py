# DARWIN HAMMER — match 3769, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1208_s9.py (gen6)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s7.py (gen3)
# born: 2026-05-29T23:51:27Z

"""
Hybrid Algorithm: physar_geomet_hybrid_m1208_s9.py
Fusing Physarum flux & conductance dynamics with geometric-algebraic structures.
Mathematical bridge: 
The Physarum model's conductance matrix update can be interpreted as a 
discrete-time dynamical system. By representing the conductance matrix 
as a geometric-algebraic multivector, we can leverage geometric operations 
to inform the conductance updates.

Parents:
- hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1208_s9.py
- hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s7.py
"""

import math
import numpy as np
from typing import Dict, List, Tuple

def physarum_flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float) -> float:
    """Flux on a single edge according to the Physarum model."""
    if edge_length <= 0.0:
        raise ValueError("edge_length must be positive")
    return conductance / edge_length * (pressure_a - pressure_b)

def update_conductance_matrix(
    C: np.ndarray,
    Q: np.ndarray,
    dt: float = 0.1,
    gain: float = 1.0,
    decay: float = 0.01,
) -> np.ndarray:
    """
    Vectorised ODE step for the conductance matrix.
    C_{ij} ← max(0, C_{ij} + dt·(gain·|Q_{ij}| – decay·C_{ij}))
    Diagonal entries are forced to zero (no self‑loops).
    """
    delta = dt * (gain * np.abs(Q) - decay * C)
    new_C = np.maximum(0.0, C + delta)
    np.fill_diagonal(new_C, 0.0)
    return new_C

class Multivector:
    """Very small GA container supporting only addition and outer product."""
    def __init__(self, components: Dict[frozenset, float], n: int):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        return f"Multivector({self.components}, {self.n})"

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
                # duplicate basis vector → cancels (grade‑2+ becomes 0)
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def geometric_conductance_update(
    C: Multivector,
    Q: Multivector,
    dt: float = 0.1,
    gain: float = 1.0,
    decay: float = 0.01,
) -> Multivector:
    """
    Geometric-algebraic conductance update.
    """
    # Represent conductance matrix as a multivector
    C_components = C.components
    Q_components = Q.components

    # Perform geometric product
    product_components = {}
    for blade_a, coeff_a in C_components.items():
        for blade_b, coeff_b in Q_components.items():
            blade, sign = _multiply_blades(blade_a, blade_b)
            product_components[blade] = product_components.get(blade, 0) + sign * coeff_a * coeff_b

    # Update multivector components
    new_components = {}
    for blade, coeff in product_components.items():
        new_coeff = max(0, coeff + dt * (gain * abs(coeff) - decay * C_components.get(blade, 0)))
        if new_coeff != 0:
            new_components[blade] = new_coeff

    return Multivector(new_components, C.n)

def hybrid_physarum_geometric_update(
    physarum_C: np.ndarray,
    physarum_Q: np.ndarray,
    geometric_C: Multivector,
    geometric_Q: Multivector,
    dt: float = 0.1,
    gain: float = 1.0,
    decay: float = 0.01,
) -> Tuple[np.ndarray, Multivector]:
    """
    Hybrid update of Physarum conductance matrix and geometric multivector.
    """
    physarum_C = update_conductance_matrix(physarum_C, physarum_Q, dt, gain, decay)
    geometric_C = geometric_conductance_update(geometric_C, geometric_Q, dt, gain, decay)
    return physarum_C, geometric_C

def smoke_test():
    # Initialize Physarum conductance matrix and geometric multivector
    physarum_C = np.random.rand(5, 5)
    physarum_Q = np.random.rand(5, 5)
    geometric_C = Multivector({frozenset(): 1.0}, 2)
    geometric_Q = Multivector({frozenset({0}): 1.0}, 2)

    # Perform hybrid update
    physarum_C, geometric_C = hybrid_physarum_geometric_update(physarum_C, physarum_Q, geometric_C, geometric_Q)

    print("Physarum Conductance Matrix:")
    print(physarum_C)
    print("Geometric Multivector:")
    print(geometric_C)

if __name__ == "__main__":
    smoke_test()