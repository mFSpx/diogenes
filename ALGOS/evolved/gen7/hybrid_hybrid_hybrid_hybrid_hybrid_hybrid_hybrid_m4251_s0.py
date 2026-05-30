# DARWIN HAMMER — match 4251, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1218_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1669_s0.py (gen6)
# born: 2026-05-29T23:54:39Z

"""
Hybrid algorithm merging the principles of 'hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1218_s4.py' 
and 'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1669_s0.py'. 

The mathematical bridge between these two algorithms lies in their use of multivectors. 
The first algorithm uses a Gaussian kernel to compute similarity between multivectors, 
while the second algorithm uses multivectors to represent conductance updates in a physarum network. 
We fuse these concepts by representing the context of the first algorithm as a multivector 
and using the B-spline basis functions from the second algorithm to approximate the radial-basis 
surrogate model used in the first algorithm. The Caputo fractional derivative is used to modulate 
the effective learning rate of the matrix update, which is integrated with the flux-based conductance 
update primitive.
"""

import math
import random
import sys
import pathlib
import numpy as np

class Multivector:
    def __init__(self, components: dict, n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[b] = result.get(blade, 0) + coef
        return Multivector(result, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({blade: coef * scalar for blade, coef in self.components.items()}, self.n)

def gaussian_kernel(x: Multivector, y: Multivector) -> float:
    """
    Compute the Gaussian kernel between two multivectors.
    
    Args:
    x (Multivector): The first multivector.
    y (Multivector): The second multivector.
    
    Returns:
    float: The Gaussian kernel value.
    """
    components_x = np.array(list(x.components.values()))
    components_y = np.array(list(y.components.values()))
    return np.exp(-np.sum((components_x - components_y) ** 2))

def b_spline_basis(x: float, k: int, i: int) -> float:
    """
    Compute the B-spline basis function.
    
    Args:
    x (float): The input value.
    k (int): The order of the B-spline.
    i (int): The index of the B-spline basis function.
    
    Returns:
    float: The value of the B-spline basis function.
    """
    if k == 0:
        return 1 if i <= x < i + 1 else 0
    else:
        return (x - i) / (k - 1) * b_spline_basis(x, k - 1, i) + (i + k - x) / (k - 1) * b_spline_basis(x, k - 1, i + 1)

def approximate_radial_basis_surrogate(x: Multivector, y: Multivector) -> float:
    """
    Approximate the radial-basis surrogate model using B-spline basis functions.
    
    Args:
    x (Multivector): The input multivector.
    y (Multivector): The output multivector.
    
    Returns:
    float: The approximated value.
    """
    return b_spline_basis(gaussian_kernel(x, y), 3, 0)

def caputo_fractional_derivative(x: float, alpha: float) -> float:
    """
    Compute the Caputo fractional derivative.
    
    Args:
    x (float): The input value.
    alpha (float): The order of the derivative.
    
    Returns:
    float: The value of the Caputo fractional derivative.
    """
    return x ** (1 - alpha) / math.gamma(2 - alpha)

def flux_based_conductance_updateprimitive(x: Multivector, y: Multivector) -> float:
    """
    Compute the flux-based conductance update primitive.
    
    Args:
    x (Multivector): The input multivector.
    y (Multivector): The output multivector.
    
    Returns:
    float: The updated value.
    """
    return gaussian_kernel(x, y) * caputo_fractional_derivative(gaussian_kernel(x, y), 0.5)

if __name__ == "__main__":
    multivector1 = Multivector({frozenset([1, 2, 3]): 1.0, frozenset(): 2.0}, 3)
    multivector2 = Multivector({frozenset([1, 2, 3]): 3.0, frozenset(): 1.0}, 3)
    print(gaussian_kernel(multivector1, multivector2))
    print(b_spline_basis(0.5, 3, 0))
    print(approximate_radial_basis_surrogate(multivector1, multivector2))
    print(caputo_fractional_derivative(1.0, 0.5))
    print(flux_based_conductance_updateprimitive(multivector1, multivector2))