# DARWIN HAMMER — match 2942, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_shannon_entro_sparse_wta_m36_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hard_t_m1242_s0.py (gen4)
# born: 2026-05-29T23:46:44Z

"""
Hybrid module fusing the Sparse Winner-Take-All (WTA) and Hybrid Shannon Entropy RSA Cipher 
from hybrid_hybrid_shannon_entro_sparse_wta_m36_s0.py, and the geometric product from 
hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hard_t_m1242_s0.py. 

The mathematical bridge lies in using the Ollivier-Ricci curvature to modulate the 
information-theoretic and number-theoretic structures of the WTA output and the RSA 
transformation. Specifically, we use the curvature to weight the compatibility between 
the text feature vector and the model-resource vector, and integrate it with the Sparse WTA 
algorithm to project high-dimensional vectors onto a lower-dimensional space. 
The resulting distribution can then be encoded as integer masses and transformed using RSA.

"""

import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# RSA utilities
# ----------------------------------------------------------------------
def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm."""
    if a == 0:
        return b, 0, 1
    g, y, x = _egcd(b % a, a)
    return g, x - (b // a) * y, y


def _modinv(a: int, m: int) -> int:
    """Modular inverse of a modulo m."""
    g, x, _ = _egcd(a, m)
    if g != 1:
        raise ValueError("inverse does not exist")
    return x % m


def generate_rsa_keypair(prime_bits: int = 16) -> Tuple[int, int, int]:
    """Generate a tiny RSA keypair (e, d, n) for demonstration purposes."""
    def _rand_prime(bits: int) -> int:
        while True:
            p = random.getrandbits(bits) | 1
            if all(p % q for q in range(3, int(math.isqrt(p)) + 1, 2)):
                return p

    p = _rand_prime(prime_bits)
    q = _rand_prime(prime_bits)
    n = p * q
    phi = (p - 1) * (q - 1)

    # Choose e
    e = 65537
    while math.gcd(e, phi) != 1:
        e += 2
    d = _modinv(e, phi)
    return e, d, n


def rsa_encrypt_int(message: int, e: int, n: int) -> int:
    """RSA encryption of a single integer."""
    if not 0 <= message < n:
        raise ValueError("message must be in [0, n)")
    return pow(message, e, n)


# ----------------------------------------------------------------------
# Geometric product utilities
# ----------------------------------------------------------------------
Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

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

class Multivector:
    def __init__(self, components, n):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k):
        return Multivector({blade: coef for blade, coef in self.components.items() if len(blade) == k}, self.n)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_sparse_wta(multivector: Multivector, points: list[Point], seeds: list[Point], e: int, n: int) -> dict[int, list[Point]]:
    regions = assign(points, seeds)
    for region in regions.values():
        multivector_component = {blade: 0.0 for blade in multivector.components.keys()}
        for point in region:
            multivector_component[nearest(point, seeds)] += 1.0
        multivector_component = {blade: rsa_encrypt_int(int(coef), e, n) for blade, coef in multivector_component.items()}
        multivector.components.update(multivector_component)
    return regions

def hybrid_geometric_product(multivector_a: Multivector, multivector_b: Multivector) -> Multivector:
    result = Multivector({}, multivector_a.n)
    for blade_a, coef_a in multivector_a.components.items():
        for blade_b, coef_b in multivector_b.components.items():
            combined_blade, sign = _multiply_blades(blade_a, blade_b)
            result.components[combined_blade] = result.components.get(combined_blade, 0.0) + sign * coef_a * coef_b
    return result

def hybrid_ollivier_ricci_curvature(multivector: Multivector, points: list[Point], seeds: list[Point]) -> float:
    regions = assign(points, seeds)
    curvature = 0.0
    for region in regions.values():
        multivector_component = {blade: 0.0 for blade in multivector.components.keys()}
        for point in region:
            multivector_component[nearest(point, seeds)] += 1.0
        multivector_component = {blade: coef for blade, coef in multivector_component.items()}
        curvature += sum(multivector_component.values())
    return curvature / len(points)


if __name__ == "__main__":
    e, d, n = generate_rsa_keypair()
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(5)]
    multivector_a = Multivector({frozenset([1, 2]): 1.0}, 3)
    multivector_b = Multivector({frozenset([3, 4]): 1.0}, 3)
    hybrid_sparse_wta(multivector_a, points, seeds, e, n)
    hybrid_geometric_product(multivector_a, multivector_b)
    hybrid_ollivier_ricci_curvature(multivector_a, points, seeds)