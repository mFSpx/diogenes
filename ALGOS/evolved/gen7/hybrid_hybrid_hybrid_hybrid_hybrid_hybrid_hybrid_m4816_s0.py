# DARWIN HAMMER — match 4816, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m1401_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_fractional_hdc_m2289_s1.py (gen6)
# born: 2026-05-29T23:58:09Z

"""
Module fusing hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m1401_s3.py and 
hybrid_hybrid_hybrid_hybrid_fractional_hdc_m2289_s1.py.

The mathematical bridge between the two parents lies in their treatment of geometric 
structures. The first parent utilizes Clifford algebra and Voronoi cells, while the 
second parent employs morphological descriptions and haversine distance. 

The fusion leverages the concept of geometric product from Clifford algebra to 
integrate the Voronoi cell representations with the morphological descriptions. 
Specifically, the geometric product is used to compute the interaction between 
Voronoi cells and morphological features.

The interface between the two parents is established through the use of multivectors 
from Clifford algebra to represent both the Voronoi cells and the morphological 
descriptions. This allows for a unified treatment of geometric structures and 
distances.

"""

import numpy as np
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

# Clifford algebra helpers
def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
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
                del lst[j:j + 2]
                n -= 2
                i = -1  
                break
            j += 1
        i += 1
    return tuple(lst), sign

def _multiply_blades(blade_a: Tuple[int, ...], blade_b: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    concatenated = blade_a + blade_b
    result_blade, sign = _blade_sign(concatenated)
    return result_blade, sign

Multivector = Dict[Tuple[int, ...], float]   

def geometric_product(A: Multivector, B: Multivector) -> Multivector:
    result: Multivector = {}
    for blade_a, coeff_a in A.items():
        for blade_b, coeff_b in B.items():
            res_blade, sign = _multiply_blades(blade_a, blade_b)
            if not res_blade:      
                key = ()
            else:
                key = res_blade
            result[key] = result.get(key, 0.0) + coeff_a * coeff_b * sign
    return {b: c for b, c in result.items() if abs(c) > 1e-12}

def scalar_multivector(value: float) -> Multivector:
    return {(): value}

def basis_blade(index: int) -> Multivector:
    return {(index,): 1.0}

# Voronoi–Fisher utilities
def generate_voronoi_cells(points: np.ndarray) -> List[Dict]:
    cells = []
    for i, pt in enumerate(points):
        theta = math.atan2(pt[1], pt[0])
        cells.append({
            'id': i,
            'center': theta,
            'blade': basis_blade(i + 1)   
        })
    return cells

def fisher_information(theta: float, center: float, width: float) -> float:
    z = (theta - center) / width
    intensity = math.exp(-0.5 * z * z)
    derivative = intensity * (-(theta - center) / (width * width))
    if intensity == 0.0:
        return 0.0
    return (derivative * derivative) / intensity

# Morphology and haversine distance
@dataclass
class Morphology:
    """Geometric description of an endpoint."""
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of geometric mean to maximal dimension, ∈ (0,1]."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    gm = (length * width * height) ** (1.0 / 3.0)
    return gm / max(length, width, height)

def haversine_distance(morphology1: Morphology, morphology2: Morphology) -> float:
    """Haversine distance between two morphologies."""
    # Convert morphologies to spherical coordinates
    lat1, lon1, r1 = math.radians(morphology1.length), math.radians(morphology1.width), morphology1.height
    lat2, lon2, r2 = math.radians(morphology2.length), math.radians(morphology2.width), morphology2.height

    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    # Modified to account for radii difference
    return math.sqrt(r1**2 + r2**2 - 2*r1*r2*math.cos(c))

# Hybrid functions
def compute_hybrid_distance(points: np.ndarray, morphology1: Morphology, morphology2: Morphology) -> float:
    voronoi_cells = generate_voronoi_cells(points)
    multivector1 = morphology_to_multivector(morphology1)
    multivector2 = morphology_to_multivector(morphology2)
    product = geometric_product(multivector1, multivector2)
    distance = 0.0
    for cell in voronoi_cells:
        blade = cell['blade']
        distance += fisher_information(cell['center'], 0.0, 1.0) * haversine_distance(Morphology(**{k: v for k, v in product.items() if k == ()}), morphology1)
    return distance

def morphology_to_multivector(morphology: Morphology) -> Multivector:
    multivector = scalar_multivector(1.0)
    multivector.update({(1,): morphology.length, (2,): morphology.width, (3,): morphology.height})
    return multivector

def hybrid_sphericity_index(points: np.ndarray, morphology: Morphology) -> float:
    voronoi_cells = generate_voronoi_cells(points)
    multivector = morphology_to_multivector(morphology)
    product = geometric_product(multivector, multivector)
    sphericity = sphericity_index(**{k: v for k, v in product.items() if k == ()})
    return sphericity

if __name__ == "__main__":
    points = np.random.rand(10, 2)
    morphology1 = Morphology(1.0, 2.0, 3.0, 4.0)
    morphology2 = Morphology(5.0, 6.0, 7.0, 8.0)
    distance = compute_hybrid_distance(points, morphology1, morphology2)
    sphericity = hybrid_sphericity_index(points, morphology1)
    print(distance, sphericity)