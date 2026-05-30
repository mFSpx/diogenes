# DARWIN HAMMER — match 1401, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_bandit_router_m1158_s0.py (gen3)
# born: 2026-05-29T23:36:05Z

import sys
import math
import random
import numpy as np
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

def bandit_ucb_update(trials: Dict[int, Tuple[int, float]]) -> Dict[int, Tuple[float, float]]:
    stats = {}
    for cid, (n, total) in trials.items():
        mu = total / n if n > 0 else 0.0
        beta = 1.0 / math.sqrt(1.0 + n)   
        stats[cid] = (mu, beta)
    return stats

def select_cell(cells: List[Dict],
                width: float,
                bandit_stats: Dict[int, Tuple[float, float]]) -> Dict:
    best_cell = None
    best_priority = -math.inf
    for cell in cells:
        cid = cell['id']
        theta = cell['center']
        F = fisher_information(theta, center=0.0, width=width)
        mu, beta = bandit_stats.get(cid, (0.0, 1.0))  
        priority = F * (mu + beta)
        if priority > best_priority:
            best_priority = priority
            best_cell = cell
    return best_cell

def update_allocation(R: Multivector, selected_cell: Dict) -> Multivector:
    return geometric_product(R, selected_cell['blade'])

# High‑level hybrid routine
def hybrid_resource_allocation(points: np.ndarray,
                               n_iterations: int = 10,
                               width: float = 0.5) -> Multivector:
    cells = generate_voronoi_cells(points)
    R: Multivector = scalar_multivector(1.0)
    trials: Dict[int, Tuple[int, float]] = {cell['id']: (0, 0.0) for cell in cells}
    for _ in range(n_iterations):
        bandit_stats = bandit_ucb_update(trials)
        selected_cell = select_cell(cells, width, bandit_stats)
        R = update_allocation(R, selected_cell)
        theta = selected_cell['center']
        reward = fisher_information(theta, center=0.0, width=width) + np.random.normal(0, 0.1)
        trials[selected_cell['id']] = (trials[selected_cell['id']][0] + 1, trials[selected_cell['id']][1] + reward)
    return R

def main():
    points = np.array([[1, 1], [2, 2], [3, 3]])
    result = hybrid_resource_allocation(points)
    print(result)

if __name__ == "__main__":
    main()