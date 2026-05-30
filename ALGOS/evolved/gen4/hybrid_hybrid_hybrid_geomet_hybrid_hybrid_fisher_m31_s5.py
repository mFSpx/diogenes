# DARWIN HAMMER — match 31, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_geometric_pro_decision_hygiene_m25_s1.py (gen2)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s4.py (gen3)
# born: 2026-05-29T23:26:31Z

import math
import random
import sys
import pathlib
import numpy as np
from typing import Tuple, List, Dict

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
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
                sign *= 1
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0) + coef
        return Multivector(result, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({blade: coef * scalar for blade, coef in self.components.items()}, self.n)


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))


def hybrid_multivector_fisher_ssim(multivector: Multivector, 
                                    fisher_center: float, 
                                    fisher_width: float, 
                                    ssim_x: np.ndarray, 
                                    ssim_y: np.ndarray) -> Multivector:
    fisher_info = fisher_score(np.mean(ssim_x), fisher_center, fisher_width)
    ssim_value = ssim(ssim_x, ssim_y)
    weighted_multivector = multivector * (fisher_info * ssim_value)
    return weighted_multivector


def hybrid_decision_hygiene(multivector: Multivector, 
                            fisher_center: float, 
                            fisher_width: float, 
                            ssim_x: np.ndarray, 
                            ssim_y: np.ndarray) -> float:
    weighted_multivector = hybrid_multivector_fisher_ssim(multivector, fisher_center, fisher_width, ssim_x, ssim_y)
    hygiene_score = weighted_multivector.scalar_part()
    return hygiene_score


def hybrid_entropy(multivector: Multivector, 
                   fisher_center: float, 
                   fisher_width: float, 
                   ssim_x: np.ndarray, 
                   ssim_y: np.ndarray) -> float:
    hygiene_score = hybrid_decision_hygiene(multivector, fisher_center, fisher_width, ssim_x, ssim_y)
    if hygiene_score <= 0:
        return 0
    entropy = -hygiene_score * math.log(hygiene_score)
    return entropy


def improved_hybrid_multivector_fisher_ssim(multivector: Multivector, 
                                            fisher_center: float, 
                                            fisher_width: float, 
                                            ssim_x: np.ndarray, 
                                            ssim_y: np.ndarray) -> Multivector:
    fisher_info = fisher_score(np.mean(ssim_x), fisher_center, fisher_width)
    ssim_value = ssim(ssim_x, ssim_y)
    weighted_multivector = multivector * (fisher_info * ssim_value)
    weighted_multivector_components = dict(weighted_multivector.components)
    for blade, coef in weighted_multivector_components.items():
        weighted_multivector_components[blade] = coef / (1 + math.exp(-coef))
    return Multivector(weighted_multivector_components, multivector.n)


def improved_hybrid_decision_hygiene(multivector: Multivector, 
                                    fisher_center: float, 
                                    fisher_width: float, 
                                    ssim_x: np.ndarray, 
                                    ssim_y: np.ndarray) -> float:
    weighted_multivector = improved_hybrid_multivector_fisher_ssim(multivector, fisher_center, fisher_width, ssim_x, ssim_y)
    hygiene_score = weighted_multivector.scalar_part()
    return hygiene_score


def improved_hybrid_entropy(multivector: Multivector, 
                           fisher_center: float, 
                           fisher_width: float, 
                           ssim_x: np.ndarray, 
                           ssim_y: np.ndarray) -> float:
    hygiene_score = improved_hybrid_decision_hygiene(multivector, fisher_center, fisher_width, ssim_x, ssim_y)
    if hygiene_score <= 0:
        return 0
    entropy = -hygiene_score * math.log(hygiene_score)
    return entropy


if __name__ == "__main__":
    multivector = Multivector({frozenset(): 1.0}, 2)
    fisher_center = 0.5
    fisher_width = 0.1
    ssim_x = np.array([1, 2, 3])
    ssim_y = np.array([1, 2, 3])
    print(improved_hybrid_decision_hygiene(multivector, fisher_center, fisher_width, ssim_x, ssim_y))
    print(improved_hybrid_entropy(multivector, fisher_center, fisher_width, ssim_x, ssim_y))