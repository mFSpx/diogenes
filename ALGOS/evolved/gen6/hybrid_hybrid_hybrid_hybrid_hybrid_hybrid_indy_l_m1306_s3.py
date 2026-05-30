# DARWIN HAMMER — match 1306, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_model__m584_s0.py (gen4)
# parent_b: hybrid_hybrid_indy_learning_hybrid_hybrid_hybrid_m1194_s1.py (gen5)
# born: 2026-05-29T23:35:10Z

import math
import numpy as np

class HybridDarwinHammer:
    def __init__(self, beta: float, alpha: float, spatial_budget: int, privacy_budget: float, decision_budget: int):
        self.beta = beta
        self.alpha = alpha
        self.spatial_budget = spatial_budget
        self.privacy_budget = privacy_budget
        self.decision_budget = decision_budget

    def calculate_resource_vector(self, entity: dict, reference_location: tuple) -> list:
        d = self.haversine_distance(entity['location'], reference_location)
        p = self.signature_collision(entity['signature']) * self.beta
        s = self.decision_hygiene(entity)
        return [d, p, s]

    def haversine_distance(self, location: tuple, reference_location: tuple) -> float:
        lat1, lon1 = math.radians(location[0]), math.radians(location[1])
        lat2, lon2 = math.radians(reference_location[0]), math.radians(reference_location[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) 
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371 * c  

    def signature_collision(self, signature: str) -> float:
        return len(signature)

    def decision_hygiene(self, entity: dict) -> float:
        return 1.0


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width ** 2))
    return (derivative ** 2) / intensity


def geometric_product(a: np.ndarray, b: np.ndarray, distance: float) -> np.ndarray:
    return np.dot(a, b) + np.cross(a, b) + distance * np.outer(a, b)


def hybrid_geometric_product(entity: dict, reference_location: tuple, hybrid_darwin_hammer: HybridDarwinHammer) -> np.ndarray:
    location = np.array(entity['location'])
    reference_location = np.array(reference_location)
    distance = hybrid_darwin_hammer.haversine_distance(entity['location'], reference_location)
    return geometric_product(location, reference_location, distance)


def hybrid_fisher_score(theta: float, center: float, width: float, distance: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width ** 2))
    return (derivative ** 2) / intensity + distance ** 2


if __name__ == "__main__":
    entity = {'location': (40.7128, -74.0060), 'signature': 'abc123'}
    reference_location = (37.7749, -122.4194)
    hybrid_darwin_hammer = HybridDarwinHammer(0.0, 0.0, 0, 0.0, 0)
    distance = hybrid_darwin_hammer.haversine_distance(entity['location'], reference_location)
    product = hybrid_geometric_product(entity, reference_location, hybrid_darwin_hammer)
    print(product)
    score = hybrid_fisher_score(0.0, 0.0, 1.0, distance)
    print(score)