# DARWIN HAMMER — match 1306, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_model__m584_s0.py (gen4)
# parent_b: hybrid_hybrid_indy_learning_hybrid_hybrid_hybrid_m1194_s1.py (gen5)
# born: 2026-05-29T23:35:10Z

import math
import numpy as np
from typing import Dict, Any, Tuple, List

# ----------------------------------------------------------------------
# Gaussian / Fisher utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity G(θ) with centre `center` and standard-deviation `width`."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.
    For a Gaussian N(center, width²) the Fisher information w.r.t. the mean is
    I = 1 / width².  The implementation follows the original code and returns
    (∂G/∂θ)² / G, which is algebraically equivalent.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width ** 2))
    return (derivative ** 2) / intensity


# ----------------------------------------------------------------------
# Haversine Distance and Resource Vector utilities
# ----------------------------------------------------------------------
class HybridDarwinHammer:
    def __init__(self, beta: float, alpha: float, spatial_budget: int, privacy_budget: float, decision_budget: int):
        """
        Initialize the HybridDarwinHammer class.

        Args:
        beta (float): Beta value.
        alpha (float): Alpha value.
        spatial_budget (int): Spatial budget.
        privacy_budget (float): Privacy budget.
        decision_budget (int): Decision budget.
        """
        self.beta = beta
        self.alpha = alpha
        self.spatial_budget = spatial_budget
        self.privacy_budget = privacy_budget
        self.decision_budget = decision_budget

    def calculate_resource_vector(self, entity: Dict[str, Any], reference_location: Tuple[float, float]) -> List[float]:
        """
        Calculate the 3-dimensional resource vector eᵢ = [ dᵢ, pᵢ, sᵢ ] for an entity.

        Args:
        entity (dict): Entity data with 'location' and 'signature'.
        reference_location (tuple): Reference location in degrees.

        Returns:
        resource_vector (list): 3-dimensional resource vector.
        """
        d = self.haversine_distance(entity['location'], reference_location)
        p = self.signature_collision(entity['signature']) * self.beta
        s = self.decision_hygiene(entity)
        return [d, p, s]

    def haversine_distance(self, location: Tuple[float, float], reference_location: Tuple[float, float]) -> float:
        """
        Calculate the haversine distance between two points.

        Args:
        location (tuple): Point coordinates (latitude, longitude).
        reference_location (tuple): Reference point coordinates (latitude, longitude).

        Returns:
        distance (float): Distance in meters.
        """
        lat1, lon1 = math.radians(location[0]), math.radians(location[1])
        lat2, lon2 = math.radians(reference_location[0]), math.radians(reference_location[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371000 * c  # Earth's radius in meters

    def signature_collision(self, signature: Any) -> float:
        """
        Calculate the signature collision metric.

        Args:
        signature (any): Entity signature.

        Returns:
        collision (float): Signature collision metric.
        """
        return len(signature)

    def decision_hygiene(self, entity: Dict[str, Any]) -> float:
        """
        Calculate the decision hygiene metric.

        Args:
        entity (dict): Entity data.

        Returns:
        hygiene (float): Decision hygiene metric.
        """
        return 1.0


# ----------------------------------------------------------------------
# Geometric Product utilities
# ----------------------------------------------------------------------
def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Geometric product of two vectors.

    Returns:
    product (np.ndarray): Geometric product.
    """
    return np.dot(a, b) * np.eye(3) + np.cross(a, b)


def hybrid_geometric_product(entity: Dict[str, Any], reference_location: Tuple[float, float], 
                             hybrid_darwin_hammer: HybridDarwinHammer) -> np.ndarray:
    """
    Hybrid geometric product of an entity's location and a reference location.

    Args:
    entity (dict): Entity data with 'location'.
    reference_location (tuple): Reference location in degrees.
    hybrid_darwin_hammer (HybridDarwinHammer): Hybrid Darwin Hammer instance.

    Returns:
    product (np.ndarray): Hybrid geometric product.
    """
    location = np.array(entity['location'])
    reference_location = np.array(reference_location)
    location = location - reference_location
    return geometric_product(location, np.array([0, 0, 1]))


# ----------------------------------------------------------------------
# Hybrid Algorithm utilities
# ----------------------------------------------------------------------
def hybrid_fisher_score(theta: float, center: float, width: float, 
                        hybrid_darwin_hammer: HybridDarwinHammer, entity: Dict[str, Any], 
                        reference_location: Tuple[float, float], eps: float = 1e-12) -> float:
    """
    Hybrid Fisher information for a Gaussian beam.

    Args:
    theta (float): Theta value.
    center (float): Center value.
    width (float): Width value.
    hybrid_darwin_hammer (HybridDarwinHammer): Hybrid Darwin Hammer instance.
    entity (dict): Entity data.
    reference_location (tuple): Reference location.

    Returns:
    score (float): Hybrid Fisher score.
    """
    distance = hybrid_darwin_hammer.haversine_distance(entity['location'], reference_location)
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width ** 2))
    return (derivative ** 2) / intensity + (distance / 6371000) ** 2


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    entity = {'location': (40.7128, -74.0060), 'signature': 'abc123'}
    reference_location = (37.7749, -122.4194)
    hybrid_darwin_hammer = HybridDarwinHammer(0.0, 0.0, 0, 0.0, 0)
    product = hybrid_geometric_product(entity, reference_location, hybrid_darwin_hammer)
    print(product)
    score = hybrid_fisher_score(0.0, 0.0, 1.0, hybrid_darwin_hammer, entity, reference_location)
    print(score)