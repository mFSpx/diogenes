# DARWIN HAMMER — match 5672, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_indy_learning_hybrid_hybrid_hybrid_m1194_s3.py (gen5)
# parent_b: hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s1.py (gen3)
# born: 2026-05-30T00:04:02Z

"""
Module for the Hybrid Bayesian-Krampus-Ollivier-Ricci-Darwin-Hammer Algorithm, 
integrating the core topologies of hybrid_hybrid_indy_learning_hybrid_hybrid_hybrid_m1194_s3 and 
hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s1.

The mathematical bridge between the two structures is the application of Bayesian inference 
to update the probabilities of the decision scores in the Darwin Hammer algorithm, 
taking into account the Ollivier-Ricci curvature of the connections between the different 
dimensions of the brain map, while using the Structural Similarity Index (SSIM) to 
inform the selection of actions in the entity-level resource computation.

The governing equations of both parents are fused through the following interface:
- The decision hygiene scores from the Darwin Hammer algorithm are used as prior 
  probabilities in the Bayesian update of the Hybrid Bayesian-Krampus-Ollivier-Ricci-Hybrid-Bandit-Router Algorithm.
- The SSIM scores from the Hybrid Bayesian-Krampus-Ollivier-Ricci-Hybrid-Bandit-Router Algorithm 
  are used to weight the resource vectors in the entity-level resource computation of the 
  Darwin Hammer algorithm.
"""

import numpy as np
import random
import math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, Iterable, List, Tuple, Optional

# Constants
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

# SSIM implementation
def compute_ssim(
    x: list[float],
    y: list[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)

class HybridDarwinHammer:
    EARTH_RADIUS_KM = 6371.0

    def __init__(
        self,
        beta: float,
        alpha: float,
        spatial_budget: float,
        privacy_budget: float,
        decision_budget: float,
        signature_db: Optional[set] = None,
        decision_scores: Optional[Dict[Any, float]] = None,
    ):
        """
        Parameters
        ----------
        beta, alpha : float
            Scaling factors for privacy and decision components.
        spatial_budget, privacy_budget, decision_budget : float
            Upper limits for the aggregated resource consumption.
        signature_db : set, optional
            Set of known signatures for collision detection.
        decision_scores : dict, optional
            Mapping from entity identifiers to pre‑computed decision hygiene scores.
        """
        self.beta = beta
        self.alpha = alpha
        self.spatial_budget = spatial_budget
        self.privacy_budget = privacy_budget
        self.decision_budget = decision_budget

        self.signature_db = signature_db if signature_db is not None else set()
        self.decision_scores = decision_scores if decision_scores is not None else {}

    def calculate_resource_vector(self, entity: Dict, reference_location: Tuple[float, float]) -> np.ndarray:
        """
        Returns a 3‑dimensional vector [d_i, p_i, s_i] for a single entity.
        """
        d = self._haversine_distance(entity["location"], reference_location)
        p = self._signature_collision(entity["signature"]) * self.beta
        s = self._hybrid_decision_hygiene(entity)
        return np.array([d, p, s], dtype=float)

    def _haversine_distance(
        self, location: Tuple[float, float], reference_location: Tuple[float, float]
    ) -> float:
        """
        Haversine distance in metres.
        """
        lat1, lon1 = map(math.radians, location)
        lat2, lon2 = map(math.radians, reference_location)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return self.EARTH_RADIUS_KM * 1000 * c

    def _signature_collision(self, signature: str) -> float:
        """
        Returns a probability of collision.
        """
        return 1.0 if signature in self.signature_db else 0.0

    def _hybrid_decision_hygiene(self, entity: Dict) -> float:
        """
        Returns a decision hygiene score using Bayesian update and SSIM.
        """
        prior = self.decision_scores.get(entity["id"], 0.5)
        payload = entity.get("payload")
        if payload:
            ssim_score = compute_ssim(payload, PROTOTYPE_VECTOR.tolist(), dynamic_range=1.0)
            likelihood = ssim_score
            posterior = prior * likelihood / (prior * likelihood + (1-prior) * (1-likelihood))
            return posterior
        return prior

def hybrid_score(packet: dict[str, list[float]]) -> float:
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        return 0.0
    try:
        payload_vec = np.asarray(payload, dtype=np.float64)
        if payload_vec.size < PROTOTYPE_VECTOR.size:
            payload_vec = np.pad(payload_vec, (0, PROTOTYPE_VECTOR.size - payload_vec.size))
        elif payload_vec.size > PROTOTYPE_VECTOR.size:
            payload_vec = payload_vec[: PROTOTYPE_VECTOR.size]
        return compute_ssim(payload_vec.tolist(), PROTOTYPE_VECTOR.tolist(), dynamic_range=1.0)
    except Exception:
        return 0.0

if __name__ == "__main__":
    # Smoke test
    hammer = HybridDarwinHammer(1.0, 1.0, 1000.0, 1.0, 1.0)
    entity = {"id": 1, "location": (0.0, 0.0), "signature": "test", "payload": [0.1, 0.2, 0.3]}
    resource_vector = hammer.calculate_resource_vector(entity, (0.0, 0.0))
    print(resource_vector)
    packet = {"payload": [0.1, 0.2, 0.3]}
    print(hybrid_score(packet))