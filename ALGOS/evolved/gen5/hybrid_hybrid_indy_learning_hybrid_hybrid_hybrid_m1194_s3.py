# DARWIN HAMMER — match 1194, survivor 3
# gen: 5
# parent_a: hybrid_indy_learning_vector_hybrid_fold_change_d_m38_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s2.py (gen4)
# born: 2026-05-29T23:33:29Z

import math
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, Iterable, List, Tuple, Optional

import numpy as np


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

    # --------------------------------------------------------------------- #
    #  Entity‑level resource computation
    # --------------------------------------------------------------------- #
    def calculate_resource_vector(self, entity: Dict, reference_location: Tuple[float, float]) -> np.ndarray:
        """
        Returns a 3‑dimensional vector [d_i, p_i, s_i] for a single entity.
        """
        d = self._haversine_distance(entity["location"], reference_location)
        p = self._signature_collision(entity["signature"]) * self.beta
        s = self._decision_hygiene(entity)
        return np.array([d, p, s], dtype=float)

    def _haversine_distance(
        self, location: Tuple[float, float], reference_location: Tuple[float, float]
    ) -> float:
        """
        Haversine distance in metres.
        """
        lat1, lon1 = map(math.radians, location)
        lat2, lon2 = map(math.radians, reference_location)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return self.EARTH_RADIUS_KM * c * 1000.0

    def _signature_collision(self, signature: str) -> int:
        """
        Returns 1 if the signature already exists in the database, else 0.
        """
        return 1 if signature in self.signature_db else 0

    def _decision_hygiene(self, entity: Dict) -> float:
        """
        Retrieves a decision hygiene score; falls back to a neutral score (0.5) if missing.
        """
        return float(self.decision_scores.get(entity.get("id"), 0.5))

    # --------------------------------------------------------------------- #
    #  Model‑level resource computation
    # --------------------------------------------------------------------- #
    def calculate_model_resource_vector(self, model: Dict, tier_factor: int) -> np.ndarray:
        """
        Returns a 2‑dimensional vector [RAM_j, α·τ_j·μ_j].
        """
        ram = float(model.get("ram", 0.0))
        mu = self._mean_privacy_risk(model.get("records", []))
        return np.array([ram, self.alpha * tier_factor * mu], dtype=float)

    def _mean_privacy_risk(self, records: List[Dict]) -> float:
        """
        Computes the arithmetic mean of 'risk' fields; returns 0.0 for empty input.
        """
        if not records:
            return 0.0
        total = sum(float(rec.get("risk", 0.0)) for rec in records)
        return total / len(records)

    # --------------------------------------------------------------------- #
    #  Matrix assembly and budget enforcement
    # --------------------------------------------------------------------- #
    def stack_resource_vectors(
        self, entity_vectors: List[np.ndarray], model_vectors: List[np.ndarray]
    ) -> np.ndarray:
        """
        Vertically stacks entity and model vectors into a single matrix A.
        """
        if not entity_vectors and not model_vectors:
            raise ValueError("At least one entity or model vector must be provided.")
        all_vectors = entity_vectors + model_vectors
        return np.vstack(all_vectors)

    def enforce_budgets(self, A: np.ndarray) -> np.ndarray:
        """
        Filters rows of A so that cumulative consumption does not exceed the three budgets.
        The columns are assumed to be ordered as [spatial, privacy, decision] for entities
        and [spatial, privacy] for models; missing columns are treated as zeros.
        """
        # Pad model rows to three columns for uniform handling
        if A.shape[1] < 3:
            pad = np.zeros((A.shape[0], 3 - A.shape[1]), dtype=float)
            A = np.hstack([A, pad])

        cum = np.cumsum(A, axis=0)
        mask = (
            (cum[:, 0] <= self.spatial_budget)
            & (cum[:, 1] <= self.privacy_budget)
            & (cum[:, 2] <= self.decision_budget)
        )
        return A[mask]

    # --------------------------------------------------------------------- #
    #  Subset selection utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def select_subset(A: np.ndarray, mask: Iterable[int]) -> np.ndarray:
        """
        Returns rows of A where the corresponding mask entry is truthy.
        """
        mask_arr = np.asarray(list(mask), dtype=bool)
        if mask_arr.shape[0] != A.shape[0]:
            raise ValueError("Mask length must match number of rows in A.")
        return A[mask_arr]

    # --------------------------------------------------------------------- #
    #  Bandit update (Thompson sampling variant)
    # --------------------------------------------------------------------- #
    @staticmethod
    def thompson_update(
        successes: int, failures: int, prior_alpha: float = 1.0, prior_beta: float = 1.0
    ) -> float:
        """
        Draws a sample from the posterior Beta distribution.
        """
        posterior_alpha = prior_alpha + successes
        posterior_beta = prior_beta + failures
        return np.random.beta(posterior_alpha, posterior_beta)

    @staticmethod
    def hybrid_update(
        token_frequency_vector: List[int],
        log_ratio: float,
        bandit_propensity: float,
        lr: float = 0.1,
    ) -> float:
        """
        Performs a bounded gradient‑like update on the bandit propensity.
        """
        gain = float(token_frequency_vector[0]) if token_frequency_vector else 0.0
        update = bandit_propensity + lr * gain * log_ratio
        return max(0.0, min(1.0, update))


def main() -> None:
    # --------------------------------------------------------------------- #
    #  Sample data and initialisation
    # --------------------------------------------------------------------- #
    signature_db = {"abcdef12345", "9876543210"}
    decision_scores = {"entity_1": 0.8, "entity_2": 0.3}

    hybrid = HybridDarwinHammer(
        beta=1.2,
        alpha=0.9,
        spatial_budget=5000.0,      # metres
        privacy_budget=5.0,         # abstract units
        decision_budget=2.0,        # abstract units
        signature_db=signature_db,
        decision_scores=decision_scores,
    )

    entities = [
        {"id": "entity_1", "location": (37.7749, -122.4194), "signature": "1234567890"},
        {"id": "entity_2", "location": (37.7840, -122.4090), "signature": "abcdef12345"},
    ]
    reference_location = (37.7859, -122.4364)

    models = [
        {"ram": 2048, "records": [{"risk": 0.4}, {"risk": 0.6}]},
        {"ram": 1024, "records": []},  # empty record list → zero privacy risk
    ]
    tier_factor = 3

    # --------------------------------------------------------------------- #
    #  Build resource matrix
    # --------------------------------------------------------------------- #
    entity_vectors = [hybrid.calculate_resource_vector(e, reference_location) for e in entities]
    model_vectors = [hybrid.calculate_model_resource_vector(m, tier_factor) for m in models]

    A = hybrid.stack_resource_vectors(entity_vectors, model_vectors)
    A_budgeted = hybrid.enforce_budgets(A)

    # --------------------------------------------------------------------- #
    #  Subset selection example (select first two rows)
    # --------------------------------------------------------------------- #
    mask = [1, 1, 0, 0]  # length must equal A.shape[0]
    selected = hybrid.select_subset(A_budgeted, mask)

    # --------------------------------------------------------------------- #
    #  Bandit update demonstration
    # --------------------------------------------------------------------- #
    token_frequency_vector = [2, 5, 1]
    log_ratio = 0.4
    bandit_propensity = 0.2
    updated_propensity = hybrid.hybrid_update(
        token_frequency_vector, log_ratio, bandit_propensity
    )

    # Thompson sampling example
    successes, failures = 7, 3
    sampled_theta = HybridDarwinHammer.thompson_update(successes, failures)

    # --------------------------------------------------------------------- #
    #  Output
    # --------------------------------------------------------------------- #
    print("Entity resource vectors:")
    for ev in entity_vectors:
        print(ev)

    print("\nModel resource vectors:")
    for mv in model_vectors:
        print(mv)

    print("\nCombined resource matrix (A):")
    print(A)

    print("\nBudget‑compliant matrix (A_budgeted):")
    print(A_budgeted)

    print("\nSelected subset:")
    print(selected)

    print("\nUpdated bandit propensity:", updated_propensity)
    print("Sampled Thompson theta:", sampled_theta)


if __name__ == "__main__":
    main()