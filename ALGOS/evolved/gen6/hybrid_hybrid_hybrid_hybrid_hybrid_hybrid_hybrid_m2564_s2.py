# DARWIN HAMMER — match 2564, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_regret_m236_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1200_s2.py (gen5)
# born: 2026-05-29T23:42:59Z

import numpy as np
from typing import List, Dict, Any, Tuple, Sequence, Union


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)


# ----------------------------------------------------------------------
# Structural Similarity Index (SSIM)
# ----------------------------------------------------------------------
def compute_ssim(
    x: Sequence[float],
    y: Sequence[float],
    *,
    dynamic_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Compute a 1‑D version of the Structural Similarity Index (SSIM).

    Parameters
    ----------
    x, y : sequence of float
        Input signals. They are converted to ``np.ndarray`` of ``float64``.
    dynamic_range : float, optional
        The difference between the maximum possible value and the minimum
        possible value of the signals. If ``None`` it is inferred from the
        data as ``max(x.max() - x.min(), y.max() - y.min())``.
    k1, k2 : float
        Small constants to stabilise the division with weak denominator.

    Returns
    -------
    float
        SSIM value in ``[-1, 1]`` (theoretical range for 1‑D signals).
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) == 0:
        raise ValueError("x and y must be non‑empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if dynamic_range is None:
        dr_x = x_arr.max() - x_arr.min()
        dr_y = y_arr.max() - y_arr.min()
        dynamic_range = max(dr_x, dr_y)
        if dynamic_range == 0.0:
            dynamic_range = 1.0  # avoid division by zero for constant signals

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)

    # Guard against pathological zero‑denominator (should not happen with c1,c2)
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _align_vector(v: Sequence[float], target_len: int) -> np.ndarray:
    """
    Pad or truncate a vector so that it matches ``target_len``.
    Padding is performed with zeros.
    """
    arr = np.asarray(v, dtype=np.float64)
    if arr.size < target_len:
        return np.pad(arr, (0, target_len - arr.size))
    if arr.size > target_len:
        return arr[:target_len]
    return arr


def hybrid_score(packet: Dict[str, Any]) -> float:
    """
    Compute the SSIM between a packet payload and the prototype vector.
    Returns ``0.0`` for malformed packets.
    """
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple, np.ndarray)):
        return 0.0
    try:
        payload_vec = _align_vector(payload, PROTOTYPE_VECTOR.size)
        return compute_ssim(payload_vec, PROTOTYPE_VECTOR)
    except Exception:
        return 0.0


def compute_health_scores(endpoints: Sequence[Dict[str, Any]]) -> np.ndarray:
    """
    Extract a numeric health score from each endpoint dictionary.
    Missing or non‑numeric entries default to ``0.0``.
    """
    scores = []
    for ep in endpoints:
        h = ep.get("health_score", 0.0)
        try:
            scores.append(float(h))
        except Exception:
            scores.append(0.0)
    return np.array(scores, dtype=np.float64)


def tropical_regret_gains(health_scores: np.ndarray, actions: Sequence[Dict[str, Any]]) -> np.ndarray:
    """
    Classic tropical (max‑plus) regret gain: the difference between the best
    health score and the intrinsic cost of each action.
    """
    if health_scores.size == 0:
        best = 0.0
    else:
        best = health_scores.max()
    gains = []
    for act in actions:
        cost = act.get("intrinsic_cost", 0.0)
        try:
            cost = float(cost)
        except Exception:
            cost = 0.0
        gains.append(best - cost)
    return np.array(gains, dtype=np.float64)


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def hybrid_regrets(
    endpoints: Sequence[Dict[str, Any]],
    actions: Sequence[Dict[str, Any]],
    ssim_scores: Union[float, Sequence[float]],
) -> np.ndarray:
    """
    Compute regret values for each action, weighted by SSIM similarity.
    ``ssim_scores`` can be a single scalar (applied uniformly) or a
    sequence whose length matches ``len(actions)``.
    """
    health_scores = compute_health_scores(endpoints)
    gains = tropical_regret_gains(health_scores, actions)

    # Normalise SSIM scores to [0,1] range (they already are, but we guard)
    if isinstance(ssim_scores, (float, int)):
        weights = np.full_like(gains, float(ssim_scores))
    else:
        weights = np.asarray(ssim_scores, dtype=np.float64)
        if weights.shape != gains.shape:
            raise ValueError(
                f"ssim_scores length {weights.shape} does not match number of actions {gains.shape}"
            )
    # Clip to a sensible range to avoid exploding regrets
    weights = np.clip(weights, 0.0, 1.0)

    return gains * weights


class StoreState:
    """
    Minimal store model: a scalar level that can be increased (inflow)
    or decreased (outflow). ``update`` returns the new level and the
    absolute change (delta).
    """

    def __init__(self, initial_level: float = 0.0):
        self.level = float(initial_level)

    def update(self, inflow: Sequence[Sequence[float]], outflow: Sequence[Sequence[float]]) -> Tuple[float, float]:
        """
        Apply inflow and outflow matrices (treated as sums of their entries).
        """
        inflow_sum = float(np.sum(inflow))
        outflow_sum = float(np.sum(outflow))
        new_level = self.level + inflow_sum - outflow_sum
        delta = new_level - self.level
        self.level = new_level
        return self.level, delta


def hybrid_update_store(
    store_state: StoreState,
    inflow: Sequence[Sequence[float]],
    outflow: Sequence[Sequence[float]],
    ssim_score: float,
) -> Tuple[float, float]:
    """
    Update the store and scale the reported delta by a single SSIM weight.
    ``ssim_score`` is clipped to ``[0,1]`` to keep the scaling bounded.
    """
    new_level, delta = store_state.update(inflow, outflow)
    weight = np.clip(float(ssim_score), 0.0, 1.0)
    return new_level, delta * weight


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a batch of SSIM scores from synthetic packets
    ssim_batch = [
        hybrid_score({"payload": [0.1, 0.2, 0.3, 0.4, 0.5]}) for _ in range(5)
    ]

    # Example endpoints and actions
    endpoints = [
        {"health_score": 0.5},
        {"health_score": 0.8},
        {"health_score": 0.3},
    ]
    actions = [
        {"action_id": "a1", "intrinsic_cost": 0.2},
        {"action_id": "a2", "intrinsic_cost": 0.1},
        {"action_id": "a3", "intrinsic_cost": 0.4},
    ]

    # Use the mean SSIM as a uniform weight
    mean_ssim = float(np.mean(ssim_batch))
    regrets = hybrid_regrets(endpoints, actions, mean_ssim)
    print("Regrets (uniform SSIM weight):", regrets)

    # Or weight each action individually (here we repeat the batch to match actions)
    per_action_ssim = ssim_batch[: len(actions)]
    regrets_weighted = hybrid_regrets(endpoints, actions, per_action_ssim)
    print("Regrets (per‑action SSIM weight):", regrets_weighted)

    # Store update example
    store = StoreState(initial_level=10.0)
    inflow = [[0.1, 0.2], [0.3, 0.4]]
    outflow = [[0.5, 0.6], [0.7, 0.8]]
    new_lvl, scaled_delta = hybrid_update_store(store, inflow, outflow, mean_ssim)
    print("Store new level:", new_lvl, "Scaled delta:", scaled_delta)