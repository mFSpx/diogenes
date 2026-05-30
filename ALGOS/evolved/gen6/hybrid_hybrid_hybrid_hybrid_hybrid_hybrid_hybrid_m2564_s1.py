# DARWIN HAMMER — match 2564, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_regret_m236_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1200_s2.py (gen5)
# born: 2026-05-29T23:42:59Z

import numpy as np
from typing import List, Dict

# Constants
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

# SSIM implementation with improved robustness to outliers
def compute_ssim(
    x: List[float],
    y: List[float],
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

    mx = np.median(x_arr)
    my = np.median(y_arr)
    vx = np.percentile(np.abs(x_arr - mx), 75) ** 2
    vy = np.percentile(np.abs(y_arr - my), 75) ** 2
    cov = np.mean((x_arr - mx) * (y_arr - my))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)
    return float(numerator / denominator)

# Hybrid routing utilities with improved handling of edge cases
def hybrid_score(packet: Dict[str, List[float]]) -> float:
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        return 0.0
    try:
        payload_vec = np.asarray(payload, dtype=np.float64)
        if payload_vec.size < PROTOTYPE_VECTOR.size:
            payload_vec = np.pad(payload_vec, (0, PROTOTYPE_VECTOR.size - payload_vec.size))
        elif payload_vec.size > PROTOTYPE_VECTOR.size:
            payload_vec = payload_vec[: PROTOTYPE_VECTOR.size]
        return compute_ssim(payload_vec, PROTOTYPE_VECTOR, dynamic_range=1.0)
    except Exception as e:
        print(f"Error computing hybrid score: {e}")
        return 0.0

# Regret-match utilities with improved numerical stability
def tropical_regret_gains(health_scores, actions):
    # Evaluate the max-plus network and return a gain per action
    gains = []
    for action in actions:
        gain = np.max(health_scores) - action['intrinsic_cost']
        gains.append(gain)
    return np.array(gains)

# Hybrid operation: compute regrets based on SSIM scores with improved weighting
def hybrid_regrets(endpoints, actions, ssim_scores):
    health_scores = np.array([endpoint['health_score'] for endpoint in endpoints])
    gains = tropical_regret_gains(health_scores, actions)
    weighted_gains = gains * np.array(ssim_scores) / np.sum(ssim_scores)
    return weighted_gains

# Hybrid operation: update store state based on SSIM scores with improved scaling
def hybrid_update_store(store_state, inflow, outflow, ssim_scores):
    new_level, delta = store_state.update(inflow, outflow)
    scaled_delta = delta * np.mean(ssim_scores)
    return new_level, scaled_delta

class StoreState:
    def __init__(self):
        self.level = 0.0
        self.delta = 0.0

    def update(self, inflow, outflow):
        self.level += np.mean(inflow) - np.mean(outflow)
        self.delta = np.mean(inflow) - np.mean(outflow)
        return self.level, self.delta

if __name__ == "__main__":
    # Smoke test
    ssim_scores = [hybrid_score({"payload": [0.1, 0.2, 0.3, 0.4, 0.5]}) for _ in range(10)]
    endpoints = [{"health_score": 0.5}, {"health_score": 0.8}]
    actions = [{"action_id": "action1", "intrinsic_cost": 0.2}, {"action_id": "action2", "intrinsic_cost": 0.1}]
    print(hybrid_regrets(endpoints, actions, ssim_scores))
    store_state = StoreState()
    inflow = [[0.1, 0.2], [0.3, 0.4]]
    outflow = [[0.5, 0.6], [0.7, 0.8]]
    new_level, delta = hybrid_update_store(store_state, inflow, outflow, ssim_scores)
    print(new_level, delta)