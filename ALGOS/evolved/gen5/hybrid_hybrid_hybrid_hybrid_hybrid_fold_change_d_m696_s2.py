# DARWIN HAMMER — match 696, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s3.py (gen4)
# parent_b: hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s0.py (gen3)
# born: 2026-05-29T23:30:35Z

"""HybridFusion Algorithm
Combines:
- Parent A: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s3.py
  (resource vector e_i = [d_i, p_i, s_i] and a VRAM‑store modulated bandit with a weight
   matrix W ∈ ℝ^{d_in×d_out} that yields expected rewards).
- Parent B: hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s0.py
  (fold‑change detection dynamics defined by the Euler‑integrated step() function and a
   policy table that stores cumulative rewards per action).

Mathematical Bridge
------------------
The bridge is built by feeding the distance component d_i of each resource vector as the
input signal u_t of the fold‑change detector. The detector’s output y_t (the “response”)
scales the expected reward computed from the resource vector via the weight matrix:
    
    r̂_i = e_i · W                     (expected reward from bandit)
    y_t = response_i(t)               (fold‑change output)
    reward_i(t) = r̂_i * y_t           (final reward used for policy update)

The virtual VRAM store S(t) is updated by a first‑order differential equation that
depends on the instantaneous reward, providing a feedback loop that modulates the
learning rate η(t) used when updating the weight matrix:

    dS/dt = α·(reward_i - S) - β·S
    η(t)   = η₀·(1 + γ·S)

Thus the three core topologies (resource vector, fold‑change dynamics, and bandit
learning) are mathematically fused into a single unified system.
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility functions (parent A)
# ----------------------------------------------------------------------
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great‑circle distance in metres between two lat/lon points."""
    R = 6371000.0  # Earth radius in metres
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_resource_vector(
    lat: float,
    lon: float,
    ref_lat: float,
    ref_lon: float,
    signature: str,
    signature_set: set[str],
    score: float,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Build the 3‑dimensional resource vector e = [d, p, s].

    d : haversine distance to a reference location.
    p : β·σ where σ = 1 if the signature collides with any other entity, else 0.
    s : external decision‑hygiene score (passed directly).
    """
    d = haversine(lat, lon, ref_lat, ref_lon)
    sigma = 1.0 if signature in signature_set else 0.0
    p = beta * sigma
    s = float(score)
    return np.array([d, p, s], dtype=float)


# ----------------------------------------------------------------------
# Fold‑change detection (parent B)
# ----------------------------------------------------------------------
def step(
    u: float,
    x: float,
    y: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay_x: float = 1.0,
    decay_y: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """Euler integration of the feed‑forward fold‑change detector."""
    if dt < 0:
        raise ValueError("dt must be non‑negative")
    ratio = u / max(abs(x), eps)
    dy = gain * ratio - decay_y * y
    dx = u - decay_x * x
    return x + dt * dx, y + dt * dy


def response_series(
    inputs: List[float],
    x0: float = 1.0,
    y0: float = 0.0,
    dt: float = 1.0,
    gain: float = 1.0,
    decay_x: float = 1.0,
    decay_y: float = 1.0,
) -> List[Tuple[float, float]]:
    """Generate (x_t, y_t) pairs for a sequence of inputs u_t."""
    x, y = x0, y0
    out: List[Tuple[float, float]] = []
    for u in inputs:
        x, y = step(u, x, y, dt, gain, decay_x, decay_y)
        out.append((x, y))
    return out


# ----------------------------------------------------------------------
# Policy handling (parent B)
# ----------------------------------------------------------------------
_POLICY: dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])  # [total_reward, count]


def reset_policy() -> None:
    _POLICY.clear()


def _reward(action: str) -> float:
    total, n = _POLICY[action]
    return total / n if n else 0.0


def _count(action: str) -> float:
    return _POLICY[action][1]


def update_policy(updates: List[Tuple[str, float]]) -> None:
    """Incrementally update the cumulative reward table."""
    for action, reward in updates:
        total, n = _POLICY[action]
        _POLICY[action] = [total + reward, n + 1]


def select_action(actions: List[str]) -> str:
    """ε‑greedy selection based on current average rewards."""
    eps = 0.1
    if random.random() < eps:
        return random.choice(actions)
    # choose action with highest average reward
    best = max(actions, key=_reward)
    return best


# ----------------------------------------------------------------------
# Hybrid Fusion Core
# ----------------------------------------------------------------------
class HybridFusion:
    def __init__(
        self,
        d_in: int,
        d_out: int,
        seed: int = 0,
        base_eta: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        dt: float = 1.0,
        store_decay: float = 0.99,
    ):
        """
        Parameters
        ----------
        d_in, d_out : dimensions of the bandit weight matrix.
        base_eta    : baseline learning rate.
        alpha, beta : coefficients for the VRAM‑store differential equation.
        gamma       : scaling of the store into the learning‑rate modulation.
        dt          : time step for all Euler integrations.
        store_decay : exponential decay applied to the store each step.
        """
        random.seed(seed)
        np.random.seed(seed)
        self.W = np.random.randn(d_in, d_out) * 0.1  # weight matrix
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dt = dt
        self.store = 0.0
        self.store_decay = store_decay

    # ------------------------------------------------------------------
    # 1️⃣ Compute expected reward from resource vectors
    # ------------------------------------------------------------------
    def expected_rewards(self, resource_vectors: np.ndarray) -> np.ndarray:
        """
        Multiply each resource vector e_i (shape (N, d_in)) by the weight matrix
        to obtain a scalar expected reward per entity (shape (N,)).
        """
        # Linear projection followed by sum across output dimension
        proj = resource_vectors @ self.W  # (N, d_out)
        return proj.mean(axis=1)  # scalar per entity

    # ------------------------------------------------------------------
    # 2️⃣ Fold‑change response using distance as input
    # ------------------------------------------------------------------
    def fold_change_responses(self, distances: np.ndarray) -> np.ndarray:
        """
        Run the fold‑change detector on the distance component of each entity.
        Returns the y‑component of the detector (shape (N,)).
        """
        responses = response_series(
            distances.tolist(),
            dt=self.dt,
            gain=1.0,
            decay_x=1.0,
            decay_y=1.0,
        )
        # extract y_t (second element) for each time step
        return np.array([y for _, y in responses], dtype=float)

    # ------------------------------------------------------------------
    # 3️⃣ Update VRAM store and learning rate
    # ------------------------------------------------------------------
    def update_store_and_eta(self, reward_series: np.ndarray) -> float:
        """
        Integrate dS/dt = α·(r - S) - β·S with Euler, then decay.
        Returns the modulated learning rate η(t).
        """
        # simple Euler step using mean reward as r
        r = reward_series.mean() if reward_series.size else 0.0
        dS = self.alpha * (r - self.store) - self.beta * self.store
        self.store += self.dt * dS
        self.store *= self.store_decay  # exponential leak
        eta = self.base_eta * (1.0 + self.gamma * self.store)
        return max(eta, 1e-8)  # avoid zero or negative learning rates

    # ------------------------------------------------------------------
    # 4️⃣ Full hybrid step (demonstrates the fused operation)
    # ------------------------------------------------------------------
    def step(
        self,
        entities: List[dict],
        actions: List[str],
        ref_location: Tuple[float, float],
    ) -> Tuple[str, float]:
        """
        Perform one hybrid iteration:
        1. Build resource vectors.
        2. Compute expected rewards.
        3. Generate fold‑change responses from distances.
        4. Fuse them into final rewards.
        5. Update VRAM store → learning rate.
        6. Perform a gradient step on W.
        7. Update the bandit policy and select an action.

        Returns the selected action and its associated reward.
        """
        # ---- 1. Resource vectors -------------------------------------------------
        sig_set = {e["signature"] for e in entities if e.get("duplicate", False)}
        e_list = [
            compute_resource_vector(
                lat=e["lat"],
                lon=e["lon"],
                ref_lat=ref_location[0],
                ref_lon=ref_location[1],
                signature=e["signature"],
                signature_set=sig_set,
                score=e["score"],
                beta=self.beta,
            )
            for e in entities
        ]
        E = np.stack(e_list)  # shape (N, 3)

        # ---- 2. Expected rewards -------------------------------------------------
        r_hat = self.expected_rewards(E)  # (N,)

        # ---- 3. Fold‑change response (using distance) ----------------------------
        distances = E[:, 0]  # d component
        y_resp = self.fold_change_responses(distances)  # (N,)

        # ---- 4. Fuse into final reward -------------------------------------------
        final_reward = r_hat * y_resp  # element‑wise product, shape (N,)

        # ---- 5. Store & learning‑rate modulation ---------------------------------
        eta = self.update_store_and_eta(final_reward)

        # ---- 6. Gradient update on W (simple SGD on mean‑squared error) ---------
        # target = final_reward (scalar per entity) -> we try to make r_hat close to target
        error = r_hat - final_reward  # (N,)
        grad_W = (E.T @ error[:, None]) / E.shape[0]  # (d_in, 1) approximates gradient
        self.W -= eta * grad_W  # update

        # ---- 7. Policy update & action selection ---------------------------------
        # Use the mean final reward as the scalar feedback for the selected action
        chosen_action = select_action(actions)
        update_policy([(chosen_action, final_reward.mean())])

        return chosen_action, final_reward.mean()


# ----------------------------------------------------------------------
# Public helper functions (required: at least three)
# ----------------------------------------------------------------------
def build_entities(num: int, ref_lat: float, ref_lon: float) -> List[dict]:
    """Generate a list of synthetic entities for testing."""
    entities = []
    for i in range(num):
        lat = ref_lat + random.uniform(-0.01, 0.01)
        lon = ref_lon + random.uniform(-0.01, 0.01)
        signature = f"sig_{random.randint(0, 5)}"
        duplicate = random.random() < 0.2  # 20 % chance of being a duplicate
        score = random.uniform(0, 1)
        entities.append(
            {
                "lat": lat,
                "lon": lon,
                "signature": signature,
                "duplicate": duplicate,
                "score": score,
            }
        )
    return entities


def run_hybrid_demo() -> None:
    """Execute a short demo of the HybridFusion system."""
    ref_loc = (37.7749, -122.4194)  # San Francisco approx.
    actions = ["explore", "exploit", "wait"]
    fusion = HybridFusion(d_in=3, d_out=1, seed=42)

    reset_policy()
    for epoch in range(5):
        ents = build_entities(num=10, ref_lat=ref_loc[0], ref_lon=ref_loc[1])
        action, reward = fusion.step(ents, actions, ref_loc)
        print(f"Epoch {epoch+1}: action={action!r}, reward={reward:.4f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_hybrid_demo()