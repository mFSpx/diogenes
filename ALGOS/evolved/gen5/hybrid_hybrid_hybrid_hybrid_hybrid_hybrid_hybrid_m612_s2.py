# DARWIN HAMMER — match 612, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m192_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m67_s1.py (gen4)
# born: 2026-05-29T23:30:12Z

import datetime as dt
import math
import random
from collections import Counter
from typing import List, Tuple, Iterable, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (risk‑health & calendar statistics)
# ----------------------------------------------------------------------
def doomsday_numpy(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Vectorised weekday calculation (Sun=0 … Sat=6)."""
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    # Convert to Python datetime once per element – NumPy does not expose weekday directly.
    py_weekday = np.fromiter(
        (
            dt.datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    # Shift: Monday=0 → Sunday=0
    return (py_weekday + 1) % 7


def weekday_counts(dates: Iterable[dt.date]) -> np.ndarray:
    """Return a length‑7 array with counts of each weekday (Sun=0 … Sat=6)."""
    counts = np.zeros(7, dtype=int)
    for d in dates:
        if isinstance(d, dt.datetime):
            d = d.date()
        # Python's weekday: Mon=0 … Sun=6 → roll to Sun=0
        counts[(d.weekday() + 1) % 7] += 1
    return counts


def gini_coefficient(values: np.ndarray) -> float:
    """Robust Gini coefficient for a 1‑D non‑negative array."""
    if values.size == 0:
        return 0.0
    values = values.astype(float)
    if np.all(values == 0):
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    # Guard against division by zero if total sum is zero (already handled above)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


def compute_health(
    reconstruction_risk_score: float,
    failures: int,
    failure_threshold: int,
    recovery_priority: float,
) -> float:
    """
    Health is a bounded confidence metric.

    health = (1 - (reconstruction_risk_score * failure_rate)) * (1 - recovery_priority)

    where failure_rate = failures / failure_threshold, capped at 1.
    The result is clipped to the unit interval.
    """
    failure_rate = min(1.0, failures / max(1, failure_threshold))
    health = (1.0 - reconstruction_risk_score * failure_rate) * (1.0 - recovery_priority)
    return float(np.clip(health, 0.0, 1.0))


# ----------------------------------------------------------------------
# Parent B utilities (information‑theoretic & geometric signals)
# ----------------------------------------------------------------------
def shannon_entropy(probs: np.ndarray) -> float:
    """
    Normalised Shannon entropy in [0, 1].

    Normalisation uses log(|support|) so the maximum entropy (uniform) maps to 1.
    """
    probs = probs.astype(float)
    if probs.size == 0:
        return 0.0
    probs = probs[probs > 0]  # remove zeros to avoid log(0)
    if probs.size == 0:
        return 0.0
    raw = -np.sum(probs * np.log(probs))
    max_entropy = np.log(probs.size)
    return float(raw / max_entropy) if max_entropy > 0 else 0.0


def curvature_proxy(adj: np.ndarray, node: int) -> float:
    """
    Bounded Ollivier‑Ricci curvature proxy.

    curvature = (avg_neighbor_degree - degree_i) / max(degree_i, 1)
    The result is clipped to [-1, 1] to avoid pathological amplification.
    """
    deg_i = float(np.sum(adj[node]))
    neighbors = np.where(adj[node] > 0)[0]
    if neighbors.size == 0:
        return 0.0
    neighbor_degs = np.sum(adj[neighbors], axis=1)
    avg_neighbor_deg = float(np.mean(neighbor_degs))
    curvature = (avg_neighbor_deg - deg_i) / max(deg_i, 1.0)
    return float(np.clip(curvature, -1.0, 1.0))


# ----------------------------------------------------------------------
# Hybrid bandit engine – deeper integration of the two parent systems
# ----------------------------------------------------------------------
class HybridBandit:
    """
    Upper‑Confidence‑Bound (UCB) bandit where each arm carries its own
    contextual curvature (graph node) and Gini weight (calendar distribution).
    The hybrid reward for arm *i* is

        r_i = health * (1 - entropy) * (1 + curvature_i) * (1 - gini_i)

    where `gini_i` is the Gini coefficient of the weekday distribution
    *restricted* to the dates associated with arm *i* (if any).  This makes the
    calendar signal arm‑specific rather than a global diagnostic.
    """

    def __init__(self, actions: List[str], action_to_node: Dict[int, int]):
        """
        Parameters
        ----------
        actions
            List of action identifiers.
        action_to_node
            Mapping from arm index to a node index in the adjacency matrix.
            Allows each arm to have a distinct curvature context.
        """
        self.actions = actions
        self.action_to_node = action_to_node
        self.n_arms = len(actions)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms, dtype=float)  # empirical mean reward

    def _hybrid_reward(
        self,
        health: float,
        entropy: float,
        curvature: float,
        gini: float,
    ) -> float:
        """
        Compute the per‑arm hybrid reward term.
        All factors are bounded in [0, 1] except curvature which lives in [-1, 1];
        the expression therefore stays within a reasonable range.
        """
        return health * (1.0 - entropy) * (1.0 + curvature) * (1.0 - gini)

    def select(
        self,
        health: float,
        entropy: float,
        curvature_vec: np.ndarray,
        gini_vec: np.ndarray,
    ) -> Tuple[int, str]:
        """
        Choose an arm using UCB on the blended estimate:
        blended_i = empirical_mean_i + hybrid_reward_i
        """
        total_counts = max(1, self.counts.sum())
        hybrid_rewards = np.fromiter(
            (
                self._hybrid_reward(
                    health,
                    entropy,
                    curvature_vec[i],
                    gini_vec[i],
                )
                for i in range(self.n_arms)
            ),
            dtype=float,
            count=self.n_arms,
        )
        blended = self.values + hybrid_rewards
        exploration = np.sqrt(2.0 * np.log(total_counts) / (self.counts + 1e-6))
        ucb = blended + exploration
        idx = int(np.argmax(ucb))
        return idx, self.actions[idx]

    def update(self, arm_idx: int, reward: float) -> None:
        """Incremental update of the empirical mean for the selected arm."""
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        self.values[arm_idx] += (reward - self.values[arm_idx]) / n

    def __repr__(self) -> str:
        return (
            f"<HybridBandit actions={self.actions} counts={self.counts.tolist()} "
            f"values={self.values.tolist()}>"
        )


# ----------------------------------------------------------------------
# High‑level hybrid decision pipeline
# ----------------------------------------------------------------------
def hybrid_decision_pipeline(
    dates: List[dt.date],
    reconstruction_risk_score: float,
    failures: int,
    failure_threshold: int,
    recovery_priority: float,
    feature_probs: np.ndarray,
    adjacency: np.ndarray,
    bandit: HybridBandit,
    action_date_map: Dict[int, List[dt.date]],
) -> str:
    """
    End‑to‑end decision making.

    Parameters
    ----------
    dates
        Global list of dates used for the *global* health computation.
    action_date_map
        Mapping from arm index to the subset of dates that are considered
        relevant for that arm (used to compute arm‑specific Gini).
    """
    # 1. Health (global, independent of actions)
    health = compute_health(
        reconstruction_risk_score, failures, failure_threshold, recovery_priority
    )

    # 2. Global entropy (same for all arms)
    entropy = shannon_entropy(feature_probs)

    # 3. Per‑arm curvature (lookup via bandit.action_to_node)
    curvature_vec = np.empty(bandit.n_arms, dtype=float)
    for i in range(bandit.n_arms):
        node = bandit.action_to_node.get(i, 0)  # default to node 0 if missing
        curvature_vec[i] = curvature_proxy(adjacency, node)

    # 4. Per‑arm Gini based on the dates assigned to each arm
    gini_vec = np.empty(bandit.n_arms, dtype=float)
    for i in range(bandit.n_arms):
        arm_dates = action_date_map.get(i, [])
        counts = weekday_counts(arm_dates) if arm_dates else np.zeros(7, dtype=int)
        gini_vec[i] = gini_coefficient(counts)

    # 5. Bandit selection
    arm_idx, action = bandit.select(health, entropy, curvature_vec, gini_vec)

    # 6. Simulated reward using the same hybrid formula (for learning)
    reward = health * (1.0 - entropy) * (1.0 + curvature_vec[arm_idx]) * (1.0 - gini_vec[arm_idx])
    bandit.update(arm_idx, reward)

    return action


# ----------------------------------------------------------------------
# Smoke test (executed when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample dates spanning two weeks
    today = dt.date.today()
    sample_dates = [today + dt.timedelta(days=i) for i in range(14)]

    # Risk / failure parameters
    recon_risk = 0.27
    failures = 3
    failure_thresh = 10
    recovery_prio = 0.15

    # Feature probability vector (must sum to 1)
    probs = np.array([0.2, 0.15, 0.25, 0.1, 0.3])

    # Simple undirected graph (5 nodes)
    adj_matrix = np.array(
        [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ],
        dtype=float,
    )

    # Actions and a deterministic mapping to graph nodes
    actions = ["ALERT", "MITIGATE", "IGNORE", "ESCALATE"]
    action_to_node = {i: i % adj_matrix.shape[0] for i in range(len(actions))}
    bandit = HybridBandit(actions, action_to_node)

    # Distribute dates among actions (simple round‑robin for demo)
    action_date_map = {i: [] for i in range(len(actions))}
    for idx, d in enumerate(sample_dates):
        action_date_map[idx % len(actions)].append(d)

    # Run the pipeline a few times to see learning in action
    for _ in range(20):
        chosen = hybrid_decision_pipeline(
            dates=sample_dates,
            reconstruction_risk_score=recon_risk,
            failures=failures,
            failure_threshold=failure_thresh,
            recovery_priority=recovery_prio,
            feature_probs=probs,
            adjacency=adj_matrix,
            bandit=bandit,
            action_date_map=action_date_map,
        )
        print(f"Chosen action: {chosen}")

    print("Final bandit state:", bandit)