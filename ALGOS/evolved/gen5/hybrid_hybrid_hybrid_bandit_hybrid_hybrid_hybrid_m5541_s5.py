# DARWIN HAMMER — match 5541, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_bandit_m534_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m42_s1.py (gen4)
# born: 2026-05-30T00:02:47Z

"""
Hybrid Algorithm: Bandit‑Sheaf Temperature Fusion

Parents:
- hybrid_hybrid_bandit_router_hybrid_hybrid_bandit_m534_s2 (Bandit router + Schoolfield temperature model)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m42_s1 (Cellular sheaf + Bayesian VRAM scheduler)

Mathematical Bridge:
The bridge is built on two scalar quantities that are naturally produced by each parent:

1. **Entropy of the bandit propensity distribution**  
   \(H(p) = -\sum_i p_i \log p_i\) captures the uncertainty (exploration) of the bandit
   policy.

2. **Temperature‑dependent scaling from the Schoolfield model**  
   \(S(T) = \rho_{25}\frac{T}{298.15}\exp\!\Big(\frac{\Delta H_{act}}{R}
   ( \frac{1}{298.15} - \frac{1}{T})\Big) /
   \bigl(1+\exp(\frac{\Delta H_{low}}{R}( \frac{1}{T_{low}}-\frac{1}{T}))+
        \exp(\frac{\Delta H_{high}}{R}( \frac{1}{T}-\frac{1}{T_{high}}))\bigr)\)

The hybrid algorithm multiplies these two scalars to obtain a **temperature‑entropy
scale** \(\tau = H(p)\,S(T)\).  This scale is used in two places:

* **Sheaf restriction update** – each restriction matrix is multiplied by \(\tau\),
  making the sheaf “warmer” (more permissive) when the bandit is uncertain or the
  ambient temperature is high.
* **VRAM allocation** – the estimated memory for each artifact is weighted by
  \(\tau\) and the bandit’s expected reward, yielding a Bayesian‑inspired,
  temperature‑aware planner.

The code below implements this fusion with three core functions:
`temperature_scaled_reward`, `sheaf_update_with_bandit`, and `plan_vram_from_bandit`.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # probability of selection (p_i)
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # gas constant in cal·K⁻¹·mol⁻¹


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def schoolfield_rate(temp_k: float, params: SchoolfieldParams) -> float:
    """
    Compute the temperature‑dependent developmental rate using the
    Schoolfield model.
    """
    if temp_k <= 0:
        raise ValueError("Temperature must be > 0 K")
    num = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * (1 / 298.15 - 1 / temp_k)
    )
    low_term = math.exp(
        (params.delta_h_low / params.r_cal) * (1 / params.t_low - 1 / temp_k)
    )
    high_term = math.exp(
        (params.delta_h_high / params.r_cal) * (1 / temp_k - 1 / params.t_high)
    )
    den = 1 + low_term + high_term
    return num / den


# ----------------------------------------------------------------------
# Data structures from Parent B
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Sheaf:
    """
    Cellular sheaf on a directed graph.

    * Nodes carry a vector space of dimension given by ``node_dims``.
    * Each directed edge (u, v) carries a linear restriction map
      src_map : ℝ^{dim(u)} → ℝ^{dim(e)} and
      dst_map : ℝ^{dim(v)} → ℝ^{dim(e)}.
    * A *section* assigns a vector to every node.
    """

    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[Any, Any], src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column size must match dim of node u")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column size must match dim of node v")
        self._restrictions[edge] = (src_map, dst_map)

    def get_restriction(self, edge: Tuple[Any, Any]) -> Tuple[np.ndarray, np.ndarray]:
        return self._restrictions[edge]

    def set_section(self, node: Any, vector: np.ndarray) -> None:
        if vector.shape[0] != self.node_dims[node]:
            raise ValueError("section vector length must match node dimension")
        self._sections[node] = vector

    def get_section(self, node: Any) -> np.ndarray:
        return self._sections[node]

    def all_sections(self) -> Dict[Any, np.ndarray]:
        return self._sections


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def temperature_scaled_reward(action: BanditAction, temp_c: float,
                              params: SchoolfieldParams) -> float:
    """
    Compute a reward that is modulated by the temperature‑dependent scale S(T).
    """
    temp_k = c_to_k(temp_c)
    scale = schoolfield_rate(temp_k, params)
    return action.expected_reward * scale


def compute_propensity_entropy(actions: List[BanditAction]) -> float:
    """
    Shannon entropy of the propensity distribution of a list of actions.
    """
    total = sum(a.propensity for a in actions)
    if total == 0:
        return 0.0
    probs = [a.propensity / total for a in actions]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)  # avoid log(0)
    return entropy


def sheaf_update_with_bandit(sheaf: Sheaf, actions: List[BanditAction],
                             temp_c: float, params: SchoolfieldParams) -> None:
    """
    Update every restriction map of the sheaf by multiplying it with the
    temperature‑entropy scale τ = H(p) * S(T).

    This makes the sheaf more flexible when the bandit is uncertain (high entropy)
    or the environment is warm (high temperature scaling).
    """
    entropy = compute_propensity_entropy(actions)
    temp_k = c_to_k(temp_c)
    temp_scale = schoolfield_rate(temp_k, params)
    tau = entropy * temp_scale

    for edge in sheaf.edges:
        src_map, dst_map = sheaf.get_restriction(edge)
        # Scale both restriction matrices in‑place
        sheaf._restrictions[edge] = (src_map * tau, dst_map * tau)


def plan_vram_from_bandit(actions: List[BanditAction],
                          total_mb: int,
                          temp_c: float,
                          params: SchoolfieldParams) -> List[VramSlotPlan]:
    """
    Produce a VRAM allocation plan where each artifact receives memory proportional to

        weight_i = τ * expected_reward_i * propensity_i

    The τ factor is the same temperature‑entropy scale used for the sheaf.
    """
    entropy = compute_propensity_entropy(actions)
    temp_k = c_to_k(temp_c)
    temp_scale = schoolfield_rate(temp_k, params)
    tau = entropy * temp_scale

    # Compute raw weights
    raw_weights = []
    for a in actions:
        w = tau * a.expected_reward * a.propensity
        raw_weights.append(max(w, 0.0))

    total_weight = sum(raw_weights) + 1e-12  # avoid division by zero
    plans: List[VramSlotPlan] = []
    for a, w in zip(actions, raw_weights):
        mb = int(round((w / total_weight) * total_mb))
        plan = VramSlotPlan(
            artifact_id=a.action_id,
            artifact_kind="bandit_action",
            action="allocate",
            estimated_mb=mb,
            reason="temperature‑entropy weighted bandit reward",
            detail={
                "propensity": a.propensity,
                "expected_reward": a.expected_reward,
                "tau": tau,
                "raw_weight": w,
            },
        )
        plans.append(plan)
    return plans


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------


def _smoke_test() -> None:
    # Create a few dummy bandit actions
    actions = [
        BanditAction("a1", propensity=0.5, expected_reward=1.2, confidence_bound=0.1, algorithm="A"),
        BanditAction("a2", propensity=0.3, expected_reward=0.8, confidence_bound=0.2, algorithm="A"),
        BanditAction("a3", propensity=0.2, expected_reward=1.5, confidence_bound=0.15, algorithm="A"),
    ]

    # Temperature parameters
    params = SchoolfieldParams()
    temp_c = 25.0  # room temperature

    # Demonstrate temperature‑scaled reward
    scaled_rewards = [temperature_scaled_reward(a, temp_c, params) for a in actions]
    print("Temperature‑scaled rewards:", scaled_rewards)

    # Build a tiny sheaf (2 nodes, 1 edge)
    node_dims = {"n1": 3, "n2": 2}
    edges = [("n1", "n2")]
    sheaf = Sheaf(node_dims, edges)
    src_map = np.random.randn(4, 3)  # dim(e)=4, dim(n1)=3
    dst_map = np.random.randn(4, 2)  # dim(e)=4, dim(n2)=2
    sheaf.set_restriction(("n1", "n2"), src_map, dst_map)

    # Update sheaf with bandit information
    sheaf_update_with_bandit(sheaf, actions, temp_c, params)
    updated_src, updated_dst = sheaf.get_restriction(("n1", "n2"))
    print("Updated restriction matrices (norms):",
          np.linalg.norm(updated_src), np.linalg.norm(updated_dst))

    # VRAM planning
    total_mb = 1024
    plans = plan_vram_from_bandit(actions, total_mb, temp_c, params)
    for p in plans:
        print(p.as_dict())


if __name__ == "__main__":
    _smoke_test()