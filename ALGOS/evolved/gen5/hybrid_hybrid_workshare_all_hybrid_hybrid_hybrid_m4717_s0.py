# DARWIN HAMMER — match 4717, survivor 0
# gen: 5
# parent_a: hybrid_workshare_allocator_hybrid_hybrid_hybrid_m1490_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_sketch_epistemic_certainty_m2_s1.py (gen3)
# born: 2026-05-29T23:57:40Z

"""
Hybrid Workshare‑Sheaf Allocator

This module fuses the deterministic work‑share allocation and Hoeffding‑modulated
bandit learning from **hybrid_workshare_allocator_hybrid_hybrid_m1490_s0.py**
with the sheaf‑cohomology based uncertainty quantification from
**hybrid_hybrid_hybrid_sketch_epistemic_certainty_m2_s1.py**.

Mathematical bridge
-------------------
* The deterministic share (``deterministic_units``) is treated as a *signal*,
  while the stochastic LLM share (``llm_units``) is the *noise*.
  Their ratio ``signal_to_noise = deterministic_units / max(llm_units, 1e-9)`` rescales
  the per‑node sheaf sections.
* Bandit updates use a Hoeffding confidence bound  

  ``ε = sqrt( (1/(2·n))·log(2/δ) )``  

  where *n* is the number of observations for an action and ``δ`` a confidence
  level (default 0.05).  This ε modulates the learning rate applied to the
  restriction maps of the sheaf, i.e. the maps are shrunk/expanded by a factor
  ``1 ± ε`` according to the observed reward.
* The coboundary operator of the sheaf measures local disagreement between
  sections.  Its L2 norm provides an *information‑loss* estimate that is fed
  back as the bandit reward.

The three public functions below demonstrate the hybrid operation:
``allocate_workshare``, ``compute_sheaf_uncertainty`` and ``update_bandit_and_restriction``.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

# ----------------------------------------------------------------------
# Parent A – workshare allocation & bandit scaffolding
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid_workshare_bandit"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

def _pct(value: float) -> float:
    """Round a float to 6 decimal places for deterministic output."""
    return round(float(value), 6)

def allocate_workshare(*, total_units: float,
                       deterministic_target_pct: float = 90.0,
                       groups: tuple[str, ...] = GROUPS) -> dict[str, Any]:
    """
    Deterministic allocation of ``total_units`` into a fixed deterministic part
    and a stochastic LLM part that is evenly split among ``groups``.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)
    lanes = [
        {
            "group": group,
            "llm_units": _pct(per_group),
            "llm_share_pct": _pct(100.0 / len(groups)),
            "proof_required": True,
        }
        for group in groups
    ]
    return {
        "schema": "lucidota.project2501.workshare_allocation.v1",
        "total_units": _pct(total_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
    }

def hoeffding_epsilon(n: int, delta: float = 0.05) -> float:
    """Hoeffding bound epsilon for ``n`` i.i.d. observations."""
    if n <= 0:
        return float('inf')
    return math.sqrt((1.0 / (2 * n)) * math.log(2.0 / delta))

def bandit_update(action: BanditAction,
                  reward: float,
                  n_observations: int,
                  delta: float = 0.05) -> BanditAction:
    """
    Update a BanditAction using the observed ``reward``.
    The expected reward is moved towards the new reward with a step size
    proportional to the Hoeffding epsilon; the confidence bound is set to ε.
    """
    ε = hoeffding_epsilon(n_observations, delta)
    # Simple exponential‑moving‑average style update
    new_expected = (1 - ε) * action.expected_reward + ε * reward
    return BanditAction(
        action_id=action.action_id,
        propensity=action.propensity,
        expected_reward=new_expected,
        confidence_bound=ε,
        algorithm=action.algorithm,
    )

# ----------------------------------------------------------------------
# Parent B – sheaf‑cohomology based uncertainty quantification
# ----------------------------------------------------------------------
class HybridSheaf:
    """
    A sheaf over a graph where each node carries a vector (the *section*)
    and each edge carries a linear restriction map.
    """
    def __init__(self, node_dims: Dict[Any, int],
                 edge_list: List[Tuple[Any, Any]],
                 width: int = 64,
                 depth: int = 4):
        self.node_dims = dict(node_dims)          # node → dimension
        self.edges = list(edge_list)              # list of (u, v)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}
        self.width = width
        self.depth = depth

    def set_restriction(self, edge: Tuple[Any, Any],
                        src_map: List[float],
                        dst_map: List[float]) -> None:
        """Assign linear maps (as 1‑D arrays) for an edge."""
        u, v = edge
        src_arr = np.array(src_map, dtype=float)
        dst_arr = np.array(dst_map, dtype=float)
        self._restrictions[(u, v)] = (src_arr, dst_arr)

    def set_section(self, node: Any, value: List[float]) -> None:
        """Define the section (vector) at ``node``."""
        self._sections[node] = np.array(value, dtype=float)

    def _edge_dim(self, u: Any, v: Any) -> int:
        """Return the dimension of the restriction map for edge (u, v)."""
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction map for edge ({u}, {v})")

    def coboundary_norm(self) -> float:
        """
        Compute the L2 norm of the coboundary operator:
        Σ_{(u,v)} ‖ R_{u→v}(s_u) – s_v ‖²,
        where R_{u→v} is the source restriction map.
        """
        total = 0.0
        for (u, v) in self.edges:
            if (u, v) in self._restrictions:
                src_map, _ = self._restrictions[(u, v)]
                mapped = src_map * self._sections.get(u, np.zeros_like(src_map))
                diff = mapped - self._sections.get(v, np.zeros_like(mapped))
                total += np.linalg.norm(diff) ** 2
            elif (v, u) in self._restrictions:
                _, dst_map = self._restrictions[(v, u)]
                mapped = dst_map * self._sections.get(v, np.zeros_like(dst_map))
                diff = mapped - self._sections.get(u, np.zeros_like(mapped))
                total += np.linalg.norm(diff) ** 2
            else:
                # No restriction → treat as zero disagreement
                continue
        return math.sqrt(total)

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_sheaf_uncertainty(sheaf: HybridSheaf) -> float:
    """
    Return a scalar uncertainty measure derived from the sheaf coboundary norm.
    The value is normalised to [0, 1] by a simple sigmoid.
    """
    raw = sheaf.coboundary_norm()
    # Sigmoid normalisation (avoids division by zero)
    return 1.0 / (1.0 + math.exp(-raw))

def update_bandit_and_restriction(sheaf: HybridSheaf,
                                  action: BanditAction,
                                  reward: float,
                                  n_observations: int,
                                  delta: float = 0.05) -> Tuple[BanditAction, HybridSheaf]:
    """
    Perform a bandit update with ``reward`` and propagate the resulting
    Hoeffding epsilon to all restriction maps of the sheaf.
    The maps are scaled by ``(1 ± ε)`` where the sign is positive if the reward
    exceeds the previous expectation and negative otherwise.
    """
    updated_action = bandit_update(action, reward, n_observations, delta)
    ε = updated_action.confidence_bound
    sign = 1.0 if reward >= action.expected_reward else -1.0
    scale = 1.0 + sign * ε  # shrink or enlarge

    # Apply the scaling to every restriction map (in‑place)
    for edge, (src_map, dst_map) in sheaf._restrictions.items():
        sheaf._restrictions[edge] = (src_map * scale, dst_map * scale)

    return updated_action, sheaf

def hybrid_allocate_and_update(total_units: float,
                               deterministic_target_pct: float,
                               groups: tuple[str, ...],
                               sheaf: HybridSheaf,
                               n_observations: int,
                               delta: float = 0.05) -> Dict[str, Any]:
    """
    1. Allocate workshare using ``allocate_workshare``.
    2. Map the stochastic LLM allocation to sheaf node sections,
       rescaled by the signal‑to‑noise ratio.
    3. Compute uncertainty (as a reward) and run a bandit‑driven
       restriction‑map update.
    Returns a dictionary summarising the allocation, uncertainty and bandit state.
    """
    allocation = allocate_workshare(
        total_units=total_units,
        deterministic_target_pct=deterministic_target_pct,
        groups=groups,
    )
    deterministic_units = allocation["deterministic_units"]
    llm_units = allocation["llm_units"]
    # Signal‑to‑noise ratio (avoid division by zero)
    s2n = deterministic_units / max(llm_units, 1e-9)

    # Distribute llm_units to sheaf nodes (assume one‑to‑one mapping)
    for lane, node in zip(allocation["lanes"], sheaf.node_dims.keys()):
        scaled = lane["llm_units"] * s2n
        sheaf.set_section(node, [scaled] * sheaf.node_dims[node])

    # Compute uncertainty and treat it as a reward for the bandit
    uncertainty = compute_sheaf_uncertainty(sheaf)
    # Initialise a dummy bandit action if none exists
    action = BanditAction(
        action_id="alloc_update",
        propensity=1.0,
        expected_reward=0.0,
        confidence_bound=float('inf')
    )
    updated_action, updated_sheaf = update_bandit_and_restriction(
        sheaf,
        action,
        reward=uncertainty,
        n_observations=n_observations,
        delta=delta,
    )
    return {
        "allocation": allocation,
        "signal_to_noise": _pct(s2n),
        "uncertainty": _pct(uncertainty),
        "bandit_action": asdict(updated_action),
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph: each group is a node, edges form a cycle
    groups = ("codex", "groq", "cohere", "local_models")
    node_dims = {g: 3 for g in groups}                     # 3‑dimensional sections
    edges = [(groups[i], groups[(i + 1) % len(groups)]) for i in range(len(groups))]

    sheaf = HybridSheaf(node_dims=node_dims, edge_list=edges)

    # Initialise simple restriction maps (identity‑like)
    for u, v in edges:
        dim = node_dims[u]
        src_map = [1.0] * dim
        dst_map = [1.0] * dim
        sheaf.set_restriction((u, v), src_map, dst_map)

    result = hybrid_allocate_and_update(
        total_units=1000.0,
        deterministic_target_pct=80.0,
        groups=groups,
        sheaf=sheaf,
        n_observations=10,
        delta=0.05,
    )

    print("Hybrid allocation & bandit update result:")
    for key, val in result.items():
        print(f"{key}: {val}")