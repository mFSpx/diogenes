# DARWIN HAMMER — match 11, survivor 5
# gen: 3
# parent_a: physarum_network.py (gen0)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py (gen2)
# born: 2026-05-29T23:25:06Z

"""Hybrid algorithm merging Physarum flux conductance dynamics (Parent A) with a contextual bandit/VRAM scheduler (Parent B).

Mathematical bridge
-------------------
- Each *BanditAction* is interpreted as an edge in a flow network.
- The edge has a **conductance** 𝒞 (from Parent A) and a **length** ℓ (a tunable hyper‑parameter).
- Node pressures are derived from the bandit’s *expected_reward* (up‑stream) and a global
  *baseline_pressure* (down‑stream).  
  The pressure difference Δp = p_up – p_down drives a **flux**
  
  q = 𝒞 / ℓ * Δp                # Eq. (1) – identical to `physarum_network.flux`
  
- The conductance evolves according to the absolute flux, exactly as in
  `physarum_network.update_conductance`:
  
  𝒞 ← max(0, 𝒞 + dt·(gain·|q| – decay·𝒞))   # Eq. (2)
  
- The updated conductance feeds back into the bandit policy: higher conductance
  raises the *propensity* (inflow) of the corresponding action, while the
  *confidence_bound* (outflow) is reduced proportionally.

Thus the two parent topologies are fused: the bandit’s stochastic decision layer
provides pressures; the Physarum‐style differential equation updates a latent
conductance matrix that, in turn, modulates the bandit’s propensities.

The module implements the hybrid dynamics, exposing three core functions:
`flux`, `update_conductance`, and `hybrid_bandit_step`.  A lightweight
`HybridBanditNetwork` class orchestrates actions, stores, and the conductance
matrix.  The `__main__` block runs a smoke test.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent A primitives (re‑implemented for self‑containment)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0,
                       gain: float = 1.0, decay: float = 0.05) -> float:
    """Physarum conductance update."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


# ----------------------------------------------------------------------
# Parent B data structures (simplified)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision – also an edge in the hybrid network."""
    action_id: str
    propensity: float            # inflow rate (used as upstream pressure)
    expected_reward: float
    confidence_bound: float      # outflow rate (used as downstream pressure)
    algorithm: str = "Hybrid"


@dataclass(frozen=True)
class BanditUpdate:
    """Observed reward for a given action."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
class HybridBanditNetwork:
    """
    Maintains a set of actions, a conductance matrix linking them,
    and a simple VRAM‑style store that modulates learning rates.
    """

    def __init__(
        self,
        action_ids: List[str],
        edge_lengths: Optional[Dict[Tuple[str, str], float]] = None,
        dt: float = 1.0,
        gain: float = 1.0,
        decay: float = 0.05,
        base_eta: float = 0.01,
        store_decay: float = 0.99,
    ) -> None:
        """
        Parameters
        ----------
        action_ids : list of unique identifiers for actions/edges.
        edge_lengths : optional explicit lengths; missing pairs default to 1.0.
        dt, gain, decay : Physarum ODE parameters.
        base_eta : baseline learning rate for bandit propensity updates.
        store_decay : exponential decay of the internal VRAM store.
        """
        self.actions: Dict[str, BanditAction] = {
            aid: BanditAction(
                action_id=aid,
                propensity=1.0,               # uniform start
                expected_reward=0.0,
                confidence_bound=1.0,
            )
            for aid in action_ids
        }
        self.dt = dt
        self.gain = gain
        self.decay = decay
        self.base_eta = base_eta
        self.store_decay = store_decay

        # Conductance matrix indexed by (src, dst). For a single‑edge model we
        # keep only self‑loops; the code nevertheless supports full matrices.
        self.conductance: Dict[Tuple[str, str], float] = {}
        self.edge_length: Dict[Tuple[str, str], float] = {}

        for src in action_ids:
            for dst in action_ids:
                key = (src, dst)
                self.conductance[key] = 0.1  # small positive seed
                self.edge_length[key] = (
                    edge_lengths.get(key, 1.0) if edge_lengths else 1.0
                )

        # Simple scalar store that influences learning rate.
        self.store: float = 0.0

    # ------------------------------------------------------------------
    # Helper: compute pressures for a given action
    # ------------------------------------------------------------------
    def _pressures(self, act: BanditAction) -> Tuple[float, float]:
        """
        Upstream pressure = propensity (interpreted as inflow).
        Downstream pressure = confidence_bound (interpreted as outflow).
        """
        return act.propensity, act.confidence_bound

    # ------------------------------------------------------------------
    # Core hybrid step
    # ------------------------------------------------------------------
    def hybrid_bandit_step(self, updates: List[BanditUpdate]) -> None:
        """
        Perform one hybrid iteration:
        1. Incorporate observed rewards → update expected_reward.
        2. Compute fluxes for every (src, dst) edge using Eq.(1).
        3. Update conductances via Eq.(2).
        4. Modulate propensities with the new conductances and a learning‑rate
           that is scaled by the internal store.
        5. Decay the store.
        """
        # 1. Update expected rewards (simple exponential moving average)
        for upd in updates:
            act = self.actions[upd.action_id]
            # EMA factor α = 0.1 (hard‑coded for brevity)
            new_reward = 0.9 * act.expected_reward + 0.1 * upd.reward
            self.actions[upd.action_id] = BanditAction(
                action_id=act.action_id,
                propensity=act.propensity,
                expected_reward=new_reward,
                confidence_bound=act.confidence_bound,
                algorithm=act.algorithm,
            )

        # 2. Compute fluxes & 3. update conductances
        fluxes: Dict[Tuple[str, str], float] = {}
        for (src, dst), 𝒞 in self.conductance.items():
            src_act = self.actions[src]
            dst_act = self.actions[dst]
            p_up, _ = self._pressures(src_act)
            _, p_down = self._pressures(dst_act)
            q = flux(𝒞, self.edge_length[(src, dst)], p_up, p_down)
            fluxes[(src, dst)] = q
            new_𝒞 = update_conductance(
                conductance=𝒞,
                q=q,
                dt=self.dt,
                gain=self.gain,
                decay=self.decay,
            )
            self.conductance[(src, dst)] = new_𝒞

        # 4. Propensity modulation
        #   propensity_i ← propensity_i + η·Σ_j 𝒞_{ij}·|q_{ij}|
        #   η is scaled by the internal store (higher store → larger step)
        eta = self.base_eta * (1.0 + self.store)
        for aid in self.actions:
            inflow = 0.0
            for other in self.actions:
                inflow += self.conductance[(other, aid)] * abs(fluxes[(other, aid)])
            act = self.actions[aid]
            new_propensity = max(0.0, act.propensity + eta * inflow)
            # Confidence bound is reduced proportionally to total outflow
            outflow = 0.0
            for other in self.actions:
                outflow += self.conductance[(aid, other)] * abs(fluxes[(aid, other)])
            new_confidence = max(0.1, act.confidence_bound - eta * outflow)
            self.actions[aid] = BanditAction(
                action_id=aid,
                propensity=new_propensity,
                expected_reward=act.expected_reward,
                confidence_bound=new_confidence,
                algorithm=act.algorithm,
            )

        # 5. Store decay / accumulation
        #   store ← store_decay·store + Σ_i (reward_i – expected_reward_i)
        reward_error = sum(
            upd.reward - self.actions[upd.action_id].expected_reward for upd in updates
        )
        self.store = self.store_decay * self.store + reward_error

    # ------------------------------------------------------------------
    # Policy: sample an action proportionally to propensity
    # ------------------------------------------------------------------
    def select_action(self) -> BanditAction:
        """Stochastic selection based on current propensities."""
        ids = list(self.actions.keys())
        props = np.array([self.actions[i].propensity for i in ids], dtype=float)
        if props.sum() == 0:
            probs = np.full_like(props, 1.0 / len(props))
        else:
            probs = props / props.sum()
        choice = self.rng.choice(ids, p=probs)
        return self.actions[choice]

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the network."""
        return {
            "actions": {k: asdict(v) for k, v in self.actions.items()},
            "conductance": {
                f"{src}->{dst}": val for (src, dst), val in self.conductance.items()
            },
            "store": self.store,
        }

# ----------------------------------------------------------------------
# Additional stand‑alone functions demonstrating hybrid operation
# ----------------------------------------------------------------------
def simulate_hybrid(num_steps: int = 10) -> List[Dict[str, Any]]:
    """
    Run a short simulation with three dummy actions.
    Returns a list of state snapshots for inspection.
    """
    hb = HybridBanditNetwork(action_ids=["A", "B", "C"])
    history = []
    rng = np.random.default_rng(42)

    for step in range(num_steps):
        # Randomly generate a synthetic reward update for each action
        updates = []
        for aid in hb.actions:
            reward = rng.normal(loc=hb.actions[aid].expected_reward, scale=1.0)
            updates.append(
                BanditUpdate(
                    context_id="ctx",
                    action_id=aid,
                    reward=reward,
                    propensity=hb.actions[aid].propensity,
                )
            )
        hb.hybrid_bandit_step(updates)
        history.append(hb.get_state())
    return history


def export_state_to_json(state: Dict[str, Any], path: Path) -> None:
    """Utility to write a snapshot to a JSON file (uses only stdlib)."""
    import json

    with path.open("w", encoding="utf-8") as fp:
        json.dump(state, fp, indent=2, sort_keys=True)


def main_demo() -> None:
    """Execute a quick demo and write the final state to a temporary file."""
    snapshots = simulate_hybrid(num_steps=15)
    final_state = snapshots[-1]
    tmp_path = Path("hybrid_demo_final_state.json")
    export_state_to_json(final_state, tmp_path)
    print(f"Demo finished – final state written to {tmp_path.resolve()}")
    # Show a few numbers for sanity
    for aid, act in final_state["actions"].items():
        print(f"Action {aid}: propensity={act['propensity']:.3f}, "
              f"expected_reward={act['expected_reward']:.3f}, "
              f"confidence_bound={act['confidence_bound']:.3f}")


if __name__ == "__main__":
    # Smoke test – should run without exceptions
    main_demo()