# DARWIN HAMMER — match 5395, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m217_s3.py (gen4)
# parent_b: hybrid_hybrid_poikilotherm__hybrid_pheromone_hyb_m2357_s0.py (gen4)
# born: 2026-05-30T00:01:36Z

"""
Hybrid algorithm combining:

- **Parent A**: Dense Associative Memory (DAM) with matrix‑based pattern storage and
  retrieval, and a Sheaf structure for restriction maps.
- **Parent B**: Poikilotherm‑Tree temperature scaling (`normalized_activity`) and
  surface pheromone dynamics (`PheromoneStore`).

**Mathematical bridge**  
Both parents rely on linear algebra.  DAM updates its weight matrix `W` with
outer‑products of pattern vectors.  The Poikilotherm model provides a
temperature‑dependent activity factor `a(T)` (Schoolfield equation) that can be
used as a scalar multiplier for any matrix update.  Likewise, pheromone decay
half‑life can be modulated by `a(T)`.  The hybrid therefore:

1. Scales the DAM Hebbian update `ΔW = a(T) * Σ p_i p_iᵀ`.
2. Scales pheromone half‑life `τ(T) = τ₀ / a(T)`.
3. During inference, combines the DAM recall vector with a pheromone‑derived
   bias vector obtained from the store.

The three core functions below illustrate this fused behaviour.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Sheaf and Dense Associative Memory (simplified)
# ----------------------------------------------------------------------
class Sheaf:
    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[Any, Any], src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node: Any, value: np.ndarray) -> None:
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("Section dimension must match node dimension")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: Any) -> np.ndarray:
        return self._sections.get(node)

    def get_restriction(self, edge: Tuple[Any, Any]) -> Tuple[np.ndarray, np.ndarray]:
        return self._restrictions.get(edge)


class DenseAssociativeMemory:
    """
    Simple binary DAM:
    - `patterns` is a (n_patterns, dim) array.
    - Weight matrix `W` = Σ p_i p_iᵀ (Hebbian learning).
    - Retrieval: sign(W @ x).
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.W = np.zeros((dim, dim), dtype=float)

    def store_patterns(self, patterns: np.ndarray, scale: float = 1.0) -> None:
        """
        Update the weight matrix with scaled Hebbian outer products.
        `scale` can be temperature‑dependent.
        """
        if patterns.ndim != 2 or patterns.shape[1] != self.dim:
            raise ValueError("Patterns must be a 2‑D array with matching dimension.")
        for p in patterns:
            self.W += scale * np.outer(p, p)

    def recall(self, cue: np.ndarray) -> np.ndarray:
        """
        Return the sign of the weighted sum.  Zero entries are left as 0.
        """
        if cue.shape != (self.dim,):
            raise ValueError("Cue vector dimension mismatch.")
        raw = self.W @ cue
        return np.sign(raw)

# ----------------------------------------------------------------------
# Parent B – Poikilotherm temperature scaling and pheromone dynamics
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal mol⁻¹ K⁻¹
K25 = 298.15   # reference temperature (K)

def normalized_activity(T: float, rho_25: float = 1.0, delta_h_activation: float = 1200.0) -> float:
    """
    Schoolfield temperature response:
        a(T) = rho_25 * exp( -ΔH/(R) * (1/T - 1/K25) )
    Returns a dimensionless activity factor.
    """
    if T <= 0:
        raise ValueError("Temperature must be positive Kelvin.")
    exponent = -delta_h_activation / R_CAL * (1.0 / T - 1.0 / K25)
    return rho_25 * math.exp(exponent)

Node = Any
Graph = Dict[Node, Set[Node]]

class PheromoneStore:
    """
    Minimal in‑memory pheromone table.
    Each record stores a concentration that decays exponentially with half‑life τ.
    """
    def __init__(self) -> None:
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    def signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
        detail: str | None = None,
    ) -> str:
        rec = {
            "pheromone_id": f"{surface_key}:{len(self._store.get(surface_key, []))}",
            "surface_key": surface_key,
            "kind": signal_kind,
            "value": float(signal_value),
            "half_life": float(half_life_seconds),
            "timestamp": random.random(),  # placeholder for time
            "detail": detail,
        }
        self._store.setdefault(surface_key, []).append(rec)
        return rec["pheromone_id"]

    def concentration(self, surface_key: str, current_time: float) -> float:
        """
        Sum decayed contributions for a given key.
        """
        total = 0.0
        for rec in self._store.get(surface_key, []):
            elapsed = max(0.0, current_time - rec["timestamp"])
            decay = 0.5 ** (elapsed / rec["half_life"])
            total += rec["value"] * decay
        return total

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def temperature_scaled_memory_update(
    dam: DenseAssociativeMemory,
    patterns: np.ndarray,
    temperature_K: float,
) -> None:
    """
    Update DAM weights using temperature‑scaled activity a(T) as the Hebbian gain.
    """
    a_T = normalized_activity(temperature_K)
    dam.store_patterns(patterns, scale=a_T)


def pheromone_signal_with_temperature(
    store: PheromoneStore,
    surface_key: str,
    base_half_life: float,
    temperature_K: float,
    signal_value: float = 1.0,
) -> str:
    """
    Emit a pheromone whose half‑life is inversely proportional to activity a(T):
        τ(T) = base_half_life / a(T)
    """
    a_T = normalized_activity(temperature_K)
    if a_T == 0:
        raise ZeroDivisionError("Activity factor a(T) evaluated to zero.")
    adjusted_half_life = base_half_life / a_T
    return store.signal(
        surface_key=surface_key,
        signal_kind="hybrid",
        signal_value=signal_value,
        half_life_seconds=adjusted_half_life,
        detail=f"temp={temperature_K:.1f}K",
    )


def hybrid_inference(
    dam: DenseAssociativeMemory,
    store: PheromoneStore,
    cue: np.ndarray,
    temperature_K: float,
    surface_key: str,
    current_time: float,
) -> np.ndarray:
    """
    Perform DAM recall, then bias the result with the pheromone concentration
    associated to `surface_key`.  The bias is scaled by activity a(T).
    """
    raw_recall = dam.recall(cue)
    pheromone_conc = store.concentration(surface_key, current_time)
    a_T = normalized_activity(temperature_K)
    bias = a_T * pheromone_conc
    # Broadcast bias to vector length; here we simply add a uniform offset.
    return raw_recall + bias


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    dim = 8
    n_patterns = 5
    temperature = 310.0  # ~37 °C in Kelvin
    base_half_life = 30.0  # seconds
    surface = "node_A"

    # Random binary patterns (+1 / -1)
    rng = np.random.default_rng(42)
    patterns = rng.choice([-1, 1], size=(n_patterns, dim))

    # Initialise components
    dam = DenseAssociativeMemory(dim=dim)
    store = PheromoneStore()

    # 1. Temperature‑scaled memory update
    temperature_scaled_memory_update(dam, patterns, temperature)

    # 2. Emit a pheromone signal with temperature‑adjusted decay
    pid = pheromone_signal_with_temperature(
        store,
        surface_key=surface,
        base_half_life=base_half_life,
        temperature_K=temperature,
        signal_value=2.0,
    )
    print(f"Pheromone record created: {pid}")

    # Simulate a small time advance
    simulated_time = 5.0  # arbitrary units; here we use the random timestamp as proxy
    # 3. Hybrid inference
    cue = rng.choice([-1, 1], size=(dim,))
    output = hybrid_inference(
        dam,
        store,
        cue,
        temperature_K=temperature,
        surface_key=surface,
        current_time=simulated_time,
    )
    print("Cue vector:", cue)
    print("Hybrid output:", output)
    # Ensure code runs without exception
    sys.exit(0)