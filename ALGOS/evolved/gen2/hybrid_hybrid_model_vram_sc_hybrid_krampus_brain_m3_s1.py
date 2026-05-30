# DARWIN HAMMER — match 3, survivor 1
# gen: 2
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s4.py (gen1)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s0.py (gen1)
# born: 2026-05-29T23:22:36Z

"""
Module fusing the VRAM planner from hybrid_model_vram_scheduler_ttt_linear_m11_s4 with 
the Krampus-Ollivier-Ricci curvature algorithm from hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s0.
The mathematical bridge lies in utilizing the VRAM planner's artifact registration mechanism 
to optimize the graph construction in the Krampus-Ollivier-Ricci curvature computation, 
enabling memory-efficient analysis of complex systems with both graph-theoretic and feature-based insights.
"""

import numpy as np
import math
import random
import sys
from collections import deque, defaultdict
from pathlib import Path
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Global constants & helpers
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
DEFAULT_BUDGET_MB = int(os.getenv("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.getenv("LUCIDOTA_VRAM_RESERVE_MB", "768"))
RUNTIME_DIR = ROOT / "04_RUNTIME" / "inference_os"

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)

def _append_runtime_receipt(receipt: Dict[str, Any], *, path: Path | None = None) -> None:
    target = path or (RUNTIME_DIR / "preemption_receipts.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, default=str) + "\n")

def _query_nvidia_smi() -> Dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"status": "missing", "message": "nvidia-smi not found"}
    cp = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,pstate",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    if cp.returncode != 0 or not cp.stdout.strip():
        return {"status": "error", "stderr": cp.stderr[-500:]}
    gpus: List[Dict[str, Any]] = []
    for line in cp.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, total, used, free, driver, pstate = parts[:7]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "total_mb": int(total),
                "used_mb": int(used),
                "free_mb": int(free),
                "driver_version": driver,
                "pstate": pstate,
            }
        )
    if not gpus:
        return {"status": "error", "stdout": cp.stdout[-500:], "stderr": cp.stderr[-500:]}
    return {"status": "ok", "selected_index": gpus[0]["index"], **gpus[0], "gpus": gpus}

# ----------------------------------------------------------------------
# VRAM planning core – deeper integration
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VramPlanner:
    """Tracks VRAM usage of multiple artifacts and provides a budget‑aware API."""

    def __init__(self, static_budget_mb: int = DEFAULT_BUDGET_MB, reserve_mb: int = DEFAULT_RESERVE_MB):
        self.static_budget_mb = static_budget_mb
        self.reserve_mb = reserve_mb
        self._artifacts: Dict[str, VramSlotPlan] = {}
        self._last_gpu_query: Dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # GPU query caching (avoid repeated nvidia‑smi calls)
    # ------------------------------------------------------------------
    def _gpu_info(self) -> Dict[str, Any]:
        if self._last_gpu_query is None:
            self._last_gpu_query = _query_nvidia_smi()
        return self._last_gpu_query

    # ------------------------------------------------------------------
    # Public budgeting helpers
    # ------------------------------------------------------------------
    def total_committed_mb(self) -> int:
        return sum(plan.estimated_mb for plan in self._artifacts.values())

    def available_budget_mb(self) -> int:
        gpu_total = self._gpu_info().get("total_mb", self.static_budget_mb)
        effective_budget = min(self.static_budget_mb, gpu_total) - self.reserve_mb
        return max(effective_budget - self.total_committed_mb(), 0)

    # ------------------------------------------------------------------
    # Artifact registration
    # ------------------------------------------------------------------
    def register(self, plan: VramSlotPlan) -> VramSlotPlan:
        """Add or replace an artifact plan. Returns the stored plan."""
        self._artifacts[plan.artifact_id] = plan
        receipt = {
            "timestamp": now_z(),
            "event": "register_artifact",
            "artifact_id": plan.artifact_id,
            "action": plan.action,
            "estimated_mb": plan.estimated_mb,
            "available_budget_mb": self.available_budget_mb(),
        }
        _append_runtime_receipt(receipt)
        return plan

    def unregister(self, artifact_id: str) -> None:
        if artifact_id in self._artifacts:
            del self._artifacts[artifact_id]
            receipt = {
                "timestamp": now_z(),
                "event": "unregister_artifact",
                "artifact_id": artifact_id,
                "available_budget_mb": self.available_budget_mb(),
            }
            _append_runtime_receipt(receipt)

# ----------------------------------------------------------------------
# Krampus-Ollivier-Ricci curvature helpers
# ----------------------------------------------------------------------
def lazy_rw_distribution(adj, node, alpha=0.5):
    """Lazy random walk distribution centred at *node*.

    Parameters
    ----------
    adj   : dict mapping node_id -> list of neighbour node_ids
    node  : the source node
    alpha : mass kept at the node itself (laziness parameter)

    Returns
    -------
    dict mapping node_id -> float probability
    """
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist

def bfs_distances(adj):
    """All-pairs shortest-path distances via BFS.

    Parameters
    ----------
    adj : dict mapping node_id -> list of neighbour node_ids

    Returns
    -------
    D        : 2-D numpy float64 array, shape (n, n)
    node_ids : sorted list of node identifiers (index into D)
    """
    node_ids = sorted(adj.keys())
    n = len(node_ids)
    idx = {v: i for i, v in enumerate(node_ids)}
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    for src in adj:
        si = idx[src]
        visited = {src}
        q = deque([(src, 0)])
        while q:
            node, dist = q.popleft()
            D[si, idx[node]] = dist
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))

    return D, node_ids

def wasserstein1_graph(mu, nu, D, node_ids):
    """Wasserstein-1 distance between discrete measures *mu* and *nu*.

    Uses a greedy iterative reweighting on the cost matrix:
    at each step ship as much mass as possible from the cheapest
    unsatisfied source to its cheapest unsatisfied sink, matching the
    north-west-corner rule on the distance-sorted cost matrix.

    Parameters
    ----------
    mu       : dict node_id -> float  (supply measure, sums to 1)
    nu       : dict node_id -> float  (demand measure, sums to 1)
    D        : 2-D numpy distance matrix (indexed by node_ids order)
    node_ids : ordered list of node identifiers

    Returns
    -------
    float  W_1 distance
    """
    n = len(node_ids)
    idx = {v: i for i, v in enumerate(node_ids)}

    supply = np.array([mu.get(v, 0.0) for v in node_ids], dtype=np.float64)
    demand = np.array([nu.get(v, 0.0) for v in node_ids], dtype=np.float64)

    # normalise numerically to avoid floating drift
    s_sum = supply.sum()
    d_sum = demand.sum()
    if s_sum > 0:
        supply /= s_sum
    if d_sum > 0:
        demand /= d_sum

    # flatten cost matrix entries and sort by distance (ascending)
    costs = []
    for i in range(n):
        for j in range(n):
            costs.append((D[i, j], i, j))
    costs.sort(key=lambda t: t[0])

    total_cost = 0.0
    supply_rem = supply.copy()
    demand_rem = demand.copy()
    eps = 1e-12

    for cost, i, j in costs:
        if supply_rem[i] < eps or demand_rem[j] < eps:
            continue
        flow = min(supply_rem[i], demand_rem[j])
        total_cost += cost * flow
        supply_rem[i] -= flow
        demand_rem[j] -= flow

    return float(total_cost)

def krampus_ollivier_ricci(adj, alpha=0.5):
    """Compute Ollivier-Ricci curvature kappa(x, y) for every edge.

    Parameters
    ----------
    adj   : dict mapping node_id -> list of neighbour node_ids
    alpha : laziness parameter for the random walk (default 0.5)

    Returns
    -------
    dict mapping (x, y) tuple -> float kappa
    """
    D, node_ids = bfs_distances(adj)
    curvatures = {}

    seen = set()
    for x in adj:
        for y in adj[x]:
            edge = (min(x, y), max(x, y))
            if edge in seen:
                continue
            seen.add(edge)

            xi = node_ids.index(x)
            yi = node_ids.index(y)
            d_xy = D[xi, yi]
            if d_xy < 1e-12:
                curvatures[(x, y)] = 1.0
                continue

            mu = lazy_rw_distribution(adj, x, alpha)
            nu = lazy_rw_distribution(adj, y, alpha)
            w1 = wasserstein1_graph(mu, nu, D, node_ids)
            curvatures[(x, y)] = (d_xy - w1) / d_xy

    return curvatures

def hybrid_optimize(vram_planner: VramPlanner, adj: dict, alpha=0.5):
    """Hybrid function optimizing the graph construction using VRAM planner.

    Parameters
    ----------
    vram_planner : VramPlanner instance
    adj          : dict mapping node_id -> list of neighbour node_ids
    alpha        : laziness parameter for the random walk (default 0.5)

    Returns
    -------
    dict mapping (x, y) tuple -> float kappa
    """
    # Register graph construction as an artifact
    graph_plan = VramSlotPlan(
        artifact_id="graph_construction",
        artifact_kind="graph",
        action="construct",
        estimated_mb=sys.getsizeof(adj),
        reason="graph construction",
        detail={"adj": adj},
    )
    vram_planner.register(graph_plan)

    # Compute Ollivier-Ricci curvature
    curvatures = krampus_ollivier_ricci(adj, alpha)

    # Unregister graph construction artifact
    vram_planner.unregister("graph_construction")

    return curvatures

def hybrid_analyze(vram_planner: VramPlanner, adj: dict, alpha=0.5):
    """Hybrid function analyzing the graph using VRAM planner and Ollivier-Ricci curvature.

    Parameters
    ----------
    vram_planner : VramPlanner instance
    adj          : dict mapping node_id -> list of neighbour node_ids
    alpha        : laziness parameter for the random walk (default 0.5)

    Returns
    -------
    tuple containing the curvatures and the available budget
    """
    curvatures = hybrid_optimize(vram_planner, adj, alpha)
    available_budget = vram_planner.available_budget_mb()

    return curvatures, available_budget

def extract_full_features(text: str) -> dict[str, float]:
    features: dict[str, float] = {}
    features["visceral_ratio"] = 0.5
    features["tech_ratio"] = 0.3
    features["legal_osint_ratio"] = 0.2
    features["ledger_density"] = 0.1
    features["recursion_score"] = 0.4
    features["directive_ratio"] = 0.6
    features["target_density"] = 0.7
    features["forensic_shield_ratio"] = 0.8
    features["poetic_entropy"] = 0.9
    features["dissociative_index"] = 0.1
    features["wrath_velocity"] = 0.2
    features["bureaucratic_weaponization_index"] = 0.3
    features["resource_exhaustion_metric"] = 0.4
    features["swarm_orchestration_density"] = 0.5
    features["logic_crucifixion_index"] = 0.6
    features["conspiracy_grounding_ratio"] = 0.7
    features["chaotic_good_tax"] = 0.8
    features["corporate_grit_tension"] = 0.9
    features["countdown_density"] = 0.1
    features["asset_structuring_weight"] = 0.2
    features["pitch_formatting_ratio"] = 0.3
    features["agent_symmetry_ratio"] = 0.4
    features["protocol_discipline"] = 0.5
    features["manic_velocity"] = 0.6
    return features

if __name__ == "__main__":
    # Create a VRAM planner instance
    vram_planner = VramPlanner()

    # Define a sample graph
    adj = {
        "A": ["B", "C"],
        "B": ["A", "D"],
        "C": ["A", "D"],
        "D": ["B", "C"],
    }

    # Run the hybrid optimization function
    curvatures = hybrid_optimize(vram_planner, adj)

    # Print the curvatures
    print("Curvatures:")
    for edge, curvature in curvatures.items():
        print(f"{edge}: {curvature}")

    # Run the hybrid analysis function
    curvatures, available_budget = hybrid_analyze(vram_planner, adj)

    # Print the curvatures and available budget
    print("\nCurvatures and Available Budget:")
    for edge, curvature in curvatures.items():
        print(f"{edge}: {curvature}")
    print(f"Available Budget: {available_budget} MB")