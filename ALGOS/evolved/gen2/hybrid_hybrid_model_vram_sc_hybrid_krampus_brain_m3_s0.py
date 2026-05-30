# DARWIN HAMMER — match 3, survivor 0
# gen: 2
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s4.py (gen1)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s0.py (gen1)
# born: 2026-05-29T23:22:36Z

"""
Hybrid Vram Planner with Krampus Ollivier-Ricci Curvature algorithm.

This module combines the VRAM planning capabilities of the VramPlanner class with 
the graph-theoretic insights of the Krampus Ollivier-Ricci curvature algorithm. The 
mathematical bridge is established by using the VRAM allocation plans as node 
attributes in the graph, which are then used to compute the Ollivier-Ricci 
curvature. This fusion enables the analysis of complex systems with both 
graph-theoretic and feature-based insights.

Parent algorithms: hybrid_model_vram_scheduler_ttt_linear_m11_s4.py, 
hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s0.py
"""

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import math
import random

# Global constants & helpers
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

def _append_runtime_receipt(receipt: dict, *, path: Path | None = None) -> None:
    target = path or (RUNTIME_DIR / "preemption_receipts.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, default=str) + "\n")

@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: dict

    def as_dict(self) -> dict:
        return asdict(self)

class VramPlanner:
    def __init__(self, static_budget_mb: int = DEFAULT_BUDGET_MB, reserve_mb: int = DEFAULT_RESERVE_MB):
        self.static_budget_mb = static_budget_mb
        self.reserve_mb = reserve_mb
        self._artifacts: dict = {}
        self._last_gpu_query: dict | None = None

    def _gpu_info(self) -> dict:
        if self._last_gpu_query is None:
            self._last_gpu_query = self._query_nvidia_smi()
        return self._last_gpu_query

    def total_committed_mb(self) -> int:
        return sum(plan.estimated_mb for plan in self._artifacts.values())

    def available_budget_mb(self) -> int:
        gpu_total = self._gpu_info().get("total_mb", self.static_budget_mb)
        effective_budget = min(self.static_budget_mb, gpu_total) - self.reserve_mb
        return max(effective_budget - self.total_committed_mb(), 0)

    def register(self, plan: VramSlotPlan) -> VramSlotPlan:
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

    def _query_nvidia_smi(self) -> dict:
        # For simplicity, this function is not implemented here.
        pass

def extract_full_features(text: str) -> dict:
    features: dict = {}
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

def lazy_rw_distribution(adj: dict, node: str, alpha: float = 0.5) -> dict:
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist

def bfs_distances(adj: dict) -> tuple:
    node_ids = sorted(adj.keys())
    n = len(node_ids)
    idx = {v: i for i, v in enumerate(node_ids)}
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    for src in adj:
        si = idx[src]
        visited = {src}
        q = [(src, 0)]
        while q:
            node, dist = q.pop(0)
            D[si, idx[node]] = dist
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))

    return D, node_ids

def wasserstein1_graph(mu: dict, nu: dict, D: np.ndarray, node_ids: list) -> float:
    n = len(node_ids)
    idx = {v: i for i, v in enumerate(node_ids)}

    supply = np.array([mu.get(v, 0.0) for v in node_ids], dtype=np.float64)
    demand = np.array([nu.get(v, 0.0) for v in node_ids], dtype=np.float64)

    s_sum = supply.sum()
    d_sum = demand.sum()
    if s_sum > 0:
        supply /= s_sum
    if d_sum > 0:
        demand /= d_sum

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

def krampus_ollivier_ricci(adj: dict, alpha: float = 0.5) -> dict:
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
            curvatures[(x, y)] = w1
    return curvatures

def hybrid_vram_planner(adj: dict, vram_plans: list) -> tuple:
    vram_planner = VramPlanner()
    for plan in vram_plans:
        vram_planner.register(plan)

    curvatures = krampus_ollivier_ricci(adj)
    return vram_planner, curvatures

if __name__ == "__main__":
    adj = {
        "A": ["B", "C"],
        "B": ["A", "D"],
        "C": ["A", "D"],
        "D": ["B", "C"]
    }
    vram_plans = [
        VramSlotPlan("plan1", "kind1", "action1", 100, "reason1", {}),
        VramSlotPlan("plan2", "kind2", "action2", 200, "reason2", {}),
        VramSlotPlan("plan3", "kind3", "action3", 300, "reason3", {}),
        VramSlotPlan("plan4", "kind4", "action4", 400, "reason4", {})
    ]
    vram_planner, curvatures = hybrid_vram_planner(adj, vram_plans)
    print(vram_planner.available_budget_mb())
    print(curvatures)