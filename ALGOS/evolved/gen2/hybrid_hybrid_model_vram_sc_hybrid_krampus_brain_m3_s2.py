# DARWIN HAMMER — match 3, survivor 2
# gen: 2
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s4.py (gen1)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s0.py (gen1)
# born: 2026-05-29T23:22:36Z

"""Hybrid VRAM‑Curvature Scheduler

This module fuses two parent algorithms:

* **Parent A** – ``model_vram_scheduler.py`` provides ``VramPlanner`` that
  tracks VRAM usage of artefacts and offers a budget‑aware API.
* **Parent B** – ``krampus_brainmap`` / ``ollivier_ricci_curvature`` computes
  Ollivier‑Ricci curvature on a graph using lazy random‑walk measures and
  Wasserstein‑1 distance.

**Mathematical bridge**

Each artefact registered in ``VramPlanner`` becomes a node in a graph.
The node attribute *estimated_mb* is interpreted as a mass weight.
When computing the lazy random‑walk distribution for Ollivier‑Ricci curvature,
the base distribution (α‑laziness) is multiplied by the artefact’s mass weight
and renormalised.  Consequently the curvature values reflect the VRAM
allocation landscape, and the planner can use curvature as a heuristic to
accept or reject new artefacts while respecting the static budget.

The hybrid system therefore integrates:


μ_i(v) = α·w_i·δ_{i=v} + (1-α)·w_i·(1/deg(i))·∑_{u∈N(i)} δ_{u=v}


where ``w_i`` is the normalised VRAM weight of node *i*.  The curvature
computed from these weighted measures is fed back into ``VramPlanner`` via
``register_with_curvature``.
"""

import json
import os
import shutil
import subprocess
import sys
import math
import random
from collections import deque, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Global constants & helpers (from Parent A)
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
# VRAM planning core – deeper integration (Parent A)
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
    """Tracks VRAM usage of multiple artefacts and provides a budget‑aware API."""

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
    # Artefact registration
    # ------------------------------------------------------------------
    def register(self, plan: VramSlotPlan) -> VramSlotPlan:
        """Add or replace an artefact plan. Returns the stored plan."""
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

    # ------------------------------------------------------------------
    # Decision helper used by TTT (Parent A)
    # ------------------------------------------------------------------
    def can_accommodate(self, mb: int) -> Tuple[bool, str]:
        if mb <= self.available_budget_mb():
            return True, "sufficient budget"
        else:
            return False, f"need {mb} MB, only {self.available_budget_mb()} MB free"

    # ------------------------------------------------------------------
    # Expose artefact meta‑data for the curvature module
    # ------------------------------------------------------------------
    def artifact_weights(self) -> Dict[str, float]:
        """Return a mapping artifact_id → weight proportional to its VRAM estimate."""
        total = self.total_committed_mb() or 1.0
        return {aid: plan.estimated_mb / total for aid, plan in self._artifacts.items()}

    def adjacency_from_artifacts(self) -> Dict[str, List[str]]:
        """
        Very simple heuristic: connect artefacts that share the same ``artifact_kind``.
        """
        kind_to_ids: Dict[str, List[str]] = defaultdict(list)
        for aid, plan in self._artifacts.items():
            kind_to_ids[plan.artifact_kind].append(aid)

        adj: Dict[str, List[str]] = defaultdict(list)
        for ids in kind_to_ids.values():
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    adj[a].append(b)
                    adj[b].append(a)
        return dict(adj)


# ----------------------------------------------------------------------
# Graph utilities (from Parent B)
# ----------------------------------------------------------------------
def bfs_distances(adj: Dict[str, List[str]]) -> Tuple[np.ndarray, List[str]]:
    """All‑pairs shortest‑path distances via BFS."""
    node_ids = sorted(adj.keys())
    n = len(node_ids)
    idx = {v: i for i, v in enumerate(node_ids)}
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    for src in node_ids:
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


def wasserstein1_graph(mu: Dict[str, float], nu: Dict[str, float],
                       D: np.ndarray, node_ids: List[str]) -> float:
    """Greedy Wasserstein‑1 distance between discrete measures."""
    n = len(node_ids)
    supply = np.array([mu.get(v, 0.0) for v in node_ids], dtype=np.float64)
    demand = np.array([nu.get(v, 0.0) for v in node_ids], dtype=np.float64)

    # Normalise numerically
    s_sum = supply.sum()
    d_sum = demand.sum()
    if s_sum > 0:
        supply /= s_sum
    if d_sum > 0:
        demand /= d_sum

    # Flatten cost matrix and sort by distance
    costs = [(D[i, j], i, j) for i in range(n) for j in range(n)]
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


def lazy_rw_distribution_weighted(adj: Dict[str, List[str]],
                                  node: str,
                                  node_weights: Dict[str, float],
                                  alpha: float = 0.5) -> Dict[str, float]:
    """
    Lazy random‑walk distribution centred at *node*.
    The base distribution (α‑laziness) is re‑weighted by ``node_weights`` and
    renormalised to a probability measure.
    """
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    base = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            base[nb] = base.get(nb, 0.0) + spread

    # Apply external weights
    weighted = {k: base[k] * node_weights.get(k, 1.0) for k in base}
    total = sum(weighted.values())
    if total == 0:
        # fallback to uniform over present keys
        return {k: 1.0 / len(weighted) for k in weighted}
    return {k: v / total for k, v in weighted.items()}


def krampus_ollivier_ricci_weighted(adj: Dict[str, List[str]],
                                   node_weights: Dict[str, float],
                                   alpha: float = 0.5) -> Dict[Tuple[str, str], float]:
    """
    Compute Ollivier‑Ricci curvature κ(x, y) for every edge using the
    weighted lazy random‑walk distributions.
    """
    D, node_ids = bfs_distances(adj)
    curvatures: Dict[Tuple[str, str], float] = {}
    seen: set[Tuple[str, str]] = set()

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

            mu = lazy_rw_distribution_weighted(adj, x, node_weights, alpha)
            nu = lazy_rw_distribution_weighted(adj, y, node_weights, alpha)
            w1 = wasserstein1_graph(mu, nu, D, node_ids)
            curvatures[(x, y)] = 1.0 - w1 / d_xy
    return curvatures


# ----------------------------------------------------------------------
# Hybrid operations (three required functions)
# ----------------------------------------------------------------------
def build_artifact_graph(planner: VramPlanner) -> Dict[str, List[str]]:
    """
    Construct an adjacency dictionary from the planner's artefacts.
    Artefacts sharing the same ``artifact_kind`` become neighbours.
    """
    return planner.adjacency_from_artifacts()


def compute_curvature_on_planner(planner: VramPlanner, alpha: float = 0.5) -> Dict[Tuple[str, str], float]:
    """
    Compute Ollivier‑Ricci curvature on the artefact graph,
    using each artefact's VRAM estimate as its mass weight.
    """
    adj = build_artifact_graph(planner)
    weights = planner.artifact_weights()
    return krampus_ollivier_ricci_weighted(adj, weights, alpha)


def register_with_curvature(planner: VramPlanner,
                            plan: VramSlotPlan,
                            alpha: float = 0.5,
                            curvature_threshold: float = 0.0) -> bool:
    """
    Attempt to register *plan*.
    The function simulates adding the new node to the current graph,
    recomputes curvatures for edges incident to the new node,
    and accepts the registration only if:

    1. The VRAM budget can accommodate the artefact, and
    2. The average curvature of its incident edges is >= ``curvature_threshold``.
       (Higher curvature indicates a “well‑connected” region of the resource graph.)

    Returns ``True`` on successful registration, ``False`` otherwise.
    """
    # Budget check first
    ok, msg = planner.can_accommodate(plan.estimated_mb)
    if not ok:
        # Budget insufficient – reject
        return False

    # Build a temporary adjacency including the prospective node
    temp_adj = planner.adjacency_from_artifacts()
    # Connect to existing nodes of the same kind (same heuristic as