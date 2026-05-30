# DARWIN HAMMER — match 2574, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1148_s0.py (gen4)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s4.py (gen2)
# born: 2026-05-29T23:42:57Z

"""Hybrid VRAM-Regret Graph Engine
This module fuses the core mathematics of two parent algorithms:

* **Parent A** – treats regret‑weighted action values as sections over a graph,
  uses the Gini coefficient to measure unevenness, the coboundary operator to
  quantify local disagreement, and a Koopman operator to forecast future
  values.

* **Parent B** – queries NVIDIA‑SMI for per‑GPU VRAM statistics, maintains a
  runtime receipt log and respects a global VRAM budget.

The mathematical bridge is the interpretation of *VRAM usage* as a
*section* on a graph whose nodes are the GPUs.  Inequality of VRAM allocation
is measured with the Gini coefficient, the coboundary operator computes the
pairwise discrepancy between allocated and actual usage, and a linear Koopman
operator (learned from a short history) forecasts the next VRAM usage vector.
The forecast can then be fed back into a budget‑allocation routine, closing the
loop between observation (Parent B) and decision‑theoretic reasoning
(Parent A)."""

import json
import os
import subprocess
import sys
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (light‑weight versions of the parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    driver_version: str
    pstate: str


# ----------------------------------------------------------------------
# Parent B – GPU query and receipt handling (trimmed)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
RUNTIME_DIR = ROOT / "04_RUNTIME" / "inference_os"


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_runtime_receipt(receipt: Dict[str, Any], *, path: Path | None = None) -> None:
    """Append a JSON‑line receipt to the runtime log."""
    target = path or (RUNTIME_DIR / "preemption_receipts.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def query_nvidia_smi() -> List[GPUInfo]:
    """Return a list of GPUInfo objects.  Falls back to a synthetic single‑GPU
    record when nvidia‑smi is unavailable (useful for CI)."""
    if not shutil.which("nvidia-smi"):
        # Synthetic fallback
        return [
            GPUInfo(
                index=0,
                name="SyntheticGPU",
                total_mb=8192,
                used_mb=random.randint(0, 4000),
                free_mb=0,  # will be recomputed
                driver_version="synthetic",
                pstate="P0",
            )
        ]

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
        # On error, also fall back to synthetic data
        return [
            GPUInfo(
                index=0,
                name="SyntheticGPU",
                total_mb=8192,
                used_mb=random.randint(0, 4000),
                free_mb=0,
                driver_version="synthetic",
                pstate="P0",
            )
        ]

    gpus: List[GPUInfo] = []
    for line in cp.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, total, used, free, driver, pstate = parts[:7]
        gpus.append(
            GPUInfo(
                index=int(idx),
                name=name,
                total_mb=int(total),
                used_mb=int(used),
                free_mb=int(free),
                driver_version=driver,
                pstate=pstate,
            )
        )
    return gpus


# ----------------------------------------------------------------------
# Parent A – Graph, Gini, Coboundary, Koopman
# ----------------------------------------------------------------------
class GPUGraph:
    """A simple undirected graph whose nodes are GPUs.  Each node stores a
    *section* – the current VRAM usage (in MB).  Edges are implicitly all‑to‑all
    (complete graph) to allow pairwise coboundary computation."""

    def __init__(self, gpu_infos: List[GPUInfo]):
        self.nodes = {gpu.index: gpu for gpu in gpu_infos}
        self._sections: Dict[int, float] = {gpu.index: float(gpu.used_mb) for gpu in gpu_infos}
        # Pre‑compute edge list for a complete graph (excluding self‑loops)
        self.edges: List[Tuple[int, int]] = [
            (u, v) for i, u in enumerate(self.nodes) for v in list(self.nodes)[i + 1 :]
        ]

    def set_section(self, node: int, usage_mb: float) -> None:
        """Update the VRAM usage section for a given GPU node."""
        if node not in self.nodes:
            raise KeyError(f"GPU {node} not present in the graph")
        self._sections[node] = float(usage_mb)

    def get_section(self, node: int) -> float:
        return self._sections[node]

    def usage_vector(self) -> np.ndarray:
        """Return a column vector of usages ordered by sorted node index."""
        ordered = [self._sections[i] for i in sorted(self.nodes)]
        return np.array(ordered, dtype=float).reshape(-1, 1)


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a 1‑D array of non‑negative numbers."""
    if values.ndim != 1:
        raise ValueError("Gini expects a 1‑D array")
    if np.any(values < 0):
        raise ValueError("Gini coefficient undefined for negative values")
    n = values.size
    if n == 0:
        return 0.0
    sorted_vals = np.sort(values)
    cumulative = np.cumsum(sorted_vals, dtype=float)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    # Gini formula: (2*Σi·xi)/(n*Σxi) - (n+1)/n
    i = np.arange(1, n + 1)
    gini = (2.0 * np.sum(i * sorted_vals)) / (n * sum_vals) - (n + 1) / n
    return float(gini)


def coboundary_norm(graph: GPUGraph) -> float:
    """Compute the L2 norm of the coboundary (pairwise differences) over all edges."""
    diffs = []
    for u, v in graph.edges:
        diff = graph.get_section(u) - graph.get_section(v)
        diffs.append(diff)
    if not diffs:
        return 0.0
    return float(np.linalg.norm(diffs, ord=2))


class KoopmanForecaster:
    """Learn a linear Koopman operator from a short history of usage vectors
    and forecast future usage.  The operator is estimated by least‑squares:
        X_{t+1} ≈ K * X_t
    """

    def __init__(self, lag: int = 1):
        self.lag = lag
        self.K: np.ndarray | None = None

    def fit(self, history: List[np.ndarray]) -> None:
        """Fit K using consecutive pairs in history."""
        if len(history) < self.lag + 1:
            raise ValueError("Not enough history to fit Koopman operator")
        X = np.hstack(history[:-1])  # shape (n, m)
        Y = np.hstack(history[1:])   # shape (n, m)
        # Solve Y = K X  =>  K = Y X^{+}
        X_pinv = np.linalg.pinv(X)
        self.K = Y @ X_pinv

    def predict(self, current: np.ndarray, steps: int = 1) -> np.ndarray:
        """Iterate the Koopman map."""
        if self.K is None:
            raise RuntimeError("KoopmanForecaster not fitted")
        pred = current.copy()
        for _ in range(steps):
            pred = self.K @ pred
            # Clip negative forecasts to zero (VRAM cannot be negative)
            pred = np.maximum(pred, 0.0)
        return pred


def allocate_budget(
    graph: GPUGraph,
    total_budget_mb: int,
    gini_target: float = 0.2,
) -> Dict[int, float]:
    """Allocate a new VRAM budget across GPUs aiming to reduce the Gini
    coefficient towards *gini_target*.  The allocation is proportional to the
    inverse of current usage, scaled to meet the total budget."""
    usages = np.array([graph.get_section(i) for i in sorted(graph.nodes)], dtype=float)
    inv = np.where(usages > 0, 1.0 / usages, 1.0)  # avoid division by zero
    weights = inv / inv.sum()
    allocation = weights * total_budget_mb
    alloc_dict = {node: float(alloc) for node, alloc in zip(sorted(graph.nodes), allocation)}
    return alloc_dict


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def hybrid_step(
    graph: GPUGraph,
    forecaster: KoopmanForecaster,
    history: deque,
    budget_mb: int,
) -> Tuple[float, float, np.ndarray]:
    """
    Perform a single hybrid iteration:
    1. Record current usage.
    2. Update history (maxlen 5).
    3. Fit/predict with Koopman.
    4. Compute Gini and coboundary.
    5. Allocate budget based on the forecast.
    Returns (gini, coboundary_norm, forecast_vector).
    """
    # 1. Record current usage
    current_vec = graph.usage_vector()
    history.append(current_vec)

    # 2. Fit forecaster if enough data
    if len(history) >= 3:
        forecaster.fit(list(history))

    # 3. Predict next usage (one step ahead)
    forecast = forecaster.predict(current_vec, steps=1) if forecaster.K is not None else current_vec

    # 4. Compute metrics
    gini = gini_coefficient(current_vec.flatten())
    cob_norm = coboundary_norm(graph)

    # 5. Allocate new budget based on forecasted usage
    # Here we simply update the sections to the forecast (clipped by budget)
    for idx, node in enumerate(sorted(graph.nodes)):
        projected = float(forecast[idx, 0])
        # Enforce that projected usage does not exceed per‑GPU total
        max_allowed = graph.nodes[node].total_mb
        new_usage = min(projected, max_allowed, budget_mb)
        graph.set_section(node, new_usage)

    return gini, cob_norm, forecast


def log_metrics(gini: float, cob_norm: float, forecast: np.ndarray) -> None:
    """Append a runtime receipt with the computed metrics."""
    receipt = {
        "timestamp": now_z(),
        "gini_coefficient": gini,
        "coboundary_norm": cob_norm,
        "forecast_usage_mb": forecast.flatten().tolist(),
    }
    _append_runtime_receipt(receipt)


def run_hybrid_demo(budget_mb: int = 4096, steps: int = 5) -> None:
    """End‑to‑end demo that queries GPUs, builds the graph, and runs a few
    hybrid steps, printing the intermediate metrics."""
    gpu_infos = query_nvidia_smi()
    graph = GPUGraph(gpu_infos)
    forecaster = KoopmanForecaster(lag=1)
    history: deque = deque(maxlen=5)

    print(f"Initial GPU usages: {[graph.get_section(i) for i in sorted(graph.nodes)]}")
    for step in range(steps):
        gini, cob_norm, forecast = hybrid_step(graph, forecaster, history, budget_mb)
        print(f"Step {step+1}: Gini={gini:.4f}, Coboundary={cob_norm:.2f}, Forecast={forecast.flatten()}")
        log_metrics(gini, cob_norm, forecast)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Use a modest budget to keep the demo lightweight
    run_hybrid_demo(budget_mb=2048, steps=3)