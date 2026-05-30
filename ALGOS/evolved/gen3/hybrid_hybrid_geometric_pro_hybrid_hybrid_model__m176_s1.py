# DARWIN HAMMER — match 176, survivor 1
# gen: 3
# parent_a: hybrid_geometric_product_hybrid_model_vram_sc_m22_s0.py (gen2)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s0.py (gen2)
# born: 2026-05-29T23:25:58Z

"""HybridGeometricVRAMCurvature
Combines:
- Parent A: geometric product (Clifford algebra) and TTT‑Linear model (gradient descent).
- Parent B: VramPlanner (resource allocation) and Ollivier‑Ricci curvature on a graph.

Mathematical bridge:
The TTT weight matrix `W` is interpreted as the adjacency matrix of a graph whose
nodes correspond to VRAM‑allocation features.  Ollivier‑Ricci curvature of this
graph is computed (via a lightweight proxy) and used to modulate the gradient
step of the TTT‑Linear update.  Simultaneously the VRAM plan is encoded as a
multivector; the geometric product of this plan with a constant multivector
produces a transformed coefficient set that can be injected back into the
learning dynamics.  Thus the Clifford‑product, gradient descent, and curvature
form a single fused update rule.
"""

import sys
import math
import random
import pathlib
import numpy as np
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os

# ----------------------------------------------------------------------
# Helper functions from Parent A (Clifford algebra)
# ----------------------------------------------------------------------
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate index cancels (e_i ^ e_i = 0)
                lst.pop(j)
                lst.pop(j)  # second element shifts to j after first pop
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def geometric_product(a, b):
    """
    Full Clifford product `ab`.
    `a` and `b` are dicts mapping frozenset blades -> scalar coefficient.
    Returns a new dict representing the multivector product.
    """
    result = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            new_coef = coef_a * coef_b * sign
            if blade_out in result:
                result[blade_out] += new_coef
            else:
                result[blade_out] = new_coef
    # prune near‑zero entries
    eps = 1e-12
    return {k: v for k, v in result.items() if abs(v) > eps}

# ----------------------------------------------------------------------
# TTT‑Linear model utilities (Parent A)
# ----------------------------------------------------------------------
def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    """Initialize weight matrix `W` of shape (d_out, d_in)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target=None):
    """Self‑supervised squared‑error loss."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W, x, target=None):
    """Gradient of `ttt_loss` w.r.t. `W`."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)

# ----------------------------------------------------------------------
# VRAM planning utilities (Parent B)
# ----------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2] if pathlib.Path(__file__).exists() else pathlib.Path.cwd()
DEFAULT_BUDGET_MB = int(os.getenv("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.getenv("LUCIDOTA_VRAM_RESERVE_MB", "768"))
RUNTIME_DIR = ROOT / "04_RUNTIME" / "inference_os"

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _rel(path):
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)

def _append_runtime_receipt(receipt: dict, *, path: pathlib.Path | None = None) -> None:
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
    """Simple VRAM planner that stores artifact plans and can emit a feature vector."""
    def __init__(self, static_budget_mb: int = DEFAULT_BUDGET_MB, reserve_mb: int = DEFAULT_RESERVE_MB):
        self.static_budget_mb = static_budget_mb
        self.reserve_mb = reserve_mb
        self._plans: dict[str, VramSlotPlan] = {}

    def add_plan(self, plan: VramSlotPlan):
        self._plans[plan.artifact_id] = plan

    def feature_vector(self):
        """
        Produce a 1‑D numpy vector where each entry corresponds to the estimated
        memory of a stored plan, ordered by artifact_id.
        """
        ids = sorted(self._plans.keys())
        vec = np.array([self._plans[i].estimated_mb for i in ids], dtype=float)
        # Normalise to the budget scale to keep values O(1)
        scale = max(self.static_budget_mb, 1.0)
        return vec / scale

# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature proxy (bridge between the two parents)
# ----------------------------------------------------------------------
def compute_curvature(W):
    """
    Compute a lightweight proxy for Ollivier‑Ricci curvature on the graph
    defined by the (symmetrised) weight matrix `W`.

    The proxy uses the absolute weight magnitude to infer transport cost:
        C_ij = 1 / (1 + |W_ij|)

    Returns a matrix `C` of the same shape as `W` with values in (0,1].
    """
    A = (W + W.T) / 2.0               # make the adjacency symmetric
    C = 1.0 / (1.0 + np.abs(A))
    return C

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def multivector_from_plan(vec):
    """
    Encode a 1‑D numpy vector as a multivector where each component becomes
    a grade‑1 blade `e_i` with coefficient equal to the vector entry.
    """
    mv = {}
    for i, coeff in enumerate(vec):
        blade = frozenset({i})
        if abs(coeff) > 1e-12:
            mv[blade] = coeff
    return mv

def curvature_scaled_grad(W, x, lr=0.01, target=None):
    """
    Compute a gradient step for `W` where the raw TTT gradient is multiplied
    element‑wise by the curvature proxy `C`.  This yields a curvature‑aware
    update direction.
    """
    g = ttt_grad(W, x, target)
    C = compute_curvature(W)
    scaled = C * g
    return W - lr * scaled

def hybrid_step(planner: VramPlanner, W, lr=0.01, target=None):
    """
    Perform one hybrid iteration:
      1. Extract a normalized feature vector `x` from the VRAM planner.
      2. Update `W` with a curvature‑scaled TTT gradient.
      3. Convert `x` to a multivector `A`.
      4. Form a constant multivector `B` (grade‑0 scalar = 1.0).
      5. Return the new `W` and the geometric product `A * B`.
    """
    x = planner.feature_vector()
    if x.size == 0:
        raise ValueError("Planner contains no plans; cannot construct feature vector.")
    # Ensure W has compatible dimensions
    if W.shape[1] != x.shape[0]:
        # Re‑initialise to matching size
        W = init_ttt(x.shape[0], W.shape[0] if W.ndim == 2 else None)
    W_new = curvature_scaled_grad(W, x, lr, target)

    # Multivector encoding of the feature vector
    A = multivector_from_plan(x)

    # Constant multivector B = 1 (scalar blade)
    B = {frozenset(): 1.0}

    product = geometric_product(A, B)   # effectively copies A but passes through the product pipeline
    return W_new, product

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a planner with a few dummy VRAM plans
    planner = VramPlanner()
    planner.add_plan(VramSlotPlan(
        artifact_id="modelA",
        artifact_kind="ckpt",
        action="load",
        estimated_mb=1024,
        reason="initial_load",
        detail={}
    ))
    planner.add_plan(VramSlotPlan(
        artifact_id="tensorB",
        artifact_kind="tensor",
        action="allocate",
        estimated_mb=256,
        reason="intermediate",
        detail={}
    ))
    planner.add_plan(VramSlotPlan(
        artifact_id="cacheC",
        artifact_kind="cache",
        action="reserve",
        estimated_mb=128,
        reason="speedup",
        detail={}
    ))

    # Initialise weight matrix compatible with feature dimension
    dim = planner.feature_vector().shape[0]
    W = init_ttt(dim, dim, scale=0.05, seed=42)

    # Run a few hybrid steps
    for step in range(3):
        loss_before = ttt_loss(W, planner.feature_vector())
        W, prod = hybrid_step(planner, W, lr=0.02)
        loss_after = ttt_loss(W, planner.feature_vector())
        print(f"Step {step+1}: loss {loss_before:.6f} → {loss_after:.6f}, "
              f"multivector size {len(prod)} blades")
    print("Hybrid execution completed without errors.")