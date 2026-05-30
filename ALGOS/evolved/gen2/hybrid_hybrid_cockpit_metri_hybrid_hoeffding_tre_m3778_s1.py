# DARWIN HAMMER — match 3778, survivor 1
# gen: 2
# parent_a: hybrid_cockpit_metrics_rectified_flow_m10_s0.py (gen1)
# parent_b: hybrid_hoeffding_tree_gini_coefficient_m13_s3.py (gen1)
# born: 2026-05-29T23:51:37Z

import numpy as np
import math
from dataclasses import dataclass
from typing import Iterable

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))

def audit_debt(exports_missing_audit_step: int) -> float:
    return float(max(0, exports_missing_audit_step))

def interpolant(x0, x1, t):
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if x0.ndim > t.ndim:
        t = t[..., np.newaxis]
    return t * x1 + (1.0 - t) * x0

def flow_target(x0, x1):
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    return x1 - x0

def flow_loss(v_pred, x0, x1):
    v_pred = np.asarray(v_pred, dtype=np.float64)
    target = flow_target(x0, x1)
    diff = v_pred - target
    return float(np.mean(diff ** 2))

def euler_solve(v_fn, x0, steps=10):
    x0 = np.asarray(x0, dtype=np.float64)
    dt = 1.0 / steps
    ts = np.linspace(0.0, 1.0 - dt, steps)
    traj = np.empty((steps + 1,) + x0.shape, dtype=np.float64)
    traj[0] = x0
    z = x0.copy()
    for k, t in enumerate(ts):
        v = v_fn(z, float(t))
        z = z + dt * np.asarray(v, dtype=np.float64)
        traj[k + 1] = z
    return traj

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: return 0.0
    if xs[0] < 0: raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2*i-n-1)*x for i,x in enumerate(xs,1))/(n*sum(xs))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split(best_gini: float, second_best_gini: float, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gini - second_best_gini
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

def evaluate_split(gini_values: Iterable[float], r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    gini_coeffs = [gini_coefficient([x]) for x in gini_values]
    best_gini = max(gini_coeffs)
    second_best_gini = max(g for g in gini_coeffs if g != best_gini) if len(gini_coeffs) > 1 else 0.0
    return should_split(best_gini, second_best_gini, r, delta, n, tie_threshold)

def hybrid_solve(v_fn, x0, claims_with_evidence, total_claims_emitted, displayed_ok, unknown_displayed_as_ok, 
                 r: float, delta: float, n: int, steps=10):
    x0 = np.asarray(x0, dtype=np.float64)
    dt = 1.0 / steps
    ts = np.linspace(0.0, 1.0 - dt, steps)
    traj = np.empty((steps + 1,) + x0.shape, dtype=np.float64)
    traj[0] = x0
    z = x0.copy()
    for k, t in enumerate(ts):
        v = v_fn(z, float(t))
        z = z + dt * np.asarray(v, dtype=np.float64)
        traj[k + 1] = z
        
        gini_values = [cockpit_honesty(displayed_ok, unknown_displayed_as_ok), anti_slop_ratio(claims_with_evidence, total_claims_emitted)]
        decision = evaluate_split(gini_values, r, delta, n)
        if decision.should_split:
            print(f"Splitting at step {k+1} with epsilon {decision.epsilon} and gain gap {decision.gain_gap}")
        else:
            print(f"Not splitting at step {k+1} with epsilon {decision.epsilon} and gain gap {decision.gain_gap}")
    
    return traj

def main():
    x0 = np.array([1.0, 2.0, 3.0])
    v_fn = lambda z, t: 10.0 * np.sin(2.0 * np.pi * t) * z
    claims_with_evidence = 10
    total_claims_emitted = 20
    displayed_ok = 5
    unknown_displayed_as_ok = 3
    r = 1.0
    delta = 0.01
    n = 100
    
    hybrid_solve(v_fn, x0, claims_with_evidence, total_claims_emitted, displayed_ok, unknown_displayed_as_ok, r, delta, n)

if __name__ == "__main__":
    main()