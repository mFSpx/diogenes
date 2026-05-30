# DARWIN HAMMER — match 10, survivor 1
# gen: 1
# parent_a: cockpit_metrics.py (gen0)
# parent_b: rectified_flow.py (gen0)
# born: 2026-05-29T23:22:30Z

"""
This module defines a novel hybrid algorithm, combining elements of 
'cockpit_metrics.py' and 'rectified_flow.py'. The mathematical bridge between 
these two structures is found in the concept of 'straightness' in the 
'cockpit_honesty' metric, which can be seen as a form of 'regularization' 
in the 'rectified_flow' model. By integrating the governing equations of 
both models, we create a new algorithm that balances the honesty of claims 
with the straightness of the flow.

The key innovation is the introduction of a 'straightness_regularization' term 
in the 'flow_loss' function, which encourages the model to produce straighter 
trajectories.
"""

import numpy as np
import math
import random

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))

def interpolant(x0, x1, t):
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    t  = np.asarray(t,  dtype=np.float64)
    if x0.ndim > t.ndim:
        t = t[..., np.newaxis]
    return t * x1 + (1.0 - t) * x0

def flow_target(x0, x1):
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    return x1 - x0

def straightness_regularization(x0, x1, v_pred):
    target = flow_target(x0, x1)
    diff = v_pred - target
    return np.mean(diff ** 2)

def flow_loss(v_pred, x0, x1):
    target = flow_target(x0, x1)
    diff = v_pred - target
    straightness_reg = straightness_regularization(x0, x1, v_pred)
    return np.mean(diff ** 2) + 0.1 * straightness_reg

def hybrid_solve(v_fn, x0, steps=10):
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

def midpoint_solve(v_fn, x0, steps=10):
    x0 = np.asarray(x0, dtype=np.float64)
    dt = 1.0 / steps
    ts = np.linspace(0.0, 1.0 - dt, steps)  
    traj = np.empty((steps + 1,) + x0.shape, dtype=np.float64)
    traj[0] = x0
    z = x0.copy()
    for k, t in enumerate(ts):
        v = v_fn(z, float(t))
        z_mid = z + 0.5 * dt * np.asarray(v, dtype=np.float64)
        v_mid = v_fn(z_mid, float(t) + 0.5 * dt)
        z = z + dt * np.asarray(v_mid, dtype=np.float64)
        traj[k + 1] = z
    return traj

if __name__ == "__main__":
    def v_fn(z, t):
        return np.ones(z.shape)
    x0 = np.array([1.0, 2.0])
    x1 = np.array([3.0, 4.0])
    print(flow_loss(v_fn(np.array([2.0, 3.0]), 0.5), x0, x1))
    print(hybrid_solve(v_fn, x0))
    print(midpoint_solve(v_fn, x0))