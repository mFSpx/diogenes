# DARWIN HAMMER — match 68, survivor 0
# gen: 2
# parent_a: tri_algo_conduit.py (gen0)
# parent_b: hybrid_cockpit_metrics_rectified_flow_m10_s1.py (gen1)
# born: 2026-05-29T23:26:48Z

"""
Novel hybrid algorithm fusing 'tri_algo_conduit.py' and 'hybrid_cockpit_metrics_rectified_flow_m10_s1.py'.
The mathematical bridge between these two structures is found in the concept of 
'signal_score' in 'tri_algo_conduit.py', which can be seen as a form of 'honesty' 
in the 'cockpit_honesty' metric of 'hybrid_cockpit_metrics_rectified_flow_m10_s1.py'. 
By integrating the governing equations of both models, we create a new algorithm 
that balances the signal quality with the straightness of the flow.

The key innovation is the introduction of a 'signal_regularization' term 
in the 'flow_loss' function, which encourages the model to produce more reliable 
signals.
"""

import numpy as np
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class ConduitDecision:
    action: str  
    confidence_gap: float
    epsilon: float
    signal_score: float
    noise_score: float
    dormancy_probability: float
    recovery_priority: float
    reason: str

def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    if not data:
        return 0.0
    chunk = data[:sample]
    return shannon_entropy(list(chunk)) / 8.0

def shannon_entropy(chunk):
    entropy = 0.0
    for x in set(chunk):
        p_x = chunk.count(x)/len(chunk)
        entropy += - p_x*math.log(p_x, 2)
    return entropy

def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> tuple[float, float]:
    size = len(data)
    entropy = _byte_entropy(data)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml")) else 0.02
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)
    signal = max(0.0, min(1.0, 0.20 + status_bonus + mime_bonus + size_bonus + keyword_bonus + structure_bonus + 0.12 * entropy))
    noise = max(0.0, min(1.0, 0.58 - 0.22 * entropy - keyword_bonus - structure_bonus + (0.12 if size < 64 else 0.0)))
    return signal, noise

def cockpit_honesty(signal_score: float) -> float:
    return signal_score

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

def straightness_regularization(signal_score, v_pred):
    target = signal_score
    diff = v_pred - target
    return np.mean(diff ** 2)

def signal_regularization(signal_score, v_pred):
    straightness_reg = straightness_regularization(signal_score, v_pred)
    return 0.1 * straightness_reg

def flow_loss(v_pred, signal_score):
    diff = v_pred - signal_score
    signal_reg = signal_regularization(signal_score, v_pred)
    return np.mean(diff ** 2) + signal_reg

def hybrid_solve(v_fn, signal_score, steps=10):
    signal_score = np.asarray(signal_score, dtype=np.float64)
    dt = 1.0 / steps
    ts = np.linspace(0.0, 1.0 - dt, steps)  
    traj = np.empty((steps + 1,), dtype=np.float64)
    traj[0] = 0.0
    z = 0.0
    for k, t in enumerate(ts):
        v = v_fn(z, float(t))
        z = z + dt * np.asarray(v, dtype=np.float64)
        traj[k + 1] = z
    return traj

def rectified_flow(signal_score, steps=10):
    def v_fn(z, t):
        return signal_score
    return hybrid_solve(v_fn, signal_score, steps)

def tri_algo_conduit_hybrid(data: bytes, status_code: int | None = None, mime: str = "", keyword_hits: int = 0, structural_links: int = 0, steps=10):
    signal_score, _ = signal_scores(data, status_code, mime, keyword_hits, structural_links)
    honesty = cockpit_honesty(signal_score)
    traj = rectified_flow(signal_score, steps)
    return ConduitDecision(
        action="burst",
        confidence_gap=0.1,
        epsilon=0.01,
        signal_score=signal_score,
        noise_score=1 - signal_score,
        dormancy_probability=0.5,
        recovery_priority=honesty,
        reason="hybrid",
    ), traj

if __name__ == "__main__":
    data = b"Hello, World!"
    decision, traj = tri_algo_conduit_hybrid(data)
    print(decision)
    print(traj)