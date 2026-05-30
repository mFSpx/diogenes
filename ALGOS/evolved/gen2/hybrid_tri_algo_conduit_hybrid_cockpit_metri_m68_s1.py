# DARWIN HAMMER — match 68, survivor 1
# gen: 2
# parent_a: tri_algo_conduit.py (gen0)
# parent_b: hybrid_cockpit_metrics_rectified_flow_m10_s1.py (gen1)
# born: 2026-05-29T23:26:48Z

"""
Hybrid Algorithm: Tri-algo Conduit + Hybrid Cockpit Metrics Rectified Flow
This module defines a novel hybrid algorithm, combining elements of 
'tri_algo_conduit.py' and 'hybrid_cockpit_metrics_rectified_flow_m10_s1.py'. 
The mathematical bridge between these two structures is found in the concept 
of 'signal scores' in the 'tri_algo_conduit.py', which can be seen as a form 
of 'claims_with_evidence' in the 'cockpit_metrics.py' model. By integrating 
the governing equations of both models, we create a new algorithm that 
balances the signal scores with the straightness of the flow.

The key innovation is the introduction of a 'signal_straightness_regularization' 
term in the 'flow_loss' function, which encourages the model to produce 
straighter trajectories with high signal scores.
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

def shannon_entropy(sequence):
    sequence = [ord(c) for c in sequence]
    sequence_len = len(sequence)
    if sequence_len <= 0:
        return 0.0

    frequency_dict = {}
    for item in sequence:
        if item not in frequency_dict:
            frequency_dict[item] = 0
        frequency_dict[item] += 1

    entropy = 0.0
    for item in frequency_dict:
        p_x = frequency_dict[item] / sequence_len
        entropy += - p_x * math.log(p_x, 2)
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

def anti_slop_ratio(signal_score: float, total_score: float) -> float:
    return 1.0 if total_score <= 0 else max(0.0, min(1.0, signal_score / total_score))

def cockpit_honesty(signal_score: float, noise_score: float) -> float:
    total = signal_score + noise_score
    return 1.0 if total <= 0 else max(0.0, min(1.0, signal_score / total))

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

def signal_straightness_regularization(signal_score, x0, x1, v_pred):
    target = flow_target(x0, x1)
    diff = v_pred - target
    anti_slop = anti_slop_ratio(signal_score, 1.0)
    return np.mean(diff ** 2) * anti_slop

def flow_loss(v_pred, x0, x1, signal_score):
    target = flow_target(x0, x1)
    diff = v_pred - target
    signal_reg = signal_straightness_regularization(signal_score, x0, x1, v_pred)
    return np.mean(diff ** 2) + 0.1 * signal_reg

def hybrid_solve(v_fn, x0, signal_score, steps=10):
    x0 = np.asarray(x0, dtype=np.float64)
    dt = 1.0 / steps
    ts = np.linspace(0.0, 1.0 - dt, steps)  
    traj = np.empty((steps + 1,) + x0.shape, dtype=np.float64)
    traj[0] = x0
    z = x0.copy()
    for k, t in enumerate(ts):
        v = v_fn(z, float(t), signal_score)
        z = z + dt * np.asarray(v, dtype=np.float64)
        traj[k + 1] = z
    return traj

def smoke_test():
    data = b'Hello, World!'
    signal, noise = signal_scores(data)
    honesty = cockpit_honesty(signal, noise)
    anti_slop = anti_slop_ratio(signal, signal + noise)
    print(f"Signal Score: {signal:.4f}, Noise Score: {noise:.4f}, Honesty: {honesty:.4f}, Anti-Slop Ratio: {anti_slop:.4f}")

if __name__ == "__main__":
    smoke_test()