# DARWIN HAMMER — match 95, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_label_foundry_m21_s2.py (gen3)
# parent_b: hybrid_cockpit_metrics_rectified_flow_m10_s0.py (gen1)
# born: 2026-05-29T23:26:51Z

import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Set, Callable
from pathlib import Path
import math
import random
import sys

@dataclass(frozen=True)
class LabelingFunctionResult: 
    lf_name: str 
    doc_id: str 
    label: int 

@dataclass(frozen=True)
class ProbabilisticLabel: 
    doc_id: str 
    label: int 
    confidence: float 

def labeling_function(name: str|None=None): 
    def deco(fn: Callable[[dict],int]): 
        fn.lf_name=name or fn.__name__ 
        return fn 
    return deco 

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]: 
    votes=defaultdict(list) 
    for batch in batches: 
        for r in batch: 
            if r.label in (0,1): votes[r.doc_id].append(r.label) 
    out=[] 
    for d,vs in votes.items(): 
        if not vs: 
            out.append(ProbabilisticLabel(d,0,0.5)); 
            continue 
        c=Counter(vs); 
        label=1 if c[1]>=c[0] else 0; 
        confidence = c[label]/len(vs) if len(vs) > 0 else 0.5
        out.append(ProbabilisticLabel(d,label,confidence)) 
    return out 

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    if total_claims_emitted <= 0: 
        return 0.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0: 
        return 0.0
    return max(0.0, min(1.0, displayed_ok / total))

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

def hybrid_labeling(batch: list[LabelingFunctionResult], claims_with_evidence: int, total_claims_emitted: int) -> list[ProbabilisticLabel]:
    labels = aggregate_labels([batch])
    slop_ratio = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    honest_labels = []
    for label in labels:
        confidence = label.confidence * slop_ratio
        honest_labels.append(ProbabilisticLabel(label.doc_id, label.label, confidence))
    return honest_labels

def euler_solve(v_fn, x0, steps=10, t_span=(0.0, 1.0)):
    x0 = np.asarray(x0, dtype=np.float64)
    dt = (t_span[1] - t_span[0]) / steps
    ts = np.linspace(t_span[0], t_span[1] - dt, steps)
    traj = np.empty((steps + 1,) + x0.shape, dtype=np.float64)
    traj[0] = x0
    z = x0.copy()
    for k, t in enumerate(ts):
        v = v_fn(z, float(t))
        z = z + dt * np.asarray(v, dtype=np.float64)
        traj[k + 1] = z
    return traj

def hybrid_solve(v_fn, x0, batch: list[LabelingFunctionResult], claims_with_evidence, total_claims_emitted, steps=10, t_span=(0.0, 1.0)):
    labels = hybrid_labeling(batch, claims_with_evidence, total_claims_emitted)
    def modified_v_fn(z, t):
        label_confidences = [label.confidence for label in labels]
        return v_fn(z, t) * np.mean(label_confidences)
    return euler_solve(modified_v_fn, x0, steps, t_span)

if __name__ == "__main__":
    batch = [LabelingFunctionResult("lf1", "doc1", 1), LabelingFunctionResult("lf2", "doc1", 0), LabelingFunctionResult("lf3", "doc2", 1)]
    claims_with_evidence = 10
    total_claims_emitted = 20
    x0 = np.array([1.0, 2.0])
    def v_fn(z, t):
        return np.array([z[0] + t, z[1] - t])
    traj = hybrid_solve(v_fn, x0, batch, claims_with_evidence, total_claims_emitted)
    print(traj)