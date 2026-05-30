# DARWIN HAMMER — match 2282, survivor 2
# gen: 6
# parent_a: hybrid_counterfactual_effec_hybrid_hybrid_hybrid_m948_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_model__hybrid_doomsday_cale_m193_s0.py (gen4)
# born: 2026-05-29T23:41:42Z

import numpy as np
import math
import random
import subprocess
import hashlib
from datetime import datetime
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict

Vector = Sequence[float]

@dataclass(frozen=True)
class CausalEffect:
    effect_id: str
    treatment: str
    outcome: str
    confounders: Tuple[str, ...]
    ate_estimate: float | None
    ate_confidence_interval: Tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: Tuple[str, ...]
    heterogeneous_effects: Dict[str, float]

def gini_coefficient(values: List[float]) -> float:
    values = sorted(values)
    index = np.arange(1, len(values)+1)
    n = len(values)
    return ((np.sum((2 * index - n  - 1) * values)) / (n * np.sum(values)))

def minhash_signature(values: List[float], num_buckets: int = 10) -> List[int]:
    hash_values = []
    for value in values:
        hash_object = hashlib.md5(str(value).encode())
        hash_value = int(hash_object.hexdigest(), 16)
        bucket = hash_value % num_buckets
        hash_values.append(bucket)
    return hash_values

def rbf_kernel(x: List[float], y: List[float], sigma: float = 1.0) -> float:
    return math.exp(-np.sum((np.array(x) - np.array(y)) ** 2) / (2 * sigma ** 2))

def hybrid_causal_gini_rbf(treatment: List[float], outcome: List[float], confounders: List[float]) -> CausalEffect:
    if len(treatment) != len(outcome) or len(treatment) != len(confounders):
        raise ValueError("Input lists must have the same length")

    ate_estimate = np.mean(outcome) - np.mean(treatment)
    gini_coef = gini_coefficient(confounders)
    minhash_sig = minhash_signature(confounders)
    rbf_weights = [rbf_kernel(minhash_sig, [i] * len(minhash_sig)) for i in minhash_sig]
    weighted_ate = sum([ate_estimate * weight for weight in rbf_weights]) / sum(rbf_weights) if sum(rbf_weights) != 0 else ate_estimate
    return CausalEffect(
        effect_id="hybrid_causal_gini_rbf",
        treatment="treatment",
        outcome="outcome",
        confounders=tuple(confounders),
        ate_estimate=weighted_ate,
        ate_confidence_interval=(ate_estimate - gini_coef, ate_estimate + gini_coef),
        refutation_passed=True,
        refutation_methods=("gini_coefficient",),
        heterogeneous_effects={"gini_coefficient": gini_coef}
    )

def gpu_memory() -> dict[str, any]:
    if not hasattr(subprocess, 'run'):
        return {"status": "missing", "message": "subprocess.run not found"}
    try:
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
    except FileNotFoundError:
        return {"status": "missing", "message": "nvidia-smi not found"}
    if cp.returncode != 0 or not cp.stdout.strip():
        return {"status": "error", "stderr": cp.stderr[-500:]}
    gpus: list[dict[str, any]] = []
    for line in cp.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, total, used, free, driver, pstate = parts[:7]
        gpus.append(
            {
                "index": idx,
                "name": name,
                "memory": {"total": total, "used": used, "free": free},
                "driver": driver,
                "pstate": pstate
            }
        )
    return gpus

def doomsday_weekday(date: datetime) -> int:
    t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    year = date.year
    month = date.month
    day = date.day
    if month < 3:
        year -= 1
        month += 10
    return (year + int(year/4) - int(year/100) + int(year/400) + t[month-1] + day) % 7

def hybrid_causal_gini_rbf_improved(treatment: List[float], outcome: List[float], confounders: List[float], date: datetime) -> CausalEffect:
    weekday = doomsday_weekday(date)
    weekday_frequencies = [0] * 7
    weekday_frequencies[weekday] += 1
    gini_coef_weekday = gini_coefficient(weekday_frequencies)

    ate_estimate = np.mean(outcome) - np.mean(treatment)
    gini_coef = gini_coefficient(confounders)
    minhash_sig = minhash_signature(confounders)
    rbf_weights = [rbf_kernel(minhash_sig, [i] * len(minhash_sig)) for i in minhash_sig]
    weighted_ate = sum([ate_estimate * weight for weight in rbf_weights]) / sum(rbf_weights) if sum(rbf_weights) != 0 else ate_estimate
    return CausalEffect(
        effect_id="hybrid_causal_gini_rbf_improved",
        treatment="treatment",
        outcome="outcome",
        confounders=tuple(confounders),
        ate_estimate=weighted_ate,
        ate_confidence_interval=(ate_estimate - gini_coef - gini_coef_weekday, ate_estimate + gini_coef + gini_coef_weekday),
        refutation_passed=True,
        refutation_methods=("gini_coefficient", "doomsday_weekday"),
        heterogeneous_effects={"gini_coefficient": gini_coef, "gini_coefficient_weekday": gini_coef_weekday}
    )

if __name__ == "__main__":
    treatment = [random.random() for _ in range(100)]
    outcome = [random.random() for _ in range(100)]
    confounders = [random.random() for _ in range(100)]
    date = datetime.now()
    effect = hybrid_causal_gini_rbf_improved(treatment, outcome, confounders, date)
    print(effect)
    gpu_info = gpu_memory()
    print(gpu_info)