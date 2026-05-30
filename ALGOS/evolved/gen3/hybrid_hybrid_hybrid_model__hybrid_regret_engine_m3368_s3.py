# DARWIN HAMMER — match 3368, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s3.py (gen2)
# parent_b: hybrid_regret_engine_hybrid_liquid_time_c_m13_s0.py (gen2)
# born: 2026-05-29T23:49:43Z

import os
import sys
import json
import math
import random
import subprocess
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Iterable, Tuple
import numpy as np

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
DEFAULT_RESERVE_MB = int(os.getenv("LUCIDOTA_VRAM_RESERVE_MB", "768"))

def _query_nvidia_smi() -> Tuple[int, int]:
    if not shutil.which("nvidia-smi"):
        return 16384, 0  
    cp = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=5,
    )
    if cp.returncode != 0 or not cp.stdout.strip():
        return 16384, 0
    try:
        total_str, used_str = cp.stdout.strip().splitlines()[0].split(",")
        return int(total_str.strip()), int(used_str.strip())
    except Exception:
        return 16384, 0

def memory_pressure_factor(total_mb: int, used_mb: int, reserve_mb: int = DEFAULT_RESERVE_MB) -> float:
    usable = max(total_mb - reserve_mb, 1)
    excess = max(used_mb - reserve_mb, 0)
    μ = 1.0 - excess / usable
    return max(0.0, min(1.0, μ))

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: Iterable[str], k: int = 64) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def compute_regret_weights(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    if not actions:
        return {}
    cf_sum: Dict[str, float] = {}
    for cf in counterfactuals:
        cf_sum[cf.action_id] = cf_sum.get(cf.action_id, 0.0) + cf.outcome_value * cf.probability

    regrets = {}
    for a in actions:
        cf_val = cf_sum.get(a.id, 0.0)
        regrets[a.id] = max(a.expected_value - cf_val, 0.0)

    max_regret = max(regrets.values()) if regrets else 0.0
    exp_vals = {aid: math.exp(r - max_regret) for aid, r in regrets.items()}
    total = sum(exp_vals.values())
    return {aid: (v / total if total > 0 else 0.0) for aid, v in exp_vals.items()}

def apply_memory_pressure(
    base_weights: Dict[str, float],
    μ: float,
) -> Dict[str, float]:
    return {aid: w * μ for aid, w in base_weights.items()}

def hybrid_select_action(
    current_tokens: Iterable[str],
    reference_signatures: Dict[str, List[int]],
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    total_mb: int,
    used_mb: int,
) -> Tuple[str, float]:
    base_weights = compute_regret_weights(actions, counterfactuals)
    μ = memory_pressure_factor(total_mb, used_mb)
    weighted = apply_memory_pressure(base_weights, μ)
    cur_sig = signature(current_tokens, k=64)
    best_id = ""
    best_score = -math.inf
    for a in actions:
        sim = similarity(cur_sig, reference_signatures.get(a.id, cur_sig))
        score = weighted.get(a.id, 0.0) * sim
        if score > best_score:
            best_score = score
            best_id = a.id
    if best_id == "":
        return ("", 0.0)
    return (best_id, best_score)

def build_reference_signatures(actions: List[MathAction], token_map: Dict[str, List[str]]) -> Dict[str, List[int]]:
    refs: Dict[str, List[int]] = {}
    for a in actions:
        tokens = token_map.get(a.id, [])
        refs[a.id] = signature(tokens, k=64)
    return refs

if __name__ == "__main__":
    actions = [
        MathAction("action1", 10.0),
        MathAction("action2", 20.0),
        MathAction("action3", 30.0),
    ]
    counterfactuals = [
        MathCounterfactual("action1", 5.0),
        MathCounterfactual("action2", 15.0),
        MathCounterfactual("action3", 25.0),
    ]
    token_map = {
        "action1": ["token1", "token2"],
        "action2": ["token3", "token4"],
        "action3": ["token5", "token6"],
    }
    reference_signatures = build_reference_signatures(actions, token_map)
    total_mb, used_mb = _query_nvidia_smi()
    current_tokens = ["token1", "token2", "token3"]
    selected_action, score = hybrid_select_action(current_tokens, reference_signatures, actions, counterfactuals, total_mb, used_mb)
    print(f"Selected action: {selected_action}, Score: {score}")