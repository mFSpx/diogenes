# DARWIN HAMMER — match 3368, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s3.py (gen2)
# parent_b: hybrid_regret_engine_hybrid_liquid_time_c_m13_s0.py (gen2)
# born: 2026-05-29T23:49:43Z

"""Hybrid VRAM‑Aware Regret‑Weighted MinHash Decision Engine.

Parents:
- hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s3.py (VRAM scheduler & runtime receipts)
- hybrid_regret_engine_hybrid_liquid_time_c_m13_s0.py (Regret‑Weighted strategy + MinHash similarity)

Mathematical bridge:
The regret‑weighted probability p_i for action i is scaled by a memory‑pressure factor μ∈[0,1] derived from the current GPU VRAM utilisation.
A MinHash signature σ(x) of the current token set x is compared to reference signatures σ_ref(i) associated with each action.
The final hybrid score for action i is

    score_i = μ · p_i · Sim(σ(x), σ_ref(i))

where Sim(·,·) is the Jaccard‑like MinHash similarity. The action with maximal score is selected.
"""

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

# ----------------------------------------------------------------------
# Data structures (from parent B)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# VRAM utilities (excerpt from parent A)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
DEFAULT_RESERVE_MB = int(os.getenv("LUCIDOTA_VRAM_RESERVE_MB", "768"))

def _query_nvidia_smi() -> Tuple[int, int]:
    """Return (total_mb, used_mb) for the first GPU.
    If nvidia‑smi is unavailable, fall back to dummy values."""
    if not shutil.which("nvidia-smi"):
        return 16384, 0  # assume 16 GiB free
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
    """Map VRAM utilisation to a scaling factor μ∈[0,1].
    Full reserve → μ≈1, heavy usage → μ≈0."""
    usable = max(total_mb - reserve_mb, 1)
    excess = max(used_mb - reserve_mb, 0)
    μ = 1.0 - excess / usable
    return max(0.0, min(1.0, μ))

# ----------------------------------------------------------------------
# MinHash utilities (from parent B)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: Iterable[str], k: int = 64) -> List[int]:
    """MinHash signature of a token multiset."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity based on equal hash positions."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

# ----------------------------------------------------------------------
# Regret‑Weighted core (from parent B) with memory scaling
# ----------------------------------------------------------------------
def compute_regret_weights(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """Standard regret‑weighted probabilities (unnormalised)."""
    if not actions:
        return {}
    # Aggregate counterfactual outcomes per action
    cf_sum: Dict[str, float] = {}
    for cf in counterfactuals:
        cf_sum[cf.action_id] = cf_sum.get(cf.action_id, 0.0) + cf.outcome_value * cf.probability

    # Regret = expected_value - counterfactual outcome
    regrets = {}
    for a in actions:
        cf_val = cf_sum.get(a.id, 0.0)
        regrets[a.id] = max(a.expected_value - cf_val, 0.0)

    # Exponential weighting (softmax‑like) to obtain probabilities
    max_regret = max(regrets.values()) if regrets else 0.0
    exp_vals = {aid: math.exp(r - max_regret) for aid, r in regrets.items()}
    total = sum(exp_vals.values())
    return {aid: (v / total if total > 0 else 0.0) for aid, v in exp_vals.items()}

def apply_memory_pressure(
    base_weights: Dict[str, float],
    μ: float,
) -> Dict[str, float]:
    """Scale each weight by the VRAM pressure factor μ."""
    return {aid: w * μ for aid, w in base_weights.items()}

# ----------------------------------------------------------------------
# Hybrid decision engine
# ----------------------------------------------------------------------
def hybrid_select_action(
    current_tokens: Iterable[str],
    reference_signatures: Dict[str, List[int]],
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    total_mb: int,
    used_mb: int,
) -> Tuple[str, float]:
    """
    Return (selected_action_id, score) using the hybrid formula:
        score_i = μ · p_i · Sim(σ(cur), σ_ref(i))

    Parameters
    ----------
    current_tokens : iterable of str
        Tokens describing the current context (e.g., model prompt).
    reference_signatures : dict action_id → MinHash signature
        Pre‑computed signatures for each action.
    actions, counterfactuals : as defined above.
    total_mb, used_mb : VRAM statistics.

    Returns
    -------
    (action_id, score) of the best action. If no action is viable, returns ("", 0.0).
    """
    # 1. Compute base regret‑weighted probabilities
    base_weights = compute_regret_weights(actions, counterfactuals)

    # 2. Derive memory pressure factor μ
    μ = memory_pressure_factor(total_mb, used_mb)

    # 3. Apply μ to the weights
    weighted = apply_memory_pressure(base_weights, μ)

    # 4. MinHash signature of the current token set
    cur_sig = signature(current_tokens, k=64)

    # 5. Combine with similarity per action
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

# ----------------------------------------------------------------------
# Helper to build reference signatures (demo purpose)
# ----------------------------------------------------------------------
def build_reference_signatures(actions: List[MathAction], token_map: Dict[str, List[str]]) -> Dict[str, List[int]]:
    """
    token_map maps action_id → list of representative tokens.
    Returns a dict action_id → MinHash signature.
    """
    refs: Dict[str, List[int]] = {}
    for a in actions:
        tokens = token_map.get(a.id, [])
        refs[a.id] = signature(tokens, k=64)
    return refs

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny action space
    actions = [
        MathAction(id="A", expected_value=10.0, cost=1.0),
        MathAction(id="B", expected_value=8.0, cost=0.5),
        MathAction(id="C", expected_value=6.0, cost=0.2),
    ]

    # Counterfactual outcomes (pretend we observed something)
    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=7.0, probability=0.9),
        MathCounterfactual(action_id="B", outcome_value=5.0, probability=0.8),
        MathCounterfactual(action_id="C", outcome_value=4.0, probability=0.7),
    ]

    # Representative token sets per action (could be prompts, tags, etc.)
    token_map = {
        "A": ["alpha", "beta", "gamma"],
        "B": ["delta", "epsilon", "zeta"],
        "C": ["eta", "theta", "iota"],
    }

    ref_sigs = build_reference_signatures(actions, token_map)

    # Current context tokens
    current_tokens = ["beta", "delta", "lambda"]

    # Query (or mock) VRAM status
    total_mb, used_mb = _query_nvidia_smi()

    selected, score = hybrid_select_action(
        current_tokens=current_tokens,
        reference_signatures=ref_sigs,
        actions=actions,
        counterfactuals=counterfactuals,
        total_mb=total_mb,
        used_mb=used_mb,
    )

    print(f"Selected action: {selected} with hybrid score {score:.4f}")
    print(f"VRAM: {used_mb}/{total_mb} MB (pressure factor applied)")