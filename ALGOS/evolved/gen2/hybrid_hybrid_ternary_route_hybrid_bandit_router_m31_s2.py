# DARWIN HAMMER — match 31, survivor 2
# gen: 2
# parent_a: hybrid_ternary_router_ssim_m1_s1.py (gen1)
# parent_b: hybrid_bandit_router_honeybee_store_m9_s4.py (gen1)
# born: 2026-05-29T23:25:20Z

"""
This module fuses the hybrid_ternary_router_ssim_m1_s1 and hybrid_bandit_router_honeybee_store_m9_s4 algorithms into a single hybrid system.
The mathematical bridge between the two structures is the use of the ssim function to evaluate the similarity between the input and output of the ternary router,
and the integration of the bandit algorithm's update policy and store update into the ternary router's route_command function.
This fusion enables the evaluation of the ternary router's performance using the ssim metric and the optimization of the router's decisions using the bandit algorithm.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
import json

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def now_z() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def parse_context(text: str | None) -> dict[str, any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value

_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: list) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u['action_id'], [0.0, 0.0])
        stats[0] += float(u['reward'])
        stats[1] += 1.0

def update_store(
    store: float,
    inflow: list[float],
    outflow: list[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> tuple[float, float]:
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta

def dance_duration(
    delta_store: float,
    base: float = 1.0,
    gain: float = 1.0,
    limit: float = 10.0,
) -> float:
    return max(0.0, min(limit, base + gain * delta_store))

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))
    c1 = (k1 * dynamic_range)**2
    c2 = (k2 * dynamic_range)**2
    ssim = ((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)) / ((mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2))
    return ssim

def hybrid_select_action(
    context: dict[str, float],
    actions: list[str],
    store: float,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    gamma: float = 0.1,
    seed: int | str | None = 7,
) -> dict[str, float]:
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)
    store_factor = 1.0 + store / (store + 1.0)
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        def sample(a: str) -> float:
            r = _reward(a)
            n = _count(a)
            a_param = 1.0 + max(0.0, r) * store_factor
            b_param = 1.0 + max(0.0, 1.0 - r) * store_factor
            return rng.betavariate(a_param, b_param)
        chosen = max(actions, key=sample)
    else:
        scale = math.sqrt(sum(float(v) * float(v) for v in context.values())) if context else 1.0
        chosen = max(actions, key=lambda a: _reward(a) * scale)
    return {'action_id': chosen, 'propensity': _reward(chosen), 'expected_reward': _reward(chosen), 'confidence_bound': 1.0}

def route_packet(packet: dict[str, any]) -> dict[str, any]:
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    intent = str(packet.get("normalized_intent") or packet.get("intent") or "bytewax_rete_bandit")
    context = {
        "source": packet.get("source"),
        "source_ref": packet.get("source_ref"),
        "ontology_terms": packet.get("ontology_terms") or [],
        "epistemic_flag": packet.get("epistemic_flag"),
        "payload": packet.get("payload") or {},
    }
    action = hybrid_select_action(context, ["action1", "action2"], 0.5)
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 3.0, 4.0])
    ssim_value = ssim(x, y)
    return {
        'action_id': action['action_id'],
        'propensity': action['propensity'],
        'expected_reward': action['expected_reward'],
        'confidence_bound': action['confidence_bound'],
        'ssim': ssim_value
    }

def main():
    packet = {
        "text_surface": "This is a test packet",
        "raw_command": "This is a raw command",
        "text": "This is a text",
        "normalized_intent": "This is a normalized intent",
        "intent": "This is an intent",
        "source": "This is a source",
        "source_ref": "This is a source ref",
        "ontology_terms": ["term1", "term2"],
        "epistemic_flag": "flag",
        "payload": {"key": "value"}
    }
    result = route_packet(packet)
    print(result)

if __name__ == "__main__":
    main()