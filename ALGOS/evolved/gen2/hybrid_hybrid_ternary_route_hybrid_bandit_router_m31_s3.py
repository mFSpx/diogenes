# DARWIN HAMMER — match 31, survivor 3
# gen: 2
# parent_a: hybrid_ternary_router_ssim_m1_s1.py (gen1)
# parent_b: hybrid_bandit_router_honeybee_store_m9_s4.py (gen1)
# born: 2026-05-29T23:25:20Z

"""
This module fuses the hybrid_ternary_router_ssim_m1_s1 and hybrid_bandit_router_honeybee_store_m9_s4 algorithms into a single hybrid system.
The mathematical bridge between the two structures is the use of the ssim function to evaluate the similarity between the input and output of the ternary router,
which is then used to update the policy of the bandit router using the reward function.
The ternary router's route_command function is used to generate a response to the input, and the ssim function is used to calculate the similarity between the input and the response.
This fusion enables the evaluation of the ternary router's performance using the ssim metric and the adaptation of the bandit router's policy using the reward function.
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import math
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.fairyfuse.fairyfuse_backend import resident_engine_from_env, route_command

RUNTIME_DIR = ROOT / "04_RUNTIME" / "fairyfuse"
DEFAULT_HEARTBEAT = RUNTIME_DIR / "ternary_router_heartbeat.jsonl"

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def parse_context(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value

def route_packet(packet: dict[str, Any]) -> dict[str, Any]:
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    intent = str(packet.get("normalized_intent") or packet.get("intent") or "bytewax_rete_bandit")
    context = {
        "source": packet.get("source"),
        "source_ref": packet.get("source_ref"),
        "ontology_terms": packet.get("ontology_terms") or [],
        "epistemic_flag": packet.get("epistemic_flag"),
        "payload": packet.get("payload") or {},
    }
    route = route_command(text[:4096], intent, context).to_dict()
    route["engine_channel"] = "cpu_fairyfuse_ternary"
    route["outbound_state"] = "draft_only"
    return route

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError("Input arrays must be the same length")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.sqrt(np.var(x))
    sigma_y = np.sqrt(np.var(y))
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
    return ssim

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
        stats = _POLICY.setdefault(u[0], [0.0, 0.0])
        stats[0] += float(u[1])
        stats[1] += 1.0

def hybrid_select_action(context: dict[str, float], actions: list[str], store: float, algorithm: str = "linucb", epsilon: float = 0.1, eta: float = 0.1, gamma: float = 0.1, seed: int | str | None = 7) -> tuple[str, float]:
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
    propensity = _reward(chosen)
    return chosen, propensity

def hybrid_route_packet(packet: dict[str, Any]) -> dict[str, Any]:
    route = route_packet(packet)
    action = hybrid_select_action(route["context"], [route["intent"]], 1.0)
    reward = ssim(np.array([0.5]), np.array([action[1]]))
    update_policy([action])
    route["reward"] = reward
    return route

def hybrid_update_store(store: float, inflow: list[float], outflow: list[float], alpha: float = 1.0, beta: float = 1.0, dt: float = 1.0) -> tuple[float, float]:
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta

if __name__ == "__main__":
    packet = {
        "text_surface": "Hello, world!",
        "normalized_intent": "greet",
        "context": {
            "source": "user",
            "source_ref": "1234567890",
            "ontology_terms": ["hello", "world"],
            "epistemic_flag": True,
            "payload": {"name": "John", "age": 30},
        },
    }
    route = hybrid_route_packet(packet)
    print(route)
    store = 1.0
    inflow = [0.5, 0.3]
    outflow = [0.2]
    new_store, delta = hybrid_update_store(store, inflow, outflow)
    print(new_store, delta)