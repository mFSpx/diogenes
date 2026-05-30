# DARWIN HAMMER — match 1, survivor 1
# gen: 1
# parent_a: ternary_router.py (gen0)
# parent_b: ssim.py (gen0)
# born: 2026-05-29T23:13:26Z

"""
This module fuses the ternary_router and ssim algorithms into a single hybrid system.
The mathematical bridge between the two structures is the use of the ssim function
to evaluate the similarity between the input and output of the ternary router.
The ternary router's route_command function is used to generate a response to the input,
and the ssim function is used to calculate the similarity between the input and the response.
This fusion enables the evaluation of the ternary router's performance using the ssim metric.
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
        raise ValueError('samples must have equal length')
    if not x.size:
        raise ValueError('samples must not be empty')
    n = len(x)
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def hybrid_route_packet(packet: dict[str, Any]) -> dict[str, Any]:
    route = route_packet(packet)
    input_text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    output_text = str(route.get("text_surface") or route.get("raw_command") or route.get("text") or "")
    input_array = np.array([ord(c) for c in input_text])
    output_array = np.array([ord(c) for c in output_text])
    similarity = ssim(input_array, output_array)
    route["similarity"] = similarity
    return route


def evaluate_ternary_router(packet: dict[str, Any]) -> float:
    route = hybrid_route_packet(packet)
    similarity = route["similarity"]
    return similarity


def generate_random_packet() -> dict[str, Any]:
    packet = {
        "text_surface": "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(100)),
        "normalized_intent": "bytewax_rete_bandit",
        "context": {},
    }
    return packet


if __name__ == "__main__":
    packet = generate_random_packet()
    route = hybrid_route_packet(packet)
    similarity = evaluate_ternary_router(packet)
    print("Route:", route)
    print("Similarity:", similarity)