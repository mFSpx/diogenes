# DARWIN HAMMER — match 21, survivor 1
# gen: 2
# parent_a: hybrid_ternary_router_ssim_m1_s1.py (gen1)
# parent_b: variational_free_energy.py (gen0)
# born: 2026-05-29T23:22:53Z

"""
This module fuses the ternary_router_ssim_m1_s1 and variational_free_energy algorithms into a single hybrid system.
The mathematical bridge between the two structures is the use of the SSIM function to evaluate the similarity between 
the input and output of the ternary router, and the variational free energy to update the belief mean of the ternary 
router based on the observation and the prediction error. This fusion enables the evaluation of the ternary router's 
performance using the SSIM metric and the variational free energy principle.
"""

import json
import os
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
    route = {"text_surface": "example response"}
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

def kl_gaussian(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p: float | np.ndarray,
    sigma_p: float | np.ndarray,
) -> float:
    mu_q = np.asarray(mu_q, dtype=float)
    sigma_q = np.asarray(sigma_q, dtype=float)
    mu_p = np.asarray(mu_p, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float)

    if np.any(sigma_q <= 0) or np.any(sigma_p <= 0):
        raise ValueError("Standard deviations must be strictly positive.")

    kl = (
        np.log(sigma_p / sigma_q)
        + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2)
        - 0.5
    )
    return float(np.sum(kl))

def free_energy_gaussian(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p_s: float | np.ndarray,
    sigma_p_s: float | np.ndarray,
    obs: float | np.ndarray,
    sigma_obs: float | np.ndarray,
) -> float:
    obs = np.asarray(obs, dtype=float)
    mu_q = np.asarray(mu_q, dtype=float)
    sigma_obs = np.asarray(sigma_obs, dtype=float)

    if np.any(sigma_obs <= 0):
        raise ValueError("sigma_obs must be strictly positive.")

    kl = kl_gaussian(mu_q, sigma_q, mu_p_s, sigma_p_s)

    recon = float(
        np.sum(
            0.5 * ((obs - mu_q) ** 2 / sigma_obs ** 2)
            + 0.5 * np.log(2.0 * np.pi * sigma_obs ** 2)
        )
    )

    return kl + recon

def belief_update(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    obs: float | np.ndarray,
    A: float | np.ndarray,
    sigma_obs: float | np.ndarray,
    eta: float = 0.1,
) -> float | np.ndarray:
    mu_q = np.atleast_1d(np.asarray(mu_q, dtype=float))
    obs = np.atleast_1d(np.asarray(obs, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    sigma_obs = np.atleast_1d(np.asarray(sigma_obs, dtype=float))

    if np.any(sigma_obs <= 0):
        raise ValueError("sigma_obs must be strictly positive.")

    pred_error = obs - np.dot(A, mu_q)
    mu_q_new = mu_q + eta * np.dot(A.T, pred_error) / sigma_obs ** 2
    return mu_q_new

def hybrid_route_packet(packet: dict[str, Any]) -> dict[str, Any]:
    route = route_packet(packet)
    input_text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    output_text = str(route.get("text_surface") or route.get("raw_command") or route.get("text") or "")
    input_array = np.array([ord(c) for c in input_text])
    output_array = np.array([ord(c) for c in output_text])
    similarity = ssim(input_array, output_array)
    route["similarity"] = similarity
    return route

def hybrid_belief_update(packet: dict[str, Any]) -> dict[str, Any]:
    route = hybrid_route_packet(packet)
    input_text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    output_text = str(route.get("text_surface") or route.get("raw_command") or route.get("text") or "")
    input_array = np.array([ord(c) for c in input_text])
    output_array = np.array([ord(c) for c in output_text])
    mu_q = np.mean(input_array)
    sigma_q = np.std(input_array)
    obs = np.mean(output_array)
    sigma_obs = np.std(output_array)
    A = np.array([[1]])
    mu_q_new = belief_update(mu_q, sigma_q, obs, A, sigma_obs)
    route["belief_mean"] = mu_q_new
    return route

def evaluate_hybrid_ternary_router(packet: dict[str, Any]) -> float:
    route = hybrid_route_packet(packet)
    similarity = route["similarity"]
    return similarity

if __name__ == "__main__":
    packet = {
        "text_surface": "example input",
        "normalized_intent": "bytewax_rete_bandit",
        "context": {},
    }
    route = hybrid_route_packet(packet)
    belief = hybrid_belief_update(packet)
    similarity = evaluate_hybrid_ternary_router(packet)
    print("Route:", route)
    print("Belief Mean:", belief["belief_mean"])
    print("Similarity:", similarity)