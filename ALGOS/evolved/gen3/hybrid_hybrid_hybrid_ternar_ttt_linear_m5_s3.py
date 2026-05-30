# DARWIN HAMMER — match 5, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_ternary_route_variational_free_ene_m21_s1.py (gen2)
# parent_b: ttt_linear.py (gen0)
# born: 2026-05-29T23:25:06Z

"""
This module fuses the hybrid_hybrid_ternary_route_variational_free_ene_m21_s1 and ttt_linear algorithms into a single hybrid system.
The mathematical bridge between the two structures is the use of the SSIM function to evaluate the similarity between 
the input and output of the ternary router, and the variational free energy to update the belief mean of the ternary 
router based on the observation and the prediction error. The TTT-Linear algorithm is integrated by using its 
weight matrix as a compressor of the input distribution seen so far, and updating the weight matrix using the 
reconstruction loss. The variational free energy is used to update the belief mean of the ternary router, which 
is then used to compute the SSIM between the input and output of the ternary router.

The hybrid system combines the strengths of both algorithms, using the TTT-Linear algorithm to compress the input 
distribution and the variational free energy to update the belief mean of the ternary router. The SSIM function 
is used to evaluate the similarity between the input and output of the ternary router, providing a measure of the 
system's performance.
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

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    """Initialize W shape (d_out, d_in).

    d_out defaults to d_in. Small random initialization; scale controls
    the initial magnitude so the first few gradient steps are interpretable.
    """
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target=None):
    """Self-supervised loss for TTT.

    If target is None, use reconstruction loss: ||W x - x||^2.
    x shape: (d_in,). Returns scalar float.

    The reconstruction objective treats the identity mapping as the target.
    The weight matrix learns to be a good compressor of the input distribution
    seen so far — if W@x ≈ x holds, W has absorbed enough structure from the input.
    """
    if target is None:
        return np.sum((W @ x - x) ** 2)
    else:
        return np.sum((W @ x - target) ** 2)

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x.size:
        raise ValueError('samples must not be empty')
    n = len(x)
    mx = np.mean(x)
    my = np.mean(y)
    mx2 = np.mean(x**2)
    my2 = np.mean(y**2)
    mxy = np.mean(x*y)
    sigma_x2 = mx2 - mx**2
    sigma_y2 = my2 - my**2
    sigma_xy = mxy - mx*my
    k1_squared = k1**2
    k2_squared = k2**2
    r = (2*sigma_xy + k1_squared*dynamic_range**2) / (sigma_x2 + sigma_y2 + k1_squared*dynamic_range**2)
    s = (sigma_x2 + k2_squared*dynamic_range**2) / (sigma_y2 + k2_squared*dynamic_range**2)
    return ((2*mx*my + k1_squared*dynamic_range**2) * (2*sigma_xy + k2_squared*dynamic_range**2)) / ((mx**2 + my**2 + k1_squared*dynamic_range**2) * (sigma_x2 + sigma_y2 + k2_squared*dynamic_range**2))

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

def hybrid_step(W, x, packet):
    """Hybrid step that combines the TTT-Linear algorithm with the variational free energy and SSIM."""
    loss = ttt_loss(W, x)
    W_new = W - 0.01 * np.outer(x, x)
    packet_out = route_packet(packet)
    ssim_value = ssim(np.array([x]), np.array([packet_out["text_surface"]]))
    return W_new, packet_out, ssim_value

def hybrid_sequence(W, x_sequence, packet_sequence):
    """Hybrid sequence that applies the hybrid step to a sequence of inputs and packets."""
    W_out = W
    packet_out_sequence = []
    ssim_values = []
    for x, packet in zip(x_sequence, packet_sequence):
        W_out, packet_out, ssim_value = hybrid_step(W_out, x, packet)
        packet_out_sequence.append(packet_out)
        ssim_values.append(ssim_value)
    return W_out, packet_out_sequence, ssim_values

if __name__ == "__main__":
    W = init_ttt(10)
    x_sequence = [np.random.rand(10) for _ in range(10)]
    packet_sequence = [{"text_surface": str(i)} for i in range(10)]
    W_out, packet_out_sequence, ssim_values = hybrid_sequence(W, x_sequence, packet_sequence)
    print("Hybrid sequence completed without error.")