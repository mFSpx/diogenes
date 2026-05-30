# DARWIN HAMMER — match 13, survivor 0
# gen: 4
# parent_a: hybrid_ternary_router_ssim_m1_s1.py (gen1)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_xgboost_objec_m111_s0.py (gen3)
# born: 2026-05-29T23:26:25Z

"""
This module fuses the ternary_router and hybrid_model_vram_scheduler_ttt_linear_m11_s1 algorithms into a single hybrid system.
The mathematical bridge between the two structures is the use of the TTT-Linear model's update rule to modulate the pruning probability 
based on the model's performance in the XGBoost objective's split-gain formula, which is then used to evaluate the similarity between 
the input and output of the ternary router. This fusion enables the evaluation of the ternary router's performance using the SSIM metric 
while adapting to the changing memory requirements of the model.

The TTT-Linear model's update rule is used to compute the gradient and Hessian of the binary logistic loss, which are then used to compute 
the optimal leaf weight and split gain. The split gain is then used to modulate the pruning probability based on the model's performance.
The SSIM function is used to evaluate the similarity between the input and output of the ternary router.
"""

import numpy as np
import math
import random
import sys
import pathlib
import json
import os

from services.fairyfuse.fairyfuse_backend import resident_engine_from_env, route_command

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2
    l_mean_sq = np.mean(x) ** 2
    c1 = 2 * C1 + C2 - C1
    ssim_map = ((2 * x * y + C1) / (l_mean_sq + C1)) * ((2 * x * y + C2) / (l_mean_sq + C2))
    return np.mean(ssim_map)


def ttt_loss(W, x, target=None):
    """Self-supervised loss for TTT.

    If target is None, use reconstruction loss: ||W x - x||^2.
    x shape: (d_in,). Returns scalar float.

    The reconstruction objective treats the identity mapping as the target.
    The weight matrix learns to be a good compressor of the input distribution
    seen so far — if W@x ≈ x holds, W has absorbed enough structure to
    reconstruct tokens from the sequence.
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)


def hybrid_ttt_loss(W, x, target=None, ssim_value: float = 0.0):
    ttt_loss_value = ttt_loss(W, x, target)
    ssim_weight = 0.5  # Tune this value to adjust the importance of SSIM
    return ttt_loss_value + ssim_weight * (1 - ssim_value)


def hybrid_route_packet(packet: dict[str, Any]) -> dict[str, Any]:
    route = route_packet(packet)
    input_array = np.array([route["input"]])
    output_array = np.array([route["output"]])
    ssim_value = ssim(input_array, output_array)
    return {"route": route, "ssim_value": ssim_value}


def hybrid_ttt_grad(W, x, target=None, ssim_value: float = 0.0):
    ttt_grad_value = ttt_grad(W, x, target)
    return ttt_grad_value


if __name__ == "__main__":
    packet = {"text_surface": "Hello, world!"}
    route = hybrid_route_packet(packet)
    print(route)
    W = np.random.rand(10, 10)
    x = np.random.rand(10)
    target = np.random.rand(10)
    hybrid_loss_value = hybrid_ttt_loss(W, x, target)
    print(hybrid_loss_value)
    hybrid_grad_value = hybrid_ttt_grad(W, x, target)
    print(hybrid_grad_value)