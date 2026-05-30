# DARWIN HAMMER — match 906, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s1.py (gen3)
# parent_b: ternary_router.py (gen0)
# born: 2026-05-29T23:31:42Z

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

# Shared linguistic resources
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCTUATION: List[str] = list("!?;:,.—-()[]{}\"'`/\\|@#$%^")

class MockRouteResult:
    def __init__(self, channel: str, state: str):
        self.channel = channel
        self.state = state

    def to_dict(self) -> Dict[str, Any]:
        return {"engine_channel": self.channel, "outbound_state": self.state}

def mock_route_command(text: str, intent: str, context: Dict[str, Any]) -> MockRouteResult:
    h = hash(intent) & 0xFFFFFFFF
    if h % 3 == 0:
        channel = "cpu_fairyfuse_ternary"
    elif h % 3 == 1:
        channel = "gpu_fairyfuse_binary"
    else:
        channel = "fallback_cpu"
    return MockRouteResult(channel, "draft_only")

def tokenize(text: str) -> List[str]:
    tokens = []
    current = []
    for ch in text:
        if ch.isalnum() or ch == "'":
            current.append(ch.lower())
        else:
            if current:
                tokens.append("".join(current))
                current = []
            if ch in PUNCTUATION:
                tokens.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens

def extract_features(text: str) -> np.ndarray:
    token_list = tokenize(text)
    cat_keys = sorted(FUNCTION_CATS.keys())
    dim = len(cat_keys) + len(PUNCTUATION)
    f = np.zeros(dim, dtype=float)

    for idx, cat in enumerate(cat_keys):
        vocab = FUNCTION_CATS[cat]
        f[idx] = sum(tok in vocab for tok in token_list)

    offset = len(cat_keys)
    for idx, punct in enumerate(PUNCTUATION):
        f[offset + idx] = token_list.count(punct)

    return f

def bilinear_project(features: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    if features.shape[0] != weight_matrix.shape[0]:
        raise ValueError("Dimension mismatch between features and weight matrix.")
    return features @ weight_matrix

def compute_reliability_score(packet: Dict[str, Any], projected: np.ndarray) -> float:
    text_len = len(str(packet.get("text_surface") or packet.get("raw_command") or ""))
    base_rho = min(1.0, text_len / 200.0)

    if packet.get("intent") or packet.get("normalized_intent"):
        base_rho = min(1.0, base_rho + 0.15)

    if packet.get("context"):
        base_rho = min(1.0, base_rho + 0.10)

    curvature = float(np.var(projected) + 1e-8)
    return base_rho * curvature

def fuse_and_route(packet: Dict[str, Any], weight_matrix: np.ndarray) -> Dict[str, Any]:
    text = str(
        packet.get("text_surface")
        or packet.get("raw_command")
        or packet.get("text")
        or ""
    )
    intent = str(
        packet.get("normalized_intent")
        or packet.get("intent")
        or "default_intent"
    )
    context = packet.get("context") or {}

    f_vec = extract_features(text)
    p_vec = bilinear_project(f_vec, weight_matrix)

    gamma = compute_reliability_score(packet, p_vec)

    s_vec = gamma * p_vec
    norm_s = np.linalg.norm(s_vec)

    threshold_low = 0.3
    threshold_high = 0.7

    if norm_s < threshold_low:
        channel = "cpu_fairyfuse_ternary"
    elif norm_s > threshold_high:
        channel = "gpu_fairyfuse_binary"
    else:
        channel = "fallback_cpu"

    return {"engine_channel": channel, "outbound_state": "draft_only"}

def improved_fuse_and_route(packet: Dict[str, Any], weight_matrix: np.ndarray) -> Dict[str, Any]:
    text = str(
        packet.get("text_surface")
        or packet.get("raw_command")
        or packet.get("text")
        or ""
    )
    intent = str(
        packet.get("normalized_intent")
        or packet.get("intent")
        or "default_intent"
    )
    context = packet.get("context") or {}

    f_vec = extract_features(text)
    p_vec = bilinear_project(f_vec, weight_matrix)

    gamma = compute_reliability_score(packet, p_vec)

    s_vec = gamma * p_vec
    norm_s = np.linalg.norm(s_vec)

    threshold_low = 0.3
    threshold_high = 0.7

    if norm_s < threshold_low:
        channel = "cpu_fairyfuse_ternary"
    elif norm_s > threshold_high:
        channel = "gpu_fairyfuse_binary"
    else:
        # Improved routing logic using a more nuanced approach
        if intent == "default_intent":
            channel = "fallback_cpu"
        elif context:
            channel = "gpu_fairyfuse_binary"
        else:
            channel = "cpu_fairyfuse_ternary"

    return {"engine_channel": channel, "outbound_state": "draft_only"}