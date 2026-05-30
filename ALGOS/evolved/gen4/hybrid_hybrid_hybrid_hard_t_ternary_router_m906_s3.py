# DARWIN HAMMER — match 906, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s1.py (gen3)
# parent_b: ternary_router.py (gen0)
# born: 2026-05-29T23:31:42Z

"""Hybrid Fusion Module: hybrid_hard_truth_math_model + ternary_router

This module mathematically fuses the core topologies of two parent algorithms:

* **Parent A** – stylometry / high‑dimensional text feature extraction.  
  It produces a numeric representation **f ∈ ℝⁿ** (n = number of linguistic
  categories + punctuation symbols).

* **Parent B** – ternary routing engine.  
  It decides a processing channel based on the textual payload, intent and a
  context dictionary.

**Mathematical Bridge**

1. **Bilinear Projection** – a weight matrix **W ∈ ℝⁿˣᵐ** (n high‑dimensional,
   m low‑dimensional) defines a bilinear form **p = fᵀ·W ∈ ℝᵐ** that projects
   the stylometric vector onto a compact model space.

2. **Reliability‑Curvature Scalar** – a scalar **γ** is built from two
   orthogonal components:  
   * reliability **ρ** (derived from packet meta‑information) and  
   * curvature **κ** (the variance of the projected vector *p*).  

   The scalar is **γ = ρ·κ**.

3. **Hybrid Fusion** – the final routing signature **s ∈ ℝᵐ** is obtained by
   element‑wise multiplication of the projected vector with the scalar:
   **s = γ · p**.

   The ternary router then maps **s** to one of three engine channels by
   thresholding the L2‑norm of **s**.

The code below implements this fusion, exposing three public functions
`extract_features`, `bilinear_project`, `compute_reliability_score` and a
convenient wrapper `fuse_and_route`.
"""

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Shared linguistic resources (from Parent A)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Helper: simple mock of Parent B's route_command (no external deps)
# ----------------------------------------------------------------------
class MockRouteResult:
    """Mimic the object returned by the real route_command."""
    def __init__(self, channel: str, state: str):
        self.channel = channel
        self.state = state

    def to_dict(self) -> Dict[str, Any]:
        return {"engine_channel": self.channel, "outbound_state": self.state}


def mock_route_command(text: str, intent: str, context: Dict[str, Any]) -> MockRouteResult:
    """
    Very lightweight stand‑in for the real FairyFuse routing routine.
    It simply returns a deterministic channel based on the intent hash.
    """
    h = hash(intent) & 0xFFFFFFFF
    # three deterministic buckets
    if h % 3 == 0:
        channel = "cpu_fairyfuse_ternary"
    elif h % 3 == 1:
        channel = "gpu_fairyfuse_binary"
    else:
        channel = "fallback_cpu"
    return MockRouteResult(channel, "draft_only")


# ----------------------------------------------------------------------
# Core Hybrid Functions
# ----------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Very simple whitespace + punctuation tokenizer."""
    tokens: List[str] = []
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
    """
    Build a high‑dimensional feature vector `f ∈ ℝⁿ` from the input text.
    The vector consists of:
      * counts for each FUNCTION_CATS category,
      * counts for each punctuation symbol.
    The ordering is deterministic: categories first (alphabetical key order),
    then punctuation in the order defined by PUNCTUATION.
    """
    token_list = tokenize(text)
    # Initialise feature vector
    cat_keys = sorted(FUNCTION_CATS.keys())
    dim = len(cat_keys) + len(PUNCTUATION)
    f = np.zeros(dim, dtype=float)

    # Category counts
    for idx, cat in enumerate(cat_keys):
        vocab = FUNCTION_CATS[cat]
        f[idx] = sum(tok in vocab for tok in token_list)

    # Punctuation counts
    offset = len(cat_keys)
    for idx, punct in enumerate(PUNCTUATION):
        f[offset + idx] = token_list.count(punct)

    return f


def bilinear_project(features: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    """
    Perform the bilinear projection `p = fᵀ·W`.
    `features` is a row vector (1 × n), `weight_matrix` is (n × m).
    Returns a column vector `p ∈ ℝᵐ`.
    """
    if features.shape[0] != weight_matrix.shape[0]:
        raise ValueError("Dimension mismatch between features and weight matrix.")
    return features @ weight_matrix  # shape (m,)


def compute_reliability_score(packet: Dict[str, Any], projected: np.ndarray) -> float:
    """
    Compute the scalar γ = ρ·κ where:
      * ρ (reliability) ∈ (0, 1] depends on packet metadata:
            - longer raw text → higher reliability,
            - presence of explicit intent → boost,
            - presence of context → boost.
      * κ (curvature) = variance of the projected vector `projected`.
    """
    text_len = len(str(packet.get("text_surface") or packet.get("raw_command") or ""))
    base_rho = min(1.0, text_len / 200.0)  # saturates at 200 chars

    if packet.get("intent") or packet.get("normalized_intent"):
        base_rho = min(1.0, base_rho + 0.15)

    if packet.get("context"):
        base_rho = min(1.0, base_rho + 0.10)

    # curvature as variance (add epsilon to avoid zero)
    curvature = float(np.var(projected) + 1e-8)

    return base_rho * curvature


def fuse_and_route(packet: Dict[str, Any], weight_matrix: np.ndarray) -> Dict[str, Any]:
    """
    End‑to‑end hybrid operation:
      1. Extract high‑dimensional features from the textual payload.
      2. Project them via the bilinear form.
      3. Compute the reliability‑curvature scalar γ.
      4. Fuse to obtain the signature vector `s = γ·p`.
      5. Use the L2‑norm of `s` to select a ternary routing channel.
    Returns the routing dictionary compatible with Parent B's output format.
    """
    # Step 1 – textual source selection (mirrors Parent B's extraction)
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

    # Feature extraction & projection
    f_vec = extract_features(text)
    p_vec = bilinear_project(f_vec, weight_matrix)

    # Scalar γ
    gamma = compute_reliability_score(packet, p_vec)

    # Fusion signature
    s_vec = gamma * p_vec

    # Norm‑based ternary decision (three buckets)
    norm = float(np.linalg.norm(s_vec))
    if norm < 0.5:
        channel = "cpu_fairyfuse_ternary"
    elif norm < 1.5:
        channel = "gpu_fairyfuse_binary"
    else:
        channel = "fallback_cpu"

    # Build routing dict (mirroring Parent B's fields)
    route = {
        "engine_channel": channel,
        "outbound_state": "draft_only",
        "signature_norm": norm,
        "reliability_scalar": gamma,
    }
    return route


# ----------------------------------------------------------------------
# Utility: deterministic weight matrix generator (seeded for reproducibility)
# ----------------------------------------------------------------------
def generate_weight_matrix(input_dim: int, output_dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(input_dim, output_dim))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample packet mimicking Parent B's expected structure
    sample_packet = {
        "text_surface": "I can't believe the results! Are they accurate?",
        "normalized_intent": "verify_accuracy",
        "context": {"user_id": 123, "session": "abc"},
        "source": "unit_test",
    }

    # Determine dimensionalities based on linguistic resources
    n_features = len(sorted(FUNCTION_CATS.keys())) + len(PUNCTUATION)
    m_proj = 8  # modest low‑dimensional space

    W = generate_weight_matrix(n_features, m_proj)

    routing_info = fuse_and_route(sample_packet, W)

    # Emit JSON‑compatible result (Parent B uses json.dumps)
    print(json.dumps(routing_info, ensure_ascii=False, indent=2, sort_keys=True))