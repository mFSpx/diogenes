# DARWIN HAMMER — match 71, survivor 0
# gen: 3
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s0.py (gen2)
# parent_b: hybrid_ternary_lens_router_hybrid_decision_hygi_m44_s0.py (gen2)
# born: 2026-05-29T23:25:41Z

"""Hybrid Fisher‑Ternary Router
This module fuses the core mathematics of two parent algorithms:

* **Parent A** – Fisher‑localization & SSIM routing.
  - Gaussian beam model → Fisher information score.
  - Structural Similarity Index (SSIM) between two 1‑D signals.

* **Parent B** – Ternary lens router & Shannon‑entropy decision hygiene.
  - Regex‑driven ternary vector (‑1, 0, +1) describing semantic categories.
  - Shannon entropy of the ternary distribution.

**Mathematical bridge**  
The Fisher score is used as a *probability‑weighting* for each component of the ternary
vector.  The weighted ternary histogram feeds the Shannon‑entropy calculation, giving an
information‑theoretic measure that respects the underlying physical confidence (Fisher).
The resulting entropy is then combined with the SSIM between the packet’s textual surface
and a reference sample to produce a unified routing metric.

The hybrid routing decision therefore integrates:

θ  → FisherScore(θ)                (continuous confidence)
v  → ternary vector (categorical evidence)
w_i = FisherScore(θ) * |v_i|       (weights)
H = - Σ p_i log2 p_i               (entropy of weighted histogram)
S = SSIM(packet_text, reference)   (structural similarity)
Decision = α·H + β·S                (α,β are tunable scalars)

"""

import math
import random
import sys
from pathlib import Path
import re
import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """One‑dimensional Structural Similarity Index."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    cov = np.cov(x, y, ddof=0)[0, 1]
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
TERNARY_DIMS = 12

# Regex catalogue – each pattern maps to a ternary dimension.
_REGEX_CATALOG = [
    re.compile(r"\b(evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I),  # 0
    re.compile(r"\b(plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I),  # 1
    re.compile(r"\b(pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|first|after|review)\b", re.I),  # 2
    re.compile(r"\b(ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I),  # 3
    re.compile(r"\b(boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I),  # 4
    re.compile(r"\b(done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I),  # 5
    re.compile(r"\b(rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I),  # 6
    re.compile(r"\b(scared|fear|anxious|uncertain|doubt|hesitant|reluctant|wary|cautious)\b", re.I),  # 7 (added for completeness)
    re.compile(r"\b(limited|scarcity|shortage|rare|few|insufficient|lack)\b", re.I),  # 8
    re.compile(r"\b(optimistic|hope|positive|confident|enthusiastic|ready|eager)\b", re.I),  # 9
    re.compile(r"\b(technical|code|bug|exception|stacktrace|runtime|compile|deploy|api|endpoint)\b", re.I),  #10
    re.compile(r"\b(user|client|customer|consumer|end‑user|stakeholder)\b", re.I),  #11
]


def ternary_vector(text: str) -> np.ndarray:
    """
    Produce a ternary vector (‑1, 0, +1) of length TERNARY_DIMS.
    For each regex pattern:
        - match → +1
        - explicit negation (preceded by "no" or "not") → -1
        - otherwise → 0
    """
    vec = np.zeros(TERNARY_DIMS, dtype=int)
    lowered = text.lower()
    for idx, pat in enumerate(_REGEX_CATALOG):
        if pat.search(lowered):
            # crude negation detection
            neg_pat = re.compile(r"\b(no|not)\s+" + pat.pattern, re.I)
            if neg_pat.search(lowered):
                vec[idx] = -1
            else:
                vec[idx] = 1
    return vec


def shannon_entropy(weights: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute Shannon entropy of a discrete distribution defined by `weights`.
    `weights` must be non‑negative; they are normalized to sum to 1.
    """
    if np.any(weights < 0):
        raise ValueError("weights must be non‑negative")
    total = np.sum(weights)
    if total < eps:
        return 0.0
    prob = weights / total
    prob = prob[prob > eps]  # discard zero‑mass entries
    return -np.sum(prob * np.log2(prob))


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def weighted_ternary_entropy(text: str, theta: float, center: float, width: float) -> float:
    """
    Compute Shannon entropy of the ternary vector derived from `text`,
    weighting each absolute component by the Fisher score of `theta`.
    """
    vec = ternary_vector(text)
    # Fisher score provides a scalar confidence; we broadcast it.
    fscore = fisher_score(theta, center, width)
    # Weight magnitude of each ternary component; zero entries stay zero.
    weights = np.abs(vec.astype(float)) * fscore
    return shannon_entropy(weights)


def hybrid_similarity_metric(packet: dict, reference_text: str,
                             center: float = 0.0, width: float = 1.0,
                             alpha: float = 0.6, beta: float = 0.4) -> dict:
    """
    Combine the weighted ternary entropy with SSIM to produce a routing score.

    Returns a dict with:
        - 'entropy': weighted ternary entropy
        - 'ssim': structural similarity index
        - 'score': α·entropy + β·ssim
        - 'decision': 'high', 'medium', or 'low' based on score thresholds
    """
    # Extract a numeric proxy for theta (packet length normalized)
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    theta = len(text) / 100.0  # arbitrary scaling; 100 chars ≈ 1.0

    entropy_val = weighted_ternary_entropy(text, theta, center, width)

    # Prepare arrays for SSIM – map characters to integer codes
    pkt_arr = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
    ref_arr = np.frombuffer(reference_text.encode('utf-8'), dtype=np.uint8)

    # Pad the shorter array to match lengths (required by ssim)
    max_len = max(pkt_arr.size, ref_arr.size)
    pkt_arr = np.pad(pkt_arr, (0, max_len - pkt_arr.size), constant_values=0)
    ref_arr = np.pad(ref_arr, (0, max_len - ref_arr.size), constant_values=0)

    ssim_val = ssim(pkt_arr.astype(float), ref_arr.astype(float))

    combined = alpha * entropy_val + beta * ssim_val

    if combined > 1.5:
        decision = "high"
    elif combined > 0.8:
        decision = "medium"
    else:
        decision = "low"

    return {
        "entropy": entropy_val,
        "ssim": ssim_val,
        "score": combined,
        "decision": decision
    }


def route_packet(packet: dict, reference_text: str,
                center: float = 0.0, width: float = 1.0) -> dict:
    """
    Produce a routing payload enriched with hybrid metrics.
    The returned dict contains the original packet fields plus the hybrid analysis.
    """
    analysis = hybrid_similarity_metric(packet, reference_text, center, width)
    routed = packet.copy()
    routed.update({
        "hybrid_entropy": analysis["entropy"],
        "hybrid_ssim": analysis["ssim"],
        "hybrid_score": analysis["score"],
        "routing_decision": analysis["decision"]
    })
    return routed


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dummy_packet = {
        "text_surface": "Please verify the source and provide the hash of the screenshot.",
        "source": "user123",
        "raw_command": None,
        "intent": "request_verification"
    }
    reference = "Verify the source and attach the hash of the screenshot for audit."
    result = route_packet(dummy_packet, reference, center=0.0, width=2.0)
    print("Hybrid routing result:")
    for k, v in result.items():
        print(f"{k}: {v}")