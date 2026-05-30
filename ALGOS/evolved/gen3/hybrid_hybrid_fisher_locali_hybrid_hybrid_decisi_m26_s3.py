# DARWIN HAMMER — match 26, survivor 3
# gen: 3
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s1.py (gen2)
# parent_b: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s0.py (gen2)
# born: 2026-05-29T23:25:24Z

"""Hybrid Fisher‑SSIM Routing with Decision‑Hygiene Pruning
Parents:
- hybrid_fisher_localization_hybrid_ternary_route_m40_s1.py (Fisher information + SSIM routing)
- hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s0.py (Decision‑hygiene scoring + decreasing pruning)

Mathematical bridge:
The Fisher score  I(θ) provides a data‑driven weighting factor for the similarity
measure (SSIM) while the Shannon entropy H of the token‑frequency distribution
acts as a feature importance weight in the hygiene score.  Both weights are
modulated by a decreasing‑pruning probability p(t) that depends on the current
time step t.  The unified decision metric is

    M = p(t) · [ w_f·SSIM(x,y) + w_h·H·Σ_i w_i·f_i ]

where w_f = I(θ)/(I(θ)+ε) and w_h = H/(H+ε) are normalized Fisher and entropy
weights, f_i are binary feature flags extracted by regexes, and w_i are the
raw counts of those features.  The metric M drives packet routing and
prioritisation in a single coherent system.
"""

import math
import random
import sys
from pathlib import Path
import re
from collections import Counter
import numpy as np

# ----------------------------------------------------------------------
# Core components from Parent A
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
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural similarity index (SSIM) for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# ----------------------------------------------------------------------
# Core components from Parent B
# ----------------------------------------------------------------------
# Regexes for feature extraction
EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)
SCARCITY_RE = re.compile(r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b", re.I)
RISK_RE = re.compile(r"\b(?:kill|harm|danger|risk|threat|dangerous|unsafe|expose|exposure|attack|breach|vulnerability)\b", re.I)


def _extract_feature_counts(text: str) -> Counter:
    """Count occurrences of each thematic regex in the supplied text."""
    counts = Counter()
    counts["evidence"] = len(EVIDENCE_RE.findall(text))
    counts["planning"] = len(PLANNING_RE.findall(text))
    counts["delay"] = len(DELAY_RE.findall(text))
    counts["support"] = len(SUPPORT_RE.findall(text))
    counts["boundary"] = len(BOUNDARY_RE.findall(text))
    counts["outcome"] = len(OUTCOME_RE.findall(text))
    counts["impulsive"] = len(IMPULSIVE_RE.findall(text))
    counts["scarcity"] = len(SCARCITY_RE.findall(text))
    counts["risk"] = len(RISK_RE.findall(text))
    return counts


def shannon_entropy(counter: Counter) -> float:
    """Compute Shannon entropy of a discrete distribution given by a Counter."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counter.values())) / total
    # Guard against log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def hygiene_score(text: str) -> float:
    """
    Decision‑hygiene score.
    It is the entropy‑weighted sum of normalized feature counts.
    """
    counts = _extract_feature_counts(text)
    entropy = shannon_entropy(counts)
    if entropy == 0:
        return 0.0
    # Normalise each count by the max count to keep values in [0,1]
    max_cnt = max(counts.values()) if counts else 1
    norm_counts = np.array([c / max_cnt for c in counts.values()], dtype=float)
    # Weighted sum where entropy acts as a global scaling factor
    return entropy * np.mean(norm_counts)


def prune_probability(t: int, decay_rate: float = 0.05) -> float:
    """
    Decreasing‑rate pruning schedule.
    Returns a probability in (0,1] that monotonically decreases with time step t.
    """
    if t < 0:
        raise ValueError("time step t must be non‑negative")
    return 1.0 / (1.0 + decay_rate * t)


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def weighted_similarity(packet_text: str,
                        reference_text: str,
                        theta: float,
                        center: float,
                        width: float) -> float:
    """
    Compute SSIM between packet and reference, then weight it by the normalized
    Fisher information.
    """
    # Convert strings to numeric vectors (ASCII codes)
    x = np.fromiter((ord(c) for c in packet_text), dtype=np.uint8)
    y = np.fromiter((ord(c) for c in reference_text), dtype=np.uint8)

    base_ssim = ssim(x, y)
    fisher = fisher_score(theta, center, width)
    w_f = fisher / (fisher + 1e-12)  # normalise to (0,1)
    return w_f * base_ssim


def hybrid_decision_metric(packet: dict,
                           reference_text: str,
                           theta: float,
                           center: float,
                           width: float,
                           t: int) -> float:
    """
    Unified metric M = p(t) * [ w_f·SSIM + w_h·hygiene ].
    """
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    # 1) Weighted similarity (SSIM × Fisher)
    sim = weighted_similarity(text, reference_text, theta, center, width)

    # 2) Hygiene component (entropy‑weighted feature score)
    h_score = hygiene_score(text)
    entropy = shannon_entropy(_extract_feature_counts(text))
    w_h = entropy / (entropy + 1e-12)  # normalise to (0,1)

    # 3) Pruning probability
    p = prune_probability(t)

    return p * (sim + w_h * h_score)


def route_packet(packet: dict,
                reference_text: str,
                theta: float,
                center: float,
                width: float,
                t: int,
                threshold: float = 0.5) -> dict:
    """
    Decide whether to forward a packet based on the hybrid decision metric.
    Returns a new dict with routing metadata.
    """
    metric = hybrid_decision_metric(packet, reference_text, theta, center, width, t)
    decision = "forward" if metric >= threshold else "drop"

    routed = {
        "original_packet": packet,
        "metric": metric,
        "decision": decision,
        "timestamp": t,
        "routing_info": {
            "similarity_weighted": weighted_similarity(
                str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or ""),
                reference_text,
                theta,
                center,
                width,
            ),
            "hygiene_score": hygiene_score(
                str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
            ),
            "prune_probability": prune_probability(t),
        },
    }
    return routed


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dummy_packet = {
        "text_surface": "Please verify the source and provide a hash of the screenshot.",
        "raw_command": None,
        "source": "user123",
        "source_ref": "session42",
        "ontology_terms": ["verification", "security"],
        "epistemic_flag": True,
        "payload": {"id": 7},
    }

    ref = "The user must confirm the source and attach a SHA256 hash of the evidence."
    theta_val = 0.3
    center_val = 0.0
    width_val = 1.0

    for step in range(0, 5):
        result = route_packet(
            packet=dummy_packet,
            reference_text=ref,
            theta=theta_val,
            center=center_val,
            width=width_val,
            t=step,
            threshold=0.4,
        )
        print(f"t={step} | metric={result['metric']:.4f} | decision={result['decision']}")
    sys.exit(0)