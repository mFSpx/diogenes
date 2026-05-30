# DARWIN HAMMER — match 26, survivor 5
# gen: 3
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s1.py (gen2)
# parent_b: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s0.py (gen2)
# born: 2026-05-29T23:25:24Z

import sys
import math
import random
import re
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for the Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# ----------------------------------------------------------------------
# Parent B components (regex feature extraction, entropy, pruning)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE    = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
SUPPORT_RE  = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
OUTCOME_RE  = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)
SCARCITY_RE = re.compile(r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b", re.I)
RISK_RE     = re.compile(r"\b(?:kill|harm|danger|risk|threat|expose|vulnerable|attack|breach|leak|dangerous)\b", re.I)


FEATURE_REGEXES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
    "risk": RISK_RE,
}


def extract_features(text: str) -> dict:
    """Count matches for each semantic regex category."""
    counts = {}
    for name, pattern in FEATURE_REGEXES.items():
        matches = pattern.findall(text)
        counts[name] = len(matches)
    return counts


def shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution."""
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs))


def hygiene_score(features: dict, center: float, width: float) -> float:
    """
    Decision‑hygiene score.
    Each feature count is treated as a theta for the Fisher information;
    the Fisher scores are used as weights for a normalized feature vector,
    and the weighted Shannon entropy is returned.
    """
    theta_vals = np.array(list(features.values()), dtype=float)
    if theta_vals.sum() == 0:
        return 0.0

    fisher_weights = np.array([fisher_score(theta, center, width) for theta in theta_vals])
    weight_sum = fisher_weights.sum()
    if weight_sum == 0:
        normalized_weights = np.ones_like(fisher_weights) / fisher_weights.size
    else:
        normalized_weights = fisher_weights / weight_sum

    probs = theta_vals / theta_vals.sum()
    return float(np.dot(normalized_weights, np.vectorize(shannon_entropy)(probs[:, None])))


def prune_probability(t: int, decay_rate: float = 0.1) -> float:
    """Decreasing pruning probability p(t) = exp(-γ·t)."""
    if t < 0:
        raise ValueError("time step t must be non‑negative")
    return math.exp(-decay_rate * t)


# ----------------------------------------------------------------------
# Hybrid functions (integration of both parents)
# ----------------------------------------------------------------------
def hybrid_similarity(packet: dict, reference_text: str,
                     t: int, center: float, width: float, 
                     gamma: float = 0.1) -> float:
    """
    Compute a unified similarity/hygiene metric.

    Steps
    -----
    1. SSIM between packet text and reference, weighted by Fisher information of
       the packet length.
    2. Decision‑hygiene entropy score derived from regex features, also weighted
       by Fisher information.
    3. Interpolate between the two using the pruning probability p(t):
       metric = (1 - p) * (ssim_weighted) + p * (entropy_score)
    """
    pkt_text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    if not pkt_text:
        ssim_val = 0.0
    else:
        x = np.fromiter((ord(c) for c in pkt_text), dtype=np.uint8, count=len(pkt_text))
        y = np.fromiter((ord(c) for c in reference_text), dtype=np.uint8, count=len(reference_text))
        if x.size < y.size:
            x = np.pad(x, (0, y.size - x.size), constant_values=0)
        elif y.size < x.size:
            y = np.pad(y, (0, x.size - y.size), constant_values=0)
        raw_ssim = ssim(x.astype(float), y.astype(float))
        length_fisher = fisher_score(len(pkt_text), center, width)
        ssim_val = raw_ssim * length_fisher

    features = extract_features(pkt_text.lower())
    entropy_val = hygiene_score(features, center, width)
    p_t = prune_probability(t, gamma)
    return (1 - p_t) * ssim_val + p_t * entropy_val


def main():
    packet = {"text_surface": "This is a test packet"}
    reference_text = "This is a reference text"
    t = 10
    center = 5.0
    width = 2.0
    print(hybrid_similarity(packet, reference_text, t, center, width))


if __name__ == "__main__":
    main()