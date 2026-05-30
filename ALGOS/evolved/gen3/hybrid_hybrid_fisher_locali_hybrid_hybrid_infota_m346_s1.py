# DARWIN HAMMER — match 346, survivor 1
# gen: 3
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s4.py (gen2)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_fisher_locali_m132_s0.py (gen2)
# born: 2026-05-29T23:28:21Z

"""Hybrid Fisher‑Infotaxis‑MinHash algorithm.

Parent A contributes the Fisher information score for a directional parameter
θ (gaussian beam model).  Parent B contributes a MinHash‑based similarity that
serves as the probability p_hit in an infotaxis‑style expected‑entropy
computation.  Both parents share the concept of *information density*: Fisher
score measures the information content of an observation angle, while the
infotaxis component measures the information gain of a hypothesised token set.
The bridge is built by using the Fisher score as a weighting factor for the
expected entropy derived from MinHash similarity.  The resulting hybrid metric
guides simultaneous selection of an optimal sensing angle and a token
hypothesis that together maximise expected information gain.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from collections import Counter
from typing import Any, List, Sequence, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Parent B building blocks (MinHash & Infotaxis)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash from a seed and a token."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Sequence[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def minhash_similarity(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
    """Jaccard‑like similarity derived from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def shannon_entropy(counter: Counter) -> float:
    """Shannon entropy of a discrete distribution given by a Counter."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for cnt in counter.values():
        if cnt == 0:
            continue
        p = cnt / total
        ent -= p * math.log(p, 2)
    return ent


def infotaxis_expected_entropy(
    current_tokens: Sequence[str],
    candidate_tokens: Sequence[str],
    k: int = 128,
) -> float:
    """
    Expected post‑action entropy for an infotaxis step.

    p_hit is approximated by MinHash similarity between the current token set
    and the hypothetical set after adding ``candidate_tokens``.
    """
    sig_current = minhash_signature(current_tokens, k)
    sig_candidate = minhash_signature(list(current_tokens) + list(candidate_tokens), k)

    p_hit = minhash_similarity(sig_current, sig_candidate)

    # Distributions for hit / miss scenarios
    hit_counter = Counter(current_tokens) + Counter(candidate_tokens)
    miss_counter = Counter(current_tokens)

    H_hit = shannon_entropy(hit_counter)
    H_miss = shannon_entropy(miss_counter)

    return p_hit * H_hit + (1.0 - p_hit) * H_miss


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_information_metric(
    theta: float,
    center: float,
    width: float,
    current_tokens: Sequence[str],
    candidate_tokens: Sequence[str],
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """
    Combined metric that rewards high Fisher information (angle) while
    minimising expected entropy (infotaxis).  The lower the metric, the more
    informative the joint choice.
    """
    f_score = fisher_score(theta, center, width)
    e_entropy = infotaxis_expected_entropy(current_tokens, candidate_tokens)

    # Normalise components to comparable scale
    f_norm = f_score / (f_score + 1.0)          # maps (0,∞) → (0,1)
    e_norm = e_entropy / (e_entropy + 1.0)     # maps (0,∞) → (0,1)

    # Hybrid: maximise Fisher (=> larger f_norm) and minimise entropy (=> smaller e_norm)
    return alpha * (1.0 - f_norm) + beta * e_norm


def best_hybrid_choice(
    angle_candidates: Sequence[float],
    token_candidate_sets: Sequence[Sequence[str]],
    center: float,
    width: float,
    current_tokens: Sequence[str],
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Tuple[float, Sequence[str]]:
    """
    Search over the Cartesian product of angle candidates and token candidate sets
    and return the pair that minimises the hybrid_information_metric.
    """
    if not angle_candidates:
        raise ValueError("angle_candidates required")
    if not token_candidate_sets:
        raise ValueError("token_candidate_sets required")

    best_pair = None
    best_score = float("inf")

    for theta in angle_candidates:
        for cand_tokens in token_candidate_sets:
            score = hybrid_information_metric(
                theta, center, width, current_tokens, cand_tokens, alpha, beta
            )
            if score < best_score:
                best_score = score
                best_pair = (theta, cand_tokens)

    return best_pair  # type: ignore


def route_packet_hybrid(
    packet: Dict[str, Any],
    reference_text: str,
    center: float,
    width: float,
    angle_candidates: List[float],
    token_candidates: List[List[str]],
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Dict[str, Any]:
    """
    High‑level routing routine that decides both a sensing angle and a token
    augmentation for a packet based on the hybrid metric.
    """
    # Extract textual payload (fallbacks mimic parent A)
    text = str(packet.get("text_surface") or packet.get("raw_command") or "")
    current_tokens = text.split()

    # Choose optimal hybrid pair
    best_angle, best_tokens = best_hybrid_choice(
        angle_candidates,
        token_candidates,
        center,
        width,
        current_tokens,
        alpha,
        beta,
    )

    # Attach decision metadata to the packet
    packet["selected_angle"] = best_angle
    packet["augmented_tokens"] = best_tokens
    packet["reference_text"] = reference_text
    packet["hybrid_metric"] = hybrid_information_metric(
        best_angle,
        center,
        width,
        current_tokens,
        best_tokens,
        alpha,
        beta,
    )
    return packet


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy packet
    pkt = {"text_surface": "alpha beta gamma"}

    # Parameters
    center_angle = 0.0
    beam_width = 1.0
    angles = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # Token hypotheses (could be generated by a language model)
    token_hyps = [
        ["delta"],
        ["epsilon", "zeta"],
        ["eta", "theta", "iota"],
    ]

    routed = route_packet_hybrid(
        packet=pkt,
        reference_text="reference payload",
        center=center_angle,
        width=beam_width,
        angle_candidates=angles,
        token_candidates=token_hyps,
        alpha=0.6,
        beta=0.4,
    )

    print("Selected angle:", routed["selected_angle"])
    print("Augmented tokens:", routed["augmented_tokens"])
    print("Hybrid metric value:", routed["hybrid_metric"])