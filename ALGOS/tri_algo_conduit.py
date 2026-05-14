#!/usr/bin/env python3
"""Tri-algo conduit: passive monitor -> Hoeffding gate -> self-righting recovery.

Pure resource-efficiency primitive for capture/ingress nodes. It keeps heavy
work dormant until a signal is statistically worth a burst, then supplies a
recovery score for quickly shedding malformed/heavy payloads back to standby.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ALGOS import hoeffding_tree, serpentina_self_righting, shannon_entropy, thanatosis


@dataclass(frozen=True)
class ConduitDecision:
    action: str  # standby | burst | recover
    confidence_gap: float
    epsilon: float
    signal_score: float
    noise_score: float
    dormancy_probability: float
    recovery_priority: float
    reason: str


def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    if not data:
        return 0.0
    chunk = data[:sample]
    return shannon_entropy.shannon_entropy(list(chunk)) / 8.0


def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> tuple[float, float]:
    """Return bounded (signal, noise) scores for an ingress candidate."""
    size = len(data)
    entropy = _byte_entropy(data)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml")) else 0.02
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)
    signal = max(0.0, min(1.0, 0.20 + status_bonus + mime_bonus + size_bonus + keyword_bonus + structure_bonus + 0.12 * entropy))
    noise = max(0.0, min(1.0, 0.58 - 0.22 * entropy - keyword_bonus - structure_bonus + (0.12 if size < 64 else 0.0)))
    return signal, noise


def recovery_from_payload(data: bytes, max_bytes: int, parse_error: bool = False) -> float:
    """Serpentina-style state complexity score for fast recovery priority."""
    size_ratio = min(4.0, len(data) / max(1, max_bytes))
    # Treat length/width/height as payload mass, nesting/shape, and remaining headroom.
    morph = serpentina_self_righting.Morphology(
        length=1.0 + size_ratio * 8.0,
        width=2.0 + (2.0 if parse_error else 0.5),
        height=max(0.5, 3.0 - size_ratio),
        mass=1.0 + size_ratio * 6.0 + (3.0 if parse_error else 0.0),
    )
    return serpentina_self_righting.recovery_priority(morph, max_index=12.0)


def decide(
    data: bytes,
    observations: int,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
    max_bytes: int = 1_500_000,
    delta: float = 1e-5,
    range_r: float = 1.0,
    tie_threshold: float = 0.03,
    standby_temperature: float = 0.35,
    parse_error: bool = False,
) -> ConduitDecision:
    """Decide whether to stay dormant, burst, or recover.

    Hoeffding is used as a conservative promotion gate: burst only when the
    score gap beats epsilon or the stream has become statistically stable.
    Thanatosis supplies the standby probability for non-promoted candidates.
    Serpentina supplies recovery priority when payload shape is dangerous.
    """
    n = max(1, observations)
    signal, noise = signal_scores(data, status_code, mime, keyword_hits, structural_links)
    split = hoeffding_tree.should_split(signal, noise, range_r, delta, n, tie_threshold=tie_threshold)
    recovery = recovery_from_payload(data, max_bytes=max_bytes, parse_error=parse_error)
    if parse_error or recovery >= 0.82:
        return ConduitDecision("recover", split.gain_gap, split.epsilon, signal, noise, 0.0, recovery, "serpentina_recovery")
    if split.should_split and signal >= noise:
        return ConduitDecision("burst", split.gain_gap, split.epsilon, signal, noise, 0.0, recovery, split.reason)
    dormancy = 1.0 - thanatosis.acceptance_probability(max(0.0, noise - signal), standby_temperature)
    return ConduitDecision("standby", split.gain_gap, split.epsilon, signal, noise, dormancy, recovery, "hoeffding_wait")
