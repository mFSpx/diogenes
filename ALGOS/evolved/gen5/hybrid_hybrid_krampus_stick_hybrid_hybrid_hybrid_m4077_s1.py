# DARWIN HAMMER — match 4077, survivor 1
# gen: 5
# parent_a: hybrid_krampus_stickers_hybrid_pheromone_inf_m107_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_ternar_m1215_s1.py (gen4)
# born: 2026-05-29T23:53:26Z

"""
Hybrid module combining the Krampus sticker text analytics (Parent A) with the Pheromone infotaxis dynamics (Parent B) and the Fisher information score integration into the bandit problem framework and the SSIM metric to evaluate the similarity between packet payloads (Parent C).

The mathematical bridge is found in the following interface:

- Parent A extracts a feature vector **f(text)** = (tokens, entropy, link_counts, …) and uses it to initialize a set of PheromoneEntry objects.
- Parent B treats each scalar feature as a pheromone signal **s** with exponential decay and aggregation.
- Parent C uses the Fisher information score and the SSIM metric to guide the selection of an optimal sensing angle, a token hypothesis, and a bandit action.

The hybrid maps **f(text)** → a set of PheromoneEntry objects where the initial signal value is the normalized feature magnitude and the half-life τ is a monotonic function of the text entropy (high entropy → slower decay).
"""

import math
import random
import sys
import pathlib
import numpy as np
from collections import Counter
from typing import Any, List, Sequence, Dict, Tuple

# ----------------------------------------------------------------------
# Parent A – text feature extraction (simplified)
# ----------------------------------------------------------------------

def normalize_ws(text: str) -> str:
    """Collapse whitespace to a single space and strip."""
    return re.sub(r"\s+", " ", str(text or "")).strip()

def token_count(text: str) -> int:
    """Count whitespace-separated tokens."""
    return len(re.findall(r"\S+", text or ""))

def shannon_entropy(symbols: List[str]) -> float:
    """Classic Shannon entropy H = -Σ p·log₂(p) for a list of symbols."""
    if not symbols:
        return 0.0
    total = len(symbols)
    freq = Counter(symbols)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())

def entropy_for_text(text: str, max_len: int = 10_000) -> float:
    """Entropy of the first `max_len` characters of `text`."""
    if not text:
        return 0.0
    snippet = list(text[:max_len])
    return shannon_entropy(snippet)

def links_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract markdown links, wikilinks and bare URLs."""
    links: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    # Markdown links [anchor](url)
    for m in re.finditer(r"\[([^\]]{0,20})\]\((.*?)\)", text):
        title, url = m.groups()
        if (title, url, text) not in seen:
            seen.add((title, url, text))
            links.append({"title": title, "url": url})

    return links

# ----------------------------------------------------------------------
# Parent B building blocks (Pheromone decay & aggregation)
# ----------------------------------------------------------------------

def pheromone_decay(s: float, tau: float) -> float:
    """Exponential decay of pheromone signal s(t) = s₀·½^{Δt/τ}"""
    return s * 0.5**(1/tau)

def aggregate_pheromones(pheromones: List[float]) -> float:
    """Sum of pheromone signals"""
    return sum(pheromones)

# ----------------------------------------------------------------------
# Parent C building blocks (Fisher score & SSIM)
# ----------------------------------------------------------------------

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    return (2 * dynamic_range + k1 * (x + y) - k2 * (x * y)) / (dynamic_range + k1 * x + k1 * y + k2 * x * y)

# ----------------------------------------------------------------------
# Hybrid module
# ----------------------------------------------------------------------

def normalize_feature(f: float) -> float:
    """Normalize feature magnitude"""
    return f / (f + 1)

def hybrid_text_features(text: str) -> Tuple[float, float, float]:
    """Hybrid feature vector from text"""
    tokens = token_count(text)
    entropy = entropy_for_text(text)
    links = len(links_from_text(text))
    return normalize_feature(tokens), normalize_feature(entropy), normalize_feature(links)

def inject_pheromones(features: Tuple[float, float, float]) -> List[float]:
    """Inject pheromone signals"""
    pheromones = []
    for f in features:
        tau = 1 / (1 + f)  # half-life τ is a monotonic function of the text entropy
        pheromone = normalize_feature(f)  # initial signal value is the normalized feature magnitude
        pheromones.append(pheromone_decay(pheromone, tau))
    return pheromones

def hybrid_similarity(x: List[float], y: List[float]) -> float:
    """Hybrid similarity metric"""
    return compute_ssim(x, y) * (1 + fisher_score(x[0], y[0], 1))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    text = "This is a sample text with multiple tokens and links"
    features = hybrid_text_features(text)
    pheromones = inject_pheromones(features)
    print(hybrid_similarity(pheromones, [0.5, 0.5, 0.5]))