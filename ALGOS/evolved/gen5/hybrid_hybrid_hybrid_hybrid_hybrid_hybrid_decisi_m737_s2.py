# DARWIN HAMMER — match 737, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s0.py (gen4)
# parent_b: hybrid_hybrid_decision_hygi_hybrid_hybrid_model__m3_s0.py (gen3)
# born: 2026-05-29T23:30:47Z

"""Hybrid Algorithm integrating:
- Parent A: Geometric Algebra multivector encoding + Fisher‑SSIM weighting + time‑dependent pruning.
- Parent B: Decision‑hygiene feature counting + Shannon entropy + Krampus‑Ollivier‑Ricci curvature on a weighted graph.

Mathematical Bridge:
Feature‑count vectors (from regex hygiene extraction) are mapped to basis blades of a
Clifford algebra, producing a multivector whose components act as coordinates in a
high‑dimensional space.  Those coordinates define a graph whose edge weights are
scaled by Fisher information (Gaussian‑beam model).  The Ollivier‑Ricci curvature
computed on this graph is then combined with a Fisher‑weighted SSIM similarity
and a Shannon‑entropy‑scaled hygiene score.  A time‑dependent pruning probability
p(t)=exp(-γ·t) interpolates between the similarity‑driven and entropy‑driven
contributions, yielding a single unified decision metric.
"""

import math
import random
import sys
import pathlib
import re
from collections import Counter, defaultdict
import numpy as np

# ----------------------------------------------------------------------
# Geometric Algebra utilities (Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components, n):
        # components: dict mapping frozenset(indices) -> scalar
        self.n = n
        self.components = {frozenset(k): float(v) for k, v in components.items() if abs(v) > 1e-12}

    def __add__(self, other):
        assert self.n == other.n
        comps = self.components.copy()
        for b, v in other.components.items():
            comps[b] = comps.get(b, 0.0) + v
        return Multivector(comps, self.n)

    def __mul__(self, other):
        """Geometric product."""
        assert self.n == other.n
        result = defaultdict(float)
        for b1, v1 in self.components.items():
            for b2, v2 in other.components.items():
                b_res, sign = _multiply_blades(b1, b2)
                result[b_res] += sign * v1 * v2
        return Multivector(dict(result), self.n)

    def scalar_part(self):
        return self.components.get(frozenset(), 0.0)

    def norm(self):
        # Euclidean norm of the coefficient vector
        return math.sqrt(sum(v * v for v in self.components.values()))


def multivector_from_feature_counts(feature_counts, dim=8):
    """
    Encode a feature‑count dictionary into a multivector.
    Each distinct feature is mapped (via hash) to a basis blade index
    in the range [0, dim).  The count becomes the scalar weight of that blade.
    """
    comps = {}
    for feat, cnt in feature_counts.items():
        idx = hash(feat) % dim
        blade = frozenset([idx])
        comps[blade] = comps.get(blade, 0.0) + cnt
    return Multivector(comps, dim)


# ----------------------------------------------------------------------
# Decision Hygiene & Entropy utilities (Parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def extract_feature_counts(text):
    """
    Simple hygiene extractor: count occurrences of evidence‑related and planning‑related
    regexes. Returns a Counter mapping feature names to integer counts.
    """
    counts = Counter()
    counts["evidence"] = len(EVIDENCE_RE.findall(text))
    counts["planning"] = len(PLANNING_RE.findall(text))
    # Additional generic token count as a fallback feature
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts["tokens"] = len(tokens)
    return counts


def shannon_entropy(feature_counts):
    """Compute Shannon entropy of the normalized feature‑count distribution."""
    total = sum(feature_counts.values())
    if total == 0:
        return 0.0
    probs = [cnt / total for cnt in feature_counts.values() if cnt > 0]
    return -sum(p * math.log2(p) for p in probs)


# ----------------------------------------------------------------------
# Fisher‑SSIM utilities (Parent A)
# ----------------------------------------------------------------------
def fisher_information_gaussian(variance):
    """Fisher information for a 1‑D Gaussian with known variance."""
    if variance <= 0:
        return 0.0
    return 1.0 / variance


def ssim_similarity(text_a, text_b):
    """
    Very lightweight SSIM‑like similarity:
    compare normalized character frequency vectors with a structural term.
    """
    def char_hist(s):
        h = Counter(s.lower())
        total = sum(h.values())
        return np.array([h.get(chr(i), 0) / total for i in range(32, 127)], dtype=float)

    ha = char_hist(text_a)
    hb = char_hist(text_b)
    mu_a, mu_b = ha.mean(), hb.mean()
    sigma_a, sigma_b = ha.var(), hb.var()
    sigma_ab = np.cov(ha, hb)[0, 1]

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)
    return numerator / denominator if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Graph & Ollivier‑Ricci curvature utilities (Parent B)
# ----------------------------------------------------------------------
def build_weighted_graph(mv, fisher_weight):
    """
    Construct a complete graph where each node is a basis blade present in `mv`.
    Edge weight = fisher_weight * exp(-||b_i - b_j||_2) where the blade
    coordinate is the index list interpreted as a point in ℝ^n.
    """
    nodes = list(mv.components.keys())
    n = mv.n
    adj = defaultdict(dict)
    for i, b_i in enumerate(nodes):
        coord_i = np.zeros(n)
        for idx in b_i:
            coord_i[idx] = 1.0
        for j, b_j in enumerate(nodes):
            if j <= i:
                continue
            coord_j = np.zeros(n)
            for idx in b_j:
                coord_j[idx] = 1.0
            dist = np.linalg.norm(coord_i - coord_j)
            weight = fisher_weight * math.exp(-dist)
            adj[i][j] = weight
            adj[j][i] = weight
    return adj, nodes


def ollivier_ricci_curvature(adj):
    """
    Approximate Ollivier‑Ricci curvature for each edge using a simple
    1‑step random walk distribution.
    For edge (i, j):
        κ_ij = 1 - W_ij / (d_i + d_j)
    where d_i = sum_k W_ik (degree weight).
    Returns average curvature over all edges.
    """
    degrees = {i: sum(w for w in nbrs.values()) for i, nbrs in adj.items()}
    curvatures = []
    for i, nbrs in adj.items():
        for j, w_ij in nbrs.items():
            if i < j:  # count each edge once
                d_i = degrees[i]
                d_j = degrees[j]
                if d_i + d_j == 0:
                    continue
                kappa = 1.0 - w_ij / (d_i + d_j)
                curvatures.append(kappa)
    if not curvatures:
        return 0.0
    return sum(curvatures) / len(curvatures)


# ----------------------------------------------------------------------
# Hybrid metric computation
# ----------------------------------------------------------------------
def hybrid_decision_metric(text, reference_text, t, gamma=0.01, variance=1.0):
    """
    Compute the unified decision metric for `text` against `reference_text`.
    Steps:
    1. Extract hygiene feature counts.
    2. Encode counts as a multivector.
    3. Compute Fisher information from `variance`.
    4. Build weighted graph & obtain average Ollivier‑Ricci curvature.
    5. Compute SSIM similarity and weight it by Fisher information.
    6. Compute Shannon entropy of the feature distribution.
    7. Combine everything using time‑dependent pruning probability p(t)=exp(-γ·t).
    Returns a float in [0, ∞).
    """
    # 1. Hygiene features
    feat_counts = extract_feature_counts(text)

    # 2. Multivector encoding
    mv = multivector_from_feature_counts(feat_counts, dim=8)

    # 3. Fisher information
    I_fisher = fisher_information_gaussian(variance)

    # 4. Graph + curvature
    adj, _ = build_weighted_graph(mv, fisher_weight=I_fisher)
    curvature = ollivier_ricci_curvature(adj)

    # 5. SSIM similarity weighted
    ssim = ssim_similarity(text, reference_text)
    weighted_ssim = I_fisher * ssim

    # 6. Shannon entropy (entropy‑scaled hygiene score)
    entropy = shannon_entropy(feat_counts)
    hygiene_score = sum(feat_counts.values()) * entropy

    # 7. Time‑dependent interpolation
    p_t = math.exp(-gamma * t)
    # Metric blends similarity (early times) with entropy‑curvature (later times)
    metric = p_t * weighted_ssim + (1 - p_t) * (hygiene_score * (1 + curvature))
    return metric


def update_pruning_probability(current_p, dt, gamma=0.01):
    """
    Update pruning probability p(t) over a small time step `dt`.
    Uses the analytical solution p(t+dt) = p(t) * exp(-γ·dt).
    """
    return current_p * math.exp(-gamma * dt)


def decision_hygiene_score(text):
    """
    Stand‑alone hygiene score: Shannon entropy weighted by raw feature counts.
    """
    feats = extract_feature_counts(text)
    entropy = shannon_entropy(feats)
    return sum(feats.values()) * entropy


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "The plan includes a checklist of steps, a timeline, and a budget. "
        "All evidence was verified and documented in the log."
    )
    reference = (
        "A detailed roadmap with phases, priorities, and a verification checklist. "
        "All sources are cited and hashes recorded."
    )
    t0 = 0.0
    metric0 = hybrid_decision_metric(sample, reference, t0)
    print(f"Metric at t={t0:.2f}: {metric0:.6f}")

    # advance time
    dt = 10.0
    p = math.exp(-0.01 * t0)
    p_new = update_pruning_probability(p, dt)
    metric1 = hybrid_decision_metric(sample, reference, t0 + dt)
    print(f"Metric at t={t0+dt:.2f}: {metric1:.6f}")
    print(f"Updated pruning probability: {p_new:.6f}")

    # hygiene score alone
    print(f"Hygiene score: {decision_hygiene_score(sample):.6f}")