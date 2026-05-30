# DARWIN HAMMER — match 22, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s0.py (gen2)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_ternary_lens__m71_s1.py (gen3)
# born: 2026-05-29T23:26:20Z

"""Hybrid algorithm combining pheromone‑based entropy (Parent A) with Fisher‑information‑driven
Gaussian beam analysis and ternary regex profiling (Parent B).

Mathematical bridge:
- Pheromone probabilities `p_i` (A) are used as mixing coefficients for a set of Gaussian
  beams representing the `TERNARY_DIMS` latent dimensions (B).
- For each dimension we compute Fisher information `I_i` of a Gaussian beam.
- The weighted Fisher vector `w_i = p_i * I_i` is normalised to a probability distribution,
  on which we evaluate Shannon entropy `H(w)`.  This entropy quantifies the uncertainty of the
  pheromone‑guided information gain across the ternary dimensions.
- Simultaneously, a decision‑hygiene score derived from evidence‑related regex matches
  (A) produces a discrete distribution that is compared to `w` via the one‑dimensional SSIM
  (B), yielding a similarity measure that fuses the two parent topologies.
The resulting composite score blends surface usage (pheromones), information‑theoretic
uncertainty (entropy of Fisher‑weighted beams), and textual decision hygiene (regex‑based
ternary vector)."""

import math
import random
import sys
import re
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks (adapted)
# ----------------------------------------------------------------------
def entropy(probabilities, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    probs = [(p / total) for p in probabilities if p > 0]
    return -sum(p * math.log(max(p, eps)) for p in probs)


def decision_hygiene_score(text: str) -> dict[str, int]:
    """Count evidence‑type tokens in *text*."""
    evidence_pat = re.compile(
        r"\b(evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
        r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
        re.I,
    )
    matches = evidence_pat.findall(text)
    return {"evidence_tokens": len(matches)}


def calculate_pheromone_probabilities(surface_key: str, limit: int, db_url: str | None = None) -> list[float]:
    """
    Stub implementation: generate *limit* random positive values and normalise them.
    The original version queried a PostgreSQL database; here we keep the signature
    while providing a deterministic fallback for testing.
    """
    random.seed(hash(surface_key) & 0xffffffff)
    raw = [random.random() + 0.1 for _ in range(limit)]
    total = sum(raw)
    return [v / total for v in raw]


# ----------------------------------------------------------------------
# Parent‑B building blocks (adapted)
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
# Ternary regex catalogue (Parent‑B)
# ----------------------------------------------------------------------
TERNARY_DIMS = 12
_REGEX_CATALOG = [
    re.compile(r"\b(evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I),  # 0
    re.compile(r"\b(plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I),  # 1
    re.compile(r"\b(pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|first|after|review)\b", re.I),  # 2
    re.compile(r"\b(ask|call|text|friend|colleague|mentor|expert|consult|discuss)\b", re.I),  # 3
    re.compile(r"\b(error|fail|exception|bug|crash|stacktrace|traceback)\b", re.I),  # 4
    re.compile(r"\b(optimize|performance|speed|latency|throughput|efficiency)\b", re.I),  # 5
    re.compile(r"\b(security|vulnerability|exploit|attack|threat|risk|patch)\b", re.I),  # 6
    re.compile(r"\b(user|client|customer|consumer|end[- ]?user)\b", re.I),  # 7
    re.compile(r"\b(deploy|release|version|build|ci|cd|pipeline)\b", re.I),  # 8
    re.compile(r"\b(data|dataset|record|row|column|table|schema)\b", re.I),  # 9
    re.compile(r"\b(model|algorithm|prediction|inference|training|validation)\b", re.I),  #10
    re.compile(r"\b(license|compliance|legal|policy|gdpr|privacy)\b", re.I),  #11
]


def ternary_vector(text: str) -> np.ndarray:
    """Map *text* to a binary vector of length ``TERNARY_DIMS`` using the regex catalogue."""
    vec = np.zeros(TERNARY_DIMS, dtype=int)
    for idx, pat in enumerate(_REGEX_CATALOG):
        if pat.search(text):
            vec[idx] = 1
    return vec


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def weighted_fisher_information(pheromones: list[float],
                                thetas: list[float],
                                centers: list[float],
                                widths: list[float]) -> np.ndarray:
    """
    Compute Fisher information for each dimension and weight it by the corresponding
    pheromone probability.  The three parameter lists must have the same length.
    Returns a NumPy array of weighted Fisher scores.
    """
    if not (len(pheromones) == len(thetas) == len(centers) == len(widths)):
        raise ValueError("All input lists must have equal length")
    fisher_vals = np.array([
        fisher_score(theta, center, width)
        for theta, center, width in zip(thetas, centers, widths)
    ], dtype=float)
    weights = np.array(pheromones, dtype=float)
    return weights * fisher_vals


def hybrid_entropy_of_fisher(pheromones: list[float],
                             thetas: list[float],
                             centers: list[float],
                             widths: list[float]) -> float:
    """
    Compute the Shannon entropy of the normalized weighted Fisher information vector.
    This quantifies the uncertainty of information gain across the pheromone‑biased
    ternary dimensions.
    """
    weighted = weighted_fisher_information(pheromones, thetas, centers, widths)
    if weighted.sum() == 0:
        raise ValueError("Weighted Fisher vector contains no mass")
    normalized = weighted / weighted.sum()
    return entropy(normalized.tolist())


def hybrid_decision_metric(surface_key: str,
                           limit: int,
                           text: str,
                           thetas: list[float] | None = None) -> dict:
    """
    End‑to‑end hybrid metric:
    1. Obtain pheromone probabilities for *surface_key*.
    2. Build Gaussian‑beam parameters for each ternary dimension
       (centers = evenly spaced, widths = constant, thetas = optional or derived).
    3. Compute entropy of the weighted Fisher information.
    4. Produce a ternary regex vector from *text* and compare it to the
       normalized weighted Fisher distribution via SSIM.
    5. Combine the decision‑hygiene count with the entropy and SSIM into a
       composite dictionary.
    """
    # 1. Pheromones
    pheromones = calculate_pheromone_probabilities(surface_key, limit)

    dim = min(len(pheromones), TERNARY_DIMS)
    pheromones = pheromones[:dim]

    # 2. Gaussian parameters (simple deterministic scheme)
    centers = np.linspace(-1.0, 1.0, dim).tolist()
    widths = [0.3 for _ in range(dim)]
    if thetas is None:
        # Use a theta that drifts with text length to inject variability
        base_theta = (len(text) % 100) / 100.0 * 2 - 1  # map to [-1,1]
        thetas = [base_theta for _ in range(dim)]
    else:
        thetas = thetas[:dim]

    # 3. Entropy of weighted Fisher
    fisher_entropy = hybrid_entropy_of_fisher(pheromones, thetas, centers, widths)

    # 4. Ternary vector and SSIM comparison
    tern_vec = ternary_vector(text).astype(float)
    weighted_fisher = weighted_fisher_information(pheromones, thetas, centers, widths)
    if weighted_fisher.sum() == 0:
        norm_fisher = np.zeros_like(weighted_fisher)
    else:
        norm_fisher = weighted_fisher / weighted_fisher.sum()
    # Pad both vectors to the same length for SSIM
    max_len = max(len(tern_vec), len(norm_fisher))
    pad_tern = np.pad(tern_vec, (0, max_len - len(tern_vec)), constant_values=0)
    pad_fish = np.pad(norm_fisher, (0, max_len - len(norm_fisher)), constant_values=0)
    similarity = ssim(pad_tern, pad_fish)

    # 5. Decision hygiene count
    dh_score = decision_hygiene_score(text)["evidence_tokens"]

    # Composite result
    return {
        "surface_key": surface_key,
        "pheromone_distribution": pheromones,
        "fisher_entropy": fisher_entropy,
        "ssim_similarity": similarity,
        "evidence_token_count": dh_score,
        "composite_score": fisher_entropy * (1 + similarity) + dh_score,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic test parameters
    surface = "example_surface"
    limit = 8
    sample_text = (
        "The evidence was verified and the source document was attached. "
        "We plan to deploy the new version after the review."
    )
    result = hybrid_decision_metric(surface, limit, sample_text)
    for k, v in result.items():
        print(f"{k}: {v}")