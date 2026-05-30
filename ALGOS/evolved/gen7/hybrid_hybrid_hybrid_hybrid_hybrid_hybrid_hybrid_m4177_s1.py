# DARWIN HAMMER — match 4177, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_rbf_su_m1210_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1456_s0.py (gen6)
# born: 2026-05-29T23:53:55Z

"""Hybrid Algorithm: Path‑Signature / RBF Surrogate + SSIM + Bayesian Workshare

Parent A – hybrid_hybrid_hybrid_path_s_hybrid_hybrid_rbf_su_m1210_s2.py  
Parent B – hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1456_s0.py  

Mathematical bridge:
* Parent A produces deterministic feature vectors from text and a radial‑basis‑function (RBF)
  similarity `K(x,y)=exp(-ε²‖x−y‖²)`.  
* Parent B consumes a prior distribution (label allocator) and a likelihood (SSIM) to
  obtain posterior edge weights via a Bayesian update.

The fusion treats the RBF kernel as the *likelihood* of two texts being stylistically
similar, while the feature‑based label allocator supplies the *prior* probabilities for
each routing group.  The posterior is then used to weight a minimum‑cost tree or to
drive a work‑share allocation.  The three public functions below demonstrate this
integration.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
Vector = List[float]


def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random feature vector from a string."""
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric", "resilience_swar",
    ]
    return {k: rnd.random() for k in keys}


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rbf_kernel(a: Vector, b: Vector, epsilon: float = 1.0) -> float:
    """Gaussian RBF similarity K(a,b)=exp(-ε²‖a−b‖²)."""
    dist = euclidean(a, b)
    return math.exp(-((epsilon * dist) ** 2))


def lead_lag_transform(path: List[Vector]) -> List[Vector]:
    """Lead‑lag transformation of a multivariate discrete path."""
    transformed: List[Vector] = []
    for i in range(len(path)):
        prev = path[i - 1] if i > 0 else path[-1]
        diff = [path[i][j] - prev[j] for j in range(len(path[i]))]
        transformed.append(diff)
    return transformed


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Structural Similarity Index (SSIM) between two 1‑D arrays."""
    mu_x, mu_y = np.mean(x), np.mean(y)
    sigma_x, sigma_y = np.std(x), np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1, c2 = (0.01 ** 2), (0.03 ** 2)
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return float(numerator / denominator)


def allocate_labels(features: Dict[str, float], groups: Tuple[str, ...] = GROUPS) -> Dict[str, float]:
    """
    Convert a feature dict into a prior distribution over routing groups.
    The conversion is deterministic: each group receives a share proportional to the
    sum of a predefined subset of features.
    """
    # Simple deterministic mapping (hard‑coded for illustration)
    group_feature_map = {
        "codex": ["operator_visceral_ratio", "psyche_poetic_entropy"],
        "groq": ["operator_tech_ratio", "psyche_dissociative_index"],
        "cohere": ["operator_legal_osint_ratio", "psyche_wrath_velocity"],
        "local_models": ["operator_ledger_density", "resilience_swar"],
    }
    raw = {}
    total = 0.0
    for g in groups:
        subset = group_feature_map.get(g, [])
        val = sum(features.get(k, 0.0) for k in subset)
        raw[g] = val
        total += val
    # Avoid division by zero – fall back to uniform prior
    if total == 0.0:
        return {g: 1.0 / len(groups) for g in groups}
    return {g: raw[g] / total for g in groups}


# ----------------------------------------------------------------------
# Hybrid operations (require integration of both parents)
# ----------------------------------------------------------------------
def hybrid_similarity(text_a: str, text_b: str, epsilon: float = 1.0, alpha: float = 0.5) -> float:
    """
    Combine RBF similarity (parent A) and SSIM (parent B) into a single metric.
    The two similarities are blended linearly with weight `alpha` for the RBF part.
    """
    # Feature vectors
    fa = list(extract_full_features(text_a).values())
    fb = list(extract_full_features(text_b).values())

    # RBF similarity
    rbf_sim = rbf_kernel(fa, fb, epsilon=epsilon)

    # SSIM operates on numpy arrays; we treat the same vectors as 1‑D signals
    ssim_sim = compute_ssim(np.array(fa), np.array(fb))

    # Linear blend
    return alpha * rbf_sim + (1.0 - alpha) * ssim_sim


def bayesian_edge_weights(texts: List[str], epsilon: float = 1.0, alpha: float = 0.5) -> Dict[Tuple[int, int], float]:
    """
    Build a fully connected graph over the supplied texts.
    For each unordered pair (i, j) we compute a posterior edge weight:
        posterior ∝ prior_i * prior_j * likelihood(i,j)

    * Prior_i is the label‑allocation distribution summed over groups (scalar).
      We use the entropy of the prior as a single scalar prior strength.
    * Likelihood(i,j) is the hybrid similarity defined above.
    The resulting weight is normalized to the interval [0,1].
    """
    # Compute priors (scalar strength) for each text
    priors: List[float] = []
    for txt in texts:
        feats = extract_full_features(txt)
        label_dist = allocate_labels(feats)
        # Use Shannon entropy as a compact prior strength (higher entropy → less confident)
        entropy = -sum(p * math.log(p + 1e-12) for p in label_dist.values())
        priors.append(math.exp(-entropy))  # convert to a weight in (0,1]

    # Compute unnormalized posterior weights
    unnorm: Dict[Tuple[int, int], float] = {}
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            likelihood = hybrid_similarity(texts[i], texts[j], epsilon=epsilon, alpha=alpha)
            weight = priors[i] * priors[j] * likelihood
            unnorm[(i, j)] = weight

    # Normalization to [0,1]
    max_w = max(unnorm.values(), default=1.0)
    if max_w == 0:
        max_w = 1.0
    return {(i, j): w / max_w for (i, j), w in unnorm.items()}


def route_and_allocate(packets: List[str], prototypes: Dict[str, str],
                       epsilon: float = 1.0, alpha: float = 0.5) -> Dict[str, List[str]]:
    """
    Route each packet (text) to the group whose prototype yields the highest hybrid similarity.
    Afterwards, allocate work‑share fractions to the groups based on the aggregated priors
    of their assigned packets.

    Returns a dict mapping each group name to the list of packet texts assigned to it.
    """
    # Pre‑compute prototype feature vectors
    proto_vecs: Dict[str, List[float]] = {
        g: list(extract_full_features(proto).values()) for g, proto in prototypes.items()
    }

    routing: Dict[str, List[str]] = {g: [] for g in prototypes}
    group_priors: Dict[str, float] = {g: 0.0 for g in prototypes}

    for pkt in packets:
        pkt_vec = list(extract_full_features(pkt).values())
        # Compute hybrid similarity to each prototype
        sims = {
            g: hybrid_similarity(pkt, prototypes[g], epsilon=epsilon, alpha=alpha)
            for g in prototypes
        }
        best_group = max(sims, key=sims.get)
        routing[best_group].append(pkt)

        # Update group prior strength (same entropy‑based scalar as in bayesian_edge_weights)
        feats = extract_full_features(pkt)
        label_dist = allocate_labels(feats)
        entropy = -sum(p * math.log(p + 1e-12) for p in label_dist.values())
        group_priors[best_group] += math.exp(-entropy)

    # Convert raw prior strengths to normalized work‑share percentages
    total_strength = sum(group_priors.values()) or 1.0
    workshare = {g: round(100.0 * group_priors[g] / total_strength, 2) for g in group_priors}
    # Attach workshare info as a side‑effect attribute for demonstration
    routing["_workshare"] = workshare  # type: ignore
    return routing


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "Artificial intelligence drives modern software engineering."
    ]

    # 1. Hybrid similarity demo
    sim01 = hybrid_similarity(sample_texts[0], sample_texts[1])
    print(f"Hybrid similarity (0,1): {sim01:.4f}")

    # 2. Bayesian edge weights demo
    edges = bayesian_edge_weights(sample_texts)
    for (i, j), w in edges.items():
        print(f"Posterior edge weight ({i},{j}): {w:.4f}")

    # 3. Routing + workshare demo
    prototypes = {
        "codex": "Code generation and software synthesis.",
        "groq": "High‑throughput inference for large language models.",
        "cohere": "Natural language understanding and generation.",
        "local_models": "On‑device lightweight neural networks."
    }
    routing = route_and_allocate(sample_texts, prototypes)
    for grp, pkts in routing.items():
        if grp == "_workshare":
            continue
        print(f"\nGroup '{grp}' received {len(pkts)} packet(s):")
        for p in pkts:
            print(f"  - {p[:40]}...")
    print("\nWorkshare allocation (% of total prior strength):")
    print(routing["_workshare"])