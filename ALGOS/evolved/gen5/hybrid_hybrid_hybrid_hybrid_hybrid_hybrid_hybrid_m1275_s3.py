# DARWIN HAMMER — match 1275, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1012_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m58_s0.py (gen4)
# born: 2026-05-29T23:34:55Z

"""Hybrid algorithm combining Bayesian decision hygiene (Parent A) with Fisher‑weighted similarity and sheaf‑based feature mapping (Parent B).

Mathematical bridge:
- Parent A provides a Bayesian update framework that operates on probabilities derived from feature‑based evidence.
- Parent B supplies a Fisher information score (derivative of a Gaussian beam) that quantifies the sensitivity of a similarity measure; this score is used as a *weight* on the structural similarity index (SSIM) between two feature vectors.
- The sheaf structure from Parent B defines linear restriction maps between feature subspaces; these maps are applied to the raw feature vectors before similarity computation, ensuring that both Bayesian and Fisher‑weighted components operate on a common transformed space.

Thus, the hybrid pipeline is:
1. Extract discrete feature counts from text (Parent A).
2. Transform each feature vector through a sheaf‑defined restriction map (Parent B).
3. Compute a perceptual hash of the transformed vector and obtain a Hamming distance‑derived angle θ.
4. Evaluate a Gaussian beam at θ, derive its Fisher score, and use this score to weight the SSIM between the two transformed vectors.
5. Feed the weighted similarity as the likelihood into the Bayesian update, yielding a final decision probability.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----- Parent A components -------------------------------------------------

def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Bayesian marginal probability rule."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Bayesian update rule."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


def extract_features(text: str) -> np.ndarray:
    """Extract feature counts from a string."""
    import re
    counts = []
    FEATURE_REGEXES = [
        ("evidence", r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b"),
        ("planning", r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b"),
        ("delay", r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|b)\b"),
        ("quality", r"\b(?:quality|high|low|grade|rating)\b"),
        ("security", r"\b(?:security|secure|vulnerability|exploit)\b"),
        ("performance", r"\b(?:performance|fast|slow|latency)\b"),
        ("compliance", r"\b(?:compliance|regulation|standard)\b"),
        ("cost", r"\b(?:cost|price|budget|expense)\b"),
        ("generic", r"\b\w{7,}\b"),
    ]
    for _, pattern in FEATURE_REGEXES:
        matches = re.findall(pattern, text, re.I)
        counts.append(len(matches))
    return np.array(counts, dtype=int)


# ----- Parent B components -------------------------------------------------

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural similarity index for 1‑D signals."""
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
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator


class Sheaf:
    """A lightweight sheaf implementation for linear restriction maps."""

    def __init__(self, node_dims: Dict[str, int], edges: List[Tuple[str, str]]):
        """
        node_dims: mapping node name -> dimension of its vector space.
        edges: list of (source, target) node identifiers.
        """
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions: Dict[Tuple[str, str], np.ndarray] = {}
        self._sections: Dict[str, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[str, str],
                        src_map: np.ndarray, dst_map: np.ndarray) -> None:
        """
        Define a linear restriction from source to target.
        src_map: matrix of shape (target_dim, source_dim)
        dst_map: matrix of shape (target_dim, source_dim) (often identical to src_map)
        """
        u, v = edge
        if src_map.shape != (self.node_dims[v], self.node_dims[u]):
            raise ValueError("src_map shape mismatch for edge")
        if dst_map.shape != src_map.shape:
            raise ValueError("dst_map must match src_map shape")
        self._restrictions[edge] = (src_map, dst_map)

    def set_section(self, node: str, vector: np.ndarray) -> None:
        """Assign a concrete vector (section) to a node."""
        if vector.shape != (self.node_dims[node],):
            raise ValueError("vector shape mismatch for node")
        self._sections[node] = vector

    def restrict(self, edge: Tuple[str, str]) -> np.ndarray:
        """
        Apply the restriction map of an edge to the source node's section,
        returning the transformed vector in the target space.
        """
        if edge not in self._restrictions:
            raise KeyError("restriction for edge not defined")
        src, tgt = edge
        src_vec = self._sections.get(src)
        if src_vec is None:
            raise KeyError(f"section for source node '{src}' not set")
        src_map, _ = self._restrictions[edge]
        return src_map @ src_vec

    def get_section(self, node: str) -> np.ndarray:
        """Retrieve the stored section for a node."""
        return self._sections[node]


# ----- Hybrid core ---------------------------------------------------------

def hybrid_transform_features(raw_vec: np.ndarray, sheaf: Sheaf,
                             source_node: str, target_node: str) -> np.ndarray:
    """
    Apply a sheaf restriction to a raw feature vector, mapping it from
    `source_node` space to `target_node` space.
    """
    sheaf.set_section(source_node, raw_vec)
    transformed = sheaf.restrict((source_node, target_node))
    return transformed


def hybrid_fisher_weighted_similarity(vec_a: np.ndarray, vec_b: np.ndarray,
                                      phash_center: int = 0,
                                      beam_center: float = 0.0,
                                      beam_width: float = 1.0) -> float:
    """
    Compute SSIM between two vectors, then weight it by a Fisher score derived
    from the angular distance between their perceptual hashes.

    Steps:
    1. Compute 64‑bit perceptual hashes of each vector.
    2. Derive a Hamming distance → angle θ ∈ [0, π] (linear mapping).
    3. Evaluate a Gaussian beam at θ and obtain its Fisher information.
    4. Return Fisher * SSIM as the final similarity measure.
    """
    # 1. Hashes
    h_a = compute_phash(vec_a.tolist())
    h_b = compute_phash(vec_b.tolist())

    # 2. Hamming distance → angle
    max_dist = 64
    ham = hamming_distance(h_a, h_b)
    theta = (ham / max_dist) * math.pi  # map to [0, π]

    # 3. Fisher information weighting
    fisher = fisher_score(theta, beam_center, beam_width)

    # 4. SSIM on the transformed vectors
    # Ensure vectors are on the same scale; map to 0‑255 dynamic range
    scale = 255.0 / max(vec_a.max(), vec_b.max(), 1e-9)
    ssim_val = ssim(vec_a * scale, vec_b * scale)

    return fisher * ssim_val


def hybrid_bayesian_decision(prior: float, false_positive: float,
                             vec_a: np.ndarray, vec_b: np.ndarray,
                             sheaf: Sheaf,
                             source_node: str, target_node: str) -> float:
    """
    Perform a Bayesian update where the likelihood is supplied by the
    Fisher‑weighted similarity of two sheaf‑transformed feature vectors.

    Returns the posterior probability that `vec_a` and `vec_b` belong to the same
    class (e.g., both trustworthy).
    """
    # Transform vectors through the sheaf
    t_a = hybrid_transform_features(vec_a, sheaf, source_node, target_node)
    t_b = hybrid_transform_features(vec_b, sheaf, source_node, target_node)

    # Compute weighted similarity as likelihood (clamped to [0,1])
    similarity = hybrid_fisher_weighted_similarity(t_a, t_b)
    likelihood = max(0.0, min(1.0, similarity))

    # Bayesian marginal and posterior
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    return posterior


# ----- Example usage -------------------------------------------------------

def _demo():
    # Sample texts
    txt1 = "The audit confirmed the security vulnerability and provided a detailed report."
    txt2 = "A thorough verification showed no exploit, and the compliance checklist is clear."

    # Extract raw feature vectors (Parent A)
    raw1 = extract_features(txt1).astype(float)
    raw2 = extract_features(txt2).astype(float)

    # Define a simple sheaf: two nodes "raw" (dim=9) and "proj" (dim=5)
    node_dims = {"raw": 9, "proj": 5}
    edges = [("raw", "proj")]
    sheaf = Sheaf(node_dims, edges)

    # Random linear map for restriction (for demonstration)
    rng = np.random.default_rng(42)
    restriction_map = rng.normal(size=(5, 9))
    sheaf.set_restriction(("raw", "proj"), restriction_map, restriction_map)

    # Prior belief that both texts are trustworthy
    prior = 0.6
    false_positive = 0.1

    posterior = hybrid_bayesian_decision(prior, false_positive,
                                          raw1, raw2,
                                          sheaf,
                                          source_node="raw",
                                          target_node="proj")
    print(f"Posterior probability of agreement: {posterior:.4f}")


if __name__ == "__main__":
    _demo()