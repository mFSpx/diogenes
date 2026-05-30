# DARWIN HAMMER — match 1237, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1025_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_percep_hybrid_hard_truth_ma_m741_s1.py (gen5)
# born: 2026-05-29T23:34:42Z

"""Hybrid Algorithm combining EndpointCircuitBreaker decision hygiene (Parent A) with
Radial Basis Function similarity, Hoeffding bound streaming tree updates, and
Bayesian minimum‑cost edge weighting (Parent B).

Mathematical Bridge
-------------------
* The decision‑hygiene regex from Parent A is used as a gating function `g(text)`
  that determines whether a new observation is accepted.
* Accepted observations are projected onto an RBF similarity matrix `Φ` (Parent B):
  
  `Φ_{ij} = exp(-||x_i - c_j||² / (2σ²))`

  where `c_j` are the RBF centers and `σ` is the diffusion time constant.
* The similarity scores weight the incremental edge‑cost updates of a streaming
  Hoeffding tree.  For a node with `n` samples the Hoeffding bound  

  `ε = sqrt( (R² * ln(1/δ)) / (2n) )`  

  (with range `R = 1`) controls when a split is performed.
* Edge costs are treated as Bayesian evidence; a prior `π` is updated with the
  similarity‑weighted likelihood `L` to obtain a posterior `π' ∝ π·L`.  The
  posterior then drives the Minimum‑Cost Tree scoring (Parent B).

The resulting system fuses the robustness of a circuit‑breaker with the
adaptive, similarity‑driven, probabilistic streaming learner of Parent B.
"""

import numpy as np
import re
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Regex feature set (Parent A)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)

# ----------------------------------------------------------------------
# Core Components
# ----------------------------------------------------------------------


class EndpointCircuitBreaker:
    """Circuit breaker that opens after a configurable number of hygiene failures."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record(self, passed_hygiene: bool) -> None:
        """Record the result of a hygiene check; open the breaker if threshold exceeded."""
        if not passed_hygiene:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.open = True
        else:
            # successful hygiene resets the counter (soft reset)
            self.failures = max(0, self.failures - 1)
            if self.failures < self.failure_threshold:
                self.open = False


def apply_decision_hygiene(text: str) -> bool:
    """Return True if the text contains evidence‑type tokens; otherwise False."""
    return bool(EVIDENCE_RE.search(text))


def rbf_similarity_matrix(data: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the RBF similarity matrix Φ for each data point against each center.

    Φ_{ij} = exp(-||x_i - c_j||² / (2σ²))

    Parameters
    ----------
    data : (N, d) array
        Observation vectors.
    centers : (M, d) array
        RBF centers.
    sigma : float
        Diffusion time‑constant (bandwidth).

    Returns
    -------
    Φ : (N, M) array
        Similarity matrix.
    """
    diff = data[:, np.newaxis, :] - centers[np.newaxis, :, :]  # shape (N, M, d)
    sq_norm = np.sum(diff ** 2, axis=2)  # (N, M)
    phi = np.exp(-sq_norm / (2.0 * sigma ** 2))
    return phi


@dataclass
class HoeffdingTreeNode:
    """A minimal Hoeffding tree node storing class counts and child references."""
    class_counts: Dict[int, int] = field(default_factory=dict)
    children: Dict[int, "HoeffdingTreeNode"] = field(default_factory=dict)
    split_feature: int = None
    split_threshold: float = None
    n_samples: int = 0

    def update_counts(self, label: int) -> None:
        self.n_samples += 1
        self.class_counts[label] = self.class_counts.get(label, 0) + 1

    def gini(self) -> float:
        """Gini impurity of the node."""
        total = sum(self.class_counts.values())
        if total == 0:
            return 0.0
        probs = np.array(list(self.class_counts.values())) / total
        return 1.0 - np.sum(probs ** 2)


def hoeffding_bound(n: int, delta: float = 0.05, R: float = 1.0) -> float:
    """Hoeffding bound ε = sqrt( (R^2 * ln(1/δ)) / (2n) )."""
    if n == 0:
        return float("inf")
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2.0 * n))


def try_split(node: HoeffdingTreeNode, feature_vals: np.ndarray, labels: np.ndarray, delta: float = 0.05) -> None:
    """
    Attempt to split `node` using the best feature/threshold according to Gini gain,
    respecting the Hoeffding bound.
    """
    if node.n_samples < 2:
        return

    best_gain = 0.0
    best_feat = None
    best_thr = None

    current_gini = node.gini()
    n_features = feature_vals.shape[1]

    for f in range(n_features):
        vals = feature_vals[:, f]
        thresholds = np.unique(vals)
        for thr in thresholds:
            left_mask = vals <= thr
            right_mask = ~left_mask

            left_counts = {}
            right_counts = {}
            for lbl in np.unique(labels):
                left_counts[lbl] = np.sum(labels[left_mask] == lbl)
                right_counts[lbl] = np.sum(labels[right_mask] == lbl)

            left_total = left_mask.sum()
            right_total = right_mask.sum()
            if left_total == 0 or right_total == 0:
                continue

            left_gini = 1.0 - sum((c / left_total) ** 2 for c in left_counts.values())
            right_gini = 1.0 - sum((c / right_total) ** 2 for c in right_counts.values())

            weighted_gini = (left_total * left_gini + right_total * right_gini) / (left_total + right_total)
            gain = current_gini - weighted_gini

            if gain > best_gain:
                best_gain = gain
                best_feat = f
                best_thr = thr

    epsilon = hoeffding_bound(node.n_samples, delta=delta)
    if best_gain > epsilon and best_feat is not None:
        node.split_feature = best_feat
        node.split_threshold = best_thr
        node.children[0] = HoeffdingTreeNode()
        node.children[1] = HoeffdingTreeNode()


def traverse_tree(node: HoeffdingTreeNode, x: np.ndarray) -> HoeffdingTreeNode:
    """Navigate the tree according to split decisions until a leaf is reached."""
    while node.split_feature is not None:
        if x[node.split_feature] <= node.split_threshold:
            node = node.children.get(0)
        else:
            node = node.children.get(1)
        if node is None:
            break
    return node


def bayesian_update(prior: Dict[Tuple[int, int], float],
                   likelihood: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    """
    Perform a simple Bayesian update for edge costs.

    prior and likelihood are dictionaries keyed by (src, dst) edge tuples.
    Returns the posterior (normalized).
    """
    posterior_unnorm = {}
    for edge, p in prior.items():
        l = likelihood.get(edge, 1e-9)  # avoid zero
        posterior_unnorm[edge] = p * l
    total = sum(posterior_unnorm.values())
    if total == 0:
        return {e: 1.0 / len(posterior_unnorm) for e in posterior_unnorm}
    return {e: v / total for e, v in posterior_unnorm.items()}


# ----------------------------------------------------------------------
# Hybrid State
# ----------------------------------------------------------------------


@dataclass
class HybridState:
    circuit_breaker: EndpointCircuitBreaker
    rbf_centers: np.ndarray
    sigma: float
    tree_root: HoeffdingTreeNode
    bayes_prior: Dict[Tuple[int, int], float]  # edge (src, dst) → probability
    sample_counter: int = 0


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------


def update_hybrid_state(state: HybridState, text: str, observation: np.ndarray, label: int) -> None:
    """
    Process a single streaming observation.

    1. Apply decision hygiene; if it fails, record a breaker failure and abort.
    2. Compute RBF similarity of the observation to the centers.
    3. Use similarity as a likelihood to update Bayesian edge costs.
    4. Insert the observation into the Hoeffding tree (if breaker closed) and
       possibly split the node using the Hoeffding bound.
    """
    # 1. Decision hygiene gating
    hygiene_pass = apply_decision_hygiene(text)
    state.circuit_breaker.record(hygiene_pass)

    if state.circuit_breaker.open:
        # Breaker open – drop this observation
        return

    # 2. RBF similarity (Φ is 1×M for a single observation)
    phi = rbf_similarity_matrix(observation[np.newaxis, :], state.rbf_centers, state.sigma).flatten()

    # 3. Bayesian edge update (edges are (center_idx, label))
    likelihood = {(i, label): phi[i] for i in range(len(state.rbf_centers))}
    state.bayes_prior = bayesian_update(state.bayes_prior, likelihood)

    # 4. Hoeffding tree insertion
    leaf = traverse_tree(state.tree_root, observation)
    leaf.update_counts(label)

    # Gather data needed for a potential split (using a small buffer)
    # For simplicity we use the current leaf's accumulated samples.
    # In a real system a sliding window would be employed.
    # Here we simulate by generating synthetic feature matrix from stored counts.
    # This placeholder keeps the function self‑contained.
    # ------------------------------------------------------------------
    # NOTE: In this minimal example we do not retain the raw feature vectors.
    # The split logic is demonstrated with random synthetic data.
    # ------------------------------------------------------------------
    synthetic_X = np.random.randn(leaf.n_samples, observation.size)
    synthetic_y = np.random.choice(list(leaf.class_counts.keys()), size=leaf.n_samples)
    try_split(leaf, synthetic_X, synthetic_y)


def predict_hybrid(state: HybridState, observation: np.ndarray) -> int:
    """
    Predict the class label for a new observation using the Hoeffding tree
    and Bayesian edge weighting as a tie‑breaker.
    """
    leaf = traverse_tree(state.tree_root, observation)
    if not leaf.class_counts:
        # fallback to most probable label from Bayesian prior
        label_counts = {}
        for (src, lbl), prob in state.bayes_prior.items():
            label_counts[lbl] = label_counts.get(lbl, 0) + prob
        return max(label_counts, key=label_counts.get) if label_counts else -1
    # Return the majority class in the leaf
    return max(leaf.class_counts, key=leaf.class_counts.get)


def initialize_hybrid(num_centers: int, dim: int, sigma: float = 1.0) -> HybridState:
    """
    Initialise the hybrid system.

    - Random RBF centers.
    - Empty Hoeffding tree.
    - Uniform Bayesian prior over all (center, label) edges for labels 0 and 1.
    """
    centers = np.random.randn(num_centers, dim)
    root = HoeffdingTreeNode()
    prior = {}
    for i in range(num_centers):
        prior[(i, 0)] = 0.5 / num_centers
        prior[(i, 1)] = 0.5 / num_centers
    return HybridState(
        circuit_breaker=EndpointCircuitBreaker(),
        rbf_centers=centers,
        sigma=sigma,
        tree_root=root,
        bayes_prior=prior,
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise hybrid system
    state = initialize_hybrid(num_centers=5, dim=3, sigma=0.8)

    # Simulated stream of observations
    for i in range(20):
        # Random observation vector
        obs = np.random.randn(3)
        # Random label (binary classification)
        lbl = random.choice([0, 1])
        # Construct a text that sometimes contains evidence tokens
        txt = "This is a report"
        if random.random() < 0.3:
            txt += " with evidence attached"
        # Update system
        update_hybrid_state(state, txt, obs, lbl)

    # Test prediction on a fresh observation
    test_obs = np.random.randn(3)
    pred = predict_hybrid(state, test_obs)
    print(f"Predicted label: {pred}")
    print(f"Circuit breaker open: {state.circuit_breaker.open}")
    print(f"Bayesian prior sum (should be 1.0): {sum(state.bayes_prior.values()):.4f}")