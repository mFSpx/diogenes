# DARWIN HAMMER — match 3283, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_pherom_m96_s1.py (gen4)
# parent_b: hybrid_hybrid_shap_attribut_dense_associative_me_m2066_s3.py (gen5)
# born: 2026-05-29T23:49:07Z

"""Hybrid SHAP‑Geometric‑Koopman‑Hopfield Module

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_pherom_m96_s1 (geometric algebra,
  Koopman operator, entropy‑weighted pheromone).
- hybrid_hybrid_shap_attribut_dense_associative_me_m2066_s3 (SHAP
  attribution, dense associative memory / modern Hopfield network).

Mathematical bridge
------------------
1. **SHAP → Multivector** – each node’s SHAP attribution `s_i` becomes the
   coefficient of the grade‑1 basis blade `e_i`.  The resulting
   `Multivector` `S = Σ_i s_i e_i` lives in `Cl(n,0)`.

2. **Koopman linearisation** – a linear operator `K ∈ ℝ^{n×n}` is estimated
   from successive SHAP‑multivector snapshots `{S_t}` such that
   `S_{t+1} ≈ K·S_t`.  Applying `K` yields a *predicted* multivector
   `Ŝ = K·S`.

3. **Hopfield retrieval** – the predicted coefficient vector `ŝ` is used as
   a query `ξ` for a dense associative memory matrix `M ∈ ℝ^{N×n}` (one row
   per graph node, built from neighbourhood feature patterns).  The modern
   Hopfield energy

        E(ξ) = -β⁻¹ log Σ_i exp(β M_i·ξ) + ½‖ξ‖²

   yields a softmax attention `α_i = softmax(β M ξ)`.  This distribution
   plays the role of a pheromone field.

4. **Entropy feedback** – the Shannon entropy `H(α)` modulates the pheromone
   strengths, closing the loop back to the stochastic action selection.

The code below implements this pipeline with three public functions and a
smoke test."""

import sys
import math
import random
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Minimal geometric algebra (grade‑1 only) -------------------------------------------------
# ----------------------------------------------------------------------
def _blade_sign(indices):
    """Bubble‑sort indices, returning sorted tuple and sign of the permutation."""
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
    return tuple(lst), sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (sets of indices)."""
    combined = list(blade_a) + list(blade_b)
    res, sign = _blade_sign(combined)
    return frozenset(res), sign


class Multivector:
    """Grade‑1 multivector (vector) in Cl(n,0)."""

    def __init__(self, components: dict, n: int):
        # keep only non‑zero components
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def to_vector(self) -> np.ndarray:
        """Return dense ℝⁿ representation (grade‑1 part only)."""
        vec = np.zeros(self.n)
        for blade, coeff in self.components.items():
            if len(blade) == 1:
                idx = next(iter(blade))
                vec[idx] = coeff
        return vec

    @staticmethod
    def from_vector(vec: np.ndarray) -> "Multivector":
        n = vec.shape[0]
        comps = {frozenset({i}): float(v) for i, v in enumerate(vec) if abs(v) > 1e-15}
        return Multivector(comps, n)

    def __add__(self, other):
        comps = self.components.copy()
        for b, v in other.components.items():
            comps[b] = comps.get(b, 0.0) + v
        return Multivector(comps, self.n)

    def __rmul__(self, scalar: float):
        comps = {b: scalar * v for b, v in self.components.items()}
        return Multivector(comps, self.n)

    __mul__ = __rmul__


# ----------------------------------------------------------------------
# SHAP utilities (from Parent B)
# ----------------------------------------------------------------------
Node = int
Graph = dict[int, set[int]]
Model = dict[int, float]


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Exact Shapley kernel weight for a given subset size."""
    return (
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / math.factorial(feature_count)
    )


def approximate_shap_values(graph: Graph, model: Model, feature_count: int) -> dict[Node, float]:
    """Linear surrogate for SHAP values using node value + neighbour values."""
    shap = {}
    for node, neighbours in graph.items():
        own = model.get(node, 0.0)
        neigh_sum = sum(model.get(nb, 0.0) for nb in neighbours)
        # weight by kernel (here we simply use uniform weighting for demo)
        shap[node] = (own + neigh_sum) / (1 + len(neighbours) + 1e-12)
    return shap


# ----------------------------------------------------------------------
# Koopman operator estimation (from Parent A)
# ----------------------------------------------------------------------
def estimate_koopman_operator(snapshots: list[Multivector]) -> np.ndarray:
    """
    Least‑squares estimate K such that S_{t+1} ≈ K·S_t.
    Returns K ∈ ℝ^{n×n}.
    """
    if len(snapshots) < 2:
        raise ValueError("At least two snapshots required")
    X = np.column_stack([s.to_vector() for s in snapshots[:-1]])   # shape (n, m-1)
    Y = np.column_stack([s.to_vector() for s in snapshots[1:]])    # shape (n, m-1)
    # Solve Y = K X  =>  K = Y X⁺
    K, residuals, rank, s = np.linalg.lstsq(X.T, Y.T, rcond=None)  # each row solves one output dim
    K = K.T  # shape (n, n)
    return K


def apply_koopman(K: np.ndarray, mv: Multivector) -> Multivector:
    """Apply linear operator K to a multivector (grade‑1 part only)."""
    vec = mv.to_vector()
    pred = K @ vec
    return Multivector.from_vector(pred)


# ----------------------------------------------------------------------
# Dense associative memory / modern Hopfield (from Parent B)
# ----------------------------------------------------------------------
def build_memory_matrix(graph: Graph, shap: dict[Node, float]) -> np.ndarray:
    """
    Memory matrix M ∈ ℝ^{N×n}. Row i encodes node i's own SHAP value and the
    values of its immediate neighbours (zero‑padded).
    """
    nodes = sorted(graph.keys())
    n = max(nodes) + 1  # dimension matches multivector size
    M = np.zeros((len(nodes), n))
    for row, node in enumerate(nodes):
        M[row, node] = shap.get(node, 0.0)
        for nb in graph[node]:
            M[row, nb] = shap.get(nb, 0.0)
    return M


def hopfield_attention(M: np.ndarray, query: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Modern Hopfield softmax attention.
    Returns a probability distribution over rows of M.
    """
    logits = beta * (M @ query)  # shape (N,)
    # numerical stability
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / exp_logits.sum()
    return probs


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector."""
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    return -np.sum(p_safe * np.log(p_safe))


# ----------------------------------------------------------------------
# Hybrid pipeline functions
# ----------------------------------------------------------------------
def shap_to_multivector(shap: dict[Node, float], dim: int) -> Multivector:
    """
    Embed SHAP attributions into a grade‑1 multivector of dimension `dim`.
    Missing indices receive coefficient 0.
    """
    comps = {frozenset({i}): float(shap.get(i, 0.0)) for i in range(dim)}
    return Multivector(comps, dim)


def hybrid_predict(
    graph: Graph,
    model: Model,
    feature_count: int,
    K: np.ndarray,
    M: np.ndarray,
    beta: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    One hybrid inference step:
    1. Approximate SHAP values.
    2. Convert to multivector S.
    3. Predict next multivector Ŝ = K·S.
    4. Use Ŝ as query ξ for Hopfield attention.
    5. Return attention distribution and its entropy.
    """
    shap = approximate_shap_values(graph, model, feature_count)
    dim = K.shape[0]
    S = shap_to_multivector(shap, dim)
    S_pred = apply_koopman(K, S)
    ξ = S_pred.to_vector()
    attention = hopfield_attention(M, ξ, beta)
    entropy = shannon_entropy(attention)
    return attention, entropy


def update_pheromone_strengths(
    base_strengths: np.ndarray, attention: np.ndarray, entropy: float, alpha: float = 0.5
) -> np.ndarray:
    """
    Modulate pheromone strengths with attention and entropy.
    `alpha` controls the influence of entropy (0 → no influence, 1 → full).
    """
    ent_factor = np.exp(-alpha * entropy)  # lower entropy → stronger reinforcement
    new_strengths = base_strengths * attention * ent_factor
    # renormalize to keep a proper probability field
    if new_strengths.sum() > 0:
        new_strengths /= new_strengths.sum()
    return new_strengths


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple graph
    G: Graph = {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1, 3},
        3: {2},
    }

    # Dummy model values (e.g., node utilities)
    model: Model = {i: random.uniform(0, 1) for i in G}

    # Feature count for SHAP kernel (use number of nodes)
    fcnt = len(G)

    # Compute two SHAP snapshots to estimate Koopman operator
    shap1 = approximate_shap_values(G, model, fcnt)
    # Perturb model slightly for second snapshot
    model_perturbed = {i: v + random.gauss(0, 0.01) for i, v in model.items()}
    shap2 = approximate_shap_values(G, model_perturbed, fcnt)

    dim = max(G) + 1
    S1 = shap_to_multivector(shap1, dim)
    S2 = shap_to_multivector(shap2, dim)

    K_est = estimate_koopman_operator([S1, S2])

    # Build memory matrix from the first SHAP snapshot
    M_mat = build_memory_matrix(G, shap1)

    # Initial pheromone strengths (uniform)
    pheromones = np.ones(len(G)) / len(G)

    # Run hybrid step
    attn, ent = hybrid_predict(G, model, fcnt, K_est, M_mat, beta=2.0)

    # Update pheromones
    new_pheromones = update_pheromone_strengths(pheromones, attn, ent, alpha=0.7)

    print("Attention distribution:", attn)
    print("Entropy:", ent)
    print("Updated pheromone strengths:", new_pheromones)