# DARWIN HAMMER — match 2710, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s0.py (gen4)
# parent_b: hybrid_hybrid_krampus_brain_ttt_linear_m4_s1.py (gen2)
# born: 2026-05-29T23:43:53Z

"""Hybrid algorithm combining:
- Parent A: ternary router with SSIM evaluation, XGBoost split‑gain modulation via TTT‑Linear update.
- Parent B: graph‑based Ollivier‑Ricci curvature with matrix updates from TTT‑Linear.

Mathematical bridge:
Node feature vectors (extracted from text) are used both as inputs to a logistic
model (TTT‑Linear) and as signals on the graph.  The gradient/hessian of the
logistic loss produce a split‑gain value G = grad²/(hess+λ) which is interpreted
as an *edge‑strength modifier*.  Edge weights are multiplied by a factor
f(G)=σ(α·G) (σ sigmoid) and the resulting weighted graph is fed to the
Ollivier‑Ricci curvature formula.  The curvature, together with the SSIM
similarity of the endpoint feature vectors, yields a pruning probability that
guides the ternary router’s decision making.  Thus the linear update rule
modulates graph topology, while the graph‑curvature‑aware pruning feeds back
into the router’s SSIM‑based evaluation."""
import sys
import math
import random
import pathlib
from typing import Dict, List, Tuple, Any
import numpy as np

# ----------------------------------------------------------------------
# Utility functions from Parent A
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Simplified SSIM for 1‑D vectors."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + C1) * (2 * cov + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(num / den)


# ----------------------------------------------------------------------
# TTT‑Linear update rule (Parent A & B)
# ----------------------------------------------------------------------
def ttt_linear_update(
    w: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    lr: float = 0.1,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Perform a Newton‑style update for logistic loss:
        w_new = w - lr * grad / (hess + eps)
    """
    direction = grad / (hess + eps)
    return w - lr * direction


def logistic_grad_hess(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gradient and Hessian of binary logistic loss."""
    grad = pred - target
    hess = pred * (1.0 - pred)
    return grad, hess


def split_gain(grad: np.ndarray, hess: np.ndarray, lam: float = 1.0) -> float:
    """XGBoost‑style split gain for a single scalar (sum over dimensions)."""
    return float((grad @ grad) / (hess.sum() + lam))


# ----------------------------------------------------------------------
# Graph utilities from Parent B
# ----------------------------------------------------------------------
def build_graph(nodes: List[int]) -> Dict[int, List[int]]:
    """Create an undirected ring graph for demonstration."""
    G = {n: [] for n in nodes}
    for i, n in enumerate(nodes):
        nxt = nodes[(i + 1) % len(nodes)]
        G[n].append(nxt)
        G[nxt].append(n)
    return G


def initialize_edge_weights(G: Dict[int, List[int]]) -> Dict[Tuple[int, int], float]:
    """Random positive weight for each undirected edge."""
    weights = {}
    for u, neigh in G.items():
        for v in neigh:
            if (v, u) in weights:
                continue
            weights[(u, v)] = random.random() + 0.1
    return weights


def ollivier_ricci_curvature(
    G: Dict[int, List[int]],
    w: Dict[Tuple[int, int], float],
) -> Dict[Tuple[int, int], float]:
    """
    Very coarse Ollivier‑Ricci curvature:
        κ_ij = 1 - (w_ij / (deg_i + deg_j))
    where deg_i is the sum of incident weights.
    """
    deg = {node: 0.0 for node in G}
    for (i, j), weight in w.items():
        deg[i] += weight
        deg[j] += weight

    curvature = {}
    for (i, j), weight in w.items():
        curvature[(i, j)] = 1.0 - weight / (deg[i] + deg[j] + 1e-8)
    return curvature


# ----------------------------------------------------------------------
# Hybrid core (the mathematical fusion)
# ----------------------------------------------------------------------
def hybrid_edge_modulation(
    G: Dict[int, List[int]],
    edge_w: Dict[Tuple[int, int], float],
    node_feat: Dict[int, np.ndarray],
    model_w: np.ndarray,
    lr: float = 0.05,
    lam: float = 1.0,
) -> Tuple[Dict[Tuple[int, int], float], Dict[int, np.ndarray]]:
    """
    1. Predict a scalar per node via logistic model: p_i = sigmoid(w·x_i).
    2. Compute gradient / hessian against a dummy target (e.g., 0.5).
    3. Derive split‑gain G_i for each node and map it to a scaling factor f_i.
    4. Scale incident edge weights by f_i (edge‑wise product of its two endpoints).
    5. Update the linear model weights using the TTT‑Linear rule.
    6. Return updated edge weights and model parameters.
    """
    # ----- step 1: node predictions -----
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    preds = {}
    for nid, vec in node_feat.items():
        preds[nid] = sigmoid(vec @ model_w)

    # ----- step 2: gradient / hessian -----
    target = 0.5  # neutral target for all nodes
    grads = {}
    hesses = {}
    for nid, p in preds.items():
        g, h = logistic_grad_hess(np.array([p]), np.array([target]))
        grads[nid] = g[0]
        hesses[nid] = h[0]

    # ----- step 3: split gain per node -----
    gains = {nid: (g ** 2) / (h + lam) for nid, (g, h) in zip(grads.keys(), zip(grads.values(), hesses.values()))}
    # map gain to scaling factor via sigmoid (smooth 0‑1)
    scale = {nid: sigmoid(gains[nid]) for nid in gains}

    # ----- step 4: edge weight modulation -----
    new_edge_w = {}
    for (i, j), w_ij in edge_w.items():
        factor = scale[i] * scale[j]  # combined influence of both endpoints
        new_edge_w[(i, j)] = w_ij * factor

    # ----- step 5: linear model update (TTT‑Linear) -----
    # aggregate gradient & hessian over all nodes
    grad_vec = np.sum([grads[nid] * node_feat[nid] for nid in grads], axis=0)
    hess_vec = np.sum([hesses[nid] * (node_feat[nid] ** 2) for nid in hesses], axis=0)
    new_model_w = ttt_linear_update(model_w, grad_vec, hess_vec, lr=lr)

    return new_edge_w, new_model_w


def prune_edges_by_curvature_ssim(
    G: Dict[int, List[int]],
    edge_w: Dict[Tuple[int, int], float],
    node_feat: Dict[int, np.ndarray],
    curvature: Dict[Tuple[int, int], float],
    ssim_thresh: float = 0.6,
) -> Dict[Tuple[int, int], float]:
    """
    Combine Ollivier‑Ricci curvature and SSIM similarity to compute a pruning
    probability for each edge:
        p_prune = (1 - κ) * (1 - SSIM)
    Edges with p_prune > 0.5 are removed (weight set to 0).
    """
    pruned = {}
    for (i, j), w in edge_w.items():
        # curvature contribution (higher curvature → more stable → less pruning)
        kappa = curvature.get((i, j), 0.0)

        # SSIM between node feature vectors (normalize to [0,255] for compatibility)
        xi = node_feat[i]
        xj = node_feat[j]
        # Rescale to typical image range for SSIM formula
        xi_img = (xi - xi.min()) / (xi.ptp() + 1e-8) * 255
        xj_img = (xj - xj.min()) / (xj.ptp() + 1e-8) * 255
        sim = ssim(xi_img, xj_img)

        p_prune = (1.0 - kappa) * (1.0 - sim)
        if p_prune > 0.5:
            pruned[(i, j)] = 0.0
        else:
            pruned[(i, j)] = w
    return pruned


def hybrid_step(
    G: Dict[int, List[int]],
    node_feat: Dict[int, np.ndarray],
    model_w: np.ndarray,
) -> Tuple[Dict[Tuple[int, int], float], np.ndarray]:
    """
    Perform one hybrid iteration:
        1. Modulate edge weights via split‑gain (TTT‑Linear) → new_edge_w.
        2. Compute curvature on the new graph.
        3. Prune edges using curvature + SSIM.
        4. Return the pruned edge map and updated model weights.
    """
    # initial edge weights (if not present, create a fresh set)
    edge_w = initialize_edge_weights(G)

    # 1. edge modulation + model update
    mod_edge_w, new_model_w = hybrid_edge_modulation(G, edge_w, node_feat, model_w)

    # 2. curvature
    curv = ollivier_ricci_curvature(G, mod_edge_w)

    # 3. pruning
    pruned_edge_w = prune_edges_by_curvature_ssim(G, mod_edge_w, node_feat, curv)

    return pruned_edge_w, new_model_w


# ----------------------------------------------------------------------
# Demo / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic graph with 5 nodes
    nodes = list(range(5))
    G = build_graph(nodes)

    # Generate deterministic pseudo‑random feature vectors for each node
    rng = np.random.default_rng(42)
    node_feat = {n: rng.random(10) * 10 for n in nodes}

    # Initialise linear model weight vector (same dimension as node features)
    model_w = rng.random(10)

    # Run a single hybrid step
    edge_weights, model_w = hybrid_step(G, node_feat, model_w)

    # Simple sanity prints
    print("Updated edge weights (non‑zero):")
    for (i, j), w in edge_weights.items():
        if w > 0:
            print(f"  ({i}, {j}) -> {w:.4f}")
    print("\nUpdated model weights:", model_w.round(4))
    sys.exit(0)