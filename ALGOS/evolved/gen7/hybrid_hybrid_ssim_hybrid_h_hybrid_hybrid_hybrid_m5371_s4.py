# DARWIN HAMMER — match 5371, survivor 4
# gen: 7
# parent_a: hybrid_ssim_hybrid_hybrid_fracti_m934_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_shap_a_hybrid_hybrid_hybrid_m2345_s0.py (gen6)
# born: 2026-05-30T00:01:27Z

"""Hybrid algorithm integrating SSIM-based similarity (Parent A) with graph curvature and SHAP attribution (Parent B).

Mathematical bridge:
- Each node is represented by a high‑dimensional hypervector (HV) generated as in Parent A.
- Node‑wise Ollivier‑Ricci curvature (approximated by inverse degree) is used as a scalar “causal effect”.
- This curvature scalar is raised to a fractional power (Parent A) to weight the binding of neighbour HVs.
- SHAP‑like attribution scores derived from curvature are employed as exponents in the Hoeffding bound,
  providing an uncertainty estimate for the SSIM similarity between original and reconstructed HVs.
The pipeline therefore fuses:
    HV generation → curvature‑weighted binding → reconstruction → SSIM similarity → Hoeffding‑based confidence.
"""

import sys
import math
import random
import pathlib
from typing import Sequence, Iterable, Optional, List, Dict, Tuple

import numpy as np

# ---------- Parent A components ----------
def ssim(x: Sequence[float], y: Sequence[float], dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x:
        raise ValueError('samples must not be empty')
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))

def random_hv(d: int = 10000, kind: str = "complex", seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"Unsupported kind {kind!r}")

def bind(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Circular convolution via FFT (binding)."""
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))

def unbind(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Approximate inverse binding (circular correlation)."""
    FY = np.fft.fft(Y)
    mag = np.abs(FY)
    inv_FY = np.conj(FY) / (mag**2 + 1e-30)
    return np.real(np.fft.ifft(np.fft.fft(Z) * inv_FY))

def fractional_power(X: np.ndarray, alpha: float) -> np.ndarray:
    """Element‑wise fractional power preserving sign."""
    return np.abs(X) ** alpha * np.sign(X)

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound ε for empirical mean r with confidence 1‑δ."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    epsilon = math.sqrt(math.log(2 / delta) / (2 * n))
    # The bridge: raise ε to the power r (or any scalar effect); we use r as exponent base.
    return epsilon ** r

# ---------- Parent B components ----------
Node = int
Graph = Dict[Node, set[Node]]

def build_adj(master_vectors: List[np.ndarray]) -> Graph:
    """Create an undirected graph where edges exist for near‑identical vectors."""
    graph: Graph = {}
    for i, vi in enumerate(master_vectors):
        graph[i] = set()
        for j, vj in enumerate(master_vectors):
            if i == j:
                continue
            # Euclidean distance threshold – tiny threshold to emulate “identical” vectors
            if np.linalg.norm(vi - vj) < 1e-6:
                graph[i].add(j)
    return graph

def node_curvature(graph: Graph) -> Dict[Node, float]:
    """
    Simple proxy for Ollivier‑Ricci curvature:
    curvature(node) = 1 / degree(node)  (higher for sparsely connected nodes).
    """
    curv: Dict[Node, float] = {}
    for node, neigh in graph.items():
        deg = len(neigh)
        curv[node] = 1.0 / deg if deg > 0 else 0.0
    return curv

def shap_like_scores(curvature: Dict[Node, float]) -> Dict[Node, float]:
    """
    Generate SHAP‑like attribution scores from curvature.
    For illustration we normalise curvature to sum‑to‑one.
    """
    total = sum(curvature.values()) + 1e-30
    return {node: val / total for node, val in curvature.items()}

# ---------- Hybrid functions ----------
def embed_nodes(num_nodes: int, dim: int, seed: Optional[int] = None) -> Dict[Node, np.ndarray]:
    """Generate a hypervector for each node."""
    rng = np.random.default_rng(seed)
    return {i: random_hv(d=dim, kind="complex", seed=int(rng.integers(0, 1 << 30))) for i in range(num_nodes)}

def propagate_hv(graph: Graph,
                 hv_dict: Dict[Node, np.ndarray],
                 curvature: Dict[Node, float],
                 shap_scores: Dict[Node, float],
                 alpha: float = 0.5) -> Dict[Node, np.ndarray]:
    """
    For each node, bind its HV with the fractional‑power‑scaled curvature of each neighbour.
    The SHAP score of the neighbour is used as an exponent on the resulting bound
    (via Hoeffding) to modulate confidence later.
    """
    out: Dict[Node, np.ndarray] = {}
    for node, neighbors in graph.items():
        agg = np.zeros_like(hv_dict[node], dtype=complex)
        if not neighbors:
            out[node] = hv_dict[node]
            continue
        for nb in neighbors:
            # Scale neighbour HV by curvature^alpha
            scaled = bind(hv_dict[nb], fractional_power(hv_dict[nb], curvature[nb] ** alpha))
            agg += scaled
        # Average and bind with own HV
        agg /= len(neighbors)
        out[node] = bind(hv_dict[node], agg)
    return out

def reconstruct_and_evaluate(original_hv: Dict[Node, np.ndarray],
                             propagated_hv: Dict[Node, np.ndarray],
                             shap_scores: Dict[Node, float],
                             delta: float = 0.05,
                             n_samples: int = 100) -> Dict[Node, Tuple[float, float]]:
    """
    For each node:
        1. Approximate inverse binding using neighbours' HVs (here we simply unbind with itself).
        2. Compute SSIM between magnitude vectors of original and reconstructed HVs.
        3. Compute Hoeffding‑based confidence using the node's SHAP score as exponent.
    Returns a mapping node → (ssim, confidence).
    """
    results: Dict[Node, Tuple[float, float]] = {}
    for node in original_hv:
        # Simple reconstruction: unbind the propagated HV with the original HV
        recon = unbind(propagated_hv[node], original_hv[node])
        # Use magnitude (real) for SSIM comparison
        orig_mag = np.abs(original_hv[node]).astype(float).tolist()
        recon_mag = np.abs(recon).astype(float).tolist()
        sim = ssim(orig_mag, recon_mag)
        # Hoeffding confidence: exponent = SHAP score (scaled to (0,1])
        exponent = max(shap_scores.get(node, 0.0), 1e-6)
        conf = hoeffding_bound(r=exponent, delta=delta, n=n_samples)
        results[node] = (sim, conf)
    return results

def hybrid_process(master_vectors: List[np.ndarray],
                   dim: int = 1024,
                   delta: float = 0.05,
                   n_samples: int = 200) -> Dict[Node, Tuple[float, float]]:
    """
    End‑to‑end hybrid pipeline:
        - Build graph from master vectors.
        - Compute curvature and SHAP‑like scores.
        - Embed nodes as hypervectors.
        - Propagate hypervectors using curvature‑weighted binding.
        - Reconstruct and evaluate similarity/confidence.
    """
    graph = build_adj(master_vectors)
    curvature = node_curvature(graph)
    shap_scores = shap_like_scores(curvature)
    hv_original = embed_nodes(len(master_vectors), dim)
    hv_propagated = propagate_hv(graph, hv_original, curvature, shap_scores)
    return reconstruct_and_evaluate(hv_original, hv_propagated, shap_scores,
                                    delta=delta, n_samples=n_samples)

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Create synthetic master vectors (5 nodes, 128‑dim real vectors)
    rng = np.random.default_rng(42)
    master_vecs = [rng.standard_normal(128) for _ in range(5)]

    results = hybrid_process(master_vecs, dim=512, delta=0.01, n_samples=500)

    for node, (sim, conf) in results.items():
        print(f"Node {node}: SSIM={sim:.4f}, Hoeffding confidence={conf:.6f}")