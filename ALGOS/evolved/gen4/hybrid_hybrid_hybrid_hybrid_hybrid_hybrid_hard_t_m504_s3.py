# DARWIN HAMMER — match 504, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_bandit_m269_s0.py (gen3)
# parent_b: hybrid_hybrid_hard_truth_ma_kan_m27_s2.py (gen2)
# born: 2026-05-29T23:29:25Z

import numpy as np
import math
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable


# ---------------------------------------------------------------------------
# Count‑Min Sketch (CM Sketch) – lightweight frequency estimator
# ---------------------------------------------------------------------------

@dataclass
class CountMinSketch:
    """
    Simple Count‑Min Sketch with pairwise‑independent hash functions.
    The sketch is used to obtain a robust estimate of word frequencies
    that complements the stylometric categorical frequencies.
    """
    width: int
    depth: int
    _table: np.ndarray = field(init=False, repr=False)
    _seeds: List[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("width and depth must be positive integers")
        self._table = np.zeros((self.depth, self.width), dtype=np.int64)
        # deterministic seeds for reproducibility
        self._seeds = [i * 0x9e3779b9 for i in range(self.depth)]

    def _hash(self, item: str, seed: int) -> int:
        h = hashlib.blake2b(digest_size=8, person=seed.to_bytes(8, "little"))
        h.update(item.encode("utf-8"))
        return int.from_bytes(h.digest(), "little") % self.width

    def add(self, item: str, count: int = 1) -> None:
        for i, seed in enumerate(self._seeds):
            idx = self._hash(item, seed)
            self._table[i, idx] += count

    def estimate(self, item: str) -> int:
        """Return the minimum count across hash rows – the CM sketch estimate."""
        return min(self._table[i, self._hash(item, seed)] for i, seed in enumerate(self._seeds))


# ---------------------------------------------------------------------------
# Sheaf structures – more faithful Laplacian construction
# ---------------------------------------------------------------------------

class Sheaf:
    """
    Cellular sheaf on a simple undirected graph.
    Each node carries a vector space of dimension `node_dims[node]`.
    Each edge carries a linear restriction map from the incident node spaces
    to a common edge space of dimension `edge_dim`.
    The sheaf Laplacian L = δᵀδ is built from these restriction maps.
    """

    def __init__(self,
                 node_dims: Dict[int, int],
                 edge_list: List[Tuple[int, int]],
                 edge_dim: int = 1) -> None:
        self.node_dims = dict(node_dims)          # node_id → dimension
        self.edges = list(edge_list)              # list of (u, v) tuples
        self.edge_dim = edge_dim
        self._validate()

    def _validate(self) -> None:
        if not self.node_dims:
            raise ValueError("node_dims cannot be empty")
        node_ids = set(self.node_dims.keys())
        for u, v in self.edges:
            if u not in node_ids or v not in node_ids:
                raise ValueError(f"Edge ({u},{v}) references undefined node")

    def _restriction_matrix(self, node: int) -> np.ndarray:
        """
        For simplicity we use a random orthonormal matrix of shape
        (edge_dim, node_dims[node]) to act as the restriction map.
        In a production system this would be supplied by domain knowledge.
        """
        rng = np.random.default_rng(seed=node)  # deterministic per node
        mat = rng.normal(size=(self.edge_dim, self.node_dims[node]))
        # Orthonormalise rows via QR decomposition
        q, _ = np.linalg.qr(mat.T)
        return q.T[:self.edge_dim, :]

    def compute_laplacian(self) -> np.ndarray:
        """
        Construct the coboundary operator δ and return L = δᵀδ.
        The resulting Laplacian lives in ℝ^{Σ dim(node)}.
        """
        total_dim = sum(self.node_dims.values())
        # Map global node index → slice in the big matrix
        node_offset: Dict[int, int] = {}
        cur = 0
        for n, d in self.node_dims.items():
            node_offset[n] = cur
            cur += d

        # Build δ as a (|E|*edge_dim) × total_dim matrix
        rows = []
        for u, v in self.edges:
            Ru = self._restriction_matrix(u)  # (edge_dim, dim_u)
            Rv = self._restriction_matrix(v)  # (edge_dim, dim_v)

            # Row block for this edge: [ ... -Ru ... +Rv ... ]
            row_block = np.zeros((self.edge_dim, total_dim))
            row_block[:, node_offset[u]: node_offset[u] + self.node_dims[u]] = -Ru
            row_block[:, node_offset[v]: node_offset[v] + self.node_dims[v]] = Rv
            rows.append(row_block)

        delta = np.vstack(rows) if rows else np.zeros((0, total_dim))
        laplacian = delta.T @ delta
        return laplacian


# ---------------------------------------------------------------------------
# Stylometry utilities – enriched with Count‑Min Sketch frequencies
# ---------------------------------------------------------------------------

FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def _tokenize(text: str) -> List[str]:
    """Very light tokenisation – keeps only alphabetic tokens."""
    return [tok for tok in (text or "").lower().split() if tok.isalpha()]


def _categorical_frequencies(tokens: List[str]) -> Dict[str, float]:
    total = max(1, len(tokens))
    cnt = Counter(tokens)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def _sketch_frequencies(tokens: List[str],
                        sketch: CountMinSketch) -> np.ndarray:
    """
    Returns a normalized vector of sketch estimates for the most frequent
    `k` tokens (k = sketch.width). The vector is L2‑normalized.
    """
    for tok in tokens:
        sketch.add(tok)

    # Extract the top‑k estimated frequencies
    estimates = np.array([sketch.estimate(tok) for tok in tokens])
    if estimates.size == 0:
        return np.zeros(sketch.width)
    # Normalize to sum to 1
    estimates = estimates.astype(float)
    estimates /= estimates.sum()
    # Pad / truncate to exactly `width` entries
    if estimates.shape[0] >= sketch.width:
        return estimates[:sketch.width]
    else:
        pad = np.zeros(sketch.width - estimates.shape[0])
        return np.concatenate([estimates, pad])


def stylometry_vector(text: str,
                     sketch_width: int = 64) -> np.ndarray:
    """
    Combine categorical stylometry frequencies with a Count‑Min Sketch
    estimate of raw word frequencies. The two parts are concatenated and
    L2‑normalized to produce a single feature vector.
    """
    tokens = _tokenize(text)
    cat_vec = np.array(list(_categorical_frequencies(tokens).values()))
    sketch = CountMinSketch(width=sketch_width, depth=4)
    sketch_vec = _sketch_frequencies(tokens, sketch)
    combined = np.concatenate([cat_vec, sketch_vec])
    norm = np.linalg.norm(combined)
    return combined if norm == 0 else combined / norm


# ---------------------------------------------------------------------------
# Kolmogorov‑Arnold Network (KAN) – simple spline‑based activation
# ---------------------------------------------------------------------------

def _spline_activation(x: np.ndarray,
                       knots: int = 5) -> np.ndarray:
    """
    Piecewise‑cubic spline activation built from equally spaced knots
    on the interval [min(x), max(x)]. This mimics the expressive
    univariate functions used in KANs without external dependencies.
    """
    if x.size == 0:
        return x
    mn, mx = x.min(), x.max()
    if mn == mx:
        # Degenerate case – fall back to a smooth non‑linear function
        return np.tanh(x)

    # Knot positions and corresponding random coefficients
    rng = np.random.default_rng(seed=42)
    knot_pos = np.linspace(mn, mx, knots)
    coeffs = rng.normal(size=knots)

    # Cubic Hermite interpolation (simplified)
    def hermite(t, p0, p1, m0, m1):
        h00 = (2 * t**3) - (3 * t**2) + 1
        h10 = t**3 - (2 * t**2) + t
        h01 = (-2 * t**3) + (3 * t**2)
        h11 = t**3 - t**2
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    # Compute slopes (finite differences) for smoothness
    slopes = np.gradient(coeffs, knot_pos)

    # Vectorised evaluation
    idx = np.searchsorted(knot_pos, x, side='right') - 1
    idx = np.clip(idx, 0, knots - 2)
    t = (x - knot_pos[idx]) / (knot_pos[idx + 1] - knot_pos[idx])
    return hermite(t,
                   coeffs[idx],
                   coeffs[idx + 1],
                   slopes[idx],
                   slopes[idx + 1])


def kan_transform(features: np.ndarray) -> np.ndarray:
    """
    Apply a lightweight KAN‑style transformation: a linear projection
    followed by a spline activation. The projection matrix is
    deterministic (seeded by feature dimension) to keep reproducibility.
    """
    if features.ndim != 1:
        raise ValueError("features must be a 1‑D array")
    dim = features.shape[0]
    target_dim = max(1, dim // 2)  # modest compression

    rng = np.random.default_rng(seed=dim)
    proj = rng.normal(size=(target_dim, dim))
    projected = proj @ features
    activated = _spline_activation(projected)
    # Final L2 normalisation to keep scale comparable
    norm = np.linalg.norm(activated)
    return activated if norm == 0 else activated / norm


# ---------------------------------------------------------------------------
# Fusion core – deep integration of sheaf spectral energy and stylometry
# ---------------------------------------------------------------------------

def _spectral_energy(laplacian: np.ndarray) -> float:
    """
    Compute a scalar energy from the sheaf Laplacian's spectrum.
    We use the sum of the non‑zero eigenvalues (trace of L) which equals
    the Frobenius norm squared for positive‑semidefinite L.
    """
    if laplacian.size == 0:
        return 1.0
    # For numerical stability we ignore eigenvalues below a tiny threshold
    eigvals = np.linalg.eigvalsh(laplacian)
    eps = 1e-12
    return float(np.sum(eigvals[eigvals > eps]))


def fuse_features(text: str,
                  node_dims: Dict[int, int],
                  edge_list: List[Tuple[int, int]],
                  sketch_width: int = 64) -> np.ndarray:
    """
    End‑to‑end hybrid pipeline:
    1. Build the sheaf and obtain its Laplacian.
    2. Derive a scalar spectral energy from the Laplacian.
    3. Extract stylometry + sketch features.
    4. Modulate the stylometry vector by the spectral energy (non‑linear scaling).
    5. Pass the result through a KAN‑style transformation.
    Returns the final feature embedding.
    """
    # 1‑2. Sheaf construction and energy
    sheaf = Sheaf(node_dims=node_dims, edge_list=edge_list)
    L = sheaf.compute_laplacian()
    energy = _spectral_energy(L)
    # Guard against zero energy
    energy = max(energy, 1e-6)

    # 3. Stylometry vector
    base_vec = stylometry_vector(text, sketch_width=sketch_width)

    # 4. Energy‑driven modulation (log‑scale to avoid explosion)
    scale = np.log1p(energy)  # log(1+energy) ≥ 0
    modulated = base_vec * scale

    # 5. KAN‑style embedding
    embedding = kan_transform(modulated)
    return embedding


# ---------------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_text = (
        "In the quiet town, the wind whispered through the ancient oaks, "
        "carrying stories of generations past."
    )
    # Example sheaf: three nodes with varying dimensions
    node_dimensions = {0: 3, 1: 2, 2: 4}
    edges = [(0, 1), (1, 2), (2, 0)]

    vec = fuse_features(sample_text, node_dimensions, edges, sketch_width=64)
    print("Hybrid embedding (dim = {}):".format(vec.shape[0]))
    print(vec)