# DARWIN HAMMER — match 2121, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_hybrid_m883_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_sketch_epistemic_certainty_m2_s2.py (gen3)
# born: 2026-05-29T23:40:59Z

"""Hybrid Fisher–Sheaf Stylometry Algorithm
Parent A: hybrid_hybrid_fisher_locali_hybrid_hybrid_hybrid_m883_s0.py
Parent B: hybrid_hybrid_hybrid_sketch_epistemic_certainty_m2_s2.py

Mathematical bridge:
- The stylometry feature matrix from Parent A is interpreted as *sections* of a sheaf
  defined over a graph of textual documents (nodes) and their semantic links (edges).
- For each node the Fisher‑information matrix (computed from the word‑frequency
  multinomial model) provides a local metric that we embed into the sheaf as the
  *restriction* maps between neighbouring nodes.  The restriction maps are
  constructed by projecting the source section onto the eigen‑basis of the
  Fisher matrix and then re‑expressing it in the destination basis.
- The Real Log Canonical Threshold (RLTL) from Parent B is approximated by the
  trace of the Fisher matrix; this trace is used as an uncertainty weight for
  the coboundary operator that measures local disagreement between sections.
Thus the algorithm jointly evaluates stylometric similarity, Fisher‑information
based confidence, and sheaf‑cohomological disagreement, yielding a unified
scalar “hybrid score’’ for any collection of texts.
"""

import sys
import math
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

# ----------------------------------------------------------------------
# 1. Stylometry utilities (derived from Parent A)
# ----------------------------------------------------------------------
FUNCTION_CATS = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def _clean_word(word: str) -> str:
    return "".join(ch for ch in word.lower() if ch.isalpha())


def words(text: str) -> list[str]:
    """Tokenise a string into alphabetic lower‑case words."""
    return [_clean_word(w) for w in (text or "").split() if _clean_word(w)]


def stylometry_vector(text: str) -> np.ndarray:
    """
    Build a fixed‑size stylometry feature vector.
    Order: [pronoun, article, preposition, auxiliary, conjunction,
            negation, quantifier, adverb_common] (counts normalized).
    """
    ws = words(text)
    total = max(1, len(ws))
    cat_counts = []
    for cat in [
        "pronoun",
        "article",
        "preposition",
        "auxiliary",
        "conjunction",
        "negation",
        "quantifier",
        "adverb_common",
    ]:
        cat_set = FUNCTION_CATS[cat]
        cnt = sum(1 for w in ws if w in cat_set)
        cat_counts.append(cnt / total)
    return np.array(cat_counts, dtype=float)


# ----------------------------------------------------------------------
# 2. Fisher‑information utilities (derived from Parent A)
# ----------------------------------------------------------------------
def fisher_information(text: str, theta_grid: np.ndarray = None) -> np.ndarray:
    """
    Approximate the Fisher‑information matrix for a multinomial model of word
    frequencies.  The parameter vector theta is the probability of each distinct
    word.  For a multinomial the Fisher matrix is diag(1/theta_i) – 1/θ_sum.
    We approximate by using empirical frequencies.
    """
    ws = words(text)
    if not ws:
        # Return a tiny identity matrix to avoid singularities.
        return np.eye(1) * 1e-6

    freq = Counter(ws)
    total = sum(freq.values())
    probs = np.array([c / total for c in freq.values()], dtype=float)

    # Diagonal part
    diag = np.diag(1.0 / np.maximum(probs, 1e-12))
    # Rank‑1 correction (since multinomial parameters sum to 1)
    ones = np.ones_like(probs)
    correction = np.outer(ones, ones) / max(1.0, total)
    fisher = diag - correction
    return fisher


def fisher_trace(text: str) -> float:
    """Scalar proxy for Fisher information: the trace of the matrix."""
    return np.trace(fisher_information(text))


# ----------------------------------------------------------------------
# 3. Sheaf core (derived from Parent B)
# ----------------------------------------------------------------------
class HybridSheaf:
    """
    Sheaf over a graph where:
    - node_dims : dimensionality of each node (derived from Fisher rank)
    - sections  : actual stylometry vectors placed on nodes
    - restrictions: linear maps between neighbouring nodes built from
      Fisher eigen‑bases.
    """

    def __init__(self, node_dims: dict, edges: list[tuple[int, int]]):
        self.node_dims = dict(node_dims)          # node_id -> dim (int)
        self.edges = list(edges)                  # list of (u, v)
        self._restrictions = {}                   # (u,v) -> (src_map, dst_map)
        self._sections = {}                       # node_id -> np.ndarray

    def set_section(self, node, vector):
        self._sections[node] = np.asarray(vector, dtype=float)

    def set_restriction(self, edge, src_map, dst_map):
        u, v = edge
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float),
                                      np.asarray(dst_map, dtype=float))

    def _apply_restriction(self, edge, source_vec):
        """Project source_vec using the stored restriction maps."""
        src_map, dst_map = self._restrictions[edge]
        # First map into source eigen‑basis, then into destination basis.
        return dst_map @ (src_map.T @ source_vec)

    def coboundary_norm(self) -> float:
        """
        Compute the global disagreement ‖δs‖² = Σ_{(u,v)} ||s_u - R_{u→v}s_v||²,
        weighted by the trace of the Fisher matrix of the destination node.
        """
        total = 0.0
        for (u, v) in self.edges:
            su = self._sections[u]
            sv = self._sections[v]
            # Apply restriction from v to u (so both live in u's space)
            if (v, u) in self._restrictions:
                sv_proj = self._apply_restriction((v, u), sv)
            else:
                # If no restriction, fall back to simple truncation/padding
                dim = su.shape[0]
                sv_proj = sv[:dim] if sv.shape[0] >= dim else np.pad(sv, (0, dim - sv.shape[0]))
            diff = su - sv_proj
            weight = max(1e-6, self.node_dims.get(v, 1))  # use node dimension as proxy for uncertainty
            total += weight * np.dot(diff, diff)
        return total


# ----------------------------------------------------------------------
# 4. Hybrid construction utilities
# ----------------------------------------------------------------------
def build_hybrid_sheaf(corpus: dict[int, str], edges: list[tuple[int, int]]) -> HybridSheaf:
    """
    Build a HybridSheaf from a corpus.
    - corpus: mapping node_id -> raw text.
    - edges: list of undirected edges (u, v).
    Steps:
    1. Compute stylometry vectors (sections).
    2. Compute Fisher trace → node dimension (rounded int, min 1).
    3. For each edge, compute restriction maps using eigen‑vectors of the
       Fisher matrices of the two incident nodes.
    """
    # 1. Sections
    sections = {nid: stylometry_vector(txt) for nid, txt in corpus.items()}

    # 2. Node dimensions from Fisher trace (rounded)
    node_dims = {}
    fisher_mats = {}
    for nid, txt in corpus.items():
        F = fisher_information(txt)
        fisher_mats[nid] = F
        # Use rank (number of eigenvalues > ε) as dimension, but keep at least len(section)
        eigvals = np.linalg.eigvalsh(F)
        rank = int(np.sum(eigvals > 1e-8))
        node_dims[nid] = max(rank, sections[nid].shape[0])

    sheaf = HybridSheaf(node_dims, edges)

    # Populate sections (pad/truncate to node_dims)
    for nid, vec in sections.items():
        dim = node_dims[nid]
        if vec.shape[0] < dim:
            padded = np.pad(vec, (0, dim - vec.shape[0]))
        else:
            padded = vec[:dim]
        sheaf.set_section(nid, padded)

    # 3. Restrictions
    for (u, v) in edges:
        Fu = fisher_mats[u]
        Fv = fisher_mats[v]

        # Eigen‑decomposition (orthonormal eigenvectors)
        _, Su = np.linalg.eigh(Fu)
        _, Sv = np.linalg.eigh(Fv)

        # Build linear maps: project from v‑space to u‑space.
        # src_map maps v‑basis → shared ambient space (identity), dst_map maps that space → u‑basis.
        src_map = Sv  # columns are eigenvectors of v
        dst_map = Su  # columns are eigenvectors of u
        sheaf.set_restriction((v, u), src_map, dst_map)

    return sheaf


def hybrid_uncertainty_score(sheaf: HybridSheaf) -> float:
    """
    Combine sheaf coboundary norm with global Fisher trace.
    Score = coboundary_norm / (1 + Σ trace(F_i))
    Smaller scores indicate higher consistency and lower uncertainty.
    """
    cob_norm = sheaf.coboundary_norm()
    total_fisher = sum(fisher_trace(txt) for txt in [])
    # The above sum would be zero because we have no texts here; we instead
    # approximate using node_dims (which already embed Fisher information).
    total_fisher = sum(dim for dim in sheaf.node_dims.values())
    return cob_norm / (1.0 + total_fisher)


def hybrid_score_for_corpus(corpus: dict[int, str], edges: list[tuple[int, int]]) -> float:
    """
    End‑to‑end hybrid score: build sheaf, compute uncertainty, and return a scalar.
    """
    sheaf = build_hybrid_sheaf(corpus, edges)
    return hybrid_uncertainty_score(sheaf)


# ----------------------------------------------------------------------
# 5. Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal example with three tiny documents.
    sample_corpus = {
        0: "I love programming and I love Python. It's great!",
        1: "You should consider learning JavaScript; it is also fun.",
        2: "They are testing the new algorithm; it seems promising."
    }
    # Simple chain graph 0-1-2
    sample_edges = [(0, 1), (1, 2)]

    score = hybrid_score_for_corpus(sample_corpus, sample_edges)
    print(f"Hybrid uncertainty score: {score:.6f}")
    # Ensure that the code runs without exception.
    sys.exit(0)