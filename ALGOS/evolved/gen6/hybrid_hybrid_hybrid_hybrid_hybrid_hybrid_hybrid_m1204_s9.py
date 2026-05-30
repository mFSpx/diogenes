# DARWIN HAMMER — match 1204, survivor 9
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m504_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_tropical_maxp_m1163_s0.py (gen5)
# born: 2026-05-29T23:34:38Z

import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------- Parent A components ----------
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
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Tokenise a string into alphabetic lower‑case words."""
    return [w for w in (text or "").lower().split() if w.isalpha()]


def lsm_vector(text: str) -> dict[str, float]:
    """Compute the proportion of each functional‑category vocabulary in the text."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stylometry_features(text: str) -> np.ndarray:
    """Return a fixed‑size feature vector (one entry per FUNCTION_CATS key)."""
    lsm = lsm_vector(text)
    return np.array(list(lsm.values()), dtype=float)


@dataclass
class Sheaf:
    """Simple sheaf representation with node dimensions and an undirected edge list."""

    node_dims: Dict[int, Tuple[float, ...]]
    edge_list: List[Tuple[int, int]]

    def compute_laplacian(self) -> np.ndarray:
        """Return a signed incidence‑derived Laplacian matrix."""
        n = len(self.node_dims)
        L = np.zeros((n, n), dtype=float)
        for u, v in self.edge_list:
            L[u, v] = -1.0
            L[v, u] = 1.0
        return L


# ---------- Parent B components ----------
class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True


def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition (max)."""
    return np.maximum(x, y)


def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication (ordinary addition)."""
    return np.add(x, y)


def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tropical matrix multiplication: (A ⊗ B)_{ij} = max_k (A_{ik} + B_{kj})."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # broadcasting to compute pairwise sums then max over k
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


# ---------- Hybrid core ----------
def tropical_closure(L: np.ndarray, steps: int = 5) -> np.ndarray:
    """
    Compute a max‑plus closure of the Laplacian matrix.
    C = L ⊕ L^{⊗2} ⊕ … ⊕ L^{⊗steps},
    where ⊕ is tropical addition (max) and ⊗ is tropical multiplication (t_matmul).
    The result encodes the strongest path weight between every pair of nodes.
    """
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix")
    n = L.shape[0]
    # initialise closure with the original matrix and identity (0 on diagonal, -inf elsewhere)
    closure = np.copy(L)
    # tropical identity: 0 on diagonal, -inf off‑diagonal
    identity = np.full((n, n), -np.inf, dtype=float)
    np.fill_diagonal(identity, 0.0)
    power = np.copy(L)

    for _ in range(1, steps):
        power = t_matmul(power, L)          # L^{⊗(k+1)}
        closure = t_add(closure, power)    # max‑accumulate
    # also consider the identity (paths of length zero)
    closure = t_add(closure, identity)
    return closure


def modulate_features(features: np.ndarray, tropical_weights: np.ndarray) -> np.ndarray:
    """
    Modulate a stylometric feature vector using tropical weights.
    The tropical_weights vector is derived from the row‑wise max‑plus influence of the graph.
    Modulation is performed via element‑wise multiplication after normalising both arrays.
    """
    if features.ndim != 1 or tropical_weights.ndim != 1:
        raise ValueError("Both inputs must be 1‑D vectors")
    # Normalise to unit L2 norm to keep scale comparable
    f_norm = np.linalg.norm(features) or 1.0
    w_norm = np.linalg.norm(tropical_weights) or 1.0
    f_unit = features / f_norm
    w_unit = tropical_weights / w_norm
    # Element‑wise scaling
    return f_unit * w_unit


def hybrid_process(
    text: str,
    sheaf: Sheaf,
    breaker: EndpointCircuitBreaker,
    steps: int = 4,
    variance_threshold: float = 0.02,
) -> Tuple[np.ndarray, bool]:
    """
    End‑to‑end hybrid pipeline integrating all components.
    Returns a modulated stylometric feature vector and a boolean indicating whether the circuit breaker is open.
    """
    laplacian = sheaf.compute_laplacian()
    tropical_weights = np.max(tropical_closure(laplacian, steps), axis=1)
    stylometric_features_vector = stylometry_features(text)
    modulated_features = modulate_features(stylometric_features_vector, tropical_weights)
    variance = np.var(modulated_features)
    if variance > variance_threshold:
        breaker.record_failure()
    else:
        breaker.record_success()
    return modulated_features, breaker.open


if __name__ == "__main__":
    node_dims = {0: (1.0, 2.0), 1: (3.0, 4.0), 2: (5.0, 6.0)}
    edge_list = [(0, 1), (1, 2), (2, 0)]
    sheaf = Sheaf(node_dims, edge_list)
    breaker = EndpointCircuitBreaker()
    text = "This is a test sentence."
    modulated_features, breaker_open = hybrid_process(text, sheaf, breaker)
    print(modulated_features)
    print(breaker_open)