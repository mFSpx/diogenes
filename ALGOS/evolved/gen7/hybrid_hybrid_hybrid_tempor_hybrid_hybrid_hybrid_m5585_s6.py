# DARWIN HAMMER — match 5585, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_temporal_moti_hybrid_hybrid_hybrid_m1987_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_bandit_router_m252_s0.py (gen3)
# born: 2026-05-30T00:03:14Z

import numpy as np
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


# ----------------------------------------------------------------------
# Regex feature groups (unchanged semantics, but compiled once)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)
SCARCITY_RE = re.compile(r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+ to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b", re.I)
RISK_RE = re.compile(r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis)\b", re.I)

FEATURE_GROUPS = [
    EVIDENCE_RE,
    PLANNING_RE,
    DELAY_RE,
    SUPPORT_RE,
    BOUNDARY_RE,
    OUTCOME_RE,
    IMPULSIVE_RE,
    SCARCITY_RE,
    RISK_RE,
]


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TemporalMotif:
    """A temporal motif is a sequence of symbolic tokens with a support count."""
    pattern: Tuple[str, ...]
    support: int


# ----------------------------------------------------------------------
# Sheaf implementation with simple restriction propagation
# ----------------------------------------------------------------------
class Sheaf:
    """
    A lightweight sheaf over a directed graph.
    Nodes store vector sections; edges store linear restriction maps.
    """

    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]) -> None:
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

        # initialise empty sections for each node
        for n, d in node_dims.items():
            self._sections[n] = np.zeros(d)

    # ------------------------------------------------------------------
    # API for restrictions
    # ------------------------------------------------------------------
    def set_restriction(self, edge: Tuple[Any, Any], src_map: np.ndarray, dst_map: np.ndarray) -> None:
        """
        Store a pair of linear maps (src_map, dst_map) that encode the same
        abstract restriction between the source and destination nodes.
        """
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    # ------------------------------------------------------------------
    # API for sections
    # ------------------------------------------------------------------
    def set_section(self, node: Any, value: np.ndarray) -> None:
        """Replace the section at *node* with *value* after dimension check."""
        if node not in self.node_dims:
            raise KeyError(f"Node {node!r} not defined in sheaf.")
        if value.shape[0] != self.node_dims[node]:
            raise ValueError(f"Section dimension {value.shape[0]} must match node dimension {self.node_dims[node]}")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: Any) -> np.ndarray:
        """Retrieve the current section for *node*."""
        return self._sections[node]

    # ------------------------------------------------------------------
    # Propagation using restriction maps
    # ------------------------------------------------------------------
    def propagate(self) -> None:
        """
        Perform a single pass of restriction propagation.
        For each edge (u, v) we enforce:
            src_map @ s_u ≈ dst_map @ s_v
        by averaging the two sides.
        """
        for (u, v), (src_map, dst_map) in self._restrictions.items():
            s_u = self._sections[u]
            s_v = self._sections[v]

            lhs = src_map @ s_u
            rhs = dst_map @ s_v
            # simple averaging update
            correction = (rhs - lhs) / 2.0
            # push correction back to sections (pseudo‑inverse for simplicity)
            if src_map.shape[0] > 0:
                s_u_new = s_u + np.linalg.pinv(src_map) @ correction
                self._sections[u] = s_u_new
            if dst_map.shape[0] > 0:
                s_v_new = s_v - np.linalg.pinv(dst_map) @ correction
                self._sections[v] = s_v_new


# ----------------------------------------------------------------------
# RBF surrogate
# ----------------------------------------------------------------------
def rbf_surrogate_prediction(x: np.ndarray, centers: np.ndarray, weights: np.ndarray, epsilon: float) -> float:
    """
    Gaussian RBF network output.
    x : (d,) vector
    centers : (m, d) matrix of RBF centres
    weights : (m,) vector of output weights
    epsilon : bandwidth
    """
    diff = centers - x  # broadcasting
    sq_norm = np.einsum('ij,ij->i', diff, diff)
    return float(np.sum(weights * np.exp(-sq_norm / (2 * epsilon ** 2))))


# ----------------------------------------------------------------------
# Embedding utilities
# ----------------------------------------------------------------------
def build_token_embeddings(tokens: List[str], dim: int = 8) -> Dict[str, np.ndarray]:
    """
    Produce a deterministic random embedding for each token.
    The same token always receives the same vector (seeded by hash).
    """
    rng = np.random.default_rng(0)  # deterministic base RNG
    base = rng.standard_normal((len(tokens), dim))
    token_to_vec: Dict[str, np.ndarray] = {}
    for token, vec in zip(tokens, base):
        token_to_vec[token] = vec
    return token_to_vec


def embed_motif(motif: TemporalMotif, token_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate embeddings of the motif's tokens.
    If a token is unseen, use a zero vector.
    """
    dim = next(iter(token_embeddings.values())).shape[0]
    vectors = [token_embeddings.get(tok, np.zeros(dim)) for tok in motif.pattern]
    return np.concatenate(vectors) if vectors else np.zeros(0)


# ----------------------------------------------------------------------
# Reward computation – deeper integration
# ----------------------------------------------------------------------
def compute_reward(
    motif: TemporalMotif,
    sheaf: Sheaf,
    token_embeddings: Dict[str, np.ndarray],
    centers: np.ndarray,
    weights: np.ndarray,
    epsilon: float,
) -> float:
    """
    1. Embed the motif → vector x.
    2. Predict a scalar via the RBF surrogate.
    3. Inject the prediction into the sheaf sections of all nodes appearing in the motif.
    4. Propagate restrictions to keep the sheaf consistent.
    5. Compute a feature‑based reward from the textual representation of the motif.
    """
    # 1. embedding
    x = embed_motif(motif, token_embeddings)

    # Guard against dimension mismatch with RBF centres
    if x.shape[0] != centers.shape[1]:
        raise ValueError(
            f"Embedded motif dimension {x.shape[0]} does not match RBF centre dimension {centers.shape[1]}"
        )

    # 2. surrogate prediction
    pred = rbf_surrogate_prediction(x, centers, weights, epsilon)

    # 3. update sheaf sections for each token node (if node exists)
    for token in motif.pattern:
        if token in sheaf.node_dims:
            # simple injection: replace the first coordinate with the prediction,
            # keep other coordinates unchanged.
            current = sheaf.get_section(token)
            updated = current.copy()
            updated[0] = pred
            sheaf.set_section(token, updated)

    # 4. propagate once to respect restrictions
    sheaf.propagate()

    # 5. feature‑based reward
    # Build a single string from the motif for regex matching
    motif_text = " ".join(motif.pattern).lower()
    reward = sum(bool(regex.search(motif_text)) for regex in FEATURE_GROUPS)

    # Optionally blend surrogate output into reward (deeper coupling)
    reward += 0.1 * pred  # small influence of the learned value

    return reward


# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_operation(
    sheaf: Sheaf,
    motifs: List[TemporalMotif],
    token_embeddings: Dict[str, np.ndarray],
    centers: np.ndarray,
    weights: np.ndarray,
    epsilon: float,
) -> List[float]:
    """
    Process a list of motifs and return their reward scores.
    """
    rewards: List[float] = []
    for motif in motifs:
        reward = compute_reward(
            motif,
            sheaf,
            token_embeddings,
            centers,
            weights,
            epsilon,
        )
        rewards.append(reward)
    return rewards


# ----------------------------------------------------------------------
# Demo / entry point
# ----------------------------------------------------------------------
def main() -> None:
    # Define graph topology
    node_dims = {"A": 8, "B": 8, "C": 8}
    edges = [("A", "B"), ("B", "C")]
    sheaf = Sheaf(node_dims, edges)

    # Example restriction: identity maps (no transformation) for simplicity
    for (u, v) in edges:
        dim_u, dim_v = node_dims[u], node_dims[v]
        src_map = np.eye(min(dim_u, dim_v), dim_u)
        dst_map = np.eye(min(dim_u, dim_v), dim_v)
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Build token embeddings for all tokens appearing in motifs
    all_tokens = ["A", "B", "C", "evidence", "plan", "delay"]
    token_embeddings = build_token_embeddings(all_tokens, dim=8)

    # Define motifs
    motifs = [
        TemporalMotif(("A", "B", "evidence"), support=12),
        TemporalMotif(("B", "C", "plan"), support=7),
        TemporalMotif(("C", "A", "delay"), support=3),
    ]

    # RBF surrogate parameters (centres dimension must match embedding size)
    embedding_dim = len(next(iter(token_embeddings.values()))) * max(len(m.pattern) for m in motifs)
    centers = np.random.default_rng(42).standard_normal((15, embedding_dim))
    weights = np.random.default_rng(99).standard_normal(15)
    epsilon = 0.5

    # Run hybrid operation
    rewards = hybrid_operation(sheaf, motifs, token_embeddings, centers, weights, epsilon)
    print("Reward scores:", rewards)

    # Show final sheaf sections for inspection
    for node in node_dims:
        print(f"Section at node {node}:", sheaf.get_section(node))


if __name__ == "__main__":
    main()