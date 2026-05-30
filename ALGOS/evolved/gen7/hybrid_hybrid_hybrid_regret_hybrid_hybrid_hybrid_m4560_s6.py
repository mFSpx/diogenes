# DARWIN HAMMER — match 4560, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_regret_engine_hybrid_hybrid_decisi_m82_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1204_s4.py (gen6)
# born: 2026-05-29T23:56:33Z

"""HybridRegretMorphologyEngine
Integrates:
- Parent A: regret‑weighted decision engine with decision‑hygiene cue extraction.
- Parent B: morphology‑based graph Laplacian using tropical (max‑plus) algebra and a circuit‑breaker.
Mathematical bridge:
The regret values of actions are interpreted as node potentials on a graph whose
edge weights are derived from the physical Morphology of the nodes.  The graph
Laplacian is built in the tropical (max‑plus) semiring; its dominant eigenvector
(in the max‑plus sense) supplies a “tropical potential” that rescales the
regret‑based probabilities.  Decision‑hygiene cue counts act as additional
multiplicative factors, completing a unified probability distribution over
actions.
"""

import re
import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (merged)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """An action with an expected value and optional cost/risk penalties."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """A counterfactual adjustment for a specific action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("width", self.width),
            ("height", self.height),
            ("mass", self.mass),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height


# ----------------------------------------------------------------------
# Decision‑hygiene cue extraction (Parent A)
# ----------------------------------------------------------------------


EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)

PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def extract_decision_hygiene_cues(text: str) -> Dict[str, int]:
    """Count evidence‑ and planning‑related cues in *text*."""
    cues = defaultdict(int)
    cues["evidence"] = len(EVIDENCE_RE.findall(text))
    cues["planning"] = len(PLANNING_RE.findall(text))
    return dict(cues)


# ----------------------------------------------------------------------
# Regret‑weighted core (Parent A)
# ----------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    epsilon: float = 1e-9,
) -> Dict[str, float]:
    """
    Produce a probability distribution over *actions* based on regret.

    Regret for an action is the expected shortfall between the counterfactual
    outcome and the action's nominal expected value, adjusted by cost and risk.
    The regrets are turned into a softmax distribution.
    """
    # Map counterfactuals for fast lookup
    cf_map: Dict[str, List[MathCounterfactual]] = defaultdict(list)
    for cf in counterfactuals:
        cf_map[cf.action_id].append(cf)

    regrets = []
    ids = []
    for act in actions:
        ids.append(act.id)
        # Base expected value reduced by cost & risk
        base = act.expected_value - act.cost - act.risk
        # Expected counterfactual outcome
        if act.id in cf_map:
            cf_vals = np.array([c.outcome_value * c.probability for c in cf_map[act.id]])
            cf_exp = cf_vals.mean()
        else:
            cf_exp = base
        regret = max(cf_exp - base, 0.0) + epsilon
        regrets.append(regret)

    regrets_arr = np.array(regrets)
    probs = _softmax(regrets_arr)
    return dict(zip(ids, probs))


# ----------------------------------------------------------------------
# Tropical‑maxplus Laplacian (Parent B)
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    import datetime, pytz

    return datetime.datetime.now(pytz.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True
            self.last_event_at = now_z()


class Sheaf:
    """
    Graph‑like structure where each node carries a Morphology.
    Edges are un‑weighted; weights are derived from node volumes.
    """

    def __init__(self, node_dims: Dict[str, Morphology], edge_list: List[Tuple[str, str]]):
        self.node_dims = dict(node_dims)  # id -> Morphology
        self.edges = list(edge_list)      # list of (src, dst)

    def _edge_weight(self, src: str, dst: str) -> float:
        """Weight inversely proportional to volume difference (always >0)."""
        vol_src = self.node_dims[src].volume
        vol_dst = self.node_dims[dst].volume
        diff = abs(vol_src - vol_dst)
        return 1.0 / (1.0 + diff)

    def compute_laplacian(self, tropical_maxplus: bool = False) -> np.ndarray:
        """
        Return the Laplacian matrix L.
        If tropical_maxplus is True, the matrix is interpreted in the
        max‑plus semiring: off‑diagonal entries are -weight, diagonal entries are
        max of incident weights.
        """
        ids = list(self.node_dims.keys())
        n = len(ids)
        idx = {node: i for i, node in enumerate(ids)}
        L = np.zeros((n, n), dtype=float)

        # accumulate weights
        for src, dst in self.edges:
            i, j = idx[src], idx[dst]
            w = self._edge_weight(src, dst)
            L[i, j] -= w
            L[j, i] -= w
            L[i, i] += w
            L[j, j] += w

        if tropical_maxplus:
            # Convert to max‑plus representation
            # Off‑diagonal: keep as negative weights (since max‑plus addition is max)
            # Diagonal: max of absolute off‑diagonal values for that row
            T = np.full_like(L, -np.inf)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        # max over incident weights (positive values)
                        incident = -L[i, :]  # because L off‑diag are negative
                        incident[i] = -np.inf
                        T[i, i] = np.max(incident) if incident.size > 0 else -np.inf
                    else:
                        T[i, j] = L[i, j]  # already negative weight
            return T
        else:
            return L


# ----------------------------------------------------------------------
# Tropical algebra utilities
# ----------------------------------------------------------------------


def _tropical_matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Max‑plus matrix–vector product: (A ⊗ x)_i = max_j (A_ij + x_j)
    """
    n = A.shape[0]
    y = np.full(n, -np.inf)
    for i in range(n):
        y[i] = np.max(A[i, :] + x)
    return y


def tropical_power_iteration(A: np.ndarray, steps: int = 15) -> np.ndarray:
    """
    Approximate the dominant eigenvector of A in the max‑plus semiring.
    The vector is normalised so that its maximum entry is zero (a common
    convention in tropical algebra).
    """
    n = A.shape[0]
    v = np.zeros(n)  # start with zero (neutral element)
    for _ in range(steps):
        v = _tropical_matvec(A, v)
        # Normalise: subtract max to avoid overflow
        v = v - np.max(v)
    return v


# ----------------------------------------------------------------------
# Hybrid operation (three demonstrative functions)
# ----------------------------------------------------------------------


def hybrid_regret_with_morphology(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    sheaf: Sheaf,
    text: str,
    epsilon: float = 1e-9,
) -> Dict[str, float]:
    """
    Core hybrid routine:
    1. Compute raw regret‑based probabilities.
    2. Build a tropical Laplacian from the Morphology graph.
    3. Obtain a tropical eigenvector (potential) and use it to rescale the
       regret probabilities.
    4. Apply decision‑hygiene cue multipliers.
    Returns a normalized probability distribution over action IDs.
    """
    # 1. Regret‑based distribution
    raw_probs = compute_regret_weighted_strategy(actions, counterfactuals, epsilon)

    # 2. Tropical Laplacian
    L_trop = sheaf.compute_laplacian(tropical_maxplus=True)

    # 3. Tropical eigenvector (potential) – one entry per node
    potential = tropical_power_iteration(L_trop)

    # Map node potentials to actions (assume action.id matches a node id)
    node_ids = list(sheaf.node_dims.keys())
    pot_map = {nid: potential[i] for i, nid in enumerate(node_ids)}

    # 4. Decision‑hygiene cue scaling
    cues = extract_decision_hygiene_cues(text)
    cue_factor = 1.0 + 0.1 * cues.get("evidence", 0) + 0.05 * cues.get("planning", 0)

    # Combine: multiply raw probability by exp(potential) to keep positivity
    combined = {}
    for act in actions:
        base = raw_probs.get(act.id, 0.0)
        node_pot = pot_map.get(act.id, -np.inf)
        # If node not present, treat potential as neutral (0)
        pot_factor = math.exp(node_pot) if node_pot != -np.inf else 1.0
        combined[act.id] = base * pot_factor * cue_factor

    # Normalise
    total = sum(combined.values()) + epsilon
    return {k: v / total for k, v in combined.items()}


def circuit_breaker_guard(sheaf: Sheaf, breaker: EndpointCircuitBreaker) -> bool:
    """
    Demonstrates the integration of the circuit‑breaker concept.
    Returns True if the breaker is open (i.e., too many failures) after a
    synthetic check of Laplacian conditioning.
    """
    try:
        L = sheaf.compute_laplacian()
        # Simple sanity: matrix must be finite and not singular (det != 0)
        cond = np.linalg.cond(L)
        if not np.isfinite(cond) or cond > 1e12:
            breaker.record_failure()
        else:
            breaker.record_success()
    except Exception:
        breaker.record_failure()
    return breaker.open


def hybrid_fisher_adjusted_weights(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    sheaf: Sheaf,
) -> np.ndarray:
    """
    Compute a weight vector for the actions where the Fisher score (variance of
    regrets) modulates the tropical Laplacian diagonal.
    Returns a 1‑D numpy array aligned with the order of *actions*.
    """
    # Regret values (not normalized)
    regrets = []
    cf_map: Dict[str, List[MathCounterfactual]] = defaultdict(list)
    for cf in counterfactuals:
        cf_map[cf.action_id].append(cf)

    for act in actions:
        base = act.expected_value - act.cost - act.risk
        if act.id in cf_map:
            cf_vals = np.array([c.outcome_value * c.probability for c in cf_map[act.id]])
            cf_exp = cf_vals.mean()
        else:
            cf_exp = base
        regret = max(cf_exp - base, 0.0)
        regrets.append(regret)

    regrets_arr = np.array(regrets)
    # Fisher score approximated as variance / (mean + epsilon)
    fisher_score = np.var(regrets_arr) / (np.mean(regrets_arr) + 1e-9)

    # Build tropical Laplacian and boost diagonal by (1 + fisher_score)
    L_trop = sheaf.compute_laplacian(tropical_maxplus=True)
    np.fill_diagonal(L_trop, L_trop.diagonal() * (1.0 + fisher_score))

    # Project the Laplacian onto the action space (assume same IDs)
    ids = [act.id for act in actions]
    node_ids = list(sheaf.node_dims.keys())
    idx_map = {nid: i for i, nid in enumerate(node_ids)}
    weights = np.zeros(len(actions))
    for i, aid in enumerate(ids):
        if aid in idx_map:
            weights[i]