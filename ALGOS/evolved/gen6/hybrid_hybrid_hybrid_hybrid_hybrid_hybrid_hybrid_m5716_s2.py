# DARWIN HAMMER — match 5716, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_gliner_m2542_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_korpus_m1983_s1.py (gen5)
# born: 2026-05-30T00:04:23Z

"""Hybrid Allocation‑Fisher‑Geometric + Bandit‑RBF‑Voronoi Fusion

Parents
-------
* hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_gliner_m2542_s1.py  (Multivector &
  Fisher‑Gini span analysis)
* hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_korpus_m1983_s1.py   (Bandit
  decision, RBF surrogate and Text‑Geometric Voronoi)

Mathematical Bridge
-------------------
Spans extracted by the GLiNER zero‑shot extractor are represented as 2‑D points
P_i = (s_i, ℓ_i) where s_i is the start index and ℓ_i = end‑start is the length.
These points serve two purposes:

1. **Voronoi seeding** – the set {P_i} becomes the seed points for a Voronoi
   diagram.  Each Voronoi cell is associated with a multivector V_i that
   encodes the geometric product of the cell’s seed with a bandit‑derived
   context vector C.

2. **Statistical weighting** – the lengths {ℓ_i} are used to compute a Fisher
   information matrix (approximated by the sample variance) and a Gini
   coefficient.  The resulting scalar w_F quantifies the diversity and
   confidence of the extracted information.

The Bandit core supplies a set of candidate actions A_j with context‑dependent
expected rewards ˆr_j.  An RBF surrogate model f̂(C) approximates the mapping
from a context vector C to a scalar reward using the Voronoi‑cell multivectors
as training samples.  The final hybrid score for an action combines:

    S_j = ̂r_j * w_F * Gini(ℓ) * ⟨V_i , V_k⟩

where ⟨·,·⟩ denotes the scalar part of the geometric product between the
multivector of the selected cell and the multivector of the action’s context.
The algorithm thus fuses statistical span analysis, Clifford geometric algebra,
bandit decision making and RBF‑surrogate approximation into a single unified
metric.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Sequence, Any
import numpy as np

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
Vector = Sequence[float]

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str

    @property
    def length(self) -> int:
        return self.end - self.start

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# ----------------------------------------------------------------------
# Clifford geometric algebra (minimal implementation)
# ----------------------------------------------------------------------
class Multivector:
    """A very small Clifford algebra where each basis element is represented by a string."""
    def __init__(self, coeffs: Dict[str, float] = None):
        self.coeffs: Dict[str, float] = coeffs if coeffs is not None else {}

    def __add__(self, other: "Multivector") -> "Multivector":
        result = self.coeffs.copy()
        for b, v in other.coeffs.items():
            result[b] = result.get(b, 0.0) + v
        return Multivector(result)

    def __mul__(self, other: Any) -> "Multivector":
        """Geometric product (simplified): scalar‑scalar multiplies, basis‑basis concatenates."""
        if isinstance(other, Multivector):
            result: Dict[str, float] = {}
            for b1, c1 in self.coeffs.items():
                for b2, c2 in other.coeffs.items():
                    # Concatenate basis identifiers; if they cancel (same) we treat as scalar
                    if b1 == b2:
                        basis = ""  # scalar part
                    else:
                        # keep sorted order to stay deterministic
                        basis = ''.join(sorted(b1 + b2))
                    result[basis] = result.get(basis, 0.0) + c1 * c2
            return Multivector(result)
        elif isinstance(other, (int, float)):
            return Multivector({b: c * other for b, c in self.coeffs.items()})
        else:
            raise TypeError(f"Unsupported multiplication with {type(other)}")

    def scalar_part(self) -> float:
        """Return the coefficient of the empty basis (the scalar part)."""
        return self.coeffs.get("", 0.0)

    def __repr__(self) -> str:
        return f"Multivector({self.coeffs})"

# ----------------------------------------------------------------------
# Statistical helpers (Fisher information & Gini)
# ----------------------------------------------------------------------
def fisher_information(lengths: np.ndarray) -> float:
    """Approximate Fisher information for a univariate Gaussian by the inverse variance."""
    if len(lengths) < 2:
        return 0.0
    var = np.var(lengths, ddof=1)
    return 0.0 if var == 0 else 1.0 / var

def gini_coefficient(values: np.ndarray) -> float:
    """Standard Gini coefficient for a 1‑D array."""
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals, dtype=float)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return gini

# ----------------------------------------------------------------------
# Simple zero‑shot span extractor (mock)
# ----------------------------------------------------------------------
def mock_gliner_extract(text: str) -> List[Span]:
    """Return a deterministic set of spans for demonstration."""
    words = text.split()
    spans: List[Span] = []
    idx = 0
    for i, w in enumerate(words):
        start = idx
        end = idx + len(w)
        spans.append(
            Span(
                start=start,
                end=end,
                text=w,
                label="WORD",
                score=random.uniform(0.6, 1.0),
                backend="mock"
            )
        )
        idx = end + 1  # account for space
    return spans

# ----------------------------------------------------------------------
# Bandit core (UCB‑like selection)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()

def register_action(action: BanditAction) -> None:
    _POLICY.setdefault(action.action_id, []).append(action.expected_reward)

def select_action_ucb(context_id: str, actions: List[BanditAction], alpha: float = 2.0) -> BanditAction:
    """Upper‑confidence‑bound selection."""
    best_score = -math.inf
    best_action = actions[0]
    for act in actions:
        rewards = _POLICY.get(act.action_id, [])
        n = len(rewards)
        avg = sum(rewards) / n if n > 0 else 0.0
        bound = alpha * math.sqrt(math.log(max(1, len(_POLICY))) / (n + 1))
        score = avg + bound
        if score > best_score:
            best_score = score
            best_action = act
    return best_action

def update_bandit(update: BanditUpdate) -> None:
    _POLICY.setdefault(update.action_id, []).append(update.reward)

# ----------------------------------------------------------------------
# RBF surrogate model (kernel regression)
# ----------------------------------------------------------------------
def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    diff = x - y
    return math.exp(-np.dot(diff, diff) / (2 * sigma ** 2))

def rbf_surrogate_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    query: np.ndarray,
    sigma: float = 1.0,
    eps: float = 1e-8
) -> float:
    """Weighted average using RBF kernel."""
    kernels = np.array([rbf_kernel(query, xi, sigma) for xi in train_X])
    if kernels.sum() < eps:
        return float(train_y.mean()) if train_y.size else 0.0
    return float((kernels @ train_y) / kernels.sum())

# ----------------------------------------------------------------------
# Voronoi seed generation from spans
# ----------------------------------------------------------------------
def spans_to_seed_points(spans: List[Span]) -> np.ndarray:
    """Map each span to a 2‑D point (start, length)."""
    points = np.array([[s.start, s.length] for s in spans], dtype=float)
    return points

def compute_voronoi_multivectors(
    seeds: np.ndarray,
    context_vec: Vector
) -> List[Multivector]:
    """
    For each seed point create a multivector that encodes the geometric product
    between the seed (as a vector basis) and the context vector.
    """
    multivectors: List[Multivector] = []
    for i, seed in enumerate(seeds):
        # Encode seed as basis e_start * e_length
        basis = f"e{int(seed[0])}_e{int(seed[1])}"
        seed_mv = Multivector({basis: 1.0})
        # Encode context as scalar multivector
        ctx_mv = Multivector({ "": float(np.linalg.norm(context_vec)) })
        # Geometric product
        mv = seed_mv * ctx_mv
        multivectors.append(mv)
    return multivectors

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_score(
    text: str,
    context_vec: Vector,
    actions: List[BanditAction]
) -> Tuple[BanditAction, float]:
    """
    Compute a hybrid metric for each action and return the best action
    together with its score.
    """
    # 1. Extract spans and compute statistical weights
    spans = mock_gliner_extract(text)
    lengths = np.array([s.length for s in spans], dtype=float)
    w_fisher = fisher_information(lengths)
    gini = gini_coefficient(lengths)

    # 2. Voronoi seeds and multivectors
    seeds = spans_to_seed_points(spans)
    region_mvs = compute_voronoi_multivectors(seeds, context_vec)

    # 3. Prepare surrogate training data (use scalar part of each region MV as target)
    train_X = seeds
    train_y = np.array([mv.scalar_part() for mv in region_mvs], dtype=float)

    # 4. Evaluate each action
    best_score = -math.inf
    best_action = actions[0]
    for act in actions:
        # surrogate prediction for the context
        pred = rbf_surrogate_predict(train_X, train_y, np.array(context_vec, dtype=float))
        # combine with bandit expected reward
        base = act.expected_reward * pred
        # fuse with statistical weights
        score = base * w_fisher * gini
        if score > best_score:
            best_score = score
            best_action = act
    return best_action, best_score

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset any global state
    reset_policy()

    # Dummy text
    sample_text = "The quick brown fox jumps over the lazy dog"

    # Dummy context vector (e.g., embedding)
    ctx = np.random.randn(5)

    # Define a few actions
    actions = [
        BanditAction(
            action_id="A1",
            propensity=0.3,
            expected_reward=0.7,
            confidence_bound=0.1,
            algorithm="ucb"
        ),
        BanditAction(
            action_id="A2",
            propensity=0.5,
            expected_reward=0.4,
            confidence_bound=0.2,
            algorithm="ucb"
        ),
        BanditAction(
            action_id="A3",
            propensity=0.2,
            expected_reward=0.9,
            confidence_bound=0.05,
            algorithm="ucb"
        ),
    ]

    # Register actions to initialise policy (optional)
    for a in actions:
        register_action(a)

    # Compute hybrid decision
    chosen, score = hybrid_score(sample_text, ctx, actions)

    print(f"Chosen action: {chosen.action_id}")
    print(f"Hybrid score: {score:.6f}")

    # Perform a mock update
    update = BanditUpdate(
        context_id="ctx1",
        action_id=chosen.action_id,
        reward=score,  # treat the hybrid score as reward
        propensity=chosen.propensity
    )
    update_bandit(update)

    # Verify that policy updated without error
    print(f"Policy size after update: {len(_POLICY)}")