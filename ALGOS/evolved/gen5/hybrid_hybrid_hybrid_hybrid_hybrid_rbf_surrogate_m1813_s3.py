# DARWIN HAMMER — match 1813, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s0.py (gen4)
# parent_b: hybrid_rbf_surrogate_tri_algo_conduit_m8_s0.py (gen1)
# born: 2026-05-29T23:38:57Z

"""Hybrid Certainty‑RBF Bandit (HCRB)

Parents:
- hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s0.py (epistemic certainty flags,
  labeling aggregation and bandit guidance)
- hybrid_rbf_surrogate_tri_algo_conduit_m8_s0.py (radial‑basis‑function surrogate
  model with linear system solve)

Mathematical bridge:
The epistemic certainty of a labeling function is expressed by a
`CertaintyFlag`.  By encoding each flag into a numeric feature vector we can
treat the flag’s `confidence_bps` as a target value and fit an RBF surrogate
model (`RBFSurrogate`).  The surrogate’s prediction – a smooth interpolation of
observed confidences – becomes the *estimated reward* for a contextual
multi‑armed bandit.  The bandit therefore selects actions using an Upper‑Confidence
Bound (UCB) that combines the surrogate‑predicted reward with an exploration term.
Thus the linear system solution of the RBF surrogate (parent B) is directly used
to drive the bandit decision process that originally consumed epistemic
certainty (parent A)."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Sequence, Callable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – CertaintyFlag and epistemic machinery
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable representation of an epistemic certainty flag."""
    label: str                       # one of EPISTEMIC_FLAGS
    confidence_bps: int              # basis‑points, 0 … 10 000
    authority_class: str             # free‑form string identifying authority
    rationale: str                   # human readable justification
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }

# ----------------------------------------------------------------------
# Parent B – Radial‑Basis‑Function surrogate utilities
# ----------------------------------------------------------------------
Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve the dense linear system A·x = b using Gauss‑Jordan elimination.
    Mirrors the implementation from the original RBF surrogate parent.
    """
    n = len(b)
    # Augment matrix
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        # Pivot selection (largest absolute value)
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        # Swap rows
        m[col], m[pivot] = m[pivot], m[col]
        # Normalise pivot row
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        # Eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

@dataclass(frozen=True)
class RBFSurrogate:
    """A fitted RBF surrogate model."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Predict a scalar response for input vector x."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

def fit_rbf(points: Iterable[Vector],
            values: Iterable[float],
            epsilon: float = 1.0,
            ridge: float = 1e-9) -> RBFSurrogate:
    """
    Fit an RBF surrogate to (points, values) by solving the regularised linear system.
    Returns an immutable RBFSurrogate instance.
    """
    centers = [tuple(map(float, p)) for p in points]
    y = [float(v) for v in values]
    if not centers or len(centers) != len(y):
        raise ValueError("points and values must be non‑empty and of equal length")
    n = len(centers)
    # Build the kernel matrix with ridge regularisation on the diagonal
    k = [
        [
            gaussian(euclidean(a, b), epsilon) + (ridge if i == j else 0.0)
            for j, b in enumerate(centers)
        ]
        for i, a in enumerate(centers)
    ]
    weights = solve_linear(k, y)
    return RBFSurrogate(centers, weights, epsilon)

# ----------------------------------------------------------------------
# Hybrid layer – encoding, training, and bandit decision
# ----------------------------------------------------------------------
_label_to_onehot: Dict[str, np.ndarray] = {
    lbl: np.eye(len(EPISTEMIC_FLAGS))[i]
    for i, lbl in enumerate(EPISTEMIC_FLAGS)
}

def encode_flag(flag: CertaintyFlag) -> np.ndarray:
    """
    Convert a CertaintyFlag into a numeric feature vector.

    Vector layout:
        [ label_onehot (5) | authority_hash (1) | normalized_confidence (1) ]
    """
    # One‑hot encoding of the epistemic label
    label_vec = _label_to_onehot[flag.label]
    # Deterministic hash of authority_class folded into [0,1]
    authority_hash = (hash(flag.authority_class) % 2**32) / 2**32
    # Normalised confidence (basis‑points → [0,1])
    confidence_norm = flag.confidence_bps / 10_000.0
    return np.concatenate([label_vec, [authority_hash, confidence_norm]])

def train_certainty_surrogate(flags: List[CertaintyFlag],
                              epsilon: float = 1.0) -> RBFSurrogate:
    """
    Fit an RBF surrogate that maps the encoded flag features to the observed confidence.
    The surrogate learns the smooth landscape of epistemic certainty across the feature space.
    """
    X = [encode_flag(f) for f in flags]
    y = [f.confidence_bps / 10_000.0 for f in flags]  # target in [0,1]
    return fit_rbf(X, y, epsilon=epsilon)

def predict_flag_confidence(surrogate: RBFSurrogate,
                            flag: CertaintyFlag) -> float:
    """
    Use the trained surrogate to predict the confidence (as a probability) for a new flag.
    """
    vec = encode_flag(flag)
    return surrogate.predict(vec)

# ----------------------------------------------------------------------
# Contextual UCB bandit that consumes surrogate predictions
# ----------------------------------------------------------------------
def ucb_select(actions: List[str],
               action_contexts: List[List[CertaintyFlag]],
               surrogate: RBFSurrogate,
               plays: List[int],
               total_plays: int,
               alpha: float = 2.0) -> str:
    """
    Upper‑Confidence‑Bound selection.

    For each action we compute the mean surrogate‑predicted confidence over its
    associated flags.  The UCB value is:

        μ̂ + sqrt(α * log(total_plays) / (n_i + 1))

    where μ̂ is the empirical mean, n_i the number of times the action has been
    played, and α controls exploration aggressiveness.
    """
    if not (len(actions) == len(action_contexts) == len(plays)):
        raise ValueError("actions, contexts, and plays must have equal length")

    ucb_values = []
    for ctx_flags, n_i in zip(action_contexts, plays):
        if not ctx_flags:
            mean_est = 0.0
        else:
            preds = [predict_flag_confidence(surrogate, f) for f in ctx_flags]
            mean_est = float(np.mean(preds))
        exploration = math.sqrt(alpha * math.log(max(total_plays, 1) + 1) / (n_i + 1))
        ucb_values.append(mean_est + exploration)

    best_index = int(np.argmax(ucb_values))
    return actions[best_index]

# ----------------------------------------------------------------------
# Demonstration functions (three required)
# ----------------------------------------------------------------------
def demo_encoding() -> None:
    """Show encoding of a random CertaintyFlag."""
    flag = CertaintyFlag(
        label=random.choice(EPISTEMIC_FLAGS),
        confidence_bps=random.randint(0, 10_000),
        authority_class="auth_" + str(random.randint(1, 5)),
        rationale="demo",
    )
    vec = encode_flag(flag)
    print("Encoded vector:", vec)

def demo_surrogate() -> RBFSurrogate:
    """Fit a surrogate on a synthetic dataset and return it."""
    # Generate synthetic flags with a monotonic relationship between label index and confidence
    flags = []
    for i, lbl in enumerate(EPISTEMIC_FLAGS):
        for _ in range(4):
            flags.append(
                CertaintyFlag(
                    label=lbl,
                    confidence_bps=2000 * i + random.randint(-200, 200),
                    authority_class="auth_" + str(random.randint(1, 3)),
                    rationale="synthetic",
                )
            )
    surrogate = train_certainty_surrogate(flags, epsilon=0.8)
    # Quick sanity check
    test_flag = flags[0]
    pred = predict_flag_confidence(surrogate, test_flag)
    print(f"Predicted confidence for first flag ({test_flag.label}): {pred:.3f}")
    return surrogate

def demo_bandit(surrogate: RBFSurrogate) -> None:
    """Run a tiny contextual bandit loop using the surrogate predictions."""
    actions = ["action_A", "action_B", "action_C"]
    # Each action gets its own mini‑batch of flags
    action_contexts = [
        [
            CertaintyFlag(
                label="FACT",
                confidence_bps=8000,
                authority_class="auth_1",
                rationale="ctx_A",
            ),
            CertaintyFlag(
                label="PROBABLE",
                confidence_bps=6000,
                authority_class="auth_2",
                rationale="ctx_A",
            ),
        ],
        [
            CertaintyFlag(
                label="POSSIBLE",
                confidence_bps=4000,
                authority_class="auth_3",
                rationale="ctx_B",
            )
        ],
        [
            CertaintyFlag(
                label="BULLSHIT",
                confidence_bps=1000,
                authority_class="auth_1",
                rationale="ctx_C",
            ),
            CertaintyFlag(
                label="SURE_MAYBE",
                confidence_bps=3000,
                authority_class="auth_2",
                rationale="ctx_C",
            ),
        ],
    ]
    plays = [0, 0, 0]
    total_plays = 0
    for step in range(10):
        chosen = ucb_select(actions, action_contexts, surrogate, plays, total_plays)
        idx = actions.index(chosen)
        plays[idx] += 1
        total_plays += 1
        print(f"Step {step+1:02d}: chose {chosen} (plays={plays})")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: Encoding ===")
    demo_encoding()
    print("\n=== Demo: Surrogate fitting ===")
    surrogate = demo_surrogate()
    print("\n=== Demo: Bandit loop ===")
    demo_bandit(surrogate)