# DARWIN HAMMER — match 3847, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hybrid_pherom_m411_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m1092_s0.py (gen5)
# born: 2026-05-29T23:52:06Z

"""Hybrid RBF‑Caputo‑Bandit Algorithm

Parents:
- hybrid_hybrid_rbf_surrogate_hybrid_hybrid_pherom_m411_s2.py (RBF surrogate with entropy‑regularised loss and pheromone‑based privacy scores)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m1092_s0.py (Caputo fractional derivative applied to a TT‑Hybrid weight matrix together with a contextual bandit framework)

Mathematical bridge:
The surrogate model supplies a probabilistic estimate of the expected reward for a given context.  
Its weight vector **w** is regularised by the Shannon entropy  H(w)=‑∑p_i log p_i with p_i=|w_i|/∑|w|.  
The Caputo fractional derivative of order α∈(0,1) is used to model a *memory‑decay* of the TT‑Hybrid weight matrix **W**:
    
    D^α W(t) ≈ (1/Γ(1‑α)) ∑_{k=0}^{n} (‑1)^k C(α,k) W(t‑k·Δt) / Δt^α

where C(α,k)=Γ(k‑α)/[Γ(‑α)Γ(k+1)].

By feeding the surrogate’s entropy‑regularised prediction into the contextual bandit,
the resulting reward signal updates **W**.  The fractional derivative then decays **W**
in a way that respects the history of updates, completing a closed‑loop hybrid system.

The module below implements this fused dynamics with three public functions:
    * train_surrogate – builds an entropy‑regularised RBF surrogate.
    * fractional_decay_step – applies a Caputo fractional decay to a weight matrix.
    * hybrid_step – runs one iteration of prediction, bandit selection and weight update.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np

Vector = Sequence[float]


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Gauss‑Jordan elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]


def shannon_entropy(weights: List[float]) -> float:
    """Entropy of absolute‑value normalised weights."""
    abs_w = np.abs(weights)
    total = np.sum(abs_w)
    if total == 0.0:
        return 0.0
    probs = abs_w / total
    # avoid log(0) by masking zero entries
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log(probs[mask])))


@dataclass(frozen=True)
class RBFSurrogate:
    """Entropy‑regularised radial‑basis surrogate."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def _phi(self, x: Vector) -> List[float]:
        return [gaussian(euclidean(x, c), self.epsilon) for c in self.centers]

    def predict(self, x: Vector) -> float:
        phi = self._phi(x)
        return float(np.dot(self.weights, phi))


def train_surrogate(
    centers: List[Tuple[float, ...]],
    targets: List[float],
    epsilon: float = 1.0,
    reg_factor: float = 1e-4,
) -> RBFSurrogate:
    """
    Build an RBF surrogate whose least‑squares loss is augmented by
    λ·H(w) where H is the Shannon entropy of the weight vector.
    λ is chosen proportional to the entropy itself (self‑regularising).
    """
    n = len(centers)
    # Build the interpolation matrix Φ
    Phi = np.zeros((n, n))
    for i, ci in enumerate(centers):
        for j, cj in enumerate(centers):
            Phi[i, j] = gaussian(euclidean(ci, cj), epsilon)

    # Solve (Φ + λI) w = y  iteratively because λ depends on w (entropy)
    w = np.linalg.lstsq(Phi + reg_factor * np.eye(n), targets, rcond=None)[0]
    for _ in range(5):  # simple fixed‑point iteration
        lam = reg_factor * shannon_entropy(w.tolist())
        A = Phi + lam * np.eye(n)
        w = np.linalg.solve(A, targets)
    return RBFSurrogate(centers=centers, weights=w.tolist(), epsilon=epsilon)


@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a candidate action."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRBFCaputo"


@dataclass(frozen=True)
class BanditUpdate:
    """Result of pulling an arm."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Simple leaky‑integrator storing a scalar level."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self._last_delta = delta
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """A bounded transformation of the last delta."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


def caputo_coefficients(alpha: float, n: int) -> np.ndarray:
    """Pre‑compute coefficients C(α,k) for k=0..n used in the Grunwald‑Letnikov scheme."""
    coeffs = np.empty(n + 1)
    for k in range(n + 1):
        coeffs[k] = ((-1) ** k) * math.gamma(k - alpha) / (math.gamma(-alpha) * math.gamma(k + 1))
    return coeffs


def fractional_decay_step(
    history: List[np.ndarray],
    dt: float,
    alpha: float,
) -> np.ndarray:
    """
    Apply a Caputo fractional derivative of order α to the latest weight matrix.
    `history` holds past weight matrices ordered from oldest to newest.
    Returns the decayed matrix that will replace the newest entry.
    """
    n = len(history) - 1
    if n < 0:
        raise ValueError("history must contain at least one matrix")
    coeffs = caputo_coefficients(alpha, n)
    factor = 1.0 / math.gamma(1.0 - alpha) / (dt ** alpha)
    # Linear combination of past matrices
    decayed = sum(coeffs[k] * history[n - k] for k in range(n + 1))
    return factor * decayed


class TTWeightMatrix:
    """TT‑Hybrid style weight matrix with fractional decay."""

    def __init__(self, shape: Tuple[int, int], alpha: float = 0.5, dt: float = 1.0):
        self.shape = shape
        self.W = np.zeros(shape)
        self.alpha = alpha
        self.dt = dt
        self._history: List[np.ndarray] = [self.W.copy()]

    def apply_update(self, grad: np.ndarray, lr: float = 0.1) -> None:
        """Gradient step followed by history bookkeeping."""
        self.W += lr * grad
        self._history.append(self.W.copy())
        # keep a reasonable history length (e.g., 20 steps)
        if len(self._history) > 20:
            self._history.pop(0)

    def fractional_decay(self) -> None:
        """Replace current matrix with its Caputo‑decayed version."""
        decayed = fractional_decay_step(self._history, self.dt, self.alpha)
        self.W = decayed
        self._history[-1] = self.W.copy()


def select_action(
    actions: List[BanditAction],
    surrogate_estimate: float,
) -> BanditAction:
    """
    Upper‑confidence bound (UCB) selection where the surrogate estimate
    nudges the expected_reward term.
    """
    best = None
    best_score = -math.inf
    for a in actions:
        # blend surrogate estimate into the reward term
        score = a.expected_reward + a.confidence_bound + 0.5 * surrogate_estimate
        if score > best_score:
            best_score = score
            best = a
    return best  # type: ignore


def hybrid_step(
    context: Vector,
    surrogate: RBFSurrogate,
    weight_matrix: TTWeightMatrix,
    store: StoreState,
    actions: List[BanditAction],
) -> BanditUpdate:
    """
    One iteration of the hybrid loop:
        1. Surrogate predicts a reward estimate for the context.
        2. An action is selected via a UCB rule that incorporates the estimate.
        3. The selected action yields a stochastic reward (simulated here).
        4. The reward updates the StoreState and the TT weight matrix.
        5. The weight matrix undergoes fractional decay.
    Returns the BanditUpdate describing what happened.
    """
    # 1. surrogate prediction
    est = surrogate.predict(context)

    # 2. action selection
    chosen = select_action(actions, est)

    # 3. simulated stochastic reward (Gaussian around expected_reward)
    reward = random.gauss(chosen.expected_reward, 0.1)

    # 4a. update StoreState (treat reward as inflow, zero outflow)
    store.update([reward], [])

    # 4b. compute gradient for weight matrix:
    #    simple outer product of context and a one‑hot of the action id hash
    action_vec = np.zeros(weight_matrix.shape[1])
    idx = hash(chosen.action_id) % weight_matrix.shape[1]
    action_vec[idx] = 1.0
    grad = np.outer(np.array(context), action_vec) * (reward - est)

    weight_matrix.apply_update(grad)

    # 5. fractional decay
    weight_matrix.fractional_decay()

    return BanditUpdate(
        context_id=str(hash(tuple(context))),
        action_id=chosen.action_id,
        reward=reward,
        propensity=chosen.propensity,
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Build a tiny surrogate
    centers = [(0.0, 0.0), (1.0, 1.0), (0.5, 0.2)]
    targets = [0.1, 0.9, 0.4]
    surrogate = train_surrogate(centers, targets, epsilon=2.0)

    # 2. Initialise TT weight matrix and store
    tt = TTWeightMatrix(shape=(2, 5), alpha=0.4, dt=1.0)
    store = StoreState()

    # 3. Define a small action set
    actions = [
        BanditAction(action_id="A", propensity=0.3, expected_reward=0.2, confidence_bound=0.05),
        BanditAction(action_id="B", propensity=0.5, expected_reward=0.5, confidence_bound=0.07),
        BanditAction(action_id="C", propensity=0.2, expected_reward=0.8, confidence_bound=0.03),
    ]

    # 4. Run a few hybrid steps
    for step in range(5):
        ctx = (random.random(), random.random())
        upd = hybrid_step(ctx, surrogate, tt, store, actions)
        print(f"Step {step+1}: ctx={ctx}, action={upd.action_id}, reward={upd.reward:.3f}, level={store.level:.3f}")

    print("Final weight matrix:\n", tt.W)
    print("Surrogate weights:", surrogate.weights)