# DARWIN HAMMER — match 5104, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hybrid_semant_m1604_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_doomsd_m988_s1.py (gen4)
# born: 2026-05-29T23:59:48Z

"""HybridRBF_Bandit
Integrates:

- Parent A: radial‑basis surrogate model with Gaussian kernels, Euclidean distances,
  and a linear system solver for coefficient estimation.
- Parent B: contextual multi‑armed bandit (LinUCB / epsilon‑greedy) that maintains a
  policy of actions (here, RBF centre indices) and updates it from observed rewards.

Mathematical bridge:
The surrogate model requires a set of RBF centres C = {c_i}.  Selecting which centre
to adapt at each learning step can be framed as a bandit problem: each centre i is an
action a_i.  The context vector is the distance‑based feature representation of the
current sample x, and the reward is defined as the reduction of the surrogate error
after updating the coefficient associated with the chosen centre.  Thus the bandit
policy directly drives the construction of the surrogate matrix A and right‑hand side
b used in the linear system A·w = b (Parent A), while the policy update follows the
standard bandit update rules (Parent B).  The resulting hybrid algorithm jointly
optimises centre selection and coefficient estimation in a single unified loop.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Bandit data structures (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: int          # index of the RBF centre
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: int
    reward: float
    propensity: float

# Global mutable policy: action_id -> [cumulative_reward, count]
_POLICY: dict[int, list[float]] = {}

def reset_policy() -> None:
    """Clear the global policy."""
    _POLICY.clear()

def update_policy(updates: list[BanditUpdate]) -> None:
    """Accumulate rewards for each action."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def _reward(action_id: int) -> float:
    """Mean reward observed for an action, or 0 if never taken."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0

def select_action(context: dict[str, float],
                  actions: list[int],
                  algorithm: str = 'linucb',
                  epsilon: float = 0.1,
                  seed: int | str | None = 7) -> BanditAction:
    """Choose an RBF centre index using a contextual bandit rule."""
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)

    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        # Simple Beta‑Bernoulli Thompson sampling on mean reward
        chosen = max(
            actions,
            key=lambda a: rng.betavariate(1 + max(0, _reward(a)), 1 + max(0, 1 - _reward(a)))
        )
    else:  # LinUCB‑like heuristic
        scale = math.sqrt(sum(v * v for v in context.values())) if context else 1.0
        chosen = max(
            actions,
            key=lambda a: _reward(a) + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1])
        )
    propensity = 1.0 / len(actions)
    exp_reward = _reward(chosen)
    conf = 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1])
    return BanditAction(chosen, propensity, exp_reward, conf, algorithm)

# ----------------------------------------------------------------------
# Radial‑basis surrogate utilities (from Parent A)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))

def rbf_feature_vector(x: np.ndarray,
                       centres: np.ndarray,
                       epsilon: float = 1.0) -> np.ndarray:
    """Compute the RBF feature vector φ(x) for all centres."""
    dists = np.linalg.norm(centres - x, axis=1)
    return np.vectorize(lambda r: gaussian(r, epsilon))(dists)

def solve_linear(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A·w = b via Gaussian elimination (no external libs)."""
    n = A.shape[0]
    M = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for col in range(n):
        # pivot
        pivot = max(range(col, n), key=lambda r: abs(M[r, col]))
        if abs(M[pivot, col]) < 1e-12:
            raise ValueError("singular surrogate system")
        if pivot != col:
            M[[col, pivot]] = M[[pivot, col]]
        # normalize pivot row
        M[col] = M[col] / M[col, col]
        # eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = M[row, col]
            M[row] -= factor * M[col]

    return M[:, -1]

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_predict(x: np.ndarray,
                   centres: np.ndarray,
                   coeffs: np.ndarray,
                   epsilon: float = 1.0) -> float:
    """RBF surrogate prediction for a single input vector."""
    phi = rbf_feature_vector(x, centres, epsilon)
    return float(np.dot(phi, coeffs))

def hybrid_update_policy_and_coeffs(
        X: np.ndarray,
        y: np.ndarray,
        centres: np.ndarray,
        coeffs: np.ndarray,
        epsilon: float = 1.0,
        algorithm: str = 'linucb',
        epsilon_greedy: float = 0.1,
        seed: int | str | None = 7) -> tuple[np.ndarray, list[BanditUpdate]]:
    """
    One learning iteration:
      1. Sample a training point (x, target).
      2. Build a context from distances to centres.
      3. Use the bandit to pick a centre to adjust.
      4. Re‑solve the linear system with the selected centre's column
         emphasized (by adding a small regularisation term).
      5. Compute reward as reduction in squared error.
      6. Return new coefficients and the generated BanditUpdate objects.
    """
    idx = random.randrange(len(X))
    x, target = X[idx], y[idx]

    # 2. Context: distances to all centres (rounded for hashability)
    dists = np.linalg.norm(centres - x, axis=1)
    context = {f"d{i}": float(d) for i, d in enumerate(dists[:10])}  # up to 10 dims

    # 3. Bandit action selection
    actions = list(range(len(centres)))
    action = select_action(context, actions, algorithm, epsilon_greedy, seed)

    # 4. Assemble surrogate matrix A and RHS b
    Phi = np.vstack([rbf_feature_vector(row, centres, epsilon) for row in X])
    A = Phi.T @ Phi
    b = Phi.T @ y

    # Emphasise the chosen centre by adding a tiny ridge term to its diagonal entry
    ridge = 1e-6
    A[action.action_id, action.action_id] += ridge

    # Solve for new coefficients
    new_coeffs = solve_linear(A, b)

    # 5. Reward = previous error - new error (positive if improvement)
    pred_before = hybrid_predict(x, centres, coeffs, epsilon)
    pred_after = hybrid_predict(x, centres, new_coeffs, epsilon)
    error_before = (pred_before - target) ** 2
    error_after = (pred_after - target) ** 2
    reward = float(error_before - error_after)  # improvement >0

    # 6. Record update for policy
    update = BanditUpdate(
        context_id=str(idx),
        action_id=action.action_id,
        reward=reward,
        propensity=action.propensity
    )
    return new_coeffs, [update]

def hybrid_train(X: np.ndarray,
                 y: np.ndarray,
                 n_centres: int = 20,
                 n_iters: int = 200,
                 epsilon: float = 1.0,
                 algorithm: str = 'linucb',
                 epsilon_greedy: float = 0.1,
                 seed: int | str | None = 7) -> tuple[np.ndarray, np.ndarray]:
    """
    Train the hybrid RBF‑bandit surrogate.
    Returns the final centre matrix and coefficient vector.
    """
    rng = random.Random(seed)
    # Initialise centres by random subset of training data
    centre_indices = rng.sample(range(len(X)), k=n_centres)
    centres = X[centre_indices].copy()

    # Initialise coefficients with ordinary least‑squares solution
    Phi = np.vstack([rbf_feature_vector(row, centres, epsilon) for row in X])
    A = Phi.T @ Phi
    b = Phi.T @ y
    coeffs = solve_linear(A, b)

    reset_policy()

    for _ in range(n_iters):
        coeffs, updates = hybrid_update_policy_and_coeffs(
            X, y, centres, coeffs,
            epsilon=epsilon,
            algorithm=algorithm,
            epsilon_greedy=epsilon_greedy,
            seed=seed
        )
        update_policy(updates)

    return centres, coeffs

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic regression: f(x) = sin(π * x) on [0,1]^2
    rng = np.random.default_rng(42)
    N = 300
    X = rng.random((N, 2))
    y = np.sin(np.pi * X[:, 0]) + np.cos(np.pi * X[:, 1])

    centres, coeffs = hybrid_train(
        X, y,
        n_centres=25,
        n_iters=150,
        epsilon=2.0,
        algorithm='linucb',
        epsilon_greedy=0.05,
        seed=123
    )

    # Evaluate on a held‑out set
    X_test = rng.random((50, 2))
    y_test = np.sin(np.pi * X_test[:, 0]) + np.cos(np.pi * X_test[:, 1])
    preds = np.array([hybrid_predict(x, centres, coeffs, epsilon=2.0) for x in X_test])
    mse = float(np.mean((preds - y_test) ** 2))
    print(f"Hybrid surrogate MSE on test set: {mse:.6f}")
    sys.exit(0)