# DARWIN HAMMER — match 5462, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_vorono_m2221_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1549_s0.py (gen6)
# born: 2026-05-30T00:02:01Z

"""Hybrid Surrogate‑Pheromone Fusion
Parent A: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_vorono_m2221_s0.py
Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1549_s0.py

Mathematical bridge
-------------------
* Parent A builds a Radial‑Basis Function (RBF) surrogate 𝑓̂(x)=∑ₖ wₖ·φ(‖x‑cₖ‖) that gives a smooth estimate of a scalar quantity
  (e.g. expected reward) at any point x in a metric space.
* Parent B uses an *entropic* MinHash signature 𝜒(p)=∑ᵢ log pᵢ derived from a probability distribution p and feeds the resulting
  log‑count ratio into a pheromone decay law ρ(t)=v₀·e^{‑t/τ}.

The fusion treats the surrogate output as a raw “reward” vector r for a set of candidate actions.
After a soft‑max transform it becomes a probability distribution p, which supplies the MinHash
signature.  The signature modulates the pheromone decay factor, producing a pheromone weight
πᵢ for each action.  The final hybrid score combines the surrogate estimate and the pheromone
weight:

    scoreᵢ = f̂(xᵢ) · (1 + πᵢ)

Thus the RBF surrogate supplies the quantitative backbone, while the entropic‑pheromone
machinery injects adaptive bias based on the global shape of the reward distribution.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

import numpy as np

Vector = np.ndarray

# ----------------------------------------------------------------------
# Parent A – RBF surrogate utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function φ(r)=exp(-(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)

def solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a·x = b with a simple Gauss‑Jordan elimination (no external libs)."""
    n = len(b)
    m = np.hstack((a.astype(float), b[:, None].astype(float)))
    for col in range(n):
        pivot = np.argmax(np.abs(m[col:, col])) + col
        if np.abs(m[pivot, col]) < 1e-12:
            raise ValueError("singular surrogate system")
        if pivot != col:
            m[[pivot, col]] = m[[col, pivot]]
        m[col] = m[col] / m[col, col]
        for row in range(n):
            if row == col:
                continue
            m[row] -= m[row, col] * m[col]
    return m[:, -1]

@dataclass(frozen=True)
class RBFSurrogate:
    """RBF surrogate model."""
    centers: np.ndarray          # shape (N, d)
    weights: np.ndarray          # shape (N,)
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Predict scalar value at point x."""
        phi_vals = [gaussian(euclidean(x, c), self.epsilon) for c in self.centers]
        return float(np.dot(self.weights, phi_vals))

def fit_rbf_surrogate(points: np.ndarray,
                     values: np.ndarray,
                     epsilon: float = 1.0,
                     ridge: float = 1e-9) -> RBFSurrogate:
    """Fit an RBF surrogate to (points, values)."""
    if points.shape[0] != values.shape[0]:
        raise ValueError("points and values must have the same length")
    N = points.shape[0]
    K = np.empty((N, N), dtype=float)
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            K[i, j] = gaussian(euclidean(a, b), epsilon)
            if i == j:
                K[i, j] += ridge
    w = solve_linear(K, values)
    return RBFSurrogate(centers=points, weights=w, epsilon=epsilon)

# ----------------------------------------------------------------------
# Parent B – Entropic MinHash and pheromone decay
# ----------------------------------------------------------------------
def entropic_minhash(prob_dist: np.ndarray) -> float:
    """
    Entropic MinHash signature χ(p)=∑_i log(p_i).
    The input must be a proper probability distribution (sum to 1, all >0).
    """
    if np.any(prob_dist <= 0):
        raise ValueError("probability distribution must be strictly positive")
    return float(np.sum(np.log(prob_dist)))

def pheromone_decay(t: float, tau: float, v0: float) -> float:
    """Exponential pheromone decay ρ(t)=v₀·exp(-t/τ)."""
    return v0 * math.exp(-t / tau)

# ----------------------------------------------------------------------
# Hybrid structures (borrowed from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

    def as_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["morphology"] = self.morphology.__dict__
        return d

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()

def compute_hybrid_pheromone_weights(
    surrogate: RBFSurrogate,
    action_features: np.ndarray,
    t: float,
    tau: float,
    v0: float
) -> np.ndarray:
    """
    For each action feature vector x_i:
        1. Obtain surrogate prediction r_i = f̂(x_i).
        2. Convert the vector of all r_i to a probability distribution via softmax.
        3. Compute entropic MinHash χ(p).
        4. Apply pheromone decay ρ(t) and combine: π_i = ρ(t) * χ(p).
    The returned array π has the same length as action_features.
    """
    # 1. Raw surrogate predictions
    raw = np.array([surrogate.predict(x) for x in action_features])
    # 2. Probability distribution
    prob = softmax(raw)
    # 3. Entropic MinHash (global signature)
    chi = entropic_minhash(prob)
    # 4. Decayed pheromone factor (scalar)
    decay = pheromone_decay(t, tau, v0)
    # 5. Uniform pheromone weight per action (the same χ·ρ for all)
    pheromone_weights = np.full_like(raw, decay * chi, dtype=float)
    return pheromone_weights

def hybrid_score(
    surrogate: RBFSurrogate,
    action_features: np.ndarray,
    t: float,
    tau: float,
    v0: float
) -> np.ndarray:
    """
    Compute the final hybrid score for each action:
        score_i = f̂(x_i) * (1 + π_i)
    where π_i is the pheromone weight from compute_hybrid_pheromone_weights.
    """
    predictions = np.array([surrogate.predict(x) for x in action_features])
    pheromone = compute_hybrid_pheromone_weights(surrogate, action_features, t, tau, v0)
    return predictions * (1.0 + pheromone)

def select_best_action(
    surrogate: RBFSurrogate,
    actions: List[BanditAction],
    action_features: np.ndarray,
    t: float,
    tau: float,
    v0: float
) -> BanditAction:
    """
    Rank actions using the hybrid score and return the action with the highest score.
    The BanditAction objects are assumed to be in the same order as action_features.
    """
    if len(actions) != action_features.shape[0]:
        raise ValueError("actions list must align with feature matrix")
    scores = hybrid_score(surrogate, action_features, t, tau, v0)
    best_idx = int(np.argmax(scores))
    return actions[best_idx]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Generate synthetic training data (10 points in 3‑D)
    train_pts = np.random.rand(10, 3)
    train_vals = np.sin(train_pts[:, 0] * 2 * math.pi) + np.cos(train_pts[:, 1] * 2 * math.pi)

    # Fit surrogate
    surrogate = fit_rbf_surrogate(train_pts, train_vals, epsilon=2.0)

    # Create 5 candidate actions with random features
    n_actions = 5
    action_feats = np.random.rand(n_actions, 3)

    # Build dummy BanditAction objects
    actions = []
    for i in range(n_actions):
        actions.append(
            BanditAction(
                action_id=f"a{i}",
                propensity=random.random(),
                expected_reward=0.0,      # placeholder, will be overridden by hybrid score
                confidence_bound=random.random(),
                algorithm="HybridSurrogatePheromone"
            )
        )

    # Hybrid parameters
    t_now = 1.5
    tau = 10.0
    v0 = 0.8

    # Compute scores and pick best
    best = select_best_action(surrogate, actions, action_feats, t_now, tau, v0)

    # Print a concise report
    print(f"Best action: {best.action_id}")
    print("Hybrid scores for all actions:")
    scores = hybrid_score(surrogate, action_feats, t_now, tau, v0)
    for act, sc in zip(actions, scores):
        print(f"  {act.action_id}: {sc:.6f}")