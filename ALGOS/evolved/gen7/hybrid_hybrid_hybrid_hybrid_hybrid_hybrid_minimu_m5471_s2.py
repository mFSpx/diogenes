# DARWIN HAMMER — match 5471, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1202_s1.py (gen6)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_jepa_energy_h_m1737_s0.py (gen4)
# born: 2026-05-30T00:02:17Z

"""Hybrid Algorithm integrating DARWIN HAMMER's TTT‑Linear + RBF surrogate (Parent A)
and Hybrid Minimum‑Cost Bandit Tree with DP‑sensitive JEPA energy (Parent B).

Mathematical bridge:
- The TTT weight matrix **W** linearly projects the load dimension into a latent
  feature space.
- The RBF surrogate uses a Gaussian kernel with width **ε**.  We bind ε to the
  differential‑privacy budget **dp_epsilon** of the bandit tree, scaled by the
  expected reward of the selected bandit action:
  
      ε = max(0.1, dp_epsilon * action.expected_reward)

  Thus the privacy‑sensitive bandit reward directly controls the smoothness of
  the surrogate, fusing the two parent topologies.
- The bandit’s expected reward also drives a gradient step on **W**, closing the
  feedback loop between the surrogate prediction and the bandit policy.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np

Vector = List[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel used by the RBF surrogate."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# ---------- Parent A components ------------------------------------------------

@dataclass(frozen=True)
class ResourceVector:
    load: float
    privacy: float

def init_ttt(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize the TTT linear weight matrix."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_step(W: np.ndarray, x: np.ndarray, eta: float, target: np.ndarray = None) -> np.ndarray:
    """One gradient descent step on the TTT loss."""
    if target is None:
        target = x
    grad = 2 * (W @ x - target) @ x.T
    return W - eta * grad

def transform_load(W: np.ndarray, load: float) -> np.ndarray:
    """Project the scalar load through the weight matrix."""
    return W @ np.array([[load]])  # shape (d_out, 1)

@dataclass
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return sum(w * gaussian(euclidean(x, list(c)), self.epsilon)
                   for w, c in zip(self.weights, self.centers))

    def gradient(self, x: Vector, target: float) -> List[float]:
        """Gradient of squared error w.r.t. each weight."""
        pred = self.predict(x)
        error = pred - target
        grads = []
        for w, c in zip(self.weights, self.centers):
            k = gaussian(euclidean(x, list(c)), self.epsilon)
            grads.append(2 * error * k)
        return grads

    def update_weights(self, grads: List[float], lr: float) -> None:
        """In‑place SGD update of the surrogate weights."""
        for i, g in enumerate(grads):
            self.weights[i] -= lr * g

# ---------- Parent B components ------------------------------------------------

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

@dataclass(frozen=True)
class Point:
    x: float
    y: float

class HybridBanditTree:
    """Simple bandit tree with differential‑privacy accounting."""
    def __init__(self):
        self._policy: Dict[str, List[float]] = {}  # action_id -> [total_reward, count]
        self.dp_epsilon: float = 0.1  # privacy budget

    def reset_policy(self) -> None:
        self._policy.clear()

    def _reward(self, action: str) -> float:
        total, n = self._policy.get(action, [0.0, 0.0])
        return total / n if n else 0.0

    def update_policy(self, updates: List[BanditUpdate]) -> None:
        for u in updates:
            stats = self._policy.setdefault(u.action_id, [0.0, 0.0])
            stats[0] += float(u.reward)
            stats[1] += 1.0

    def expected_rewards(self) -> List[BanditAction]:
        """Return a list of actions with their current expected rewards."""
        actions = []
        for aid, (total, cnt) in self._policy.items():
            exp_reward = total / cnt if cnt else 0.0
            actions.append(BanditAction(
                action_id=aid,
                propensity=cnt / max(1, sum(v[1] for v in self._policy.values())),
                expected_reward=exp_reward,
                confidence_bound=math.sqrt(2 * math.log(1.0 / 0.05) / max(1, cnt)),
                algorithm="HybridBanditTree"
            ))
        return actions

    def choose_action(self) -> BanditAction:
        """Select the action with the highest upper confidence bound."""
        actions = self.expected_rewards()
        if not actions:
            # default dummy action
            return BanditAction("default", 1.0, 0.0, 0.0, "HybridBanditTree")
        # Upper Confidence Bound (UCB) = expected_reward + confidence_bound
        return max(actions, key=lambda a: a.expected_reward + a.confidence_bound)

# ---------- Fusion core ---------------------------------------------------------

@dataclass
class HybridModel:
    W: np.ndarray
    surrogate: RBFSurrogate
    bandit: HybridBanditTree
    eta: float = 0.01          # learning rate for W
    lr_surrogate: float = 0.05 # learning rate for surrogate weights

def hybrid_predict(resource: ResourceVector, model: HybridModel) -> float:
    """Full forward pass: project load, predict with surrogate, update privacy."""
    latent = transform_load(model.W, resource.load).flatten()
    # Bridge: adapt surrogate epsilon using current DP budget & last chosen action reward
    # (if no action yet, use baseline epsilon)
    action = model.bandit.choose_action()
    model.surrogate.epsilon = max(0.1, model.bandit.dp_epsilon * max(0.01, action.expected_reward))
    pred = model.surrogate.predict(latent.tolist())
    # privacy update (example multiplicative rule)
    resource_privacy = resource.privacy * pred
    # (the function returns the prediction; privacy can be accessed via the object)
    return pred, resource_privacy, action

def hybrid_update(resource: ResourceVector,
                  prediction: float,
                  reward: float,
                  action: BanditAction,
                  model: HybridModel) -> None:
    """Back‑propagation style update linking bandit reward to both W and surrogate."""
    # 1) Bandit policy update
    upd = BanditUpdate(context_id="ctx", action_id=action.action_id,
                       reward=reward, propensity=action.propensity)
    model.bandit.update_policy([upd])

    # 2) Gradient step on W using the reward as a target for the latent projection
    x = np.array([[resource.load]])
    target = np.array([[reward]])  # treat reward as desired transformed load
    model.W = ttt_step(model.W, x, model.eta, target)

    # 3) Update surrogate weights using the prediction error
    latent = transform_load(model.W, resource.load).flatten().tolist()
    grads = model.surrogate.gradient(latent, reward)
    model.surrogate.update_weights(grads, model.lr_surrogate)

def hybrid_step(resource: ResourceVector, model: HybridModel) -> Tuple[float, float]:
    """
    Executes one hybrid iteration:
    - forward pass (prediction, privacy, action)
    - defines a reward (here: prediction * privacy as a toy objective)
    - performs the joint update
    Returns the new privacy value and prediction.
    """
    prediction, new_privacy, action = hybrid_predict(resource, model)
    # Define a simple reward signal; in practice this would be domain‑specific
    reward = prediction * new_privacy
    hybrid_update(resource, prediction, reward, action, model)
    return prediction, new_privacy

# ---------- Smoke test ----------------------------------------------------------

if __name__ == "__main__":
    # Initialise components
    dim_in = 1
    dim_out = 4
    W = init_ttt(dim_in, dim_out, scale=0.05, seed=42)

    # Create dummy RBF surrogate with random centers and weights
    rng = np.random.default_rng(1)
    centers = [tuple(rng.uniform(-1, 1, dim_out)) for _ in range(6)]
    weights = list(rng.uniform(-0.5, 0.5, 6))
    surrogate = RBFSurrogate(centers=centers, weights=weights, epsilon=1.0)

    bandit = HybridBanditTree()
    # Seed bandit with two actions
    bandit._policy["action_A"] = [0.0, 0.0]
    bandit._policy["action_B"] = [0.0, 0.0]

    model = HybridModel(W=W, surrogate=surrogate, bandit=bandit)

    # Example resource vector
    resource = ResourceVector(load=0.73, privacy=1.0)

    # Run a few hybrid steps
    for step in range(5):
        pred, priv = hybrid_step(resource, model)
        print(f"Step {step+1}: prediction={pred:.4f}, privacy={priv:.4f}")

    # Verify that epsilon has been adapted
    print(f"Final surrogate epsilon: {model.surrogate.epsilon:.4f}")
    # Verify bandit policy state
    print("Bandit policy:", bandit._policy)