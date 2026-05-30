# DARWIN HAMMER — match 1410, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_hybrid_m524_s5.py (gen5)
# parent_b: hybrid_path_signature_kan_m30_s2.py (gen1)
# born: 2026-05-29T23:36:07Z

"""Hybrid Krampus‑RBF Bandit with Path‑Signature Context Embedding

This module fuses two parent algorithms:

* **Parent A** – a kernelised contextual bandit that uses a Gaussian RBF kernel
  on abstract “vibe” vectors to compute reward estimates and Upper‑Confidence‑Bounds.
* **Parent B** – a path‑signature toolkit that turns a multivariate time‑series
  into a collection of geometric descriptors (lead‑lag transform, level‑1 and
  level‑2 signatures, optional B‑spline basis).

**Mathematical bridge**

A context in the bandit is represented by a time‑series `path ∈ ℝ^{T×d}`.
Parent B maps this path to a feature vector  

\[
\phi(\text{path}) = \big[\,\text{lead‑lag}(\text{path}),\;
                      \text{sig}_1(\text{path}),\;
                      \operatorname{vec}(\text{sig}_2(\text{path}))\,\big]
\]

which lives in a Hilbert space ℋ.  Parent A’s Gaussian kernel  

\[
k(x,x') = \exp\!\big(-\varepsilon^{2}\,\|x-x'\|^{2}\big)
\]

is then applied to these signature‑based embeddings, yielding a
kernel‑weighted contextual bandit.  The hybrid therefore exploits
sequential geometry (via signatures) while retaining the proven
exploration‑exploitation mechanism of the Krampus‑RBF router.

The implementation provides:
* `lead_lag_transform`, `signature_level1`, `signature_level2` – feature extraction.
* `gaussian_kernel` – RBF similarity.
* `HybridRouter` – kernelised bandit that stores contexts, actions and rewards.
* Helper functions that showcase the hybrid operation.

A minimal smoke test at the bottom runs a few update‑select cycles on synthetic data."""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridKrampusRBF_Signature"


@dataclass(frozen=True)
class BanditUpdate:
    """Record of an observed (context, action, reward)."""
    context_id: str
    action_id: str
    reward: float
    propensity: float
    feature_vec: np.ndarray


# --------------------------------------------------------------------------- #
# Parent B – Path‑signature utilities
# --------------------------------------------------------------------------- #

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Lead‑lag transform (Chevyrev‑Kormilitzin convention).

    Args:
        path: (T, d) array.

    Returns:
        (2T‑1, 2d) array where even rows contain (X_t, X_t) and odd rows
        contain (X_{t+1}, X_t).
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path: np.ndarray) -> np.ndarray:
    """Level‑1 signature: total increment."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path: np.ndarray) -> np.ndarray:
    """Level‑2 signature (iterated integral tensor)."""
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running = path[:-1] - path[0]               # (T‑1, d)
    # Tensor contraction via matrix multiplication
    return running.T @ increments               # (d, d)


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Cubic B‑spline basis evaluation (Cox‑de Boor)."""
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)
    G = len(grid)
    # Build clamped knot vector
    knots = np.concatenate((
        np.full(k, grid[0]),
        grid,
        np.full(k, grid[-1])
    ))
    N = len(x)
    # Initialise zero‑order basis
    B = np.zeros((N, G + k))
    for i in range(G + k):
        B[:, i] = (knots[i] <= x) & (x < knots[i + 1])
    # Recursion
    for deg in range(1, k + 1):
        for i in range(G + k - deg):
            left = (x - knots[i]) / (knots[i + deg] - knots[i] + 1e-12)
            right = (knots[i + deg + 1] - x) / (knots[i + deg + 1] - knots[i + 1] + 1e-12)
            B[:, i] = left * B[:, i] + right * B[:, i + 1]
    # Return only the first G columns (the usable basis functions)
    return B[:, :G]


# --------------------------------------------------------------------------- #
# Parent A – Kernelised bandit utilities
# --------------------------------------------------------------------------- #

def gaussian_kernel(x: np.ndarray, y: np.ndarray, eps: float = 1.0) -> float:
    """RBF kernel k(x, y) = exp(-eps^2 * ||x-y||^2)."""
    diff = x - y
    return math.exp(- (eps ** 2) * np.dot(diff, diff))


# --------------------------------------------------------------------------- #
# Hybrid feature construction
# --------------------------------------------------------------------------- #

def compute_context_features(path: np.ndarray,
                             eps: float = 1.0,
                             include_bspline: bool = False) -> np.ndarray:
    """Build the hybrid feature vector from a raw time‑series.

    The vector consists of:
        - flattened lead‑lag transform,
        - level‑1 signature,
        - flattened level‑2 signature,
        - optional B‑spline coefficients (evaluated on a fixed grid).

    All components are concatenated and finally L2‑normalised so that the
    Gaussian kernel operates on a unit sphere, improving numerical stability.
    """
    # Lead‑lag (flattened)
    ll = lead_lag_transform(path).ravel()

    # Signatures
    sig1 = signature_level1(path)
    sig2 = signature_level2(path).ravel()

    parts = [ll, sig1, sig2]

    if include_bspline:
        # Simple uniform grid on each dimension
        d = path.shape[1]
        grid = np.linspace(np.min(path), np.max(path), num=5)
        # Evaluate basis for each dimension independently and flatten
        bs = np.concatenate([bspline_basis(path[:, dim], grid) for dim in range(d)], axis=1)
        parts.append(bs.ravel())

    feature = np.concatenate(parts).astype(float)

    # L2 normalisation (avoid division by zero)
    norm = np.linalg.norm(feature) + 1e-12
    return feature / norm


# --------------------------------------------------------------------------- #
# Hybrid router (kernelised contextual bandit)
# --------------------------------------------------------------------------- #

class HybridRouter:
    """Kernelised contextual bandit that uses path‑signature embeddings."""

    def __init__(self,
                 actions: List[str],
                 eps: float = 1.0,
                 beta: float = 1.0,
                 include_bspline: bool = False):
        """
        Args:
            actions: list of admissible action identifiers.
            eps: kernel length‑scale.
            beta: exploration coefficient for UCB.
            include_bspline: whether to augment features with B‑spline basis.
        """
        self.actions = actions
        self.eps = eps
        self.beta = beta
        self.include_bspline = include_bspline

        # Storage for observed triples (feature, action, reward)
        self._features: List[np.ndarray] = []
        self._action_ids: List[str] = []
        self._rewards: List[float] = []

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #

    def update_policy(self,
                      context_id: str,
                      path: np.ndarray,
                      action_id: str,
                      reward: float,
                      propensity: float = 1.0) -> BanditUpdate:
        """Incorporate a new observation into the policy."""
        if action_id not in self.actions:
            raise ValueError(f"Unknown action {action_id}")

        feat = compute_context_features(path,
                                        eps=self.eps,
                                        include_bspline=self.include_bspline)

        self._features.append(feat)
        self._action_ids.append(action_id)
        self._rewards.append(reward)

        return BanditUpdate(context_id, action_id, reward, propensity, feat)

    def select_action(self,
                      context_id: str,
                      path: np.ndarray) -> BanditAction:
        """Select an action for the supplied context using kernel‑UCB."""
        x = compute_context_features(path,
                                     eps=self.eps,
                                     include_bspline=self.include_bspline)

        # Compute kernel sums per action
        action_to_num = {a: 0.0 for a in self.actions}
        action_to_den = {a: 0.0 for a in self.actions}
        action_to_weighted_reward = {a: 0.0 for a in self.actions}

        for feat, a, r in zip(self._features, self._action_ids, self._rewards):
            k = gaussian_kernel(x, feat, eps=self.eps)
            action_to_den[a] += k
            action_to_weighted_reward[a] += k * r

        # If we have never seen an action, give it a small denominator to avoid div‑by‑0
        for a in self.actions:
            if action_to_den[a] == 0.0:
                action_to_den[a] = 1e-12

        # Compute UCB values
        ucb_values = {}
        for a in self.actions:
            mean = action_to_weighted_reward[a] / action_to_den[a]
            confidence = self.beta * math.sqrt(1.0 / action_to_den[a])
            ucb = mean + confidence
            ucb_values[a] = (mean, confidence, ucb)

        # Choose action with maximal UCB
        chosen = max(self.actions, key=lambda a: ucb_values[a][2])
        mean, confidence, ucb = ucb_values[chosen]

        # Propensity is the softmax over UCBs (optional, here uniform for simplicity)
        propensity = 1.0 / len(self.actions)

        return BanditAction(action_id=chosen,
                            propensity=propensity,
                            expected_reward=mean,
                            confidence_bound=confidence)

    # ------------------------------------------------------------------- #
    # Diagnostic helpers (optional)
    # ------------------------------------------------------------------- #

    def get_history(self) -> List[BanditUpdate]:
        """Return a list of all stored updates as BanditUpdate objects."""
        history = []
        for idx, (feat, a, r) in enumerate(zip(self._features,
                                               self._action_ids,
                                               self._rewards)):
            history.append(BanditUpdate(f"ctx_{idx}", a, r, 1.0, feat))
        return history


# --------------------------------------------------------------------------- #
# Demonstration functions (the required three+ hybrid operations)
# --------------------------------------------------------------------------- #

def demo_feature_construction():
    """Show the hybrid feature vector for a random walk."""
    T, d = 10, 3
    path = np.cumsum(np.random.randn(T, d), axis=0)
    feat = compute_context_features(path, eps=0.5, include_bspline=True)
    print("Feature vector shape:", feat.shape)
    print("First 5 entries:", feat[:5])


def demo_kernel_estimate():
    """Compute a kernel‑weighted reward estimate for a toy dataset."""
    actions = ["left", "right"]
    router = HybridRouter(actions, eps=0.8, beta=0.5)

    # Simulate two past observations
    for _ in range(2):
        path = np.cumsum(np.random.randn(8, 2), axis=0)
        router.update_policy("ctx", path, random.choice(actions), reward=random.random())

    # New context
    new_path = np.cumsum(np.random.randn(8, 2), axis=0)
    x_feat = compute_context_features(new_path, eps=0.8)
    # Manual kernel estimate for action "left"
    num = den = 0.0
    for feat, a, r in zip(router._features, router._action_ids, router._rewards):
        if a == "left":
            k = gaussian_kernel(x_feat, feat, eps=0.8)
            den += k
            num += k * r
    est = num / (den + 1e-12)
    print(f"Kernel‑weighted estimate for 'left': {est:.4f}")


def demo_full_cycle():
    """Run a short online learning loop."""
    actions = ["up", "down", "stay"]
    router = HybridRouter(actions, eps=1.0, beta=1.0)

    for t in range(5):
        # Generate synthetic context (random walk)
        path = np.cumsum(np.random.randn(12, 4), axis=0)

        # Choose action based on current policy
        act = router.select_action(f"t{t}", path)
        # Simulated reward: higher if action matches sign of last increment on dim 0
        reward = 1.0 if (act.action_id == "up" and path[-1, 0] > path[-2, 0]) else \
                 1.0 if (act.action_id == "down" and path[-1, 0] < path[-2, 0]) else \
                 0.5 if act.action_id == "stay" else 0.0
        router.update_policy(f"t{t}", path, act.action_id, reward)

        print(f"Step {t}: selected {act.action_id}, reward {reward:.2f}, "
              f"UCB {act.expected_reward + act.confidence_bound:.3f}")


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=== Demo: Feature construction ===")
    demo_feature_construction()
    print("\n=== Demo: Kernel estimate ===")
    demo_kernel_estimate()
    print("\n=== Demo: Full online cycle ===")
    demo_full_cycle()
    print("\nAll demos executed successfully.")