# DARWIN HAMMER — match 538, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_capybara_opti_m52_s2.py (gen3)
# born: 2026-05-29T23:29:34Z

"""Hybrid Fusion of Darwin Hammer Decision-Hygiene Bandit and RBF Surrogate Optimizer.

Parent A: `hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s0.py`
- Defines a 3‑dimensional resource vector **eᵢ = [dᵢ, pᵢ, sᵢ]**.
- Uses a virtual VRAM store **z(t)** that obeys a first‑order ODE and
  modulates the learning rate **η(t) = base_eta * (1 + z(t))**.
- Contains a weight matrix **W ∈ ℝ^{d_out×d_in}** that linearly maps the
  resource vector before feeding it to a bandit update.

Parent B: `hybrid_hybrid_hybrid_rbf_su_hybrid_capybara_opti_m52_s2.py`
- Implements an RBF surrogate model **ĥ(x) = Σ w_k·exp(-ε²‖x‑c_k‖²)**.
- Provides utilities for Euclidean distance, Gaussian kernel, linear‑system
  solution, and a social‑interaction operator used in swarm‑style optimisation.

Mathematical Bridge
-------------------
The bridge is built by **(i)** feeding the *transformed* resource vector
`x = W·e` into the RBF surrogate to obtain a predicted reward **r̂**, and
**(ii)** using that reward both to (a) drive the VRAM store ODE and (b) update
the weight matrix **W**.  The surrogate’s centres are the historic transformed
vectors, so the surrogate evolves together with the bandit, creating a fully
coupled hybrid system.

The core equations are:

1. Resource vector  
   `e = [d, β·σ, s]`  (`σ∈{0,1}` signature collision flag).

2. Linear transformation  
   `x = W @ e`.

3. RBF surrogate prediction  
   `r̂ = Σ_{k} w_k·exp(-ε²‖x‑c_k‖²)`.

4. Store dynamics (Euler step)  
   `z_{t+Δ} = z_t + Δ·[α·(r̂‑z_t) ‑ β·z_t]`.

5. Learning‑rate modulation  
   `η = base_eta·(1 + z_{t+Δ})`.

6. Weight‑matrix update (gradient‑like)  
   `W ← W + η·r̂·(e·1ᵀ)` (where `1` is a column of ones of size `d_out`).

7. Surrogate weight update – solve the linear system  
   `K·w = y` with `K_{ij}=exp(-ε²‖c_i‑c_j‖²)`.

The implementation below realises these equations and provides three
demonstration functions.

"""

import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Gauss‑Jordan elimination for a dense linear system."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

# ----------------------------------------------------------------------
# RBF Surrogate (Parent B)
# ----------------------------------------------------------------------
class RBFSurrogate:
    def __init__(self, epsilon: float = 1.0):
        self.centers: List[np.ndarray] = []
        self.weights: List[float] = []
        self.epsilon = epsilon

    def _kernel_matrix(self, X: List[np.ndarray]) -> np.ndarray:
        n = len(X)
        K = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = gaussian(euclidean(X[i], X[j]), self.epsilon)
        return K

    def fit(self, X: List[np.ndarray], y: List[float]) -> None:
        """Re‑fit the surrogate on the given data."""
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        if not X:
            return
        K = self._kernel_matrix(X)
        w = solve_linear(K.tolist(), y)
        self.centers = X
        self.weights = w

    def add_point(self, x: np.ndarray, y_val: float) -> None:
        """Incrementally add a new observation and re‑solve."""
        self.centers.append(x)
        self.weights.append(0.0)  # placeholder
        self.fit(self.centers, self.weights[:-1] + [y_val])

    def predict(self, x: np.ndarray) -> float:
        if not self.centers:
            return 0.0
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

# ----------------------------------------------------------------------
# Hybrid Fusion Core (Parent A + B)
# ----------------------------------------------------------------------
class HybridFusionRBF:
    def __init__(
        self,
        d_in: int,
        d_out: int,
        seed: int = 0,
        base_eta: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1.0,
        store_decay: float = 0.99,
        epsilon: float = 1.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=0.1, size=(d_out, d_in))
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.store = 0.0
        self.store_decay = store_decay
        self.surrogate = RBFSurrogate(epsilon=epsilon)

    # ------------------------------------------------------------------
    # 1️⃣ Resource vector construction (from Parent A)
    # ------------------------------------------------------------------
    def resource_vector(
        self,
        distance_m: float,
        signature_collision: bool,
        hygiene_score: float,
        beta_sig: float = 1.0,
    ) -> np.ndarray:
        """Return e = [d, β·σ, s] as a column vector."""
        sigma = 1.0 if signature_collision else 0.0
        return np.array([distance_m, beta_sig * sigma, hygiene_score], dtype=float)

    # ------------------------------------------------------------------
    # 2️⃣ Linear transformation (Parent A)
    # ------------------------------------------------------------------
    def transform(self, e: np.ndarray) -> np.ndarray:
        """x = W @ e."""
        return self.W @ e

    # ------------------------------------------------------------------
    # 3️⃣ Surrogate prediction (Parent B)
    # ------------------------------------------------------------------
    def predict_reward(self, x: np.ndarray) -> float:
        """Predict reward using the RBF surrogate."""
        return self.surrogate.predict(x)

    # ------------------------------------------------------------------
    # 4️⃣ Store ODE integration (Parent A)
    # ------------------------------------------------------------------
    def update_store(self, reward: float) -> None:
        """Euler integration of dz/dt = α(r‑z) ‑ β·z."""
        dz = self.alpha * (reward - self.store) - self.beta * self.store
        self.store += self.dt * dz
        self.store *= self.store_decay  # simulate memory eviction

    # ------------------------------------------------------------------
    # 5️⃣ Learning‑rate modulation & weight update (Parent A)
    # ------------------------------------------------------------------
    def update_weights(self, e: np.ndarray, reward: float) -> None:
        """W ← W + η·r·(e·1ᵀ) where η = base_eta·(1+store)."""
        eta = self.base_eta * (1.0 + self.store)
        outer = np.outer(e, np.ones(self.W.shape[0]))
        self.W += eta * reward * outer.T  # shape matches W (d_out×d_in)

    # ------------------------------------------------------------------
    # 6️⃣ Full hybrid step
    # ------------------------------------------------------------------
    def step(
        self,
        distance_m: float,
        signature_collision: bool,
        hygiene_score: float,
        beta_sig: float = 1.0,
    ) -> Tuple[float, np.ndarray]:
        """
        Perform one hybrid iteration:
        1. Build resource vector e.
        2. Transform to x = W·e.
        3. Predict reward r̂ via surrogate.
        4. Update store and weights using r̂.
        5. Append (x, r̂) to surrogate data.
        Returns (predicted_reward, transformed_vector).
        """
        e = self.resource_vector(distance_m, signature_collision, hygiene_score, beta_sig)
        x = self.transform(e)
        reward = self.predict_reward(x)

        # Update dynamics
        self.update_store(reward)
        self.update_weights(e, reward)

        # Incrementally enrich surrogate
        self.surrogate.add_point(x, reward)

        return reward, x

# ----------------------------------------------------------------------
# Demonstration Functions (requirement: at least 3)
# ----------------------------------------------------------------------
def compute_resource_vector_example() -> np.ndarray:
    """Example: compute a resource vector for a synthetic entity."""
    hf = HybridFusionRBF(d_in=3, d_out=4, seed=42)
    # Simulated inputs
    distance = 1234.5          # metres
    collision = True          # signature duplicated
    hygiene = 0.78             # decision‑hygiene score
    return hf.resource_vector(distance, collision, hygiene, beta_sig=2.0)

def rbf_prediction_example() -> float:
    """Example: predict reward for a random transformed vector."""
    hf = HybridFusionRBF(d_in=3, d_out=4, seed=7, epsilon=0.5)
    # Populate surrogate with a few random points
    for _ in range(5):
        e = np.random.rand(3)
        x = hf.transform(e)
        # fake reward based on norm
        fake_reward = -np.linalg.norm(x)
        hf.surrogate.add_point(x, fake_reward)
    # New query
    e_new = np.array([500.0, 1.0, 0.9])
    x_new = hf.transform(e_new)
    return hf.predict_reward(x_new)

def hybrid_iteration_example() -> Tuple[float, np.ndarray]:
    """Run a single hybrid iteration and return its outputs."""
    hf = HybridFusionRBF(d_in=3, d_out=4, seed=123, base_eta=0.02, alpha=0.8, beta=0.3, dt=0.5)
    # First iteration (surrogate empty → reward = 0)
    reward, x = hf.step(
        distance_m=2500.0,
        signature_collision=False,
        hygiene_score=0.65,
        beta_sig=1.5,
    )
    return reward, x

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Resource vector demo
    e = compute_resource_vector_example()
    print("Resource vector e:", e)

    # 2. RBF surrogate prediction demo
    pred = rbf_prediction_example()
    print("RBF surrogate prediction:", pred)

    # 3. Full hybrid iteration demo
    reward, transformed = hybrid_iteration_example()
    print("Hybrid step -> reward:", reward, "transformed x:", transformed)

    # Quick sanity: run a few more steps to ensure no exceptions
    hf = HybridFusionRBF(d_in=3, d_out=4, seed=999)
    for t in range(3):
        r, x = hf.step(
            distance_m=1000.0 + t * 100,
            signature_collision=(t % 2 == 0),
            hygiene_score=0.5 + 0.1 * t,
            beta_sig=1.0,
        )
        print(f"t={t} | reward={r:.4f} | store={hf.store:.4f}")

    sys.exit(0)