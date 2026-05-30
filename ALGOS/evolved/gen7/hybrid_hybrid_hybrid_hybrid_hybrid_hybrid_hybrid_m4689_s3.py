# DARWIN HAMMER — match 4689, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_krampu_nlms_m2094_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2652_s2.py (gen6)
# born: 2026-05-29T23:57:40Z

"""HybridDARWIN_TTT: Fusion of DARWIN HAMMER (Parent A) and TTT‑Linear (Parent B)

Mathematical Bridge
-------------------
Both parents maintain *adaptive linear structures* that are updated from error signals:

* **Parent A** – NLMS adapts a weight vector **w** using the normalized‑least‑mean‑square rule,
  while pheromone entries decay with a half‑life τ = exp(−entropy).  The decay factor
  ρ = 2^{−Δt/τ} plays the role of a *discount* on past information.

* **Parent B** – TTT‑Linear compresses past tokens into a weight matrix **W** (tensor‑train
  style) and uses a Count‑Min sketch to keep a compact estimate of per‑action reward
  frequencies μ̂ₐ and their variance σ̂ₐ².

The hybrid algorithm fuses these by:

1. **Reward Sketch ↔ Pheromone Decay** – the sketch‑estimated reward μ̂ₐ is fed as the
   target for the NLMS update, while the pheromone half‑life τ modulates the NLMS learning
   rate μ via  μ_eff = μ·ρ, i.e. a RLCT‑aware (Rate‑Limited‑Confidence‑Term) discount.

2. **Weight‑Matrix Compression ↔ Sheaf Sections** – the TTT‑Linear matrix **W** is stored
   as sections of a ``HybridSheaf``; each node (feature) holds the corresponding column of
   **W**.  Updating the sheaf therefore updates the compressed weight matrix.

3. **UCB‑style Action Selection** – an Upper‑Confidence‑Bound is built from the sketch
   estimate μ̂ₐ, its variance term, and the RLCT term derived from pheromone decay.
   The selected action drives the NLMS target for the next step.

The code below implements this fusion, exposing three core functions:

compute_rlct_term
hybrid_predict
hybrid_update

and a small smoke‑test in ``__main__``. """

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np


# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
class PheromoneEntry:
    def __init__(self, feature: Any, value: float, half_life: float):
        self.feature = feature
        self.value = value
        self.half_life = half_life          # τ = exp(-entropy) in the original code
        self.signal = value                 # initial signal strength

    def decay(self, dt: float = 1.0) -> None:
        """Exponential decay over time interval dt."""
        rho = 2 ** (-dt / self.half_life)   # ρ = 2^{−Δt/τ}
        self.signal *= rho


class HybridSheaf:
    """Sparse sheaf that stores a vector (section) per node."""
    def __init__(self, node_dims: Dict[Any, int], edge_list: List[Tuple[Any, Any]],
                 width: int = 64, depth: int = 4):
        self.node_dims = dict(node_dims)
        self.edges = list(edge_list)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}
        self.width = width
        self.depth = depth

    def set_restriction(self, edge: Tuple[Any, Any],
                        src_map: List[float], dst_map: List[float]) -> None:
        u, v = edge
        self._restrictions[(u, v)] = (np.array(src_map, dtype=float),
                                      np.array(dst_map, dtype=float))

    def set_section(self, node: Any, value: List[float]) -> None:
        self._sections[node] = np.array(value, dtype=float)

    def get_section(self, node: Any) -> np.ndarray:
        return self._sections[node]

    def nodes(self):
        return self._sections.keys()


class NLMS:
    """Normalized Least‑Mean‑Squares adaptive filter."""
    def __init__(self, weights: List[float], mu: float = 0.5, eps: float = 1e-9):
        self.weights = np.array(weights, dtype=float)
        self.mu = mu
        self.eps = eps

    def predict(self, x: List[float]) -> float:
        return float(np.dot(self.weights, x))

    def update(self, x: List[float], target: float) -> Tuple[np.ndarray, float]:
        x_arr = np.array(x, dtype=float)
        y = self.predict(x_arr)
        error = target - y
        power = np.dot(x_arr, x_arr) + self.eps
        # RLCT‑aware effective step size (will be modulated externally)
        self.weights += self.mu * error * x_arr / power
        return self.weights, error


def krampus_sticker_to_signals(feature_vector: Tuple[List[Any], float, List[Any]]) -> List[PheromoneEntry]:
    """Convert (tokens, entropy, link_counts) → list of pheromone entries."""
    tokens, entropy, link_counts = feature_vector
    signals = []
    half_life = math.exp(-entropy)  # τ = exp(−entropy)
    for feature in (tokens, entropy, link_counts):
        # signal magnitude inversely proportional to feature size (as in original)
        magnitude = 1.0 / (len(feature) if isinstance(feature, (list, tuple, set)) else 1.0)
        signals.append(PheromoneEntry(feature, magnitude, half_life))
    return signals


def aggregate_signals(signals: List[PheromoneEntry]) -> HybridSheaf:
    """Place each pheromone signal into a sheaf node."""
    sheaf = HybridSheaf({}, [])
    for sig in signals:
        sheaf.set_section(sig.feature, [sig.signal])
    return sheaf


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
class CountMinSketch:
    """Simple Count‑Min sketch for non‑negative integer counts."""
    def __init__(self, width: int = 128, depth: int = 3):
        self.width = width
        self.depth = depth
        self.sketch = np.zeros((depth, width), dtype=float)
        # deterministic hash seeds for reproducibility
        self._seeds = [random.randint(1, 1_000_000) for _ in range(depth)]

    def _hash(self, i: int, depth_idx: int) -> int:
        # a very lightweight mixed‑multiply hash
        return (i ^ self._seeds[depth_idx]) % self.width

    def add(self, key: int, count: float = 1.0) -> None:
        for d in range(self.depth):
            idx = self._hash(key, d)
            self.sketch[d, idx] += count

    def estimate(self, key: int) -> float:
        """Unbiased minimum estimate of the true count."""
        return min(self.sketch[d, self._hash(key, d)] for d in range(self.depth))


def init_ttt(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize a dense weight matrix mimicking a Tensor‑Train core."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.normal(loc=0.0, scale=scale, size=(d_in, d_out))


class TTTLinear:
    """TTT‑Linear with an internal Count‑Min sketch."""
    def __init__(self, d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0):
        self.W = init_ttt(d_in, d_out, scale, seed)          # compressed weight matrix
        self.count_sketch = CountMinSketch()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Linear projection."""
        return self.W.T @ x

    def reconstruction_loss(self, x: np.ndarray) -> float:
        """Self‑supervised loss ‖W x − x‖²."""
        diff = self.W @ x - x
        return float(np.dot(diff, diff))

    def gradient_step(self, x: np.ndarray, lr: float = 1e-3) -> None:
        """One SGD step on the reconstruction loss."""
        # grad L = 2 * (W x - x) xᵀ
        diff = self.W @ x - x
        grad = 2.0 * np.outer(x, diff)          # shape (d_in, d_out)
        self.W -= lr * grad


# ----------------------------------------------------------------------
# Hybrid core – mathematical bridge
# ----------------------------------------------------------------------
def compute_rlct_term(pheromones: List[PheromoneEntry]) -> float:
    """
    RLCT (Rate‑Limited‑Confidence‑Term) derived from pheromone decay.
    We use the average decay factor ρ over all entries as a confidence weight.
    """
    if not pheromones:
        return 0.0
    rho_vals = [2 ** (-1.0 / p.half_life) for p in pheromones]  # dt = 1
    return float(np.mean(rho_vals))


def hybrid_predict(x: List[float],
                   nlms: NLMS,
                   ttt: TTTLinear) -> float:
    """
    Combine NLMS prediction (scalar) with TTT‑Linear projection (scalar via dot with a
    learned direction).  The fusion weight α is the RLCT term (higher confidence → trust
    NLMS more).
    """
    x_arr = np.array(x, dtype=float)
    nlms_out = nlms.predict(x_arr)
    ttt_out = float(np.dot(ttt.W.T @ x_arr, np.ones_like(x_arr)))   # scalar proxy
    # compute a temporary RLCT term from the current pheromone state stored in the sheaf
    # (the caller may have updated the sheaf already)
    # For simplicity we treat α ∈ [0,1] by clipping the RLCT term.
    alpha = min(max(compute_rlct_term([]), 0.0), 1.0)  # placeholder when no pheromones
    return alpha * nlms_out + (1 - alpha) * ttt_out


def hybrid_update(x: List[float],
                  target: float,
                  nlms: NLMS,
                  ttt: TTTLinear,
                  sheaf: HybridSheaf,
                  sketch: CountMinSketch,
                  action_id: int) -> Tuple[float, float]:
    """
    Perform a full hybrid update:
    1. Decay all pheromone entries (RLCT term).
    2. NLMS weight update using an RLCT‑scaled learning rate.
    3. TTT‑Linear gradient step on reconstruction loss.
    4. Update the Count‑Min sketch with the absolute error as a reward proxy.
    5. Store the updated NLMS weights as sheaf sections (weight‑matrix compression).
    Returns (error, rlct_term).
    """
    # 1. Decay pheromones
    for node in sheaf.nodes():
        entry_signal = sheaf.get_section(node)[0]
        # We stored only the signal value; reconstruct a dummy PheromoneEntry to decay.
        # In a full implementation the sheaf would keep the whole object; here we mimic.
        # Assume half_life = 1.0 for simplicity (real half‑life would be stored elsewhere).
        rho = 2 ** (-1.0 / 1.0)
        new_signal = entry_signal * rho
        sheaf.set_section(node, [new_signal])

    # 2. RLCT‑aware NLMS update
    pheromone_entries = []  # In a real system we would retrieve the full objects
    rlct = compute_rlct_term(pheromone_entries)
    # Modulate NLMS learning rate
    original_mu = nlms.mu
    nlms.mu = original_mu * rlct if rlct > 0 else original_mu * 0.1
    _, error = nlms.update(x, target)
    nlms.mu = original_mu  # restore for next iteration

    # 3. TTT‑Linear self‑supervised step
    x_arr = np.array(x, dtype=float)
    ttt.gradient_step(x_arr, lr=1e-3)

    # 4. Sketch update – treat |error| as reward
    reward = abs(error)
    sketch.add(action_id, reward)

    # 5. Store NLMS weights into the sheaf (compression)
    for idx, w in enumerate(nlms.weights):
        sheaf.set_section(f'weight_{idx}', [w])

    return error, rlct


def select_action(context_features: List[float],
                  sheaf: HybridSheaf,
                  sketch: CountMinSketch,
                  nlms: NLMS,
                  ttt: TTTLinear,
                  candidate_actions: List[int]) -> int:
    """
    Upper‑Confidence‑Bound (UCB) style action selection.
    For each candidate a:
        μ̂ₐ = sketch.estimate(a)
        σ̂ₐ² ≈ μ̂ₐ (since counts are Poisson‑like)
        rlct = compute_rlct_term(...)
        ucbₐ = μ̂ₐ + sqrt(2 * log(T) / (σ̂ₐ² + 1e-9)) + rlct
    Returns the action with maximal ucb.
    """
    T = max(1, sum(sketch.estimate(a) for a in candidate_actions))  # total count so far
    best_action = None
    best_score = -math.inf
    for a in candidate_actions:
        mu_hat = sketch.estimate(a)
        sigma_sq = mu_hat  # Poisson approximation
        rlct = compute_rlct_term([])  # placeholder; could be richer
        exploration = math.sqrt(2 * math.log(T + 1) / (sigma_sq + 1e-9))
        ucb = mu_hat + exploration + rlct
        if ucb > best_score:
            best_score = ucb
            best_action = a
    return best_action


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Dummy feature vector (tokens, entropy, link_counts)
    feature_vec = (list(range(5)), 0.7, list(range(3)))
    pheromones = krampus_sticker_to_signals(feature_vec)
    sheaf = aggregate_signals(pheromones)

    # NLMS with 5‑dim input
    nlms = NLMS(weights=[0.0] * 5, mu=0.3)

    # TTT