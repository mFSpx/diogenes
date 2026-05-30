# DARWIN HAMMER — match 5243, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_temporal_moti_hybrid_hybrid_hybrid_m1987_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_bandit_label_foundry_m21_s0.py (gen3)
# born: 2026-05-30T00:00:51Z

"""Hybrid Temporal‑Motif Sheaf + Sketch‑Bandit Optimizer
====================================================

Parent A: *Hybrid Temporal‑Motif Sheaf‑RBF Optimizer* – provides a cellular
sheaf whose sections are resource vectors **eᵢ**, a linear map **W**, and an
RBF surrogate that predicts a reward **r̂ᵢ** from the transformed vectors
**xᵢ = W·eᵢ**.

Parent B: *Hybrid Bandit‑Label‑Foundry* – supplies a Count‑Min sketch that
estimates the empirical mean reward **μᵢ** (and a variance proxy) and a
HyperLogLog estimator that yields the number of distinct contexts **D**.
These statistics are used for exploration‑exploitation decisions.

Mathematical bridge
-------------------
For each sheaf node *i* we compute


e_i   – current sheaf section (resource vector)
x_i   = W @ e_i                         (linear projection)
r̂_i  = φ(x_i; C, σ)                    (RBF surrogate with centres C)
μ_i   = sketch.mean(i)                  (Count‑Min estimate of reward)
D    = hll.cardinality()                (distinct‑context estimate)

α ∈ [0,1]  – blending coefficient (chosen from D)
r_i  = α·r̂_i + (1‑α)·μ_i                (hybrid reward)



The blended reward **rᵢ** drives three coupled updates:

1. **Weight matrix** – stochastic gradient‑like step  
   `W ← W + η·(r_i‑μ_i)·e_i.T`
2. **Sheaf sections** – gradient descent on a simple quadratic loss  
   `e_i ← e_i – η·(r_i‑μ_i)·W.T·1`
3. **Global scalar** `z` – integrates total reward  
   `z ← z + η·∑_i r_i`

The Count‑Min sketch is also updated with the raw projection **x_i** so that
future μ_i reflect the evolving dynamics, while the HyperLogLog count **D**
modulates the blending coefficient α = min(1, log₂(D+1)/10).

The code below implements this hybrid system, provides three demonstration
functions, and ends with a smoke test that executes without error.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Sheaf and RBF surrogate
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int

class Sheaf:
    """Cellular sheaf with linear restriction maps.

    `node_dims` maps node identifiers to the dimension of their section.
    `edges` is a list of (src, dst) tuples; restriction maps are identity for
    simplicity.
    """
    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        # initialise a random section for each node
        self.sections: Dict[Any, np.ndarray] = {
            n: np.random.randn(d).astype(np.float64)
            for n, d in self.node_dims.items()
        }

    def get_section(self, node: Any) -> np.ndarray:
        return self.sections[node]

    def set_section(self, node: Any, value: np.ndarray) -> None:
        assert value.shape == (self.node_dims[node],)
        self.sections[node] = value

class RBFSurrogate:
    """Radial‑Basis‑Function surrogate.

    `centres` : (k, d) array of centre vectors.
    `sigma`   : bandwidth.
    `weights` : (k,) coefficients learned implicitly via the hybrid reward.
    """
    def __init__(self, centres: np.ndarray, sigma: float = 1.0):
        self.centres = np.asarray(centres, dtype=np.float64)  # shape (k, d)
        self.sigma = float(sigma)
        self.weights = np.ones(self.centres.shape[0], dtype=np.float64)

    def phi(self, x: np.ndarray) -> np.ndarray:
        """Compute RBF activations for a single input vector x."""
        diff = self.centres - x  # (k, d)
        sq_norm = np.einsum('kd,kd->k', diff, diff)
        return np.exp(-sq_norm / (2.0 * self.sigma ** 2))

    def predict(self, x: np.ndarray) -> float:
        """RBF surrogate prediction r̂ = w·φ(x)."""
        phi_x = self.phi(x)
        return float(np.dot(self.weights, phi_x))

# ----------------------------------------------------------------------
# Parent B – Count‑Min Sketch and HyperLogLog
# ----------------------------------------------------------------------
class CountMinSketch:
    """Simple Count‑Min sketch for non‑negative updates."""
    def __init__(self, width: int = 2000, depth: int = 5, seed: int = 0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.float64)
        rng = np.random.RandomState(seed)
        self.seeds = rng.randint(0, 2**31 - 1, size=depth, dtype=np.int64)

    def _hash(self, i: int, key: int) -> int:
        h = hash((key, self.seeds[i]))
        return h % self.width

    def add(self, key: int, value: float = 1.0) -> None:
        for i in range(self.depth):
            idx = self._hash(i, key)
            self.tables[i, idx] += value

    def estimate(self, key: int) -> float:
        """Return the minimum count over the hash rows."""
        mins = [self.tables[i, self._hash(i, key)] for i in range(self.depth)]
        return float(min(mins))

class HyperLogLog:
    """Very lightweight HyperLogLog for cardinality estimation."""
    def __init__(self, p: int = 10):
        self.p = p
        self.m = 1 << p
        self.registers = np.zeros(self.m, dtype=np.uint8)

    def _rho(self, w: int) -> int:
        """Position of first 1-bit (1‑based)."""
        rho = 1
        while w & (1 << (64 - rho)) == 0 and rho <= 64:
            rho += 1
        return rho

    def add(self, value: Any) -> None:
        h = hash(value) & ((1 << 64) - 1)  # 64‑bit hash
        idx = h >> (64 - self.p)
        w = h << self.p
        self.registers[idx] = max(self.registers[idx], self._rho(w))

    def cardinality(self) -> float:
        """Estimate distinct count."""
        Z = 1.0 / np.sum(2.0 ** -self.registers)
        E = self.alpha_mm() * Z
        # Small range correction
        if E <= 2.5 * self.m:
            V = np.count_nonzero(self.registers == 0)
            if V != 0:
                E = self.m * math.log(self.m / V)
        return E

    def alpha_mm(self) -> float:
        if self.m == 16:
            return 0.673 * self.m * self.m
        if self.m == 32:
            return 0.697 * self.m * self.m
        if self.m == 64:
            return 0.709 * self.m * self.m
        return (0.7213 / (1 + 1.079 / self.m)) * self.m * self.m

# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def extract_motifs(sequences: List[List[str]]) -> List[TemporalMotif]:
    """Very naive motif extractor: counts all length‑2 subsequences."""
    counter: Dict[Tuple[str, str], int] = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            pat = (seq[i], seq[i + 1])
            counter[pat] = counter.get(pat, 0) + 1
    motifs = [TemporalMotif(pattern=pat, support=sup) for pat, sup in counter.items()]
    return motifs

def initialise_hybrid(
    sheaf_nodes: Dict[Any, int],
    edges: List[Tuple[Any, Any]],
    motif_sequences: List[List[str]],
) -> Tuple[Sheaf, np.ndarray, RBFSurrogate, CountMinSketch, HyperLogLog, float]:
    """Create all objects needed for the hybrid algorithm."""
    sheaf = Sheaf(sheaf_nodes, edges)

    # random linear map W: output dim = 3 (for illustration)
    out_dim = 3
    W = np.random.randn(out_dim, max(sheaf_nodes.values())).astype(np.float64)

    # extract motifs and use their support as RBF centres (projected to out_dim)
    motifs = extract_motifs(motif_sequences)
    if motifs:
        centres = np.vstack([np.random.randn(out_dim) for _ in motifs])
    else:
        centres = np.random.randn(1, out_dim)
    rbf = RBFSurrogate(centres, sigma=1.5)

    cms = CountMinSketch(width=1024, depth=4, seed=42)
    hll = HyperLogLog(p=10)

    z = 0.0  # global scalar
    return sheaf, W, rbf, cms, hll, z

def hybrid_step(
    sheaf: Sheaf,
    W: np.ndarray,
    rbf: RBFSurrogate,
    cms: CountMinSketch,
    hll: HyperLogLog,
    z: float,
    eta: float = 0.01,
) -> float:
    """Perform a single hybrid iteration over all nodes."""
    # 1. Compute blended rewards per node
    rewards: List[float] = []
    for node, dim in sheaf.node_dims.items():
        e = sheaf.get_section(node)                     # (d,)
        x = W @ e                                        # (out_dim,)
        r_hat = rbf.predict(x)                           # surrogate prediction

        key = hash(node) & ((1 << 31) - 1)                # deterministic non‑negative key
        cms.add(key, float(np.linalg.norm(x)))           # update sketch with magnitude
        mu = cms.estimate(key)                           # sketch estimate

        # update HyperLogLog with the raw vector (as bytes)
        hll.add(tuple(x.tolist()))

        D = hll.cardinality()
        alpha = min(1.0, math.log2(D + 1) / 10.0)          # blending coefficient

        r = alpha * r_hat + (1.0 - alpha) * mu
        rewards.append(r)

        # 2. Update weight matrix W (gradient‑like)
        grad_W = (r - mu) * np.outer(e, np.ones(W.shape[0]))
        W += eta * grad_W.T  # shape (out_dim, d)

        # 3. Update sheaf section (simple gradient descent on quadratic loss)
        e_new = e - eta * (r - mu) * W.T @ np.ones(W.shape[0])
        sheaf.set_section(node, e_new)

    # 4. Update global scalar z
    z += eta * sum(rewards)
    return z

# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def demo_sheaf_creation() -> Sheaf:
    """Create a tiny sheaf and return it."""
    nodes = {'A': 3, 'B': 3, 'C': 3}
    edges = [('A', 'B'), ('B', 'C')]
    return Sheaf(nodes, edges)

def demo_sketch_and_hll() -> Tuple[CountMinSketch, HyperLogLog]:
    """Populate a sketch and HLL with dummy data and return them."""
    cms = CountMinSketch(width=256, depth=3, seed=7)
    hll = HyperLogLog(p=8)
    for i in range(100):
        val = random.randint(0, 50)
        cms.add(val, 1.0)
        hll.add(val)
    return cms, hll

def demo_hybrid_iteration() -> None:
    """Run a few hybrid steps on synthetic data."""
    # synthetic motif sequences (e.g., token streams)
    seqs = [
        ['a', 'b', 'c', 'a'],
        ['b', 'c', 'd'],
        ['a', 'd', 'c', 'b'],
    ]
    sheaf_nodes = {'n1': 4, 'n2': 4}
    edges = [('n1', 'n2')]
    sheaf, W, rbf, cms, hll, z = initialise_hybrid(sheaf_nodes, edges, seqs)

    for step in range(5):
        z = hybrid_step(sheaf, W, rbf, cms, hll, z, eta=0.005)
        print(f"Step {step+1}: global scalar z = {z:.4f}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Simple sheaf demo
    sh = demo_sheaf_creation()
    print("Sheaf sections after creation:")
    for n in sh.node_dims:
        print(f"  {n}: {sh.get_section(n)}")

    # 2. Sketch / HLL demo
    cms_demo, hll_demo = demo_sketch_and_hll()
    sample_key = 42
    print(f"Sketch estimate for key {sample_key}: {cms_demo.estimate(sample_key):.2f}")
    print(f"HLL cardinality estimate: {hll_demo.cardinality():.2f}")

    # 3. Hybrid iteration demo
    demo_hybrid_iteration()