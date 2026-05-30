# DARWIN HAMMER — match 3642, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1858_s0.py (gen6)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_infotaxis_min_m967_s1.py (gen5)
# born: 2026-05-29T23:51:01Z

"""Hybrid Algorithm merging RLCT‑Grokking, Pheromone‑Modulated MinHash, and Physarum‑Sheaf dynamics.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1858_s0.py (RLCT estimation, pheromone signals, regret‑weighted MinHash)
- hybrid_hybrid_physarum_netw_hybrid_infotaxis_min_m967_s1.py (Physarum‑Sheaf flux, weighted discrepancy, MinHash Jaccard similarity)

Mathematical bridge:
Both parents rely on a MinHash‑based similarity measure.  
In this fusion the Jaccard similarity of MinHash signatures modulates the
Physarum‑Sheaf gain α, while the pheromone signal associated with each edge
acts as an additional scalar multiplier for that similarity.  
The resulting “effective similarity”

    S_eff(u,v) = pheromone(u,v) · Jaccard(σ_u, σ_v)

feeds directly into the RL‑canonical‑threshold‑guided regression that
optimises a free‑energy‑like loss landscape, and drives the flux update

    q_{uv}^{new} = q_{uv} + η·S_eff(u,v)·(d_{uv} – q_{uv})

where d_{uv} is the weighted discrepancy derived from the RLCT regression.
Thus the three core topologies (log‑log regression, pheromone‑scaled MinHash,
and Physarum‑Sheaf flux) are mathematically fused into a single unified system.
"""

import math
import random
import sys
import pathlib
import hashlib
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Certainty infrastructure (from parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.datetime.utcnow().isoformat() + "Z")


# ----------------------------------------------------------------------
# Core Hybrid System
# ----------------------------------------------------------------------
class HybridSystem:
    """
    Encapsulates the fused dynamics:
    * RLCT regression (log‑log linear fit)
    * MinHash‑based similarity with pheromone scaling
    * Physarum‑Sheaf flux update driven by the effective similarity
    """

    def __init__(self, nodes: List[Any]):
        self.nodes = nodes
        self.num_nodes = len(nodes)

        # Edge‑wise pheromone signals, default 1.0 (no modulation)
        self.pheromone: Dict[Tuple[int, int], float] = {
            (i, j): 1.0 for i in range(self.num_nodes) for j in range(self.num_nodes) if i != j
        }

        # Flux matrix q_uv (Physarum)
        self.flux: np.ndarray = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)

        # Discrepancy matrix d_uv (derived from RLCT regression)
        self.discrepancy: np.ndarray = np.zeros_like(self.flux)

        # MinHash signatures per node
        self.signatures: List[np.ndarray] = [np.full(64, np.iinfo(np.uint64).max, dtype=np.uint64) for _ in nodes]

    # ------------------------------------------------------------------
    # 1️⃣ RLCT estimation (Parent A)
    # ------------------------------------------------------------------
    def estimate_rlct(self, train_losses_per_n: List[float], n_values: List[int]) -> Tuple[float, float]:
        """
        Performs a simple ordinary‑least‑squares fit of
            y = log(loss)  vs.  x = log(log(n))
        Returns (slope, intercept) which serve as proxies for the RLCT
        and a constant term.
        """
        losses = np.asarray(train_losses_per_n, dtype=np.float64)
        ns = np.asarray(n_values, dtype=np.float64)

        if np.any(ns <= math.e):
            raise ValueError("all n_values must be > e for log(log(n)) to be positive")
        if losses.shape != ns.shape:
            raise ValueError("train_losses_per_n and n_values must have the same length")

        y = np.log(np.maximum(losses, 1e-300))
        x = np.log(np.log(ns))

        x_centered = x - x.mean()
        y_centered = y - y.mean()

        var_x = (x_centered ** 2).sum()
        if var_x < 1e-15:
            raise RuntimeError("variance of x is too small for regression")

        slope = (x_centered * y_centered).sum() / var_x
        intercept = y.mean() - slope * x.mean()
        return float(slope), float(intercept)

    # ------------------------------------------------------------------
    # 2️⃣ MinHash utilities (Parent B)
    # ------------------------------------------------------------------
    @staticmethod
    def _hash(seed: int, token: str) -> int:
        data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
        return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

    @staticmethod
    def minhash_signature(tokens: Iterable[str], num_perm: int = 64) -> np.ndarray:
        """
        Computes a MinHash signature (uint64 array) for a set of tokens.
        """
        if not tokens:
            # Empty set → maximal hash values
            return np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)

        signature = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        for i in range(num_perm):
            min_val = np.iinfo(np.uint64).max
            for token in tokens:
                hv = HybridSystem._hash(i, token)
                if hv < min_val:
                    min_val = hv
            signature[i] = min_val
        return signature

    @staticmethod
    def jaccard_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
        """
        Jaccard similarity between two MinHash signatures.
        """
        if sig1.shape != sig2.shape:
            raise ValueError("signatures must have the same length")
        matches = (sig1 == sig2).sum()
        return matches / sig1.size

    # ------------------------------------------------------------------
    # 3️⃣ Pheromone‑scaled similarity (bridge between A & B)
    # ------------------------------------------------------------------
    def effective_similarity(self, u: int, v: int) -> float:
        """
        Computes S_eff(u,v) = pheromone(u,v) * Jaccard(σ_u, σ_v)
        """
        base_sim = self.jaccard_similarity(self.signatures[u], self.signatures[v])
        pher = self.pheromone.get((u, v), 1.0)
        return pher * base_sim

    # ------------------------------------------------------------------
    # 4️⃣ Physarum‑Sheaf flux update (Parent B) modulated by S_eff
    # ------------------------------------------------------------------
    def physarum_update(self, learning_rate: float = 0.1) -> None:
        """
        Performs one synchronous update of the flux matrix:
            q_uv ← q_uv + η * S_eff(u,v) * (d_uv - q_uv)
        where d_uv is the weighted discrepancy (here derived from RLCT slope).
        """
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if u == v:
                    continue
                s_eff = self.effective_similarity(u, v)
                delta = self.discrepancy[u, v] - self.flux[u, v]
                self.flux[u, v] += learning_rate * s_eff * delta

    # ------------------------------------------------------------------
    # 5️⃣ Helper to populate discrepancy matrix from RLCT slope
    # ------------------------------------------------------------------
    def set_discrepancy_from_rlct(self, slope: float, scale: float = 1.0) -> None:
        """
        Populates the discrepancy matrix with a simple function of the RLCT slope:
            d_uv = scale * |slope|  (identical for all edges)
        """
        value = scale * abs(slope)
        self.discrepancy.fill(value)

    # ------------------------------------------------------------------
    # 6️⃣ Public API – a single hybrid iteration
    # ------------------------------------------------------------------
    def hybrid_iteration(
        self,
        token_sets: List[Iterable[str]],
        train_losses_per_n: List[float],
        n_values: List[int],
        learning_rate: float = 0.1,
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Executes a full hybrid step:
        1. Compute MinHash signatures for each node.
        2. Estimate RLCT (slope, intercept) from supplied loss data.
        3. Fill discrepancy matrix using the RLCT slope.
        4. Update flux using pheromone‑scaled similarity.
        Returns the updated flux matrix and the RLCT parameters.
        """
        if len(token_sets) != self.num_nodes:
            raise ValueError("token_sets length must match number of nodes")

        # 1. Signatures
        self.signatures = [self.minhash_signature(tokens) for tokens in token_sets]

        # 2. RLCT regression
        slope, intercept = self.estimate_rlct(train_losses_per_n, n_values)

        # 3. Discrepancy
        self.set_discrepancy_from_rlct(slope)

        # 4. Flux update
        self.physarum_update(learning_rate=learning_rate)

        return self.flux.copy(), (slope, intercept)

    # ------------------------------------------------------------------
    # 7️⃣ Pheromone manipulation utilities
    # ------------------------------------------------------------------
    def deposit_pheromone(self, u: int, v: int, amount: float) -> None:
        """Increases pheromone on edge (u,v) by a positive amount."""
        if amount < 0:
            raise ValueError("pheromone amount must be non‑negative")
        self.pheromone[(u, v)] = self.pheromone.get((u, v), 1.0) + amount

    def evaporate_pheromone(self, decay: float = 0.01) -> None:
        """Applies exponential decay to all pheromone values."""
        if not 0 <= decay <= 1:
            raise ValueError("decay must be in [0,1]")
        for edge in self.pheromone:
            self.pheromone[edge] *= (1.0 - decay)


# ----------------------------------------------------------------------
# Stand‑alone helper functions (demonstrate hybrid operation)
# ----------------------------------------------------------------------
def generate_synthetic_tokens(vocab_size: int = 1000, length: int = 20) -> List[str]:
    """Creates a random list of token strings."""
    return [f"tok{random.randint(0, vocab_size)}" for _ in range(length)]


def synthetic_loss_series(base: float = 0.5, noise: float = 0.05, size: int = 10) -> Tuple[List[float], List[int]]:
    """Generates synthetic (loss, n) pairs for RLCT regression."""
    n_vals = [random.randint(20, 200) for _ in range(size)]
    losses = [base / (math.log(n) ** 0.5) + random.uniform(-noise, noise) for n in n_vals]
    return losses, n_vals


def main_smoke_test() -> None:
    # Create a tiny graph with 4 nodes
    nodes = ["A", "B", "C", "D"]
    hs = HybridSystem(nodes)

    # Random token sets per node
    token_sets = [generate_synthetic_tokens() for _ in nodes]

    # Synthetic loss data for RLCT estimation
    losses, ns = synthetic_loss_series(size=12)

    # Run a hybrid iteration
    flux, (slope, intercept) = hs.hybrid_iteration(
        token_sets=token_sets,
        train_losses_per_n=losses,
        n_values=ns,
        learning_rate=0.2,
    )

    # Simple sanity prints (no external dependencies)
    print("RLCT slope:", slope)
    print("Flux matrix after update:\n", np.round(flux, 4))

    # Demonstrate pheromone dynamics
    hs.deposit_pheromone(0, 1, amount=0.5)
    hs.evaporate_pheromone(decay=0.1)
    print("Pheromone (0,1):", hs.pheromone[(0, 1)])

if __name__ == "__main__":
    main_smoke_test()