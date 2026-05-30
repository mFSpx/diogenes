# DARWIN HAMMER — match 5551, survivor 4
# gen: 6
# parent_a: hybrid_model_vram_scheduler_hybrid_hybrid_bandit_m188_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1512_s6.py (gen5)
# born: 2026-05-30T00:02:51Z

import sys
import random
import math
import hashlib
from pathlib import Path
from collections import Counter
from typing import Mapping, Hashable, Set, List, Dict, Tuple, Iterable, Optional

import numpy as np
from dataclasses import dataclass, asdict

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
ValuesMap = Mapping[Node, List[float]]
TokensMap = Mapping[Node, Set[str]]

# ----------------------------------------------------------------------
# Utilities from Parent A (perceptual similarity)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """
    Perceptual hash: 1 bit per value indicating >= average.
    Supports arbitrary length by folding into a 64‑bit integer.
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    # Use a sliding window of 64 bits; if more values exist we fold them
    for i, v in enumerate(values):
        bit = int(v >= avg)
        bits = (bits << 1) | bit
        if i == 63:                     # first 64 bits are set, now fold
            break
    # Fold remaining values (if any) into the existing 64‑bit pattern
    for v in values[64:]:
        bits ^= int(v >= avg) << (v % 64)
    return bits & ((1 << 64) - 1)


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return bin(a ^ b).count("1")


def minhash_signature(tokens: Set[str], num_hashes: int = 128) -> List[int]:
    """
    Deterministic MinHash signature for a set of tokens.
    Uses a fast MD5‑based hash with different seeds.
    """
    if not tokens:
        return [0] * num_hashes
    signatures = []
    for seed in range(num_hashes):
        min_hash = None
        for tok in tokens:
            h = int(hashlib.md5((tok + str(seed)).encode()).hexdigest(), 16)
            if (min_hash is None) or (h < min_hash):
                min_hash = h
        signatures.append(min_hash or 0)
    return signatures


def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity for MinHash signatures."""
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def node_similarity(
    phash_a: int,
    phash_b: int,
    sig_a: List[int],
    sig_b: List[int],
    weight_phash: float = 0.5,
) -> float:
    """
    Combined similarity:
        - Normalised Hamming similarity (1 - distance/64)
        - MinHash Jaccard similarity
    Weighted by ``weight_phash`` (rest goes to MinHash).
    """
    ham_sim = 1.0 - hamming_distance(phash_a, phash_b) / 64.0
    mh_sim = jaccard_similarity(sig_a, sig_b)
    return weight_phash * ham_sim + (1.0 - weight_phash) * mh_sim


# ----------------------------------------------------------------------
# Utilities from Parent B (distributed leader)
# ----------------------------------------------------------------------
def compute_store(store: float, inflow: float, outflow: float, dt: float) -> float:
    """
    Simple Euler integration of a storage differential equation:
        dS/dt = inflow - outflow
    The store is kept non‑negative.
    """
    store += (inflow - outflow) * dt
    return max(store, 0.0)


def update_linear_weight(W: np.ndarray, eta: float, grad: np.ndarray) -> np.ndarray:
    """
    Stochastic gradient descent update with scalar learning rate ``eta``.
    The function works in‑place but also returns the matrix for convenience.
    """
    W -= eta * grad
    return W


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class BanditAction:
    """Container for a single bandit decision."""
    action: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    node_id: Optional[Node] = None  # identifier of the node this action refers to


@dataclass
class VramSlotPlan:
    """Result of the scheduler – a lightweight, serialisable plan."""
    plan_id: str
    description: str
    action: str
    total_vram: int
    status: str
    metadata: Dict[str, any]

    def as_dict(self) -> Dict[str, any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
class HybridVrBanditScheduler:
    """
    A hybrid scheduler that:
      * tracks a VRAM store (Parent B)
      * learns a low‑rank linear model for allocating VRAM (Parent A)
      * uses a fused similarity metric to bias bandit propensities (fusion point)
    """

    def __init__(
        self,
        budget_mb: int,
        reserve_mb: int,
        base_model_mb: int,
        adapter_mb: int,
        embedding_mb: int,
        rank: int = 8,
    ):
        # VRAM accounting -------------------------------------------------
        self.store = 0.0
        self.reserve = float(reserve_mb)
        self.capacity = float(budget_mb)

        # Low‑rank linear model -------------------------------------------
        # W = U @ V.T where U,V ∈ ℝ^{budget × rank}
        self.rank = rank
        self.U = np.random.randn(budget_mb, rank) * 0.01
        self.V = np.random.randn(budget_mb, rank) * 0.01

        # Bandit state ----------------------------------------------------
        self.bandit_action = BanditAction(action="", propensity=0.0, expected_reward=0.0, confidence_bound=0.0)

        # Cached node representations (for similarity) --------------------
        self.node_phash: Dict[Node, int] = {}
        self.node_minhash: Dict[Node, List[int]] = {}

    # ------------------------------------------------------------------
    # Store handling
    # ------------------------------------------------------------------
    def update_store(self, inflow: float, outflow: float, dt: float) -> None:
        self.store = compute_store(self.store, inflow, outflow, dt)

    # ------------------------------------------------------------------
    # Linear model utilities
    # ------------------------------------------------------------------
    def _predict(self, propensity_vec: np.ndarray) -> np.ndarray:
        """Predict VRAM allocation using the low‑rank factorisation."""
        # propensity_vec is 1‑D, shape (budget,)
        return (self.U @ self.V.T) @ propensity_vec

    def _gradient(self, estimate: np.ndarray, target: float, propensity_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gradient of squared error (estimate - target)^2 w.r.t. U and V.
        Returns (grad_U, grad_V) with the same shapes as U and V.
        """
        error = estimate - target
        # dL/dW = 2 * error * propensity_vec
        dL_dW = 2.0 * error[:, None] * propensity_vec[None, :]  # shape (budget, budget)

        # Using W = U @ V.T, the gradients are:
        grad_U = dL_dW @ self.V
        grad_V = dL_dW.T @ self.U
        return grad_U, grad_V

    def update_linear_weight(self, eta: float, grad_U: np.ndarray, grad_V: np.ndarray) -> None:
        self.U = update_linear_weight(self.U, eta, grad_U)
        self.V = update_linear_weight(self.V, eta, grad_V)

    # ------------------------------------------------------------------
    # Similarity handling (fusion point)
    # ------------------------------------------------------------------
    def _ensure_node_representation(self, node: Node, values: List[float], tokens: Set[str]) -> None:
        """Cache perceptual hash and MinHash for a node if not already present."""
        if node not in self.node_phash:
            self.node_phash[node] = compute_phash(values)
        if node not in self.node_minhash:
            self.node_minhash[node] = minhash_signature(tokens)

    def fused_similarity(self, node_a: Node, node_b: Node) -> float:
        """Public API – similarity between two nodes using the fused metric."""
        phash_a = self.node_phash.get(node_a, 0)
        phash_b = self.node_phash.get(node_b, 0)
        sig_a = self.node_minhash.get(node_a, [])
        sig_b = self.node_minhash.get(node_b, [])
        return node_similarity(phash_a, phash_b, sig_a, sig_b)

    # ------------------------------------------------------------------
    # Bandit interaction
    # ------------------------------------------------------------------
    def take_action(self, node: Node, values: List[float], tokens: Set[str]) -> None:
        """
        Simulate a bandit decision for ``node``.
        The node's similarity to a (hard‑coded) leader node biases the propensity.
        """
        # Ensure we have a representation for the current node
        self._ensure_node_representation(node, values, tokens)

        # Leader node is a fixed placeholder – in a real system this would be dynamic
        leader_node = "LEADER"
        if leader_node not in self.node_phash:
            # Dummy leader representation (could be loaded from a config)
            self._ensure_node_representation(leader_node, [0.5] * len(values), {"leader"})

        # Base propensity is uniform; we tilt it with similarity
        base_propensity = random.uniform(0.0, 1.0)
        similarity = self.fused_similarity(node, leader_node)
        self.bandit_action.propensity = base_propensity * (0.5 + 0.5 * similarity)  # keep in [0,1]

        # Mock reward and confidence – in practice these would come from downstream metrics
        self.bandit_action.expected_reward = random.uniform(0.0, 1.0)
        self.bandit_action.confidence_bound = random.uniform(0.0, 0.2)
        self.bandit_action.action = f"allocate_{node}"
        self.bandit_action.node_id = node

    def get_vram_slot_plan(self) -> VramSlotPlan:
        """
        Produce a plan based on the current linear model and the most recent bandit action.
        """
        # Build a propensity vector: high values where the bandit suggests allocation
        prop_vec = np.zeros(self.U.shape[0])
        if self.bandit_action.node_id is not None:
            # Map node identifier to an index – for demo we hash modulo budget
            idx = hash(self.bandit_action.node_id) % prop_vec.size
            prop_vec[idx] = self.bandit_action.propensity

        # Predict allocation
        estimate = self._predict(prop_vec)

        # Gradient step towards expected reward (treated as a scalar target)
        target = self.bandit_action.expected_reward * self.capacity  # scale to VRAM units
        grad_U, grad_V = self._gradient(estimate, target, prop_vec)

        # Learning rate schedule – simple decay
        eta = 1e-3 / (1.0 + 0.01 * np.linalg.norm(grad_U) + 0.01 * np.linalg.norm(grad_V))

        self.update_linear_weight(eta, grad_U, grad_V)

        # Summarise the plan
        total_alloc = int(np.clip(np.sum(estimate), 0, self.capacity - self.reserve))
        plan = VramSlotPlan(
            plan_id=str(random.randint(1_000_000, 9_999_999)),
            description="Hybrid VRAM allocation plan",
            action=self.bandit_action.action,
            total_vram=total_alloc,
            status="pending",
            metadata={
                "propensity": self.bandit_action.propensity,
                "expected_reward": self.bandit_action.expected_reward,
                "similarity_to_leader": self.fused_similarity(self.bandit_action.node_id, "LEADER")
                if self.bandit_action.node_id is not None else None,
                "learning_rate": eta,
            },
        )
        return plan


# ----------------------------------------------------------------------
# Hybrid interface (public façade)
# ----------------------------------------------------------------------
class HybridInterface:
    def __init__(
        self,
        budget_mb: int,
        reserve_mb: int,
        base_model_mb: int,
        adapter_mb: int,
        embedding_mb: int,
        rank: int = 8,
    ):
        self.scheduler = HybridVrBanditScheduler(
            budget_mb, reserve_mb, base_model_mb, adapter_mb, embedding_mb, rank=rank
        )

    def propose_allocation(self, node: Node, values: List[float], tokens: Set[str]) -> VramSlotPlan:
        """
        High‑level entry point:
          1. Update bandit state for ``node``.
          2. Return a VRAM allocation plan.
        """
        self.scheduler.take_action(node, values, tokens)
        return self.scheduler.get_vram_slot_plan()

    def update_vram_store(self, inflow: float, outflow: float, dt: float) -> None:
        self.scheduler.update_store(inflow, outflow, dt)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    interface = HybridInterface(
        budget_mb=4096,
        reserve_mb=768,
        base_model_mb=1800,
        adapter_mb=128,
        embedding_mb=384,
        rank=8,
    )
    # Example node data
    node_id = "node_42"
    values = [random.random() for _ in range(100)]
    tokens = {f"tok_{i}" for i in range(20)}

    plan = interface.propose_allocation(node_id, values, tokens)
    print(plan.as_dict())

    # Simulate VRAM dynamics
    interface.update_vram_store(inflow=200.0, outflow=150.0, dt=0.5)
    print(f"Current store: {interface.scheduler.store:.2f} MB")