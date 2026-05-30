# DARWIN HAMMER — match 5611, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_omni_chaotic__m1545_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m2613_s3.py (gen5)
# born: 2026-05-30T00:03:23Z

"""
Hybrid Algorithm Fusion:
- Parent A: Hybrid Allocation‑LTC & Fractional‑Memory Tree Cost with Chaotic Omni‑Engine + JEPA.
- Parent B: Hybrid Bandit‑Sketch Privacy Store + Hybrid Regret Endpoint (MinHash similarity).

Mathematical Bridge:
Both parents rely on a *feature‑weight inner product* to produce a scalar score:
    • A uses a latent vector **z** (size D) and a weight matrix **W** in the JEPA energy term.
    • B uses a feature vector **φ** (derived from regex‑based tokenisation) and a weight vector **θ** for optimistic bandit reward.
We unify them by treating the JEPA latent **z** as the bandit feature vector **φ**, and by feeding the
MinHash similarity (from B) as an additional regularisation term into the JEPA energy function (from A).
The resulting hybrid energy for a node *i* is

    E_i = ‖y_i – ŷ_i‖² + λ₁·⟨z_i, θ⟩ + λ₂·S_i

where
    – y_i , ŷ_i are true/predicted embeddings,
    – ⟨z_i, θ⟩ is the bandit‑style inner‑product reward,
    – S_i is the MinHash similarity between the node’s token set and a privacy‑risk reference set,
    – λ₁, λ₂ are scalar mixing coefficients.

The LTC temporal factor τ(day) multiplies the contribution of each node’s reward,
and a Caputo‑type fractional sum (α∈(0,1]) aggregates tree‑cost penalties across the graph.

The code below implements this fused mathematics in a self‑contained, executable module.
"""

import numpy as np
import math
import random
import re
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple

# ----------------------------------------------------------------------
# Helper utilities (shared by both parent lineages)
# ----------------------------------------------------------------------
def caputo_weighted_sum(values: np.ndarray, alpha: float) -> float:
    """Fractional (Caputo) weighted sum: Σ v_i * (i+1)^{alpha-1}."""
    idx = np.arange(1, values.size + 1, dtype=np.float64)
    weights = idx ** (alpha - 1.0)
    return float(np.sum(values * weights))

def minhash_signature(token_set: set, num_perm: int = 128) -> np.ndarray:
    """Simple MinHash signature using deterministic 64‑bit hashing."""
    max_hash = (1 << 64) - 1
    sig = np.full(num_perm, max_hash, dtype=np.uint64)
    for token in token_set:
        token_bytes = token.encode('utf-8')
        h = int(hashlib.sha256(token_bytes).hexdigest(), 16)
        for i in range(num_perm):
            combined = (h ^ (i * 0x9e3779b97f4a7c15)) & max_hash
            if combined < sig[i]:
                sig[i] = combined
    return sig

def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    return float(np.mean(sig1 == sig2))

def count_min_sketch_update(sketch: np.ndarray, item: str, hash_fns: List[Callable[[str], int]]) -> None:
    """Update a Count‑Min Sketch in‑place."""
    for i, fn in enumerate(hash_fns):
        idx = fn(item) % sketch.shape[1]
        sketch[i, idx] += 1

def count_min_sketch_query(sketch: np.ndarray, item: str, hash_fns: List[Callable[[str], int]]) -> int:
    """Query the minimum count for an item."""
    mins = []
    for i, fn in enumerate(hash_fns):
        idx = fn(item) % sketch.shape[1]
        mins.append(sketch[i, idx])
    return int(min(mins))

# ----------------------------------------------------------------------
# Core hybrid structures
# ----------------------------------------------------------------------
@dataclass
class Node:
    uid: str
    latent_z: np.ndarray          # latent vector (JEPA / bandit feature)
    tokens: set                   # token set for MinHash similarity
    true_embedding: np.ndarray    # ground‑truth vector y_i
    pred_embedding: np.ndarray    # predicted vector ŷ_i (from JEPA encoder)
    time_constant: float = 1.0   # τ_sys(t) from LTC

class HybridGraph:
    """Graph of active nodes with latent variables and token sets."""
    def __init__(self, root_uuid: str, latent_dim: int = 16):
        self.root_uuid = root_uuid
        self.latent_dim = latent_dim
        self.nodes: Dict[str, Node] = {}
        self._init_root()

    def _init_root(self):
        z = np.random.randn(self.latent_dim)
        tokens = {"root", "init"}
        y = np.random.randn(self.latent_dim)
        ŷ = np.random.randn(self.latent_dim)
        self.nodes[self.root_uuid] = Node(
            uid=self.root_uuid,
            latent_z=z,
            tokens=tokens,
            true_embedding=y,
            pred_embedding=ŷ,
            time_constant=1.0,
        )

    def add_node(self, uid: str, tokens: set):
        z = np.random.randn(self.latent_dim)
        y = np.random.randn(self.latent_dim)
        ŷ = np.random.randn(self.latent_dim)
        self.nodes[uid] = Node(
            uid=uid,
            latent_z=z,
            tokens=tokens,
            true_embedding=y,
            pred_embedding=ŷ,
            time_constant=1.0,
        )

    def get_latent_matrix(self) -> np.ndarray:
        """Stack latent vectors of all nodes (shape N×D)."""
        return np.stack([n.latent_z for n in self.nodes.values()], axis=0)

    def update_time_constants(self, day_of_week: int, gating_fn: Callable[[int], float]) -> None:
        τ = gating_fn(day_of_week)
        for n in self.nodes.values():
            n.time_constant = τ

# ----------------------------------------------------------------------
# Hybrid operations (three required functions)
# ----------------------------------------------------------------------
def hybrid_ltc(day_of_week: int, gating_fn: Callable[[int], float]) -> float:
    """
    Compute the LTC effective time constant τ_sys(t) for a given day.
    The gating function encodes learned temporal modulation.
    """
    base = 0.8  # baseline decay factor
    τ = base * gating_fn(day_of_week)
    return float(τ)

def bandit_optimistic_reward(
    latent_z: np.ndarray,
    weight_vec: np.ndarray,
    confidence: float,
    similarity: float,
    λ_sim: float = 0.3,
) -> float:
    """
    Optimistic bandit reward using inner‑product (⟨z,θ⟩) plus a similarity bonus.
    """
    base_reward = float(np.dot(latent_z, weight_vec))
    optimistic = base_reward + confidence
    # similarity term acts like a privacy‑risk adjustment
    return optimistic + λ_sim * similarity

def hybrid_step(
    graph: HybridGraph,
    day_of_week: int,
    gating_fn: Callable[[int], float],
    weight_vec: np.ndarray,
    reference_tokens: set,
    cms_sketch: np.ndarray,
    cms_hash_fns: List[Callable[[str], int]],
    α_frac: float = 0.7,
    λ_ltc: float = 0.5,
    λ_energy: float = 0.2,
) -> Tuple[float, Dict[str, float]]:
    """
    Perform one hybrid iteration:
      1. Update LTC time constants.
      2. Compute MinHash similarity for each node.
      3. Estimate optimistic bandit reward.
      4. Form JEPA‑style energy term.
      5. Aggregate a fractional tree‑cost penalty via Caputo sum.
      6. Update a Count‑Min Sketch with node identifiers (privacy store).

    Returns:
        total_energy: aggregated scalar energy for the graph.
        reward_dict: mapping node uid → optimistic reward.
    """
    # 1. LTC update
    τ = hybrid_ltc(day_of_week, gating_fn)
    graph.update_time_constants(day_of_week, gating_fn)

    # Pre‑compute reference MinHash signature once
    ref_sig = minhash_signature(reference_tokens)

    rewards = {}
    energies = []
    tree_costs = []

    for uid, node in graph.nodes.items():
        # 2. MinHash similarity
        node_sig = minhash_signature(node.tokens)
        sim = minhash_similarity(node_sig, ref_sig)

        # 3. Bandit optimistic reward (confidence term proportional to τ)
        confidence = λ_ltc * τ
        reward = bandit_optimistic_reward(
            node.latent_z, weight_vec, confidence, sim
        )
        rewards[uid] = reward

        # 4. JEPA‑style prediction error energy
        err = np.linalg.norm(node.true_embedding - node.pred_embedding) ** 2
        energy = err + λ_energy * reward + λ_energy * sim
        energies.append(energy)

        # 5. Tree‑cost contribution (use reward as a proxy cost)
        tree_costs.append(reward)

        # 6. Update privacy sketch
        count_min_sketch_update(cms_sketch, uid, cms_hash_fns)

    total_energy = float(np.sum(energies))

    # Fractional aggregation of tree costs (Caputo weighted sum)
    cost_agg = caputo_weighted_sum(np.array(tree_costs), alpha=α_frac)

    # Combine both scalar aggregates into a final metric
    final_metric = total_energy + λ_ltc * cost_agg

    return final_metric, rewards

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # 1. Initialise hybrid graph
    hg = HybridGraph(root_uuid="root-node", latent_dim=16)
    hg.add_node("node-A", {"evidence", "plan", "risk"})
    hg.add_node("node-B", {"delay", "support", "outcome"})

    # 2. Define gating function (simple sinusoidal modulation)
    def gating(day: int) -> float:
        return 1.0 + 0.3 * math.sin(2 * math.pi * day / 7)

    # 3. Bandit weight vector (random but fixed)
    θ = np.random.randn(16)

    # 4. Reference token set for privacy similarity
    reference = {"evidence", "plan", "risk", "support"}

    # 5. Initialise Count‑Min Sketch (3 hash rows, 2³⁶ columns)
    sketch_rows = 3
    sketch_cols = 2 ** 12
    cms = np.zeros((sketch_rows, sketch_cols), dtype=np.int32)

    # Simple deterministic hash functions using built‑in hash
    def make_hash(seed: int) -> Callable[[str], int]:
        def _h(s: str) -> int:
            return hash((seed, s))
        return _h

    cms_hashes = [make_hash(i) for i in range(sketch_rows)]

    # 6. Run a single hybrid step for day 3 (Wednesday)
    metric, node_rewards = hybrid_step(
        graph=hg,
        day_of_week=3,
        gating_fn=gating,
        weight_vec=θ,
        reference_tokens=reference,
        cms_sketch=cms,
        cms_hash_fns=cms_hashes,
        α_frac=0.6,
    )

    print(f"Hybrid metric (energy + fractional cost): {metric:.4f}")
    for uid, rew in node_rewards.items():
        print(f"  Node {uid} → optimistic reward = {rew:.4f}")

    # Verify sketch query (should be ≥1 after update)
    for uid in hg.nodes:
        cnt = count_min_sketch_query(cms, uid, cms_hashes)
        assert cnt >= 1, f"Sketch count for {uid} unexpectedly low."
    print("Smoke test completed successfully.")