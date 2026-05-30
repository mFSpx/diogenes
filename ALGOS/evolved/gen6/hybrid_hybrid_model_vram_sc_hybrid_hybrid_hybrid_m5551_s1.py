# DARWIN HAMMER — match 5551, survivor 1
# gen: 6
# parent_a: hybrid_model_vram_scheduler_hybrid_hybrid_bandit_m188_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1512_s6.py (gen5)
# born: 2026-05-30T00:02:51Z

"""
Hybrid VRAM-Bandit Distributed Scheduler

This module fuses the 'Hybrid VRAM-Bandit Scheduler' with the 'Hybrid Distributed Leader' algorithm.
The mathematical bridge is the application of node similarity metrics to the bandit-produced propensity and confidence_bound.
This allows the distributed scheduler to modulate the learning rate of the VRAM cost estimation matrix based on the similarity between nodes.
"""

import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path(".")
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768
DEFAULT_BASE_MODEL_MB = 1800
DEFAULT_ADAPTER_MB = 128
DEFAULT_EMBEDDING_MB = 384

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class Node:
    id: str
    values: List[float]
    tokens: Set[str]

# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
class HybridVrBanditDistributedScheduler:
    def __init__(self, num_nodes: int, num_actions: int):
        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.nodes = [Node(f"node_{i}", [], set()) for i in range(num_nodes)]
        self.bandit_actions = [BanditAction(f"action_{i}", 0.0, 0.0, 0.0, "algorithm") for i in range(num_actions)]
        self.vram_slot_plans = []

    def compute_phash(self, values: List[float]) -> int:
        if not values:
            return 0
        avg = sum(values) / len(values)
        bits = 0
        for v in values[:64]:
            bits = (bits << 1) | int(v >= avg)
        return bits

    def hamming_distance(self, a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    def minhash_signature(self, tokens: Set[str], num_hashes: int = 7) -> List[int]:
        signatures = []
        for seed in range(num_hashes):
            def hash_fn(x: str) -> int:
                return int(hashlib.md5((x + str(seed)).encode()).hexdigest(), 16)
            hashed = [hash_fn(tok) for tok in tokens]
            signatures.append(min(hashed) if hashed else 0)
        return signatures

    def jaccard_similarity(self, sig_a: List[int], sig_b: List[int]) -> float:
        if not sig_a or not sig_b:
            return 0.0
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)

    def node_similarity(self, phash_a: int, phash_b: int, sig_a: List[int], sig_b: List[int], weight_phash: float = 0.5) -> float:
        ham_sim = 1.0 - self.hamming_distance(phash_a, phash_b) / 64.0
        mh_sim = self.jaccard_similarity(sig_a, sig_b)
        return weight_phash * ham_sim + (1 - weight_phash) * mh_sim

    def update_node_values(self, node_id: str, new_values: List[float]):
        for node in self.nodes:
            if node.id == node_id:
                node.values = new_values

    def update_bandit_action(self, action_id: str, new_propensity: float, new_confidence_bound: float):
        for action in self.bandit_actions:
            if action.action_id == action_id:
                action.propensity = new_propensity
                action.confidence_bound = new_confidence_bound

    def estimate_vram_cost(self, node_id: str, action_id: str) -> float:
        node = next((node for node in self.nodes if node.id == node_id), None)
        action = next((action for action in self.bandit_actions if action.action_id == action_id), None)
        if node and action:
            phash_node = self.compute_phash(node.values)
            phash_action = self.compute_phash([action.propensity, action.confidence_bound])
            sig_node = self.minhash_signature(node.tokens)
            sig_action = self.minhash_signature({action.algorithm})
            similarity = self.node_similarity(phash_node, phash_action, sig_node, sig_action)
            return similarity * action.propensity
        return 0.0

    def generate_vram_slot_plan(self, node_id: str, action_id: str) -> VramSlotPlan:
        estimated_mb = int(self.estimate_vram_cost(node_id, action_id))
        return VramSlotPlan(node_id, "node", "allocate", estimated_mb, "estimated", {"node_id": node_id, "action_id": action_id})

    def add_vram_slot_plan(self, plan: VramSlotPlan):
        self.vram_slot_plans.append(plan)


if __name__ == "__main__":
    scheduler = HybridVrBanditDistributedScheduler(5, 3)
    scheduler.update_node_values("node_0", [1.0, 2.0, 3.0])
    scheduler.update_bandit_action("action_0", 0.5, 0.2)
    plan = scheduler.generate_vram_slot_plan("node_0", "action_0")
    scheduler.add_vram_slot_plan(plan)
    print(plan.as_dict())