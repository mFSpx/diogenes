# DARWIN HAMMER — match 3817, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1705_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_vorono_m2100_s0.py (gen5)
# born: 2026-05-29T23:51:49Z

"""
HYBRID Algorithm: Pheromone-Voronoi Label Foundry Fusion

This module mathematically fuses the core topologies of two mathematical algorithms:
1. `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1705_s2.py` (Parent A): pheromone
   tracking and Bayesian edge weights.
2. `hybrid_hybrid_hybrid_hybrid_hybrid_vorono_m2100_s0.py` (Parent B): Voronoi
   partitioning and hybrid endpoint circuit breakers with serpentina self-righting
   and weak supervision labeling primitives.

The mathematical bridge between these structures is the concept of "recovery priority,"
which is used to determine the likelihood of an endpoint recovering from a failure.
This recovery priority is calculated based on the morphology of the endpoint, and
this value is then used to adjust the pheromone signal decay rate to ensure robust
labeling and endpoint management.

In the context of labeling, the recovery priority is used to weight the confidence
of the aggregated label, giving a hybrid probabilistic label. This is achieved by
multiplying the confidence of the aggregated label by the recovery priority.

We will denote the confidence of the aggregated label as `confidence_A` and the
recovery priority as `m.recovery_priority()`. The hybrid probabilistic label is
given by:

    confidence_hybrid = confidence_A × m.recovery_priority()

The three core functions below demonstrate this fusion:
1. `document_signature` – builds the lead-lag signature from label votes.
2. `hybrid_aggregate_labels` – aggregates labels and rescales confidence with
   the recovery priority.
3. `hybrid_select_action` – selects a bandit action using the signature as
   contextual features.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (shared)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Vector = np.ndarray


@dataclass(frozen=True)
class Document:
    """Identifier and embedding vector."""
    id: str
    embedding: Vector


@dataclass(frozen=True)
class Node:
    """Graph node containing spatial, semantic and label information."""
    id: str
    point: Point
    document: Document
    label: str


@dataclass
class Edge:
    """Edge between two nodes with Bayesian parameters."""
    src: Node
    dst: Node
    prior: float          # Bayesian prior
    likelihood: float     # Bayesian likelihood
    false_positive: float # Bayesian false‑positive rate


# ----------------------------------------------------------------------
# Pheromone subsystem (from Algorithm A)
# ----------------------------------------------------------------------
class HybridPheromoneSystem:
    """Tracks decaying pheromone signals for arbitrary surface keys."""
    def __init__(self):
        self.pheromones: Dict[str, Dict] = {}

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        """Create or update a pheromone entry, returning the current signal value."""
        now = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
        else:
            entry = self.pheromones[surface_key]
            elapsed = (now - entry["created_time"]).total_seconds()
            decay_factor = 0.5 ** (elapsed / entry["half_life_seconds"])
            entry["signal_value"] *= decay_factor
            # Introduce a non-linear pheromone update
            entry["signal_value"] *= math.exp(-elapsed / half_life_seconds)
            return entry["signal_value"]


# ----------------------------------------------------------------------
# Voronoi partitioning and labeling structures (from Algorithm B)
# ----------------------------------------------------------------------
class VoronoiPartitioning:
    """Voronoi partitioning algorithm."""
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def calculate_distance(self, node1: Node, node2: Node) -> float:
        """Calculate the distance between two nodes."""
        return math.sqrt((node1.point[0] - node2.point[0]) ** 2 + (node1.point[1] - node2.point[1]) ** 2)

    def update_recovery_priority(self, node: Node, recovery_priority: float):
        """Update the recovery priority of a node."""
        node.label = f"recovery_priority={recovery_priority:.2f}"

    def calculate_recovery_priority(self, node: Node) -> float:
        """Calculate the recovery priority of a node."""
        # Calculate the recovery priority based on the morphology of the endpoint
        return 1.0 / (1.0 + math.exp(-node.point[0] / 10.0))


# ----------------------------------------------------------------------
# Hybrid labeling and pheromone structures
# ----------------------------------------------------------------------
class HybridLabeling:
    """Hybrid labeling algorithm."""
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def document_signature(self, labels: List[LabelingFunctionResult]) -> Vector:
        """Build the lead-lag signature from label votes."""
        signature = np.zeros(len(labels))
        for i, label in enumerate(labels):
            signature[i] = label.label
        return signature

    def hybrid_aggregate_labels(self, labels: List[LabelingFunctionResult]) -> ProbabilisticLabel:
        """Aggregate labels and rescale confidence with the recovery priority."""
        confidence_A = np.mean([label.label for label in labels])
        recovery_priority = self.nodes[0].label.split("=")[1]
        recovery_priority = float(recovery_priority)
        confidence_hybrid = confidence_A * recovery_priority
        return ProbabilisticLabel(confidence_hybrid=confidence_hybrid)

    def hybrid_select_action(self, signature: Vector) -> str:
        """Select a bandit action using the signature as contextual features."""
        # Use the signature to select a bandit action
        action = "action=" + str(np.argmax(signature))
        return action


# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
class HybridAlgorithm:
    """Hybrid algorithm that fuses pheromone tracking and Voronoi partitioning."""
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.pheromone_system = HybridPheromoneSystem()
        self.voronoi_partitioning = VoronoiPartitioning(nodes)
        self.hybrid_labeling = HybridLabeling(nodes)

    def run(self):
        """Run the hybrid algorithm."""
        labels = []
        for node in self.nodes:
            labels.append(LabelingFunctionResult(lf_name="lf1", doc_id=node.document.id, label=node.label))
        signature = self.hybrid_labeling.document_signature(labels)
        confidence_hybrid = self.hybrid_labeling.hybrid_aggregate_labels(labels).confidence_hybrid
        action = self.hybrid_labeling.hybrid_select_action(signature)
        print(f"Signature: {signature}")
        print(f"Confidence Hybrid: {confidence_hybrid:.2f}")
        print(f"Action: {action}")

        # Update the recovery priority of each node
        for node in self.nodes:
            recovery_priority = self.voronoi_partitioning.calculate_recovery_priority(node)
            self.voronoi_partitioning.update_recovery_priority(node, recovery_priority)

        # Update the pheromone signals
        for node in self.nodes:
            key = node.id
            signal_kind = "confidence"
            signal_value = confidence_hybrid
            half_life_seconds = 60.0  # 1 minute
            self.pheromone_system.calculate_pheromone_signal(key, signal_kind, signal_value, half_life_seconds)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    nodes = [
        Node(id="node1", point=(1.0, 2.0), document=Document(id="doc1", embedding=np.array([1.0, 2.0])), label="label1"),
        Node(id="node2", point=(3.0, 4.0), document=Document(id="doc2", embedding=np.array([3.0, 4.0])), label="label2"),
    ]
    hybrid_algorithm = HybridAlgorithm(nodes)
    hybrid_algorithm.run()