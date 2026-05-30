# DARWIN HAMMER — match 4343, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s3.py (gen3)
# parent_b: hybrid_gini_coefficient_hybrid_hybrid_rbf_su_m344_s0.py (gen4)
# born: 2026-05-29T23:54:59Z

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Hashable, Sequence, List, Dict, Set, Tuple, Iterable

# ----------------------------------------------------------------------
# Module docstring
# ----------------------------------------------------------------------

"""Hybrid VRAM-Privacy-Morphology Scheduler with Gini-coefficient Guided Hoeffding Tree

Parents:
- hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s2.py (risk → expected VRAM)
- hybrid_gini_coefficient_hybrid_hybrid_rbf_su_m344_s0.py (gini coefficient to guide hoeffding tree)

Mathematical bridge:
The hybrid scheduler integrates the weighted expected VRAM load from the first parent with the Gini coefficient guided Hoeffding tree decision process from the second parent. The reconstruction risk score from the first parent is used to compute the weighted expected VRAM load, while the Gini coefficient is used to guide the splitting process in the Hoeffding tree. The radial basis function (RBF) is used to model the similarity between nodes in the graph, which informs the decision to split in the Hoeffding tree.
"""

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a logical entity."""
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class ModelSpec:
    """Combined specification used by the hybrid scheduler."""
    tier: ModelTier
    morphology: Morphology
    unique_quasi_identifiers: int
    total_records: int


# ----------------------------------------------------------------------
# Utilities from parent A
# ----------------------------------------------------------------------

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Probability that a record can be re-identified."""
    if total_records < 1:
        return 0.0
    return unique_quasi_identifiers / total_records


def weighted_expected_vram_load(model_specs: List[ModelSpec], vram_budget: float) -> float:
    """Compute the weighted expected VRAM load."""
    total_load = 0.0
    for model_spec in model_specs:
        risk_score = reconstruction_risk_score(model_spec.unique_quasi_identifiers, model_spec.total_records)
        morphology_scaling_factor = compute_morphology_scaling_factor(model_spec.morphology)
        total_load += risk_score * morphology_scaling_factor * model_spec.tier.ram_mb
    return total_load


def compute_morphology_scaling_factor(morphology: Morphology) -> float:
    """Compute the morphology scaling factor from its geometric description."""
    return morphology.mass / (morphology.length * morphology.width * morphology.height)


# ----------------------------------------------------------------------
# Utilities from parent B
# ----------------------------------------------------------------------

def gini_coefficient(values: Iterable[float]) -> float:
    """Compute the Gini coefficient."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Compute the Gaussian function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute the Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Compute the pHash."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Compute the Hamming distance."""
    return (a ^ b).bit_count()


def similarity_matrix(features: Dict[Hashable, Sequence[float]]) -> Tuple[np.ndarray, List[Hashable]]:
    """Compute the similarity matrix."""
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(list(features[ni]))
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = compute_phash(list(features[nj]))
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
    return S, nodes


def gini_coefficient_guided_hoeffding_tree(features: Dict[Hashable, Sequence[float]]) -> List[Hashable]:
    """Guided Hoeffding tree using Gini coefficient."""
    S, nodes = similarity_matrix(features)
    gini_values = []
    for node in nodes:
        gini_value = gini_coefficient(features[node])
        gini_values.append(gini_value)
    best_node = nodes[np.argmax(gini_values)]
    return [best_node]


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------

def hybrid_scheduling(model_specs: List[ModelSpec], vram_budget: float) -> List[ModelSpec]:
    """Hybrid scheduling using weighted expected VRAM load and Gini coefficient guided Hoeffding tree."""
    if not model_specs:
        return []
    weighted_expected_load = weighted_expected_vram_load(model_specs, vram_budget)
    if weighted_expected_load <= vram_budget:
        return model_specs
    else:
        best_node = gini_coefficient_guided_hoeffding_tree({model_spec.tier.name: model_spec.tier.ram_mb for model_spec in model_specs})
        selected_model_specs = [model_spec for model_spec in model_specs if model_spec.tier.name == best_node[0]]
        return selected_model_specs


def hybrid_splitting(features: Dict[Hashable, Sequence[float]], vram_budget: float) -> List[Hashable]:
    """Hybrid splitting using Gini coefficient and weighted expected VRAM load."""
    S, nodes = similarity_matrix(features)
    gini_values = []
    for node in nodes:
        gini_value = gini_coefficient(features[node])
        gini_values.append(gini_value)
    best_node = nodes[np.argmax(gini_values)]
    selected_features = {node: features[node] for node in nodes if node != best_node}
    weighted_expected_load = weighted_expected_vram_load([ModelSpec(ModelTier("tier1", 1024, "high"), Morphology(1.0, 1.0, 1.0, 1.0), 100, 1000) for _ in selected_features], vram_budget)
    if weighted_expected_load <= vram_budget:
        return [best_node]
    else:
        return gini_coefficient_guided_hoeffding_tree(selected_features)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    model_specs = [ModelSpec(ModelTier("tier1", 1024, "high"), Morphology(1.0, 1.0, 1.0, 1.0), 100, 1000),
                   ModelSpec(ModelTier("tier2", 2048, "low"), Morphology(2.0, 2.0, 2.0, 2.0), 200, 2000)]
    vram_budget = 4096.0
    selected_model_specs = hybrid_scheduling(model_specs, vram_budget)
    print(selected_model_specs)