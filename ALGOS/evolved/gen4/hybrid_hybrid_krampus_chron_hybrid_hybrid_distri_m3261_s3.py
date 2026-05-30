# DARWIN HAMMER — match 3261, survivor 3
# gen: 4
# parent_a: hybrid_krampus_chrono_hybrid_possum_filter_m34_s0.py (gen3)
# parent_b: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s3.py (gen2)
# born: 2026-05-29T23:50:18Z

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""
    confidence: float = 0.0


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat,lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6371000 * math.sqrt(h)


def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Max‑plus (tropical) matrix multiplication.
    (A ⊗ B)_ij = max_k (A_ik + B_kj)
    """
    # Ensure column vector shape for B
    if B.ndim == 1:
        B = B[:, None]
    # Broadcast addition then max over k
    # Result shape: (A_rows, B_cols)
    result = np.max(A[:, :, None] + B[None, :, :], axis=1)
    return result.squeeze()


def tropical_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element‑wise tropical addition (max)."""
    return np.maximum(x, y)


def hoeffding_bound(n: int, delta: float, epsilon: float) -> float:
    """Hoeffding bound for a Bernoulli‑like variable."""
    if n <= 0:
        return float("inf")
    return math.sqrt(
        (math.log(2.0 / delta) + (n - 1) * math.log(1.0 / (1.0 - epsilon))) / (2.0 * n)
    )


def laplace_noise(scale: float) -> float:
    """Sample Laplace(0, scale)."""
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


def compute_resource_vectors(entities: List[Entity]) -> np.ndarray:
    """
    Build a resource vector e_i for each entity:
    [distance_to_centroid, signature_flag, confidence]
    """
    # centroid of all lat/lon pairs
    lats = np.array([e.lat for e in entities])
    lons = np.array([e.lon for e in entities])
    centroid = (float(np.mean(lats)), float(np.mean(lons)))

    vectors = []
    for e in entities:
        dist = haversine_m((e.lat, e.lon), centroid)
        sig = 1.0 if e.address_signature else 0.0
        vectors.append([dist, sig, e.confidence])
    return np.array(vectors, dtype=float)


def hybrid_algorithm(
    entities: List[Entity],
    adjacency_matrix: np.ndarray,
    temperature: float,
    delta: float,
    epsilon: float,
    dp_scale: float = 1.0,
) -> List[Entity]:
    """
    Integrated hybrid algorithm with deeper mathematical coupling.

    Steps
    -----
    1. Build resource vectors **eᵢ**.
    2. Propagate broadcast strength **b** using true tropical matrix multiplication.
    3. Fuse **eᵢ** and **b** with tropical addition to obtain a combined score.
    4. Add Laplace noise for differential‑privacy protection.
    5. Apply a per‑entity Hoeffding bound (based on local degree) to select candidates.
    6. Accept candidates via a simulated‑annealing style probability that respects the bound.
    """
    if len(entities) == 0:
        return []

    # 1. Resource vectors
    e = compute_resource_vectors(entities)                     # shape (N, 3)

    # 2. Broadcast strength vector b (tropical power iteration)
    b = np.ones(len(entities), dtype=float)
    for _ in range(10):
        b = tropical_matmul(adjacency_matrix, b)

    # 3. Fuse: we treat the first component of e (distance) as a scalar that can be
    #    added tropically to b; the other two components stay as auxiliary information.
    #    The fused scalar score is max(e[:,0], b).
    fused_scalar = tropical_add(e[:, 0], b)

    # 4. Differential privacy noise
    noisy_score = fused_scalar + np.vectorize(laplace_noise)(dp_scale)

    # 5. Candidate selection with per‑entity Hoeffding bound.
    #    Use the out‑degree (or in‑degree, symmetric here) as the sample size n_i.
    degrees = adjacency_matrix.sum(axis=1).astype(int)
    candidate_mask = np.array(
        [
            noisy_score[i] > hoeffding_bound(max(1, degrees[i]), delta, epsilon)
            for i in range(len(entities))
        ]
    )
    candidate_indices = np.where(candidate_mask)[0]

    # 6. Simulated‑annealing acceptance.
    accepted = []
    for idx in candidate_indices:
        entity = entities[idx]
        # delta_score measures how far the noisy score exceeds the bound.
        bound_i = hoeffding_bound(max(1, degrees[idx]), delta, epsilon)
        delta_score = noisy_score[idx] - bound_i
        # Higher excess → higher acceptance probability, tempered by temperature.
        prob = 1.0 - math.exp(-max(delta_score, 0.0) / max(temperature, 1e-9))
        if random.random() < prob:
            accepted.append(entity)

    # Return leaders sorted by descending confidence (or any other meaningful metric)
    accepted.sort(key=lambda e: e.confidence, reverse=True)
    return accepted


if __name__ == "__main__":
    # Example usage
    entities = [
        Entity("1", 37.7749, -122.4194, "A", confidence=0.8, address_signature="sig1"),
        Entity("2", 34.0522, -118.2437, "B", confidence=0.9, address_signature=""),
        Entity("3", 40.7128, -74.0060, "C", confidence=0.85, address_signature="sig3"),
    ]
    # Simple undirected adjacency (symmetric)
    adjacency_matrix = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=float,
    )
    temperature = 0.05
    delta = 0.01
    epsilon = 0.1
    dp_scale = 0.5

    leaders = hybrid_algorithm(
        entities,
        adjacency_matrix,
        temperature,
        delta,
        epsilon,
        dp_scale=dp_scale,
    )
    for l in leaders:
        print(l)