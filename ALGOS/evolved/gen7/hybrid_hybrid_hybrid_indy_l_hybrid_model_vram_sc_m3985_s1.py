# DARWIN HAMMER — match 3985, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_indy_learning_hybrid_hybrid_rbf_su_m1638_s3.py (gen6)
# parent_b: hybrid_model_vram_scheduler_ttt_linear_m11_s3.py (gen1)
# born: 2026-05-29T23:52:53Z

"""
Hybrid Algorithm: Fusing Ollivier-Ricci Curvature with GPU Memory Scheduling

This module combines the mathematical structures of two parent algorithms:
1. hybrid_hybrid_indy_learning_hybrid_hybrid_rbf_su_m1638_s3.py (Ollivier-Ricci Curvature)
2. hybrid_model_vram_scheduler_ttt_linear_m11_s3.py (GPU Memory Scheduling)

The mathematical bridge between the two lies in the use of pairwise distance matrices.
The Ollivier-Ricci curvature calculation relies on pairwise distances between region vectors,
while the GPU memory scheduling algorithm uses a similar concept to allocate memory.

The hybrid algorithm integrates these concepts by using the pairwise distance matrix
to inform the GPU memory allocation decisions.

"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple
import math
import json
import hashlib
import os
import sys

# ----------------------------------------------------------------------
# Global constants and utilities
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TERMS = (
    "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "ACTION", "EVENT", "TIME", "EVIDENCE",
    "CLAIM", "HYPOTHESIS", "SIGNAL", "PATTERN", "TOOL", "ALGORITHM", "BOOK",
    "SOURCE", "LEAD", "LOCATION", "LAW", "RULE",
)

def sha256_json(value: any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()

# ----------------------------------------------------------------------
# Geometric algebra utilities
# ----------------------------------------------------------------------
def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.dot(a, b)
    wedge = np.cross(a, b)
    return np.concatenate((np.atleast_1d(dot), wedge))

# ----------------------------------------------------------------------
# Ollivier-Ricci curvature (enhanced)
# ----------------------------------------------------------------------
def pairwise_distances(vectors: np.ndarray) -> np.ndarray:
    diff = vectors[:, None, :] - vectors[None, :, :]
    return np.linalg.norm(diff, axis=-1)

def ollivier_ricci_curvature(
    region_vectors: List[np.ndarray],
    region_centroids: List[np.ndarray],
    alpha: float = 0.5,
) -> float:
    distances = pairwise_distances(region_vectors)
    centroids_distances = pairwise_distances(region_centroids)
    curvature = 1 - (distances / centroids_distances)
    return np.mean(curvature)

# ----------------------------------------------------------------------
# GPU Memory Scheduling
# ----------------------------------------------------------------------
@dataclass
class GPU:
    index: int
    name: str
    total_memory: int
    used_memory: int
    free_memory: int

def query_gpu() -> List[GPU]:
    gpus = []
    gpu_info = os.popen("nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits").readlines()
    for line in gpu_info:
        parts = line.strip().split(",")
        gpu = GPU(
            index=int(parts[0]),
            name=parts[1],
            total_memory=int(parts[2]),
            used_memory=int(parts[3]),
            free_memory=int(parts[4]),
        )
        gpus.append(gpu)
    return gpus

def allocate_memory(gpus: List[GPU], memory需求: int) -> Tuple[int, int]:
    gpus.sort(key=lambda gpu: gpu.free_memory, reverse=True)
    for gpu in gpus:
        if gpu.free_memory >= memory需求:
            gpu.used_memory += memory需求
            gpu.free_memory -= memory需求
            return gpu.index, gpu.free_memory
    return -1, -1

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def hybrid_ollivier_ricci_curvature(gpus: List[GPU], region_vectors: List[np.ndarray], region_centroids: List[np.ndarray]) -> Tuple[float, int, int]:
    curvature = ollivier_ricci_curvature(region_vectors, region_centroids)
    gpu_index, free_memory = allocate_memory(gpus, 1024)  # Allocate 1GB
    return curvature, gpu_index, free_memory

def hybrid_memory_allocation(region_vectors: List[np.ndarray], region_centroids: List[np.ndarray], memory需求: int) -> Tuple[float, int, int]:
    gpus = query_gpu()
    curvature, gpu_index, free_memory = hybrid_ollivier_ricci_curvature(gpus, region_vectors, region_centroids)
    return curvature, gpu_index, free_memory

def hybrid_pairwise_distance(region_vectors: List[np.ndarray]) -> np.ndarray:
    distances = pairwise_distances(np.array(region_vectors))
    return distances

if __name__ == "__main__":
    region_vectors = [np.random.rand(10) for _ in range(5)]
    region_centroids = [np.random.rand(10) for _ in range(5)]
    gpus = query_gpu()
    curvature, gpu_index, free_memory = hybrid_ollivier_ricci_curvature(gpus, region_vectors, region_centroids)
    print(f"Curvature: {curvature}, GPU Index: {gpu_index}, Free Memory: {free_memory}")
    distances = hybrid_pairwise_distance(region_vectors)
    print(distances)