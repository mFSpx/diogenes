# DARWIN HAMMER — match 4442, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_model__hybrid_ssim_hybrid_d_m478_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_vorono_m1361_s4.py (gen4)
# born: 2026-05-29T23:55:51Z

import os
import sys
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def hybrid_curvature(
    graph: Dict[int, List[int]],
    node_features: Dict[int, np.ndarray],
    alpha: float,
    vram_weights: Dict[int, float],
    ssims: Dict[Tuple[int, int], float] = None,
) -> Dict[int, float]:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError('alpha must be in [0,1]')

    if ssims is None:
        ssims = {}
        for i, nbrs in graph.items():
            for u in nbrs:
                if (i, u) not in ssims and (u, i) not in ssims:
                    ssims[(i, u)] = ssim(node_features[i], node_features[u])

    curvature: Dict[int, float] = {}
    for i, nbrs in graph.items():
        w_i = vram_weights.get(i, 1.0)
        deg = len(nbrs) if nbrs else 1  
        self_term = alpha * w_i
        neighbor_term = 0.0
        for u in nbrs:
            s = ssims.get((i, u), ssims.get((u, i), 0.0))
            neighbor_term += s
        neighbor_term = (1 - alpha) * w_i * (neighbor_term / deg)
        curvature[i] = self_term + neighbor_term
    return curvature

def voronoi_partition(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
) -> List[int]:
    if not seeds:
        raise ValueError('seed list must not be empty')
    assignments = []
    for p in points:
        dists = [(p[0] - s[0]) ** 2 + (p[1] - s[1]) ** 2 for s in seeds]
        assignments.append(int(np.argmin(dists)))
    return assignments

def wta_selector(
    region_assignments: List[int],
    scores: List[float],
    top_k: int = 1,
) -> List[int]:
    agg: Dict[int, float] = {}
    for reg, sc in zip(region_assignments, scores):
        agg[reg] = agg.get(reg, 0.0) + sc
    sorted_regs = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    return [reg for reg, _ in sorted_regs[:top_k]]

def hybrid_step(
    graph: Dict[int, List[int]],
    node_features: Dict[int, np.ndarray],
    vram_weights: Dict[int, float],
    alpha: float,
    fisher_center: float,
    fisher_width: float,
    voronoi_seeds: List[Tuple[float, float]],
    top_k_regions: int = 1,
) -> Tuple[Dict[int, float], List[int], List[int]]:
    curvature = hybrid_curvature(graph, node_features, alpha, vram_weights)

    avg_ssim: Dict[int, float] = {}
    for i, nbrs in graph.items():
        if not nbrs:
            avg_ssim[i] = 0.0
            continue
        sims = [ssim(node_features[i], node_features[u]) for u in nbrs]
        avg_ssim[i] = sum(sims) / len(sims)

    fisher_vals = {
        i: fisher_score(avg_ssim[i], fisher_center, fisher_width)
        for i in avg_ssim
    }

    points = [(avg_ssim[i], fisher_vals[i]) for i in avg_ssim]
    region_assignments = voronoi_partition(points, voronoi_seeds)

    scores = [avg_ssim[i] * fisher_vals[i] for i in avg_ssim]
    winning_regions = wta_selector(region_assignments, scores, top_k_regions)

    return curvature, region_assignments, winning_regions

def improved_hybrid_step(
    graph: Dict[int, List[int]],
    node_features: Dict[int, np.ndarray],
    vram_weights: Dict[int, float],
    alpha: float,
    fisher_center: float,
    fisher_width: float,
    voronoi_seeds: List[Tuple[float, float]],
    top_k_regions: int = 1,
) -> Tuple[Dict[int, float], List[int], List[int]]:
    curvature = hybrid_curvature(graph, node_features, alpha, vram_weights)

    avg_ssim: Dict[int, float] = {}
    for i, nbrs in graph.items():
        if not nbrs:
            avg_ssim[i] = 0.0
            continue
        sims = []
        for u in nbrs:
            if (i, u) not in avg_ssim and (u, i) not in avg_ssim:
                sims.append(ssim(node_features[i], node_features[u]))
            else:
                sims.append(avg_ssim.get(i, 0.0))
        avg_ssim[i] = sum(sims) / len(sims)

    fisher_vals = {
        i: fisher_score(avg_ssim[i], fisher_center, fisher_width)
        for i in avg_ssim
    }

    points = [(avg_ssim[i], fisher_vals[i]) for i in avg_ssim]
    region_assignments = voronoi_partition(points, voronoi_seeds)

    scores = [avg_ssim[i] * fisher_vals[i] for i in avg_ssim]
    winning_regions = wta_selector(region_assignments, scores, top_k_regions)

    # Apply anisotropic Ollivier-Ricci curvature
    anisotropic_curvature = {}
    for i, region in enumerate(region_assignments):
        anisotropic_curvature[i] = curvature[i] * (1 + fisher_vals[i])

    return anisotropic_curvature, region_assignments, winning_regions