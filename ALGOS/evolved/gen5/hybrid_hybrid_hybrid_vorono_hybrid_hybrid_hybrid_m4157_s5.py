# DARWIN HAMMER — match 4157, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s1.py (gen3)
# born: 2026-05-29T23:54:00Z

import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

Point = Tuple[float, float]

def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def voronoi_assign(points: np.ndarray, sites: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - sites[None, :, :]  
    dists = np.linalg.norm(diff, axis=2)                  
    return np.argmin(dists, axis=1)                       

def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W: np.ndarray, X: np.ndarray) -> float:
    diff = W @ X.T - X.T
    return float(np.sum(diff ** 2))

def ttt_grad(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    diff = W @ X.T - X.T                     
    return 2.0 * diff @ X                   

def ttt_step(W: np.ndarray, X: np.ndarray, eta: float) -> np.ndarray:
    grad = ttt_grad(W, X)
    return W - eta * grad

class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = now_z()
        if self.failures >= self.failure_threshold:
            self.open = True

    def is_open(self) -> bool:
        return self.open

def ssim_pointcloud(original: np.ndarray, transformed: np.ndarray, dynamic_range: float = 1.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if original.shape != transformed.shape:
        raise ValueError("point clouds must have the same shape")
    mu_x = original.mean(axis=0)
    mu_y = transformed.mean(axis=0)
    sigma_x2 = ((original - mu_x) ** 2).mean(axis=0)
    sigma_y2 = ((transformed - mu_y) ** 2).mean(axis=0)
    sigma_xy = ((original - mu_x) * (transformed - mu_y)).mean(axis=0)

    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = numerator / denominator
    return float(ssim_map.mean())

def hybrid_transform(points: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (W @ points.T).T  

def hybrid_update(points: np.ndarray, sites: np.ndarray, W: np.ndarray, eta: float, breaker: EndpointCircuitBreaker) -> Tuple[np.ndarray, np.ndarray, float]:
    transformed = hybrid_transform(points, W)
    assign = voronoi_assign(transformed, sites)
    new_sites = np.array([points[assign == i].mean(axis=0) for i in range(len(sites))])
    new_W = ttt_step(W, points, eta)
    loss = ttt_loss(new_W, points)
    if loss < ttt_loss(W, points):
        breaker.record_success()
    else:
        breaker.record_failure()
    return new_sites, new_W, loss

def improved_hybrid(points: np.ndarray, sites: np.ndarray, W: np.ndarray, eta: float, breaker: EndpointCircuitBreaker, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, float]:
    for _ in range(max_iter):
        new_sites, new_W, loss = hybrid_update(points, sites, W, eta, breaker)
        if breaker.is_open():
            break
        sites, W = new_sites, new_W
    return sites, W, loss

def main():
    points = np.random.rand(100, 2)
    sites = np.random.rand(5, 2)
    W = init_ttt(2)
    breaker = EndpointCircuitBreaker()
    eta = 0.01
    new_sites, new_W, loss = improved_hybrid(points, sites, W, eta, breaker)
    print(f"Final Loss: {loss}")
    print(f"Final Sites: {new_sites}")
    print(f"Final W: {new_W}")

if __name__ == "__main__":
    main()