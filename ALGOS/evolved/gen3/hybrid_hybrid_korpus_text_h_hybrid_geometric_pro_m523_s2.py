# DARWIN HAMMER — match 523, survivor 2
# gen: 3
# parent_a: hybrid_korpus_text_hybrid_krampus_brain_m43_s0.py (gen2)
# parent_b: hybrid_geometric_product_voronoi_partition_m4_s0.py (gen1)
# born: 2026-05-29T23:29:20Z

"""
This module defines a hybrid algorithm that combines the minhash operation from 
hybrid_korpus_text_hybrid_krampus_brain_m43_s0.py and the Voronoi partitioning 
from hybrid_geometric_product_voronoi_partition_m4_s0.py. The mathematical 
bridge between these structures is the application of Voronoi partitioning to 
the minhash signatures, allowing for the creation of regions of similar text 
data. The hybrid algorithm integrates these two operations by using the 
minhash operation to generate compact representations of the text data and 
then applying Voronoi partitioning to these representations.

The hybrid operation can be used to analyze and visualize the structure of 
text data, identifying regions of similar content and patterns in the data.
"""

import numpy as np
import re
import math
import random
import sys
from pathlib import Path
from collections import deque

Point = tuple[float, float]

def minhash_for_text(text: str, k: int = 64) -> list[int]:
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    shingles = [text[i:i+5] for i in range(len(text)-4)]
    signature = np.random.randint(0, 1000000, size=k)
    for s in shingles:
        hash_value = hash(s) % k
        signature[hash_value] = min(signature[hash_value], hash(s) % 1000000)
    return signature.tolist()

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def hybrid_minhash_voronoi(texts: list[str], k: int = 64, num_seeds: int = 5) -> dict[int, list[str]]:
    """
    Applies minhash operation to the input texts and then applies Voronoi 
    partitioning to the resulting signatures.

    Args:
    texts (list[str]): The input texts.
    k (int, optional): The number of minhash buckets. Defaults to 64.
    num_seeds (int, optional): The number of seeds for Voronoi partitioning. Defaults to 5.

    Returns:
    dict[int, list[str]]: A dictionary where each key is a seed index and each value is a list of texts assigned to that seed.
    """
    signatures = [minhash_for_text(text, k) for text in texts]
    points = [(signature[0], signature[1]) for signature in signatures]
    seeds = random.sample(points, num_seeds)
    return assign(points, seeds)

def hybrid_minhash_voronoi_region(texts: list[str], k: int = 64, num_seeds: int = 5) -> dict[int, list[Point]]:
    """
    Applies minhash operation to the input texts and then applies Voronoi 
    partitioning to the resulting signatures.

    Args:
    texts (list[str]): The input texts.
    k (int, optional): The number of minhash buckets. Defaults to 64.
    num_seeds (int, optional): The number of seeds for Voronoi partitioning. Defaults to 5.

    Returns:
    dict[int, list[Point]]: A dictionary where each key is a seed index and each value is a list of points assigned to that seed.
    """
    signatures = [minhash_for_text(text, k) for text in texts]
    points = [(signature[0], signature[1]) for signature in signatures]
    seeds = random.sample(points, num_seeds)
    return assign(points, seeds)

def hybrid_minhash_voronoi_centroid(texts: list[str], k: int = 64, num_seeds: int = 5) -> dict[int, Point]:
    """
    Applies minhash operation to the input texts and then applies Voronoi 
    partitioning to the resulting signatures. Calculates the centroid of each region.

    Args:
    texts (list[str]): The input texts.
    k (int, optional): The number of minhash buckets. Defaults to 64.
    num_seeds (int, optional): The number of seeds for Voronoi partitioning. Defaults to 5.

    Returns:
    dict[int, Point]: A dictionary where each key is a seed index and each value is the centroid of the points assigned to that seed.
    """
    signatures = [minhash_for_text(text, k) for text in texts]
    points = [(signature[0], signature[1]) for signature in signatures]
    seeds = random.sample(points, num_seeds)
    regions = assign(points, seeds)
    centroids = {}
    for seed, points in regions.items():
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        centroid = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
        centroids[seed] = centroid
    return centroids

if __name__ == "__main__":
    texts = ["This is a test text", "Another test text", "Text text text"]
    print(hybrid_minhash_voronoi(texts))
    print(hybrid_minhash_voronoi_region(texts))
    print(hybrid_minhash_voronoi_centroid(texts))