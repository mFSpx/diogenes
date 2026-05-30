# DARWIN HAMMER — match 2223, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_korpus_text_h_hybrid_geometric_pro_m523_s5.py (gen3)
# parent_b: hybrid_hybrid_hard_truth_ma_kan_m27_s3.py (gen2)
# born: 2026-05-29T23:41:19Z

"""
Hybrid Text-Voronoi Geometric Product with Stylometry-KAN Model (HTVGP-SKAN)

This module fuses two parent algorithms:
* **Parent A** – `hybrid_hybrid_korpus_text_h_hybrid_geometric_pro_m523_s5.py` 
  provides a minhash signature for arbitrary text, a rich feature extraction, 
  and a geometric product between multivectors.
* **Parent B** – `hybrid_hybrid_hard_truth_ma_kan_m27_s3.py` 
  combines stylometric feature extraction with Kolmogorov-Arnold Networks (KAN).

The mathematical bridge is built by interpreting the minhash signature of a text 
as a compact high-dimensional hash vector, projecting its first two components 
to a 2-D point, and using these points to seed a Voronoi diagram. Each Voronoi 
region aggregates the stylometric feature vectors of its member texts into a 
Multivector. The geometric product is then applied between the multivectors of 
adjacent Voronoi cells, yielding a hybrid operation that couples textual similarity 
(minhash) with Clifford geometric algebra and stylometric analysis.

"""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import re

def _clean_text(text: str) -> str:
    """Normalize whitespace, lower-case, and strip."""
    return re.sub(r"\s+", " ", text or "").strip().lower()

def minhash_for_text(text: str, k: int = 64) -> list[int]:
    """Return a minhash signature of length *k* for *text*."""
    text = _clean_text(text)
    if len(text) < 5:
        # not enough shingles – return a zero signature
        return [0] * k
    shingles = [hashlib.md5((text[i:i+5]).encode()).hexdigest() for i in range(len(text)-4)]
    minhash = [0] * k
    for shingle in shingles:
        minhash[int(shingle, 16) % k] += 1
    return minhash

def stylometry_features(text: str) -> np.ndarray:
    """Return a stylometric feature vector for *text*."""
    words = text.split()
    word_count = len(words)
    function_word_count = sum(1 for word in words if word in ["the", "a", "an"])
    pronoun_count = sum(1 for word in words if word in ["i", "me", "my", "mine", "myself"])
    return np.array([word_count, function_word_count, pronoun_count])

def text_signature_point(text: str, k: int = 64) -> np.ndarray:
    """Return a 2-D point representing the minhash signature of *text*."""
    minhash = minhash_for_text(text, k)
    return np.array([minhash[0], minhash[1]])

def voronoi_partition_texts(texts: list[str], seed_count: int = 5) -> list[list[str]]:
    """Return a list of Voronoi regions, each containing a list of texts."""
    points = [text_signature_point(text) for text in texts]
    voronoi_regions = [[] for _ in range(seed_count)]
    for i, point in enumerate(points):
        min_distance = float('inf')
        closest_region = -1
        for j, seed in enumerate(points[:seed_count]):
            distance = np.linalg.norm(point - seed)
            if distance < min_distance:
                min_distance = distance
                closest_region = j
        voronoi_regions[closest_region].append(texts[i])
    return voronoi_regions

def region_stylometry_vectors(regions: list[list[str]]) -> list[np.ndarray]:
    """Return a list of stylometric feature vectors, one for each Voronoi region."""
    stylometry_vectors = []
    for region in regions:
        feature_vectors = [stylometry_features(text) for text in region]
        stylometry_vector = np.mean(feature_vectors, axis=0)
        stylometry_vectors.append(stylometry_vector)
    return stylometry_vectors

def geometric_product(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """Return the geometric product of two vectors."""
    return np.outer(vector1, vector2) + np.outer(vector2, vector1)

def region_geometric_products(regions: list[list[str]]) -> list[np.ndarray]:
    """Return a list of geometric products, one for each pair of adjacent Voronoi regions."""
    stylometry_vectors = region_stylometry_vectors(regions)
    geometric_products = []
    for i in range(len(stylometry_vectors) - 1):
        product = geometric_product(stylometry_vectors[i], stylometry_vectors[i+1])
        geometric_products.append(product)
    return geometric_products

if __name__ == "__main__":
    texts = ["This is a test text.", "This is another test text.", "And another one."]
    regions = voronoi_partition_texts(texts)
    products = region_geometric_products(regions)
    print(products)