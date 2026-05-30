# DARWIN HAMMER — match 4207, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_hard_t_m2229_s0.py (gen6)
# parent_b: hybrid_hybrid_korpus_text_h_hybrid_geometric_pro_m523_s5.py (gen3)
# born: 2026-05-29T23:54:15Z

"""
Hybrid Multivector-Linguistic Scoring Module

This module fuses two parent algorithms:

* **Parent A** – `hybrid_hybrid_geometric_pro_hybrid_hybrid_hard_t_m2229_s0.py`  
  Provides a hybrid geometric-linguistic scoring module that combines a geometric product 
  via Clifford algebra with Bayesian/Fisher-information based tree scoring.

* **Parent B** – `hybrid_hybrid_korpus_text_h_hybrid_geometric_pro_m523_s5.py`  
  Supplies a Hybrid Text-Voronoi Geometric Product (HTVGP) that fuses minhash 
  signatures with Clifford-algebra multivectors.

The mathematical bridge between the two parents is built by interpreting the minhash 
signature of a text as a compact high-dimensional hash vector and projecting its first 
two components to a 2-D point. Those points seed a Voronoi diagram that partitions 
a corpus of texts. Each Voronoi region aggregates the feature dictionaries of its 
member texts into a Multivector. The geometric product is then applied between the 
multivectors of adjacent Voronoi cells, and scaled by the Fisher information of 
the text distribution.

The resulting hybrid operation couples textual similarity (minhash) with Clifford 
geometric algebra and Bayesian/Fisher-information based scoring.
"""

import math
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import re
from typing import Dict, List, Tuple, FrozenSet

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades.

    components: dict mapping frozenset(basis_indices) -> float coefficient.
                frozenset() is the scalar (grade-0) blade.
    n: dimension of the base vector space.
    """

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result_components = {}
        for k1, v1 in self.components.items():
            for k2, v2 in other.components.items():
                k = frozenset(k1 | k2)
                v = v1 * v2
                if k in result_components:
                    result_components[k] += v
                else:
                    result_components[k] = v
        return Multivector(result_components, self.n)

    def scalar_part(self) -> float:
        """Extract the scalar (grade-0) part."""
        return self.components.get(frozenset(), 0.0)

def fisher_information(p: Dict[str, float]) -> float:
    """Compute the Fisher information of a probability distribution."""
    return sum([v * (1 / v) for v in p.values()])

def minhash_for_text(text: str, k: int = 64) -> list[int]:
    """Return a minhash signature of length *k* for *text*."""
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    if len(text) < 5:
        return [0] * k
    shingles = [text[i:i+3] for i in range(len(text)-2)]
    return [hash(shingle) % (2**32) for shingle in shingles[:k]]

def text_signature_point(text: str, k: int = 64) -> Tuple[float, float]:
    """minhash → 2-D point."""
    minhash = minhash_for_text(text, k)
    return minhash[0] / (2**32), minhash[1] / (2**32)

def voronoi_partition_texts(texts: List[str], seed_count: int = 5) -> Dict[int, List[str]]:
    """texts → Voronoi regions."""
    points = [text_signature_point(text) for text in texts]
    regions = defaultdict(list)
    for i, point in enumerate(points):
        closest_seed = np.argmin([np.linalg.norm(np.array(point) - np.array(p)) for p in points[:seed_count]])
        regions[closest_seed].append(texts[i])
    return regions

def region_multivector(region: List[str]) -> Multivector:
    """Aggregate feature dictionaries into a Multivector."""
    features = Counter([word for text in region for word in re.findall(r"\b\w+\b", text)])
    components = {frozenset([i]): v for i, v in enumerate(features.values())}
    return Multivector(components, len(features))

def region_geometric_products(regions: Dict[int, List[str]]) -> Dict[Tuple[int, int], float]:
    """Multivector construction + geometric products between neighboring regions."""
    multivectors = {i: region_multivector(region) for i, region in regions.items()}
    products = {}
    for i in regions:
        for j in regions:
            if i != j:
                product = multivectors[i] * multivectors[j]
                products[(i, j)] = product.scalar_part()
    return products

def hybrid_score(text: str, edge_belief: Dict[str, float]) -> float:
    """Compute the hybrid score."""
    p_text = Counter(re.findall(r"\b\w+\b", text))
    p_text = {k: v / sum(p_text.values()) for k, v in p_text.items()}
    mv_text = Multivector({frozenset([i]): v for i, v in enumerate(p_text.values())}, len(p_text))
    mv_edge = Multivector({frozenset([i]): v for i, v in enumerate(edge_belief.values())}, len(edge_belief))
    product = mv_text * mv_edge
    fisher_info = fisher_information(p_text)
    return product.scalar_part() * fisher_info

if __name__ == "__main__":
    texts = ["This is a test.", "This test is only a test.", "Test test test."]
    regions = voronoi_partition_texts(texts)
    products = region_geometric_products(regions)
    print(products)
    text = "This is a test."
    edge_belief = {"a": 0.5, "b": 0.5}
    print(hybrid_score(text, edge_belief))