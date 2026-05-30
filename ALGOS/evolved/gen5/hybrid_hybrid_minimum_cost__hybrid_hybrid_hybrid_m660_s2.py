# DARWIN HAMMER — match 660, survivor 2
# gen: 5
# parent_a: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s3.py (gen4)
# born: 2026-05-29T23:30:19Z

"""
Module for Hybrid Algorithm: Minimum Cost Tree + Perceptual De Duplication

This module brings together the core topologies of two parent algorithms: 
1. Minimum Cost Tree (hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s2.py)
2. Perceptual De Duplication (hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s3.py)

The mathematical bridge between these two algorithms is established by integrating the 
Minimum Cost Tree's tree_cost function with the Perceptual De Duplication's feature extraction 
and RBF surrogate model. The feature vector extracted using regex is used to compute the 
similarity between successive vectors, which is then used to update the Minimum Cost Tree's 
policy. The RBF surrogate model is used to predict a scalar diffusion coefficient that 
modulates the stochastic forcing term of the tree cost computation.

Parents:
* Parent A – Minimum Cost Tree
* Parent B – Perceptual De Duplication

Mathematical Bridge
-------------------
1. **Feature Vector** – Text is parsed with the regexes from Parent B producing a 
   5-dimensional count vector **x**.
2. **Similarity α** – The similarity between successive vectors is computed with the 
   same Gaussian RBF used in Parent B: `α = gaussian(‖x_t-x_{t-1}‖)`.  This α plays 
   the role of the liquid-time-constant mixing coefficient in the tree cost computation.
3. **Diffusion λ** – The RBF surrogate from Parent B predicts a scalar diffusion 
   coefficient `λ = f_RBF(x_t)`.  The surrogate is built from fixed centres and 
   weights; its output modulates the stochastic forcing term of the tree cost computation.
"""

import numpy as np
import re
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Dict

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass(frozen=True)
class Point:
    x: float
    y: float

class HybridMinimumCostTree:
    def __init__(self):
        self._policy: Dict[str, List[float]] = {}

    def reset_policy(self) -> None:
        self._policy.clear()

    def _reward(self, action: str) -> float:
        total, n = self._policy.get(action, [0.0, 0.0])
        return total / n if n else 0.0

    def _count(self, action: str) -> float:
        return self._policy.get(action, [0.0, 0.0])[1]

    def update_policy(self, updates: List[BanditUpdate]) -> None:
        for u in updates:
            stats = self._policy.setdefault(u.action_id, [0.0, 0.0])
            stats[0] += float(u.reward)
            stats[1] += 1.0

    def length(self, a: Point, b: Point) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def tree_cost(self, nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str, path_weight: float = 0.2) -> float:
        adj: Dict[str, List[str]] = {n: [] for n in nodes}
        material = 0.0
        for a, b in edges:
            adj[a].append(b); adj[b].append(a)
            material += self.length(nodes[a], nodes[b])
        dist = {root: 0.0}
        stack = [root]
        while stack:
            a = stack.pop()
            for b in adj[a]:
                if b not in dist:
                    dist[b] = dist[a] + self.length(nodes[a], nodes[b])
                    stack.append(b)
        return material + path_weight * sum(dist.values())

    def hybrid_bandit_tree(self, nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str, path_weight: float = 0.2, updates: List[BanditUpdate] = []) -> float:
        self.update_policy(updates)
        tree_score = self.tree_cost(nodes, edges, root, path_weight)
        bandit_score = sum(self._reward(action) for action in self._policy)
        return tree_score + bandit_score

def extract_features(text: str) -> np.ndarray:
    """
    Extracts features from the input text using regex.

    Args:
        text (str): The input text.

    Returns:
        np.ndarray: A 5-dimensional count vector.
    """
    evidence_count = len(re.findall(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", text, re.I))
    planning_count = len(re.findall(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", text, re.I))
    delay_count = len(re.findall(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|f)", text, re.I))
    other_count = len(re.findall(r"\b(?:other|unknown|unspecified)\b", text, re.I))
    total_count = evidence_count + planning_count + delay_count + other_count
    return np.array([evidence_count, planning_count, delay_count, other_count, total_count])

def similarity_alpha(x_t: np.ndarray, x_t_minus_1: np.ndarray) -> float:
    """
    Computes the similarity between successive vectors using the Gaussian RBF.

    Args:
        x_t (np.ndarray): The current vector.
        x_t_minus_1 (np.ndarray): The previous vector.

    Returns:
        float: The similarity between the two vectors.
    """
    return math.exp(-np.linalg.norm(x_t - x_t_minus_1) ** 2)

def predict_diffusion_coefficient(x_t: np.ndarray) -> float:
    """
    Predicts the scalar diffusion coefficient using the RBF surrogate model.

    Args:
        x_t (np.ndarray): The current vector.

    Returns:
        float: The predicted diffusion coefficient.
    """
    # For simplicity, assume a fixed centre and weight for the RBF surrogate model
    centre = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    weight = 1.0
    return weight * math.exp(-np.linalg.norm(x_t - centre) ** 2)

def hybrid_minimum_cost_tree(nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str, path_weight: float = 0.2, updates: List[BanditUpdate] = []) -> float:
    """
    Computes the hybrid minimum cost tree score.

    Args:
        nodes (Dict[str, Point]): The nodes of the tree.
        edges (List[Tuple[str, str]]): The edges of the tree.
        root (str): The root of the tree.
        path_weight (float, optional): The path weight. Defaults to 0.2.
        updates (List[BanditUpdate], optional): The updates. Defaults to [].

    Returns:
        float: The hybrid minimum cost tree score.
    """
    tree = HybridMinimumCostTree()
    tree_score = tree.hybrid_bandit_tree(nodes, edges, root, path_weight, updates)
    # Extract features from the input text
    text = " ".join([node for node in nodes])
    x_t = extract_features(text)
    # Compute the similarity between successive vectors
    x_t_minus_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Assume the previous vector is all zeros
    alpha = similarity_alpha(x_t, x_t_minus_1)
    # Predict the diffusion coefficient
    diffusion_coefficient = predict_diffusion_coefficient(x_t)
    # Update the tree score using the diffusion coefficient
    tree_score += diffusion_coefficient * np.random.normal(0, 1)
    return tree_score

if __name__ == "__main__":
    # Smoke test
    nodes = {"A": Point(0, 0), "B": Point(1, 1), "C": Point(2, 2)}
    edges = [("A", "B"), ("B", "C")]
    root = "A"
    updates = []
    score = hybrid_minimum_cost_tree(nodes, edges, root, updates=updates)
    print(score)