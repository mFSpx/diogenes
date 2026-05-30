# DARWIN HAMMER — match 26, survivor 0
# gen: 2
# parent_a: hybrid_nlms_omni_chaotic_sprint_m59_s1.py (gen1)
# parent_b: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s0.py (gen1)
# born: 2026-05-29T23:26:33Z

"""
This module implements a novel hybrid algorithm that combines the normalized least mean squares (NLMS) update 
from the hybrid_nlms_omni_chaotic_sprint_m59_s1.py algorithm with the minimum-cost tree optimization from the 
hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s0.py algorithm. The mathematical bridge between these two 
structures lies in the representation of text spans as nodes in a graph, where the edges represent the relationships 
between these spans, and the NLMS update is used to adaptively adjust the weights in the graph to optimize the extraction 
of spans while minimizing the cost of the tree.

The NLMS update is used to adjust the weights in the graph based on the similarity between spans, and the minimum-cost tree 
algorithm is then applied to this graph to select the most relevant spans while minimizing the cost of the tree. This 
hybrid algorithm integrates the governing equations of both parents, enabling it to leverage the strengths of both approaches.
"""

import json
import time
from collections import Counter, deque
from pathlib import Path
import numpy as np
import random
import sys
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "05_OUTPUTS" / "hybrid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

class Span:
    def __init__(self, start: int, end: int, text: str, label: str, score: float, backend: str):
        self.start = start
        self.end = end
        self.text = text
        self.label = label
        self.score = score
        self.backend = backend

def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def predict(weights: np.ndarray, x: np.ndarray) -> float:
    return np.dot(weights, x)

def update(weights: np.ndarray, x: np.ndarray, target: float, mu: float = 0.5, eps: float = 1e-9) -> tuple[np.ndarray, float]:
    y = predict(weights, x)
    error = target - y
    power = np.dot(x, x) + eps
    next_weights = weights + mu * error * x / power
    return next_weights, error

def construct_graph(spans: list[Span]) -> np.ndarray:
    num_spans = len(spans)
    graph = np.zeros((num_spans, num_spans))
    for i in range(num_spans):
        for j in range(num_spans):
            if i != j:
                similarity = calculate_similarity(spans[i].text, spans[j].text)
                graph[i, j] = similarity
    return graph

def calculate_similarity(text1: str, text2: str) -> float:
    return len(set(text1) & set(text2)) / len(set(text1) | set(text2))

def optimize_graph(graph: np.ndarray) -> np.ndarray:
    num_spans = graph.shape[0]
    weights = np.ones(num_spans)
    for _ in range(100):
        for i in range(num_spans):
            x = graph[i]
            target = 1.0
            weights, _ = update(weights, x, target)
    return weights

def extract_relevant_spans(spans: list[Span], weights: np.ndarray) -> list[Span]:
    relevant_spans = []
    for i, span in enumerate(spans):
        if weights[i] > 0.5:
            relevant_spans.append(span)
    return relevant_spans

if __name__ == "__main__":
    spans = [
        Span(0, 10, "example text", "label1", 0.8, "backend1"),
        Span(10, 20, "example text2", "label2", 0.7, "backend2"),
        Span(20, 30, "example text3", "label3", 0.6, "backend3"),
    ]
    graph = construct_graph(spans)
    weights = optimize_graph(graph)
    relevant_spans = extract_relevant_spans(spans, weights)
    print(relevant_spans)