# DARWIN HAMMER — match 2456, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_percep_m598_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m759_s1.py (gen5)
# born: 2026-05-29T23:42:26Z

import numpy as np
import math
import random
import sys
import pathlib

def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                if j < len(lst):
                    lst.pop(j)  
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def geometric_product(a, b):
    result = {}
    for blade_a, coeff_a in a.items():
        for blade_b, coeff_b in b.items():
            combined, sign = _multiply_blades(blade_a, blade_b)
            if combined in result:
                result[combined] += sign * coeff_a * coeff_b
            else:
                result[combined] = sign * coeff_a * coeff_b
    return result

def rbf_activation(x: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    dists = np.linalg.norm(centers - x, axis=1)
    return np.exp(- (dists ** 2) / (2 * sigma ** 2))

def construct_similarity_graph(weights: np.ndarray) -> dict:
    n = len(weights)
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            sim = 1.0 - abs(weights[i] - weights[j]) / (1.0 + abs(weights[i] - weights[j]))
            graph[i].append((j, sim))
            graph[j].append((i, sim))
    return graph

def hybrid_operation(rbf_weights: np.ndarray, clifford_input: dict) -> dict:
    graph = construct_similarity_graph(rbf_weights)
    clifford_product = geometric_product(clifford_input, clifford_input)
    result = {}
    for blade, coeff in clifford_product.items():
        rbf_output = rbf_activation(np.array(list(blade)), np.array(list(range(len(rbf_weights)))), 1.0)
        result[blade] = coeff * np.mean(rbf_output)
    return result

def prim_mst(graph: dict) -> list:
    mst = []
    visited = set()
    start_node = next(iter(graph))
    visited.add(start_node)
    while len(visited) < len(graph):
        min_edge = None
        for node in visited:
            for edge in graph[node]:
                if edge[0] not in visited:
                    if min_edge is None or edge[1] < min_edge[1]:
                        min_edge = (node, edge[0], edge[1])
        mst.append(min_edge)
        visited.add(min_edge[1])
    return mst

def improved_hybrid_operation(rbf_weights: np.ndarray, clifford_input: dict) -> dict:
    graph = construct_similarity_graph(rbf_weights)
    mst = prim_mst(graph)
    clifford_product = geometric_product(clifford_input, clifford_input)
    result = {}
    for blade, coeff in clifford_product.items():
        rbf_output = rbf_activation(np.array(list(blade)), np.array(list(range(len(rbf_weights)))), 1.0)
        result[blade] = coeff * np.mean(rbf_output)
        for edge in mst:
            node1, node2, sim = edge
            result[blade] *= (1 + sim * np.mean(rbf_output))
    return result

if __name__ == "__main__":
    rbf_weights = np.random.rand(10)
    clifford_input = {frozenset([1, 2, 3]): 1.0, frozenset([4, 5, 6]): 2.0}
    result = improved_hybrid_operation(rbf_weights, clifford_input)
    print(result)
    graph = construct_similarity_graph(rbf_weights)
    mst = prim_mst(graph)
    print(mst)