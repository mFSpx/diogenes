# DARWIN HAMMER — match 2537, survivor 2
# gen: 5
# parent_a: hybrid_distributed_leader_e_perceptual_dedupe_m16_s2.py (gen1)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s4.py (gen4)
# born: 2026-05-29T23:42:49Z

import numpy as np
import math
import random
import sys

Node = int
Graph = dict[int, set[int]]
FeatureVec = list[float]

def compute_phash(values: list[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2
    l_mean_sq = np.mean(x) ** 2
    c1 = 2 * C1 + C2 - C1
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    ssim_map = ((2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)) * ((2 * sigma_xy + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2))
    return np.mean(ssim_map)

def cluster_by_phash(hashes: dict[Node, int], max_distance: int = 4) -> list[list[Node]]:
    clusters = []
    for node, h in hashes.items():
        for c in clusters:
            if hamming_distance(h, c[0]) <= max_distance:
                c.append(node)
                break
        else:
            clusters.append([node])
    return clusters

def get_similarity_matrix(nodes: list[Node], feature_vectors: list[FeatureVec]) -> np.ndarray:
    hashes = {n: compute_phash(vs) for n, vs in enumerate(feature_vectors)}
    clusters = cluster_by_phash(hashes)
    sim_matrix = np.zeros((len(nodes), len(nodes)))
    for i, n in enumerate(nodes):
        for j, m in enumerate(nodes):
            if n == m:
                continue
            sim_matrix[i, j] = ssim(np.array(feature_vectors[n]), np.array(feature_vectors[m]))
    np.fill_diagonal(sim_matrix, 1.0)
    return sim_matrix

def distributed_leader_election(graph: Graph, similarity_matrix: np.ndarray) -> list[Node]:
    leaders = set()
    undecided = set(graph.keys())
    phase = 0
    while undecided:
        for node in undecided.copy():
            neighbors = graph[node]
            if len(neighbors) == 0:
                leaders.add(node)
                undecided.remove(node)
                continue
            avg_sim = np.mean([similarity_matrix[node, n] for n in neighbors])
            prob = 1 / (2 ** phase)
            if random.random() < prob * avg_sim:
                leaders.add(node)
                undecided.remove(node)
        phase += 1
    return list(leaders)

def ternary_routing(packets: list[dict[str, any]], similarity_matrix: np.ndarray) -> list[dict[str, any]]:
    routed_packets = []
    packet_ids = [packet["id"] for packet in packets]
    for packet in packets:
        sim_scores = similarity_matrix[packet_ids.index(packet["id"]), :]
        max_sim_idx = np.argmax(sim_scores)
        max_sim = sim_scores[max_sim_idx]
        if max_sim > 0.5:
            routed_packets.append(packets[max_sim_idx])
        else:
            leader_node = random.choice(distributed_leader_election({i: set() for i in range(len(packets))}, similarity_matrix))
            routed_packets.append({"route": leader_node, "id": packet["id"]})
    return routed_packets

def hybrid_algorithm(graph: Graph, feature_vectors: list[FeatureVec]) -> list[dict[str, any]]:
    similarity_matrix = get_similarity_matrix(list(graph.keys()), feature_vectors)
    packets = [{"id": i, "text": f"Packet {i}", "intent": "test"} for i in range(len(feature_vectors))]
    routed_packets = ternary_routing(packets, similarity_matrix)
    return routed_packets

if __name__ == "__main__":
    nodes = [0, 1, 2, 3, 4]
    feature_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    graph = {n: set() for n in nodes}
    print(hybrid_algorithm(graph, feature_vectors))