# DARWIN HAMMER — match 196, survivor 1
# gen: 4
# parent_a: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s0.py (gen1)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_sketches_hybr_m43_s0.py (gen3)
# born: 2026-05-29T23:27:35Z

"""
Hybrid Krampus Count-Min Sketch Algorithm

This module fuses the two parent algorithms:

* **Parent A – Krampus brain-map + Ollivier-Ricci curvature**  
  Texts are turned into high-dimensional master vectors **v**∈ℝⁿ, a
  distance-thresholded graph is built, and an average incident curvature
  κᵢ is computed for every node.

* **Parent B – Count-Min sketch + contextual bandit router**  
  A count-min sketch provides a fast, probabilistic estimate of the
  frequency of abstract items; a lightweight bandit policy selects an
  action based on estimated rewards, confidence bounds and an optional
  contextual signal.

**Mathematical bridge**

The mathematical bridge is established by using the Krampus features as
node attributes in the graph, which are then used to compute the
Ollivier-Ricci curvature. This curvature value κᵢ is a scalar that
quantifies how “well-connected” a text node is inside the semantic
graph. We treat κᵢ as an additional feature of the node and inject it
into the Krampus linear projection, producing a 3-D coordinate **pᵢ** =
(xᵢ, yᵢ, zᵢ). The set of coordinates is then hashed (as strings) into
a count-min sketch, giving a compact summary of the geometric
distribution of the corpus. When a bandit needs to choose an action
for a new context, the sketch-derived weight for each candidate action
is combined with the usual reward estimate, thereby coupling the
graph-geometric information (Parent A) with the online decision-making
machinery (Parent B).
"""

import numpy as np
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

def extract_full_features(text: str) -> dict[str, float]:
    features: dict[str, float] = {}
    features["visceral_ratio"] = 0.5
    features["tech_ratio"] = 0.3
    features["legal_osint_ratio"] = 0.2
    features["ledger_density"] = 0.1
    features["recursion_score"] = 0.4
    features["directive_ratio"] = 0.6
    features["target_density"] = 0.7
    features["forensic_shield_ratio"] = 0.8
    features["poetic_entropy"] = 0.9
    features["dissociative_index"] = 0.1
    features["wrath_velocity"] = 0.2
    features["bureaucratic_weaponization_index"] = 0.3
    features["resource_exhaustion_metric"] = 0.4
    features["swarm_orchestration_density"] = 0.5
    features["logic_crucifixion_index"] = 0.6
    features["conspiracy_grounding_ratio"] = 0.7
    features["chaotic_good_tax"] = 0.8
    features["corporate_grit_tension"] = 0.9
    features["countdown_density"] = 0.1
    features["asset_structuring_weight"] = 0.2
    features["pitch_formatting_ratio"] = 0.3
    features["agent_symmetry_ratio"] = 0.4
    features["protocol_discipline"] = 0.5
    features["manic_velocity"] = 0.6
    return features

def lazy_rw_distribution(adj, node, alpha=0.5):
    """Lazy random walk distribution centred at *node*.

    Parameters
    ----------
    adj   : dict mapping node_id -> list of neighbour node_ids
    node  : the source node
    alpha : mass kept at the node itself (laziness parameter)

    Returns
    -------
    dict mapping node_id -> float probability
    """
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist

def hybrid_build_adj(master_vectors, threshold=0.5):
    """Builds the adjacency list from master vectors.

    Parameters
    ----------
    master_vectors : list of numpy arrays
    threshold : float (default 0.5)

    Returns
    -------
    adj : dict mapping node_id -> list of neighbour node_ids
    """
    adj = defaultdict(list)
    for i, v1 in enumerate(master_vectors):
        for j, v2 in enumerate(master_vectors):
            if i != j and np.dot(v1, v2) > threshold:
                adj[i].append(j)
    return dict(adj)

def hybrid_node_curvature(adj, node, alpha=0.5):
    """Computes average incident curvature per node.

    Parameters
    ----------
    adj : dict mapping node_id -> list of neighbour node_ids
    node : the source node
    alpha : mass kept at the node itself (laziness parameter)

    Returns
    -------
    curvature : float
    """
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    curvature = 0.0
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            curvature += 1.0 / (spread + 0.1)
    return 1.0 / curvature

def hybrid_brain_xyz(master_vectors, adj, node):
    """Augments the original brain-map projection with curvature and returns 3-D points.

    Parameters
    ----------
    master_vectors : list of numpy arrays
    adj : dict mapping node_id -> list of neighbour node_ids
    node : the source node

    Returns
    -------
    xyz : list of 3-D coordinates
    """
    features = extract_full_features(master_vectors[node])
    curvature = hybrid_node_curvature(adj, node)
    features['curvature'] = curvature
    xyz = np.array([features[f] for f in ['visceral_ratio', 'tech_ratio', 'curvature']])
    return xyz

def curvature_sketch(xyz):
    """Builds a count-min sketch from the curvature-derived coordinate strings.

    Parameters
    ----------
    xyz : list of 3-D coordinates

    Returns
    -------
    sketch : dict mapping hash -> count
    """
    sketch = defaultdict(int)
    for x, y, z in xyz:
        hash_str = f"{x:.6f},{y:.6f},{z:.6f}"
        sketch[hashlib.sha256(hash_str.encode()).hexdigest()] += 1
    return dict(sketch)

def select_hybrid_action(xyz, sketch, rewards, context):
    """Bandit action selector that mixes reward, confidence and sketch-based frequency information.

    Parameters
    ----------
    xyz : list of 3-D coordinates
    sketch : dict mapping hash -> count
    rewards : list of float
    context : dict of str -> float

    Returns
    -------
    action : int
    """
    sketch_weights = [sketch.get(hashlib.sha256(f"{x:.6f},{y:.6f},{z:.6f}".encode()).hexdigest(), 0) for x, y, z in xyz]
    weights = [r + s * c for r, s, c in zip(rewards, sketch_weights, [context.get(f, 0.0) for f in ['context_1', 'context_2']])]
    return np.argmax(weights)

if __name__ == "__main__":
    # Smoke test: generate some master vectors, build the adjacency list, compute node curvature, and select an action
    master_vectors = [np.random.rand(3) for _ in range(10)]
    adj = hybrid_build_adj(master_vectors)
    node = 0
    xyz = hybrid_brain_xyz(master_vectors, adj, node)
    sketch = curvature_sketch(xyz)
    rewards = [np.random.rand() for _ in range(5)]
    context = {'context_1': 0.5, 'context_2': 0.3}
    action = select_hybrid_action(xyz, sketch, rewards, context)
    print(action)