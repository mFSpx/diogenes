# DARWIN HAMMER — match 58, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s4.py (gen3)
# born: 2026-05-29T23:26:37Z

"""
Hybrid Algorithm: Fisher‑SSIM Sheaf‑Associative Memory Fusion

Parents:
- hybrid_fisher_localization_hybrid_ternary_route (Algorithm A)
- hybrid_sketch_dense_associative_memory (Algorithm B)

Mathematical Bridge:
The Fisher information of a Gaussian beam (Algorithm A) is used as a *weight* on the
energy contributed by each node of a sheaf‑based associative memory (Algorithm B).
Conversely, the Structural Similarity Index (SSIM) from Algorithm A quantifies the
agreement between node sections (vectors) and a reference (or packet) vector; this
scalar modulates the restriction maps of the sheaf, effectively shaping the
information flow across edges.  The resulting hybrid system therefore couples
continuous‑parameter weighting (Fisher) with discrete‑topology similarity (SSIM) in a
single energy‑based decision framework.
"""

import json
import math
import random
import sys
from pathlib import Path
import numpy as np

__all__ = [
    "gaussian_beam",
    "fisher_score",
    "ssim",
    "Sheaf",
    "DenseAssociativeMemory",
    "hybrid_energy",
    "weighted_fisher_energy",
    "similarity_modulated_update",
    "hybrid_route_packet",
]

# ----------------------------------------------------------------------
# Algorithm A building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index (SSIM) for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
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


# ----------------------------------------------------------------------
# Algorithm B building blocks
# ----------------------------------------------------------------------
class Sheaf:
    """Discrete sheaf with node dimensions and edge restrictions."""

    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims          # {node_id: dimension}
        self.edges = edges                  # [(u, v), ...]
        self._restrictions = {}             # {(u, v): (src_map, dst_map)}
        self._sections = {}                 # {node_id: vector}

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float),
                                     np.asarray(dst_map, dtype=float))

    def set_section(self, node: any, value: np.ndarray) -> None:
        if value.shape != (self.node_dims[node],):
            raise ValueError("section shape must match node dimension")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: any) -> np.ndarray:
        return self._sections[node]

    def get_restriction(self, edge: tuple):
        return self._restrictions[edge]

    def nodes(self):
        return self.node_dims.keys()


class DenseAssociativeMemory:
    """Fully‑connected Hopfield‑like associative memory."""

    def __init__(self, patterns: np.ndarray, beta: float = 1.0):
        self.patterns = np.asarray(patterns, dtype=float)   # shape (P, N)
        self.beta = beta

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z = z - np.max(z)
        e = np.exp(z)
        return e / e.sum()

    def _lse(self, z: np.ndarray) -> float:
        m = np.max(z)
        return m + np.log(np.exp(z - m).sum())

    def energy(self, xi: np.ndarray) -> float:
        xi = np.asarray(xi, dtype=float)
        scores = self.beta * (self.patterns @ xi)               # (P,)
        lse_term = self._lse(scores) / self.beta
        quadratic_term = 0.5 * xi @ xi
        # negative log‑softmax sum = cross‑entropy with uniform target
        cross_entropy = -np.log(self._softmax(scores)).sum()
        return cross_entropy + lse_term + quadratic_term


def hybrid_energy(sheaf: Sheaf, dam: DenseAssociativeMemory) -> float:
    """Mean DAM energy over all nodes that have a defined section."""
    energies = []
    for node in sheaf.nodes():
        if node in sheaf._sections:
            energies.append(dam.energy(sheaf.get_section(node)))
    return np.mean(energies) if energies else 0.0


# ----------------------------------------------------------------------
# Hybrid Functions (bridge between A and B)
# ----------------------------------------------------------------------
def weighted_fisher_energy(sheaf: Sheaf,
                           dam: DenseAssociativeMemory,
                           theta_center: float,
                           theta_width: float) -> float:
    """
    Compute a Fisher‑weighted mean energy.

    For each node we treat the L2‑norm of its section as a proxy for the
    “theta” parameter of the Gaussian beam.  The Fisher score derived from
    (theta, theta_center, theta_width) weights the node's DAM energy.
    """
    weighted_vals = []
    for node in sheaf.nodes():
        if node not in sheaf._sections:
            continue
        section = sheaf.get_section(node)
        theta = np.linalg.norm(section)  # scalar proxy
        weight = fisher_score(theta, theta_center, theta_width)
        node_energy = dam.energy(section)
        weighted_vals.append(weight * node_energy)
    return np.mean(weighted_vals) if weighted_vals else 0.0


def similarity_modulated_update(sheaf: Sheaf,
                                reference: np.ndarray,
                                center: float,
                                width: float) -> None:
    """
    Adjust every restriction map of the sheaf by the SSIM similarity between
    the source/target sections and a reference vector.

    The similarity scalar (0 ≤ s ≤ 1) multiplies the rows of both src_map and
    dst_map, effectively pruning (when s≈0) or preserving (when s≈1) the flow.
    """
    for edge in sheaf.edges:
        u, v = edge
        src_section = sheaf.get_section(u)
        dst_section = sheaf.get_section(v)
        # Compute SSIM separately for source and destination against the reference
        sim_src = ssim(src_section, reference, dynamic_range=src_section.max() - src_section.min() + 1e-9)
        sim_dst = ssim(dst_section, reference, dynamic_range=dst_section.max() - dst_section.min() + 1e-9)
        # Blend the two similarities (geometric mean) to obtain a single scaling factor
        scale = math.sqrt(max(0.0, min(1.0, sim_src * sim_dst)))
        src_map, dst_map = sheaf.get_restriction(edge)
        sheaf._restrictions[edge] = (src_map * scale, dst_map * scale)


def hybrid_route_packet(packet: dict,
                        sheaf: Sheaf,
                        dam: DenseAssociativeMemory,
                        center: float,
                        width: float) -> any:
    """
    Route a packet to the most promising node.

    1. Convert packet text into a simple numeric vector (length‑based embedding).
    2. Compute SSIM similarity between this vector and each node's section.
    3. Combine similarity with Fisher‑weighted node energy to obtain a score.
    4. Return the node identifier with the highest score.
    """
    # ----- Step 1: naive embedding -----
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    # embedding: first character code, length, and a random component (deterministic via hash)
    rng = random.Random(hash(text))
    embed = np.array([
        ord(text[0]) if text else 0,
        len(text),
        rng.random() * 10.0
    ], dtype=float)

    best_node = None
    best_score = -np.inf

    for node in sheaf.nodes():
        if node not in sheaf._sections:
            continue
        section = sheaf.get_section(node)

        # Ensure same dimension for SSIM by zero‑padding the shorter vector
        if embed.size < section.size:
            pad = np.zeros(section.size - embed.size)
            vec = np.concatenate([embed, pad])
        elif embed.size > section.size:
            vec = embed[:section.size]
        else:
            vec = embed

        sim = ssim(vec, section, dynamic_range=section.max() - section.min() + 1e-9)

        theta = np.linalg.norm(section)
        fisher_w = fisher_score(theta, center, width)

        node_energy = dam.energy(section)

        # Higher similarity and lower energy are desirable; we invert energy
        score = sim * fisher_w - node_energy
        if score > best_score:
            best_score = score
            best_node = node

    return best_node


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny sheaf with two nodes
    node_dims = {"A": 3, "B": 3}
    edges = [("A", "B")]
    sheaf = Sheaf(node_dims, edges)

    # Random sections
    rng = np.random.default_rng(42)
    sheaf.set_section("A", rng.normal(size=3))
    sheaf.set_section("B", rng.normal(size=3))

    # Random restriction maps (identity for simplicity)
    src_map = np.eye(3)
    dst_map = np.eye(3)
    sheaf.set_restriction(("A", "B"), src_map, dst_map)

    # Dense associative memory with two stored patterns
    patterns = rng.normal(size=(2, 3))
    dam = DenseAssociativeMemory(patterns, beta=0.8)

    # Compute hybrid energies
    base_energy = hybrid_energy(sheaf, dam)
    fisher_energy = weighted_fisher_energy(sheaf, dam, theta_center=1.0, theta_width=0.5)

    print(f"Base mean energy: {base_energy:.4f}")
    print(f"Fisher‑weighted mean energy: {fisher_energy:.4f}")

    # Apply similarity‑modulated update using a reference vector
    reference_vec = np.array([0.0, 1.0, 0.5])
    similarity_modulated_update(sheaf, reference_vec, center=0.0, width=1.0)

    # Route a dummy packet
    packet = {"text_surface": "Hello world!"}
    chosen_node = hybrid_route_packet(packet, sheaf, dam, center=0.0, width=1.0)
    print(f"Packet routed to node: {chosen_node}")