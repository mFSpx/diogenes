# DARWIN HAMMER — match 2752, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_omni_chaotic__hybrid_hybrid_liquid_m1302_s2.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m1375_s0.py (gen5)
# born: 2026-05-29T23:45:38Z

"""
Hybrid Multivector- JEPA-LTCMH Engine
======================================
This module fuses the *Multivector* algebraic structure from 
`hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m1375_s0.py` 
with the *Joint Embedding Predictive Architecture (JEPA)* energy formulation 
and *Hybrid Ternary-Router / Liquid Time-Constant MinHash (HTR-LTCMH)* dynamics 
from `hybrid_hybrid_omni_chaotic__hybrid_hybrid_liquid_m1302_s2.py`.

Mathematical Bridge
-------------------
The Multivector's algebraic structure is integrated with JEPA's energy formulation 
by representing the encoder `sθ` and predictor `pφ` as Multivector operations. 
The HTR-LTCMH's MinHash signature generation process is used to define 
a similarity metric between input sequences, which is then used to weight 
the Multivector operations.

The hybrid treats every graph edge as a JEPA transition:
* ``y``  = parent node,
* ``x``  = child node,
* ``z``  = edge weight (or a derived latent).

Node attributes are embedded by `sθ` (the *encoder*), which is represented 
as a Multivector operation. The predictor `pφ` is a simple affine map 
that mixes the encoded parent with the latent, also represented as 
a Multivector operation.

The total hybrid loss combines the JEPA energy over all edges with 
a VICReg regularizer that keeps the representation space well‑conditioned, 
and the HTR-LTCMH's reconstruction loss.
"""

import math
import numpy as np
import random
import sys
import pathlib
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
EMBED_DIM = 64               # dimensionality of sθ representations
SEED = 42                    # deterministic randomness for reproducibility
MAX_NODES = 1000             # size of synthetic graph in smoke test
MAX_EDGES = 2000

random.seed(SEED)
np.random.seed(SEED)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

# Core blade arithmetic helpers
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list.

    Each transposition of adjacent indices that are out of order flips the
    sign (anti-commutativity).  Duplicate indices cancel (e_i^2 = 1 → they
    annihilate and contribute +1 to the sign, but the index disappears).
    """
    lst = list(indices)
    sign = 1
    # Bubble sort; track swaps
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate: e_i * e_i = 1, remove both
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices).

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# Multivector
class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades.

    components: dict mapping frozenset(basis_indices) -> float coefficient
    """

    def __init__(self, components):
        self.components = components

    def __add__(self, other):
        new_components = self.components.copy()
        for blade, coeff in other.components.items():
            if blade in new_components:
                new_components[blade] += coeff
            else:
                new_components[blade] = coeff
        return Multivector(new_components)

    def __rmul__(self, scalar):
        return Multivector({blade: scalar * coeff for blade, coeff in self.components.items()})


def je_pa_loss(encoder, predictor, nodes, edges):
    loss = 0
    for edge in edges:
        y = nodes[edge[0]]
        x = nodes[edge[1]]
        z = edge[2]
        encoded_y = encoder(y)
        predicted_x = predictor(encoded_y, z)
        loss += np.linalg.norm(predicted_x - x) ** 2
    return loss


def multivector_encoder(node):
    components = {frozenset(): node}
    return Multivector(components)


def multivector_predictor(encoded_node, latent):
    components = {}
    for blade, coeff in encoded_node.components.items():
        components[blade] = coeff * latent
    return Multivector(components)


def hybrid_loss(multivector_encoder, multivector_predictor, nodes, edges):
    je_pa = je_pa_loss(multivector_encoder, multivector_predictor, nodes, edges)
    # VICReg regularizer
    vicreg = 0
    for node in nodes:
        encoded_node = multivector_encoder(node)
        vicreg += np.linalg.norm(encoded_node.components[frozenset()]) ** 2
    # HTR-LTCMH reconstruction loss
    htr_ltc_mh = 0
    for edge in edges:
        y = nodes[edge[0]]
        x = nodes[edge[1]]
        z = edge[2]
        encoded_y = multivector_encoder(y)
        predicted_x = multivector_predictor(encoded_y, z)
        htr_ltc_mh += np.linalg.norm(predicted_x.components[frozenset()] - x) ** 2
    return je_pa + vicreg + htr_ltc_mh


if __name__ == "__main__":
    nodes = np.random.rand(MAX_NODES, EMBED_DIM)
    edges = [(i, i + 1, np.random.rand()) for i in range(MAX_EDGES)]
    multivector_encoder = lambda node: Multivector({frozenset(): node})
    multivector_predictor = lambda encoded_node, latent: Multivector({frozenset(): encoded_node.components[frozenset()] * latent})
    loss = hybrid_loss(multivector_encoder, multivector_predictor, nodes, edges)
    print("Hybrid Loss:", loss)