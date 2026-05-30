# DARWIN HAMMER — match 80, survivor 3
# gen: 1
# parent_a: omni_chaotic_sprint.py (gen0)
# parent_b: jepa_energy.py (gen0)
# born: 2026-05-29T23:26:42Z

"""Hybrid Omni-JEPA Engine
================================
This module fuses the *Chaotic Omni Sprint* graph‑processing core (Parent A) with the
Joint Embedding Predictive Architecture (JEPA) energy formulation (Parent B).

Mathematical Bridge
-------------------
- In the original ChaoticOmniEngine each active graph item defines a directed
  edge *(parent_uuid → uuid)* together with a scalar *weight*.
- JEPA models a temporal transition *y → x* with a latent variable *z* and defines
  the prediction energy  

  ``E(x, y, z) = || sθ(x) – pφ(sθ(y), z) ||₂²``  

  where ``sθ`` is an encoder and ``pφ`` a predictor.
- The hybrid treats every graph edge as a JEPA transition:
  * ``y``  = parent node,
  * ``x``  = child node,
  * ``z``  = edge weight (or a derived latent).
- Node attributes are embedded by ``sθ`` (the *encoder*).  The predictor
  ``pφ`` is a simple affine map that mixes the encoded parent with the latent.
- The total hybrid loss combines the JEPA energy over all edges with a VICReg
  regularizer that keeps the representation space well‑conditioned.

The implementation below stays within the allowed imports (numpy, standard
library) and provides three public functions that illustrate the hybrid
operation.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
EMBED_DIM = 64               # dimensionality of sθ representations
SEED = 42                    # deterministic randomness for reproducibility
MAX_NODES = 1000             # size of synthetic graph in smoke test
MAX_EDGES = 2000

random.seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------
# Synthetic graph generation (stand‑in for the DB‑driven fetch in Parent A)
# ----------------------------------------------------------------------
def generate_synthetic_graph(
    n_nodes: int = MAX_NODES,
    n_edges: int = MAX_EDGES,
) -> List[Dict]:
    """Create a list of graph items mimicking the schema returned by
    ``ChaoticOmniEngine.execute_seismic_ray_trace``.

    Each item contains:
        - ``item_uuid``: unique identifier (string)
        - ``parent_uuid``: identifier of the predecessor node (may be None)
        - ``term``: a textual label used by the encoder
        - ``weight``: a scalar edge weight (float)
        - ``detail``: placeholder payload (dict)
    """
    nodes = [f"node_{i}" for i in range(n_nodes)]
    items: List[Dict] = []

    # Ensure at least one root node without a parent
    root = nodes[0]
    items.append(
        {
            "item_uuid": root,
            "parent_uuid": None,
            "term": f"root_term_{root}",
            "weight": 1.0,
            "detail": {"info": "root"},
        }
    )

    for _ in range(n_edges):
        child = random.choice(nodes)
        parent = random.choice(nodes)
        # avoid self‑loops for clarity
        if child == parent:
            continue
        items.append(
            {
                "item_uuid": child,
                "parent_uuid": parent,
                "term": f"term_{child}",
                "weight": random.uniform(0.1, 2.0),
                "detail": {"info": f"edge_{parent}_to_{child}"},
            }
        )
    return items


# ----------------------------------------------------------------------
# JEPA components (Parent B)
# ----------------------------------------------------------------------
def encoder(node: Dict) -> np.ndarray:
    """Deterministic embedding of a graph node.

    The embedding is a linear projection of a hash‑derived vector onto the
    ``EMBED_DIM`` sphere.  This mimics ``sθ`` without external ML libraries.
    """
    # Hash the term string to a reproducible integer seed
    term = node["term"]
    term_seed = abs(hash(term)) % (2**32 - 1)
    rng = np.random.default_rng(term_seed)

    # Random vector then L2‑normalize to lie on the unit sphere
    vec = rng.normal(size=EMBED_DIM)
    norm = np.linalg.norm(vec) + 1e-9
    return vec / norm


# Predictor parameters (shared across all calls)
_PREDICTOR_WEIGHT_MATRIX = np.random.normal(
    loc=0.0, scale=0.1, size=(EMBED_DIM, EMBED_DIM)
)  # shape (D, D)
_PREDICTOR_BIAS = np.random.normal(loc=0.0, scale=0.01, size=EMBED_DIM)


def predictor(encoded_parent: np.ndarray, latent_z: float) -> np.ndarray:
    """Linear predictor ``pφ`` that mixes the parent embedding with a scalar latent.

    ``pφ(sθ(y), z) = W · (sθ(y) * z) + b``

    The multiplication by ``z`` scales the parent representation, modelling
    edge‑specific dynamics.
    """
    scaled = encoded_parent * latent_z
    return _PREDICTOR_WEIGHT_MATRIX @ scaled + _PREDICTOR_BIAS


def jepa_energy(
    encoded_child: np.ndarray, predicted_child: np.ndarray
) -> float:
    """Squared L2 distance between actual and predicted child embeddings."""
    diff = encoded_child - predicted_child
    return float(np.dot(diff, diff))


# ----------------------------------------------------------------------
# VICReg regularizer (from Parent B)
# ----------------------------------------------------------------------
def vicreg_regularizer(
    representations: np.ndarray,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
) -> float:
    """Compute the VICReg loss for a batch of representations.

    Parameters
    ----------
    representations: (N, D) array where N is batch size.
    sim_coeff, std_coeff, cov_coeff: weighting hyper‑parameters.
    eps: small constant for numerical stability.

    Returns a scalar regularization term.
    """
    N, D = representations.shape

    # Invariance term (here we set it to zero because we have no paired views)
    invariance = 0.0

    # Standard deviation term – encourages each dimension to have std >= 1
    std = np.sqrt(representations.var(axis=0) + eps)
    std_loss = np.mean(np.maximum(0.0, 1.0 - std))

    # Covariance term – off‑diagonal entries should be close to zero
    repr_centered = representations - representations.mean(axis=0)
    cov = (repr_centered.T @ repr_centered) / (N - 1)
    cov_off_diag = cov - np.diag(np.diag(cov))
    cov_loss = (cov_off_diag ** 2).sum() / D

    return sim_coeff * invariance + std_coeff * std_loss + cov_coeff * cov_loss


# ----------------------------------------------------------------------
# Hybrid core: combine graph traversal with JEPA energy
# ----------------------------------------------------------------------
def hybrid_compute_energy(
    graph_items: List[Dict],
) -> Tuple[float, int]:
    """Compute the total JEPA energy over all directed edges in ``graph_items``.

    The function:
        1. Encodes every distinct node once (caching results).
        2. For each edge (parent → child) obtains the latent ``z`` from the edge
           ``weight`` and evaluates ``E = || sθ(child) – pφ(sθ(parent), z) ||²``.
        3. Adds a VICReg regularizer on the set of all node embeddings.

    Returns
    -------
    total_energy: float – sum of edge energies plus regularizer.
    edge_count: int – number of evaluated edges.
    """
    # ------------------------------------------------------------------
    # 1. Build a cache of node embeddings
    # ------------------------------------------------------------------
    embedding_cache: Dict[str, np.ndarray] = {}
    for item in graph_items:
        uid = item["item_uuid"]
        if uid not in embedding_cache:
            embedding_cache[uid] = encoder(item)

    # ------------------------------------------------------------------
    # 2. Accumulate edge energies
    # ------------------------------------------------------------------
    total_edge_energy = 0.0
    edge_cnt = 0
    for item in graph_items:
        parent_uid = item["parent_uuid"]
        if parent_uid is None:
            continue  # root node has no incoming edge
        child_uid = item["item_uuid"]
        z = float(item["weight"])

        enc_parent = embedding_cache[parent_uid]
        enc_child = embedding_cache[child_uid]

        pred_child = predictor(enc_parent, z)
        total_edge_energy += jepa_energy(enc_child, pred_child)
        edge_cnt += 1

    # ------------------------------------------------------------------
    # 3. VICReg regularizer on the whole set of embeddings
    # ------------------------------------------------------------------
    all_embeddings = np.stack(list(embedding_cache.values()))
    reg_term = vicreg_regularizer(all_embeddings)

    total_energy = total_edge_energy + reg_term
    return total_energy, edge_cnt


# ----------------------------------------------------------------------
# Demonstration functions (required ≥3)
# ----------------------------------------------------------------------
def demo_graph_encoding(num_nodes: int = 10) -> None:
    """Showcase the encoder on a tiny synthetic graph."""
    demo_items = generate_synthetic_graph(n_nodes=num_nodes, n_edges=num_nodes * 2)
    uniq_nodes = {it["item_uuid"] for it in demo_items}
    print(f"Encoding {len(uniq_nodes)} unique nodes...")
    for uid in list(uniq_nodes)[:5]:
        dummy_item = {"item_uuid": uid, "term": f"term_{uid}"}
        vec = encoder(dummy_item)
        print(f"Node {uid[:8]}... → norm {np.linalg.norm(vec):.3f}")


def demo_energy_computation() -> None:
    """Compute JEPA energy on a small random graph and print the result."""
    items = generate_synthetic_graph(n_nodes=30, n_edges=80)
    energy, edges = hybrid_compute_energy(items)
    print(f"Computed hybrid JEPA energy over {edges} edges: {energy:.4f}")


def demo_vicreg_on_embeddings() -> None:
    """Generate random embeddings and evaluate the VICReg regularizer."""
    batch = np.random.normal(size=(64, EMBED_DIM))
    loss = vicreg_regularizer(batch)
    print(f"VICReg regularizer value on random batch: {loss:.6f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    start = time.time()
    print("[Hybrid Omni‑JEPA] Running smoke tests...")

    demo_graph_encoding()
    demo_energy_computation()
    demo_vicreg_on_embeddings()

    elapsed = time.time() - start
    print(f"[Hybrid Omni‑JEPA] Completed in {elapsed:.2f}s")