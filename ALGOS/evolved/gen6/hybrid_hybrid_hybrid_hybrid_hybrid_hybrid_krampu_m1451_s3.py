# DARWIN HAMMER — match 1451, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m612_s2.py (gen5)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_hybrid_ternar_m55_s0.py (gen4)
# born: 2026-05-29T23:36:35Z

import datetime as dt
import math
import numpy as np
from collections import Counter
from typing import List, Tuple, Iterable, Dict, Optional


def doomsday_numpy(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """
    Vectorised weekday calculation (Sun=0 … Sat=6).

    Parameters
    ----------
    years, months, days : np.ndarray
        Arrays of equal shape containing integer year, month and day values.

    Returns
    -------
    np.ndarray
        Array of the same shape containing the weekday index where Sunday == 0.
    """
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    # Convert each date to a Python datetime and ask for its weekday (Mon=0 … Sun=6)
    py_weekday = np.fromiter(
        (
            dt.datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    # Shift so that Sunday becomes 0
    return (py_weekday + 1) % 7


def weekday_counts(dates: Iterable[dt.date]) -> np.ndarray:
    """
    Return a length‑7 array with counts of each weekday (Sun=0 … Sat=6).

    Parameters
    ----------
    dates : iterable of datetime.date or datetime.datetime

    Returns
    -------
    np.ndarray
        Integer counts for each weekday.
    """
    counts = np.zeros(7, dtype=int)
    for d in dates:
        if isinstance(d, dt.datetime):
            d = d.date()
        counts[(d.weekday() + 1) % 7] += 1
    return counts


def gini_coefficient(values: np.ndarray) -> float:
    """
    Robust Gini coefficient for a 1‑D non‑negative array.

    Parameters
    ----------
    values : np.ndarray
        Non‑negative numbers.

    Returns
    -------
    float
        Gini coefficient in [0, 1].
    """
    if values.size == 0:
        return 0.0
    values = values.astype(float)
    if np.all(values == 0):
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


def compute_health(
    curvature: float,
    degree: int,
    max_degree: int,
    gini: float,
    recovery_priority: float = 0.5,
) -> float:
    """
    A composite health metric that blends curvature, degree imbalance and a
    recovery priority term.

    Parameters
    ----------
    curvature : float
        Ollivier‑Ricci curvature for the node (typically in [-1, 1]).
    degree : int
        Number of neighbours of the node.
    max_degree : int
        Maximum possible degree (graph size - 1).
    gini : float
        Gini coefficient of the degree distribution of the whole graph.
    recovery_priority : float, optional
        Weight given to the recovery term; default 0.5.

    Returns
    -------
    float
        Health in [0, 1] where 1 is perfectly healthy.
    """
    # Normalise degree to a failure‑rate like quantity
    degree_rate = min(degree / max_degree, 1.0)

    # Curvature contributes positively when it is high (close to 1) and
    # negatively when it is low (close to -1).  We map it linearly to [0,1].
    curvature_score = (curvature + 1) / 2.0

    # Combine the three signals; the Gini term penalises graphs with
    # highly skewed degree distributions.
    health = (
        (1 - degree_rate)          # fewer neighbours → less overload
        * curvature_score         # well‑curved neighbourhoods are healthier
        * (1 - gini)               # more egalitarian degree spread is better
        * (1 - recovery_priority) # higher priority → lower health (more at risk)
    )
    return float(np.clip(health, 0.0, 1.0))


def hybrid_build_adj(matrix: np.ndarray, radius: float = 1.0) -> List[Tuple[int, int]]:
    """
    Build an undirected adjacency list from Euclidean distances.

    Parameters
    ----------
    matrix : np.ndarray, shape (n, d)
        Coordinate matrix for n nodes in d‑dimensional space.
    radius : float, optional
        Edge is created when Euclidean distance < radius. Default 1.0.

    Returns
    -------
    list of tuple
        Each tuple (i, j) with i < j denotes an undirected edge.
    """
    n = matrix.shape[0]
    adj: List[Tuple[int, int]] = []
    # Compute the full distance matrix once (more efficient than nested loops)
    dists = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    # Upper‑triangular mask excludes self‑loops and duplicate edges
    mask = np.triu(dists < radius, k=1)
    i_idx, j_idx = np.where(mask)
    adj = list(zip(i_idx.tolist(), j_idx.tolist()))
    return adj


def _neighbor_dict(adj: List[Tuple[int, int]], n: int) -> Dict[int, List[int]]:
    """
    Helper to convert edge list to a dict mapping each node to its neighbours.
    """
    neigh: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j in adj:
        neigh[i].append(j)
        neigh[j].append(i)
    return neigh


def hybrid_node_curvature(
    adj: List[Tuple[int, int]],
    matrix: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature per node using uniform probability
    measures on the 1‑step neighbourhood (including the node itself).

    The curvature κ(i, j) for an edge (i, j) is approximated by

        κ(i, j) = 1 - W₁(μ_i, μ_j) / d(i, j)

    where μ_i is the uniform distribution over {i} ∪ N(i) and d(i, j) is the
    Euclidean distance between the two nodes.  The node curvature is the
    average of κ over all incident edges.

    Parameters
    ----------
    adj : list of tuple
        Edge list produced by ``hybrid_build_adj``.
    matrix : np.ndarray, shape (n, d)
        Coordinate matrix.
    alpha : float, optional
        Smoothing factor for self‑weight in the probability measure.
        ``alpha=0`` gives pure uniform over neighbours, ``alpha=1`` puts all mass on
        the node itself. Default 0.5.

    Returns
    -------
    np.ndarray, shape (n,)
        Average curvature per node (in [-1, 1]).
    """
    n = matrix.shape[0]
    neigh = _neighbor_dict(adj, n)

    # Pre‑compute Euclidean distances for all edges
    edge_dist = {}
    for i, j in adj:
        edge_dist[(i, j)] = np.linalg.norm(matrix[i] - matrix[j])

    curvature = np.zeros(n, dtype=float)

    for i in range(n):
        incident = [(i, j) if i < j else (j, i) for j in neigh[i]]
        if not incident:
            curvature[i] = 0.0
            continue

        kappa_sum = 0.0
        for (u, v) in incident:
            # Build the two probability vectors μ_u and μ_v
            nu_u = neigh[u] + [u]  # include self
            nu_v = neigh[v] + [v]

            # Uniform weights with optional self‑bias
            w_u = np.full(len(nu_u), (1 - alpha) / len(neigh[u]) if neigh[u] else 0.0)
            w_v = np.full(len(nu_v), (1 - alpha) / len(neigh[v]) if neigh[v] else 0.0)
            # Add the self‑mass
            w_u = np.append(w_u, alpha)
            w_v = np.append(w_v, alpha)

            # Build the cost matrix between support points
            pts_u = matrix[nu_u]
            pts_v = matrix[nu_v]
            cost = np.linalg.norm(pts_u[:, None, :] - pts_v[None, :, :], axis=2)

            # Solve the optimal transport problem with the simple linear‑program
            # because the supports are tiny (≤ n).  We use the classic
            # Hungarian algorithm via scipy if it were available, but to keep the
            # dependency footprint minimal we employ a greedy matching which is
            # sufficient for our synthetic use‑case.
            # Greedy matching: pair smallest remaining cost.
            flat_cost = cost.ravel()
            idx = np.argsort(flat_cost)
            used_u = np.zeros(len(nu_u), dtype=bool)
            used_v = np.zeros(len(nu_v), dtype=bool)
            transport = 0.0
            for k in idx:
                iu, jv = divmod(k, len(nu_v))
                if used_u[iu] or used_v[jv]:
                    continue
                amount = min(w_u[iu], w_v[jv])
                transport += amount * cost[iu, jv]
                w_u[iu] -= amount
                w_v[jv] -= amount
                if w_u[iu] == 0:
                    used_u[iu] = True
                if w_v[jv] == 0:
                    used_v[jv] = True
                if np.all(used_u) or np.all(used_v):
                    break

            # Ollivier‑Ricci curvature for this edge
            d_ij = edge_dist[(u, v)]
            kappa = 1.0 - transport / d_ij if d_ij != 0 else 0.0
            kappa_sum += kappa

        curvature[i] = kappa_sum / len(incident)

    return curvature


def krampus_ollivier_sheaf(
    matrix: np.ndarray,
    curvature: np.ndarray,
    adj: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Produce a 3‑dimensional embedding that blends the original coordinates
    with curvature‑driven diffusion.

    For each node we perform a curvature‑weighted average of its neighbours'
    positions and then concatenate the scalar curvature as a fourth coordinate.
    The final output is therefore an (n, 4) array; the first three columns are
    spatial, the last column encodes curvature.

    Parameters
    ----------
    matrix : np.ndarray, shape (n, 3)
        Original 3‑D coordinates.
    curvature : np.ndarray, shape (n,)
        Node curvature values.
    adj : list of tuple
        Edge list.

    Returns
    -------
    np.ndarray, shape (n, 4)
        Enriched embedding.
    """
    n = matrix.shape[0]
    neigh = _neighbor_dict(adj, n)

    enriched = np.empty((n, 4), dtype=float)

    for i in range(n):
        nb = neigh[i]
        if nb:
            # Curvature‑based weights: higher curvature → larger influence
            raw_weights = curvature[nb] + 1e-8  # avoid zero division
            weights = raw_weights / raw_weights.sum()
            blended = (matrix[nb] * weights[:, None]).sum(axis=0)
        else:
            blended = matrix[i]  # isolated node stays put

        enriched[i, :3] = blended
        enriched[i, 3] = curvature[i]

    return enriched


def hybrid_health(
    matrix: np.ndarray,
    curvature: np.ndarray,
    adj: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Compute a health score for each node using curvature, degree information,
    and the global degree‑distribution Gini coefficient.

    Parameters
    ----------
    matrix : np.ndarray, shape (n, d)
        Coordinate matrix (unused directly but kept for API symmetry).
    curvature : np.ndarray, shape (n,)
        Node curvature.
    adj : list of tuple
        Edge list.

    Returns
    -------
    np.ndarray, shape (n,)
        Health scores in [0, 1].
    """
    n = matrix.shape[0]
    neigh = _neighbor_dict(adj, n)

    # Global degree distribution Gini (captures heterogeneity)
    degrees = np.array([len(neigh[i]) for i in range(n)], dtype=int)
    gini = gini_coefficient(degrees.astype(float))

    health = np.empty(n, dtype=float)
    for i in range(n):
        health[i] = compute_health(
            curvature=curvature[i],
            degree=len(neigh[i]),
            max_degree=n - 1,
            gini=gini,
            recovery_priority=0.5,
        )
    return health


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Demo: generate a synthetic point cloud, build the graph, compute curvature,
    # produce the enriched embedding and evaluate node health.
    # ----------------------------------------------------------------------
    np.random.seed(0)
    # 10 random points in 3‑D space
    matrix = np.random.rand(10, 3)

    # Build adjacency based on Euclidean proximity
    adj = hybrid_build_adj(matrix, radius=0.7)

    # Compute Ollivier‑Ricci curvature per node
    curvature = hybrid_node_curvature(adj, matrix, alpha=0.5)

    # Produce the curvature‑augmented embedding (3‑D + curvature)
    enriched = krampus_ollivier_sheaf(matrix, curvature, adj)

    # Compute health scores
    health = hybrid_health(matrix, curvature, adj)

    # Display results
    np.set_printoptions(precision=4, suppress=True)
    print("Enriched embedding (x, y, z, curvature):")
    print(enriched)
    print("\nHealth scores:")
    print(health)