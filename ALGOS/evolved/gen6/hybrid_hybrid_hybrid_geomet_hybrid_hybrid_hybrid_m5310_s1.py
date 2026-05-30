# DARWIN HAMMER — match 5310, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m1375_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_model__hybrid_doomsday_cale_m193_s1.py (gen4)
# born: 2026-05-30T00:01:10Z

"""
Hybrid algorithm merging:

- Parent A: Clifford geometric product, Voronoi partitioning, Fisher‑JEPA hyperdimensional encoding.
- Parent B: Doomsday weekday calculation and Gini inequality coefficient.

Mathematical bridge:
The geometric product supplies a scalar (dot) distance between points, which drives the Voronoi
assignment.  The set of dates attached to each Voronoi cell is reduced to a weekday
frequency distribution; the Gini coefficient of this distribution is evaluated using the
scalar distances (squared Euclidean) as the weighting “numeric distribution”.  Finally the
resulting Gini value together with the cell identifier is deterministically hashed into a
bipolar hypervector, providing a single high‑dimensional representation that fuses both
parental structures.
"""

import math
import random
import sys
import pathlib
import datetime
import numpy as np

# ----------------------------------------------------------------------
# Clifford algebra helpers (Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)
        elif lst[i] == lst[i + 1]:
            # cancel identical indices (square to scalar)
            lst.pop(i)
            lst.pop(i)
            n -= 2
            i = max(i - 1, 0)
        else:
            i += 1
    return frozenset(lst), sign


def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return result, sign


class Multivector:
    """Very small multivector container used for geometric product."""

    def __init__(self, components=None):
        # components: dict mapping frozenset of basis indices to scalar coefficient
        self.components = components if components is not None else {}

    def __add__(self, other):
        res = self.components.copy()
        for k, v in other.components.items():
            res[k] = res.get(k, 0.0) + v
        return Multivector(res)

    def __mul__(self, other):
        """Geometric product."""
        result = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result[blade_res] = result.get(blade_res, 0.0) + coeff_a * coeff_b * sign
        return Multivector(result)

    def scalar_part(self):
        """Return the scalar (grade‑0) component."""
        return self.components.get(frozenset(), 0.0)


def vector_to_mv(vec):
    """Convert a 1‑D numpy array to a grade‑1 multivector."""
    comps = {}
    for i, val in enumerate(vec):
        if val != 0:
            comps[frozenset({i})] = float(val)
    return Multivector(comps)


def geometric_product(a, b):
    """Geometric product of two vectors given as 1‑D numpy arrays.
    Returns a Multivector whose scalar part equals the dot product.
    """
    mv_a = vector_to_mv(a)
    mv_b = vector_to_mv(b)
    return mv_a * mv_b


def squared_distance(p, q):
    """Scalar squared Euclidean distance via geometric product."""
    diff = p - q
    gp = geometric_product(diff, diff)
    return gp.scalar_part()


# ----------------------------------------------------------------------
# Doomsday weekday (Parent B)
# ----------------------------------------------------------------------
def doomsday_weekday(date):
    """
    Compute the weekday index (0=Monday … 6=Sunday) using the Doomsday algorithm.
    """
    y = date.year
    m = date.month
    d = date.day

    # anchor day for the century
    century = y // 100
    anchor = (5 * (century % 4) + 2) % 7

    # year part
    yy = y % 100
    doomsday = (yy + yy // 4 + anchor) % 7

    # month offsets (for Gregorian calendar)
    month_offsets = {
        1: 3 if not is_leap_year(y) else 4,
        2: 28 if not is_leap_year(y) else 29,
        3: 14,
        4: 4,
        5: 9,
        6: 6,
        7: 11,
        8: 8,
        9: 5,
        10: 10,
        11: 7,
        12: 12,
    }
    offset = month_offsets[m]
    weekday = (doomsday + d - offset) % 7
    return weekday


def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


# ----------------------------------------------------------------------
# Gini coefficient (Parent B)
# ----------------------------------------------------------------------
def gini_coefficient(values, weights=None):
    """
    Compute the Gini coefficient of a non‑negative value list.
    If weights are provided they are used as the numeric distribution.
    """
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    if weights is not None:
        w = np.array(weights, dtype=float)
        # use weights as frequencies for sorting
        idx = np.argsort(arr)
        arr = arr[idx]
        w = w[idx]
        cumw = np.cumsum(w) / w.sum()
        cumx = np.cumsum(arr * w) / (arr * w).sum()
    else:
        arr = np.sort(arr)
        cumw = np.linspace(1 / len(arr), 1, len(arr))
        cumx = np.cumsum(arr) / arr.sum()
    B = np.trapz(cumx, cumw)  # area under Lorenz curve
    return 1 - 2 * B


# ----------------------------------------------------------------------
# Hyperdimensional encoding (Parent A)
# ----------------------------------------------------------------------
def deterministic_bipolar_vector(seed_int, dim=10000):
    """
    Produce a bipolar (+1 / -1) hypervector of length `dim`
    deterministically from an integer seed.
    """
    rng = random.Random(seed_int)
    return np.array([1 if rng.random() < 0.5 else -1 for _ in range(dim)], dtype=int)


def encode_hypervector(gini_value, cell_id, dim=10000):
    """
    Combine the Gini coefficient (float) and Voronoi cell identifier (int)
    into a single hypervector via XOR‑like combination of two bipolar vectors.
    """
    # Scale gini into an integer seed (0 … 2^31‑1)
    gini_seed = int(gini_value * (2**31 - 1))
    hv_gini = deterministic_bipolar_vector(gini_seed, dim)
    hv_cell = deterministic_bipolar_vector(cell_id, dim)
    # element‑wise multiplication acts as XOR for bipolar vectors
    return hv_gini * hv_cell


# ----------------------------------------------------------------------
# Voronoi partition using geometric product distance (Parent A)
# ----------------------------------------------------------------------
def voronoi_partition(points, seeds):
    """
    Assign each point to the nearest seed using squared_distance derived from
    the geometric product. Returns a list of cell indices parallel to `points`.
    """
    assignments = []
    for p in points:
        dists = [squared_distance(p, s) for s in seeds]
        assignments.append(int(np.argmin(dists)))
    return assignments


# ----------------------------------------------------------------------
# Hybrid operation demonstrating the fused topology
# ----------------------------------------------------------------------
def hybrid_hypervector(dates, points, seeds, dim=10000):
    """
    Full hybrid pipeline:
    1. Voronoi partition of `points` with `seeds` using geometric‑product distance.
    2. For each cell, collect the weekdays of the corresponding `dates`.
    3. Compute a Gini coefficient for the weekday frequency distribution,
       weighting each weekday by the squared distance of its point to the cell seed.
    4. Encode each cell's Gini value and cell id into a bipolar hypervector.
    5. Aggregate (element‑wise sum) all cell hypervectors and return the sign‑binarized result.
    """
    if len(dates) != len(points):
        raise ValueError("dates and points must have the same length")

    assignments = voronoi_partition(points, seeds)

    # Prepare containers per cell
    cell_weekdays = {i: [] for i in range(len(seeds))}
    cell_weights = {i: [] for i in range(len(seeds))}

    for date, pt, cell in zip(dates, points, assignments):
        wd = doomsday_weekday(date)
        cell_weekdays[cell].append(wd)
        # weight = inverse distance (larger distance -> smaller weight)
        dist = math.sqrt(squared_distance(pt, seeds[cell]))
        weight = 1.0 / (dist + 1e-9)
        cell_weights[cell].append(weight)

    # Encode each cell
    hypervectors = []
    for cell_id in range(len(seeds)):
        weekdays = cell_weekdays[cell_id]
        if not weekdays:
            continue
        # frequency of each weekday (0‑6)
        freq = np.bincount(weekdays, minlength=7)
        # Use the previously collected weights as the numeric distribution
        gini = gini_coefficient(freq, weights=cell_weights[cell_id])
        hv = encode_hypervector(gini, cell_id, dim)
        hypervectors.append(hv)

    if not hypervectors:
        # No data – return a zero vector
        return np.zeros(dim, dtype=int)

    # Aggregate by summation and then binarize to bipolar form
    agg = np.sum(hypervectors, axis=0)
    return np.where(agg >= 0, 1, -1)


# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def example_geometric_distance():
    """Show geometric‑product based distance between two random 3‑D points."""
    a = np.random.randn(3)
    b = np.random.randn(3)
    return math.sqrt(squared_distance(a, b))


def example_gini_from_dates(dates):
    """Compute Gini coefficient of weekday distribution for a list of dates."""
    weekdays = [doomsday_weekday(d) for d in dates]
    freq = np.bincount(weekdays, minlength=7)
    return gini_coefficient(freq)


def example_voronoi_random():
    """Create a tiny random Voronoi assignment and return the cell list."""
    pts = np.random.randn(10, 2)
    seeds = np.random.randn(3, 2)
    return voronoi_partition(pts, seeds)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # generate synthetic data
    num_points = 50
    points = np.random.randn(num_points, 2)
    seeds = np.random.randn(5, 2)

    # random dates within the last 10 years
    base = datetime.datetime.now()
    dates = [
        base - datetime.timedelta(days=random.randint(0, 365 * 10))
        for _ in range(num_points)
    ]

    hv = hybrid_hypervector(dates, points, seeds, dim=4096)
    print("Hybrid hypervector shape:", hv.shape)
    print("First 20 components:", hv[:20])
    print("Geometric distance example:", example_geometric_distance())
    print("Gini of random dates:", example_gini_from_dates(dates))
    print("Voronoi assignment example:", example_voronoi_random())