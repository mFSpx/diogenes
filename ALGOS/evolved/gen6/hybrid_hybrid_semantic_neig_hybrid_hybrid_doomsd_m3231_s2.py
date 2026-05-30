# DARWIN HAMMER — match 3231, survivor 2
# gen: 6
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s0.py (gen2)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_hybrid_hybrid_m2697_s2.py (gen5)
# born: 2026-05-29T23:48:39Z

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, sqrt
from sys import exit
from pathlib import Path

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class Document:
    id: str
    vector: list[float]

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def _cos(a, b):
    den = sqrt(sum(x * x for x in a)) * sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den

def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    dates = np.stack([years, months, days], axis=-1).astype('datetime64[D]')
    np_weekday = dates.astype('datetime64[D]').astype('datetime64[ns]').astype('int64')
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (datetime.utcfromtimestamp(int(d.astype('datetime64[s]'))).weekday() for d in flat),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    return (py_weekday + 1) % 7

def gini_coefficient_numpy(values: np.ndarray) -> float:
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    xs = np.sort(values.astype(float))
    if xs.size == 0 or xs.sum() == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = xs.size
    i = np.arange(1, n + 1)  # 1‑based index
    numerator = np.sum((2 * i - n - 1) * xs)
    denominator = n * xs.sum()
    return abs(numerator / denominator)

def weekday_counts(dates: list) -> np.ndarray:
    years, months, days = [], [], []
    for d in dates:
        if isinstance(d, datetime):
            y, m, day = d.year, d.month, d.day
        else:
            y, m, day = d
        years.append(y)
        months.append(m)
        days.append(day)
    years, months, days = np.array(years), np.array(months), np.array(days)
    return doomsday_numpy(years, months, days)

def hybrid_recovery_priority(dates: list, documents: list[Document], m: Morphology) -> float:
    if not dates or not documents:
        return 0.0
    weekday_counts_array = weekday_counts(dates)
    gini_coefficient = gini_coefficient_numpy(weekday_counts_array)
    sphericity = sphericity_index(m.length, m.width, m.height)
    semantic_recoverypriority = recovery_priority(m)
    document_similarities = [_cos(document.vector, [sphericity, gini_coefficient]) for document in documents]
    return semantic_recoverypriority * np.prod(document_similarities)

def test_hybrid_recovery_priority():
    dates = [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)]
    documents = [Document("1", [0.5, 0.5]), Document("2", [0.3, 0.7])]
    m = Morphology(1.0, 1.0, 1.0, 1.0)
    return hybrid_recovery_priority(dates, documents, m)

if __name__ == "__main__":
    print(test_hybrid_recovery_priority())