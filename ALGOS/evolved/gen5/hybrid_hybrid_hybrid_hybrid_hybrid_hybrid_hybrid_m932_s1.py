# DARWIN HAMMER — match 932, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_geomet_m165_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m192_s3.py (gen4)
# born: 2026-05-29T23:31:47Z

import numpy as np
from collections import Counter
from datetime import date as dt
from typing import Dict, List, Tuple, Union

class Multivector:
    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items()})

def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (
            dt.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    return (py_weekday + 1) % 7

def weekday_counts(
    dates: List[Union[dt, Tuple[int, int, int]]],
) -> np.ndarray:
    years, months, days = [], [], []
    for d in dates:
        if isinstance(d, dt):
            y, m, day = d.year, d.month, d.day
        elif isinstance(d, tuple) and len(d) == 3:
            y, m, day = d
        else:
            raise ValueError("Date object not recognized")
        years.append(y)
        months.append(m)
        days.append(day)
    return np.array([years, months, days])

def health_score(reconstruction_risk_score: float, failure_rate: float, recovery_priority: float) -> float:
    return (1 - (reconstruction_risk_score * failure_rate)) * (1 - recovery_priority)

def decision_hygiene_score(multivector: Multivector, health_score: float) -> float:
    return multivector.scalar_part() * health_score

def calculate_weekday_weights(weekday_counts: np.ndarray) -> Dict[int, float]:
    weights = {}
    for i in range(7):
        weights[i] = np.sum(weekday_counts == i) / len(weekday_counts)
    return weights

def hybrid_operation(
    dates: List[Union[dt, Tuple[int, int, int]]],
    reconstruction_risk_score: float,
    failure_rate: float,
    recovery_priority: float,
) -> float:
    weekday_count = weekday_counts(dates)
    weekday_numpy = doomsday_numpy(weekday_count[0], weekday_count[1], weekday_count[2])
    weekday_weights = calculate_weekday_weights(weekday_numpy)
    multivector = Multivector({frozenset([i]): weight for i, weight in weekday_weights.items()}, 1)
    health = health_score(reconstruction_risk_score, failure_rate, recovery_priority)
    decision_hygiene = decision_hygiene_score(multivector, health)
    return decision_hygiene

if __name__ == "__main__":
    dates = [dt(2022, 1, 1), dt(2022, 1, 2), dt(2022, 1, 3)]
    reconstruction_risk_score = 0.5
    failure_rate = 0.2
    recovery_priority = 0.1
    result = hybrid_operation(dates, reconstruction_risk_score, failure_rate, recovery_priority)
    print(result)