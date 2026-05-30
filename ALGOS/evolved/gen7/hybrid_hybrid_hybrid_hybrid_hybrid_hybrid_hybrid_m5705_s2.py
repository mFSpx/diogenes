# DARWIN HAMMER — match 5705, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_doomsd_hybrid_nlms_hybrid_h_m1409_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_hoeffding_tre_m1469_s1.py (gen6)
# born: 2026-05-30T00:04:23Z

import datetime as dt
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Hashable, Sequence

import numpy as np

FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def words(text: str) -> List[str]:
    return [word for word in (text or "").lower().split() if word.isalpha()]

def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

def weekday_sakamoto(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)

def gini_coefficient(values: np.ndarray) -> float:
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non‑negative")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2 * i - n - 1) * x)
    denominator = n * x.sum()
    return float(numerator / denominator)

def predict(weights: np.ndarray, x: np.ndarray) -> float:
    return np.dot(weights, x)

def hoeffding_bound(gini: float, n: int) -> float:
    return math.sqrt((gini * (1 - gini) * math.log(2)) / (2 * n))

def hybrid_predict(text: str, weights: np.ndarray) -> float:
    lsm = lsm_vector(text)
    gini = gini_coefficient(np.array(list(lsm.values())))
    hoeffding = hoeffding_bound(gini, len(lsm))
    lsm_array = np.array(list(lsm.values()))
    prediction = predict(weights, lsm_array)
    return prediction + hoeffding

def adaptive_hybrid_predict(text: str, weights: np.ndarray, learning_rate: float) -> float:
    lsm = lsm_vector(text)
    gini = gini_coefficient(np.array(list(lsm.values())))
    hoeffding = hoeffding_bound(gini, len(lsm))
    lsm_array = np.array(list(lsm.values()))
    prediction = predict(weights, lsm_array)
    new_weights = weights + learning_rate * (prediction - hoeffding)
    return prediction + hoeffding, new_weights

def main():
    years = np.array([2022, 2023, 2024])
    months = np.array([1, 2, 3])
    days = np.array([1, 2, 3])
    weekdays = weekday_sakamoto(years, months, days)
    print(weekdays)

    text = "This is a test sentence."
    weights = np.array([0.1, 0.2, 0.3])
    prediction, new_weights = adaptive_hybrid_predict(text, weights, learning_rate=0.01)
    print(prediction)

if __name__ == "__main__":
    main()