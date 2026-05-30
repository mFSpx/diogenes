# DARWIN HAMMER — match 3037, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_decisi_m153_s2.py (gen3)
# parent_b: hybrid_ternary_router_ssim_m1_s1.py (gen1)
# born: 2026-05-29T23:47:24Z

import numpy as np
from collections import Counter
from typing import Any, Dict, List, Tuple
import re
import math

# ----------------------------------------------------------------------
# Parent A – Linguistic function categories
# ----------------------------------------------------------------------
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
        "no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren".split()
    ),
}

NEGATIVE_WEIGHTS = 0.5
POSITIVE_WEIGHTS = 1.0

def lsm_vector(text: str) -> np.ndarray:
    words = text.split()
    vector = np.zeros(len(FUNCTION_CATS))
    for word in words:
        for cat, funcs in FUNCTION_CATS.items():
            if word.lower() in funcs:
                vector[list(FUNCTION_CATS.keys()).index(cat)] += 1
    return vector / np.sum(vector) if np.sum(vector) != 0 else vector

def feature_vector(text: str) -> np.ndarray:
    regex_counts = Counter(re.findall(r'\b\w+\b', text))
    vector = np.array(list(regex_counts.values()))
    weights = np.where(vector > 0, POSITIVE_WEIGHTS, -NEGATIVE_WEIGHTS)
    return vector * weights

def combined_similarity(text_a: str, text_b: str, alpha: float = 0.5) -> float:
    f_a = lsm_vector(text_a)
    f_b = lsm_vector(text_b)
    c_a = feature_vector(text_a)
    c_b = feature_vector(text_b)
    h_a = np.concatenate((f_a, c_a))
    h_b = np.concatenate((f_b, c_b))
    similarity = np.dot(h_a, h_b) / (np.linalg.norm(h_a) * np.linalg.norm(h_b)) if np.linalg.norm(h_a) * np.linalg.norm(h_b) != 0 else 0
    return alpha * similarity + (1 - alpha) * ssim(h_a, h_b)

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 1.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError("Input vectors must have the same length")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    ssim_value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)) if (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2) != 0 else 0
    return ssim_value

if __name__ == "__main__":
    text_a = "This is a test sentence."
    text_b = "This test sentence is only a test."
    similarity = combined_similarity(text_a, text_b)
    print(f"Similarity: {similarity:.4f}")