# DARWIN HAMMER — match 27, survivor 4
# gen: 2
# parent_a: hybrid_hard_truth_math_model_pool_m8_s4.py (gen1)
# parent_b: kan.py (gen0)
# born: 2026-05-29T23:25:23Z

import hashlib
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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

def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)

def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))

    handcrafted = [
        total_words / 500.0,
        sum(len(w) for w in ws) / total_words / 12.0,
        (text.count("\n") + 1) / 200.0,
        sum(text.count(p) for p in "!?") / total_chars,
        sum(text.count(p) for p in ";:") / total_chars,
        sum(text.count(p) for p in "-—") / total_chars,
        sum(1 for ch in text if ch.isupper()) / total_chars,
        sum(1 for ch in text if ch in PUNCT) / total_chars,
    ]

    lsm = list(lsm_vector(text).values())
    needed = dim - len(handcrafted)
    if needed > 0:
        repeats = (needed + len(lsm) - 1) // len(lsm)
        lsm = (lsm * repeats)[:needed]
    else:
        lsm = lsm[: dim - len(handcrafted)]

    vec = np.asarray(handcrafted + lsm, dtype=np.float64)
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate(
        (
            np.full(k, grid[0]),
            grid,
            np.full(k, grid[-1]),
        )
    )
    n_basis = len(grid) - 1

    B = np.where(
        (x[:, None] >= t[:-1]) & (x[:, None] < t[1:]), 1.0, 0.0
    )

    for d in range(1, k):
        denom1 = t[d + np.arange(len(t) - k - 1)] - t[np.arange(len(t) - k - 1)]
        denom2 = t[d + 1 + np.arange(len(t) - k - 1)] - t[1 + np.arange(len(t) - k - 1)]

        term1 = np.where(
            denom1 > 0,
            ((x[:, None] - t[np.arange(len(t) - k - 1)]) / denom1) * B[:, :-1],
            0.0,
        )
        term2 = np.where(
            denom2 > 0,
            ((t[d + 1 + np.arange(len(t) - k - 1)] - x[:, None]) / denom2) * B[:, 1:],
            0.0,
        )
        B = term1 + term2

    return B[:, :n_basis]

@dataclass
class KanLayerParams:
    coeff: np.ndarray  
    grid: np.ndarray   
    order: int         

def init_hybrid_layer(in_dim: int, out_dim: int, grid_size: int = 10, order: int = 3) -> KanLayerParams:
    grid = np.linspace(-1.0, 1.0, grid_size)
    n_basis = grid_size - 1
    rng = np.random.default_rng()
    coeff = rng.normal(loc=0.0, scale=0.1, size=(out_dim, in_dim, n_basis))
    return KanLayerParams(coeff=coeff, grid=grid, order=order)

def kan_layer(x: np.ndarray, params: KanLayerParams) -> np.ndarray:
    N, in_dim = x.shape
    out_dim, _, n_basis = params.coeff.shape
    k = params.order

    basis = np.empty((N, in_dim, n_basis))
    for i in range(in_dim):
        basis[:, i, :] = bspline_basis(x[:, i], params.grid, k)

    return np.einsum('ijk,ikl->ijl', basis, params.coeff)

def hybrid_feature_vector(text: str, params: KanLayerParams) -> np.ndarray:
    features = stylometry_features(text)
    return kan_layer(features[None, :], params)[0]

def hybrid_predict(text: str, params: KanLayerParams) -> np.ndarray:
    return hybrid_feature_vector(text, params)