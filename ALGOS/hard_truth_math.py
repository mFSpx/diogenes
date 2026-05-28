#!/usr/bin/env python3
"""Hard-truth telemetry algorithms for LUCIDOTA.

Pure/reusable math: LSM vectors, stylometry features/classifier helpers,
ISO/vector parsing. Runtime scripts own DB reads/writes only.
No LLM calls.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from collections import Counter
from typing import Any

import numpy as np

FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PERSONA_PATTERNS: dict[str, re.Pattern[str]] = {
    "ponyboy": re.compile(r"\bponyboy\b", re.I),
    "northern_strike": re.compile(r"\bnorthern\.?strike\b", re.I),
    "lucidota": re.compile(r"\blucidota\b|\bluci\b", re.I),
    "indy_reads": re.compile(r"\bindy[_ -]?reads\b|\bindy\b", re.I),
    "zachary": re.compile(r"\bzachary\b|\bzach\b", re.I),
    "operator": re.compile(r"\boperator\b", re.I),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {cat: sum(cnt[w] for w in vocab) / total for cat, vocab in FUNCTION_CATS.items()}


def lsm_score(a: dict[str, float], b: dict[str, float]) -> tuple[float, dict[str, float]]:
    detail: dict[str, float] = {}
    vals: list[float] = []
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = round(score, 6)
        vals.append(score)
    return float(sum(vals) / len(vals)), detail



def parse_iso(s: Any) -> dt.datetime | None:
    if not s:
        return None
    try:
        raw = str(s).replace("Z", "+00:00")
        val = dt.datetime.fromisoformat(raw)
        return val if val.tzinfo else val.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None



def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))
    vals: list[float] = [
        total_words / 500.0,
        sum(len(w) for w in ws) / total_words / 12.0,
        (text.count("\n") + 1) / 200.0,
        sum(text.count(p) for p in "!?") / total_chars,
        sum(text.count(p) for p in ";:") / total_chars,
        sum(text.count(p) for p in "-—") / total_chars,
        sum(1 for ch in text if ch.isupper()) / total_chars,
        len(re.findall(r"\b[A-Z]{2,}\b", text)) / total_words,
    ]
    lv = lsm_vector(text)
    vals.extend(lv.get(cat, 0.0) for cat in sorted(FUNCTION_CATS))
    vec = np.zeros(dim, dtype=np.float64)
    base = np.array(vals[: min(len(vals), dim)], dtype=np.float64)
    vec[: len(base)] = base
    cleaned = " ".join(ws)
    for i in range(max(0, len(cleaned) - 2)):
        gram = cleaned[i : i + 3]
        idx = 20 + (stable_hash(gram) % max(1, dim - 20))
        vec[idx] += 1.0
    norm = np.linalg.norm(vec[20:])
    if norm > 0:
        vec[20:] /= norm
    return vec


def persona_label(text: str) -> str:
    hits = [(label, len(rx.findall(text))) for label, rx in PERSONA_PATTERNS.items()]
    hits = [(l, c) for l, c in hits if c > 0]
    if not hits:
        return ""
    hits.sort(key=lambda x: (-x[1], x[0]))
    return hits[0][0]



def train_hinge_classifier(samples: list[dict[str, Any]], epochs: int = 5) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels = sorted({s["label"] for s in samples if s.get("label")})
    if not labels:
        return [], np.zeros((0, 96)), np.zeros(96)
    X = np.vstack([stylometry_features(s["text"]) for s in samples])
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sigma
    W = np.zeros((len(labels), Xs.shape[1]), dtype=np.float64)
    lr = 0.03
    for _ in range(epochs):
        for x, s in zip(Xs, samples):
            if not s.get("label"):
                continue
            yi = labels.index(s["label"])
            for ci in range(len(labels)):
                y = 1.0 if ci == yi else -1.0
                margin = y * float(W[ci].dot(x))
                if margin < 1.0:
                    W[ci] += lr * y * x
    return labels, W, np.concatenate([mu, sigma])


def softmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    z = scores - np.max(scores)
    exp = np.exp(z)
    return exp / max(float(exp.sum()), 1e-9)



def parse_vector(text: str) -> np.ndarray | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    try:
        return np.fromstring(s, sep=",", dtype=np.float64)
    except ValueError:
        return None


