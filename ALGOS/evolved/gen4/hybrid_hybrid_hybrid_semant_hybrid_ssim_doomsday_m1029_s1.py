# DARWIN HAMMER — match 1029, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_semantic_neig_hybrid_sketches_hybr_m57_s0.py (gen3)
# parent_b: hybrid_ssim_doomsday_calendar_m82_s0.py (gen1)
# born: 2026-05-29T23:32:29Z

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, sqrt, sin, pi
from random import random
from sys import exit
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Dict, List, Sequence

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

    def validate(self) -> None:
        if min(self.length, self.width, self.height) <= 0:
            raise ValueError("dimensions must be positive")

@dataclass(frozen=True)
class Document:
    id: str
    vector: List[float]

    def validate(self) -> None:
        if not self.vector:
            raise ValueError("document vector must not be empty")

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

def sphericity_index(morphology: Morphology) -> float:
    morphology.validate()
    return (morphology.length * morphology.width * morphology.height) ** (1.0 / 3.0) / max(morphology.length, morphology.width, morphology.height)

def flatness_index(morphology: Morphology) -> float:
    morphology.validate()
    return (morphology.length + morphology.width) / (2.0 * morphology.height)

def hybrid_ssim(x: Sequence[float], y: Sequence[float], dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x:
        raise ValueError('samples must not be empty')
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))

def generate_periodic_signal(days: int) -> Sequence[float]:
    signal = [0.0] * days
    for i in range(days):
        signal[i] = sin(2 * pi * i / 7)
    return signal

def calculate_recovery_priority(morphology: Morphology) -> float:
    return sphericity_index(morphology) * flatness_index(morphology)

def compare_signals_with_recovery_priority(x: Sequence[float], y: Sequence[float], morphology: Morphology) -> float:
    recovery_priority = calculate_recovery_priority(morphology)
    similarity = hybrid_ssim(x, y)
    return similarity * recovery_priority

def calculate_document_similarity(document: Document, signal: Sequence[float], morphology: Morphology) -> float:
    document.validate()
    morphology.validate()
    document_vector = document.vector
    similarity = hybrid_ssim(document_vector, signal)
    recovery_priority = calculate_recovery_priority(morphology)
    return similarity * recovery_priority

def main() -> None:
    morphology = Morphology(1.0, 2.0, 3.0, 4.0)
    document = Document("id", [1.0, 2.0, 3.0])
    signal = generate_periodic_signal(14)
    similarity = compare_signals_with_recovery_priority(signal, signal, morphology)
    document_similarity = calculate_document_similarity(document, signal, morphology)
    print(f"Similarity between signals: {similarity}")
    print(f"Similarity between document and signal: {document_similarity}")

if __name__ == "__main__":
    main()