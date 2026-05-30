# DARWIN HAMMER — match 3077, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_krampus_stick_hybrid_hybrid_hybrid_m1008_s1.py (gen5)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s6.py (gen2)
# born: 2026-05-29T23:47:35Z

"""
This module fuses the core topologies of hybrid_hybrid_krampus_stick_hybrid_hybrid_hybrid_m1008_s1.py (Parent A)
and hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s6.py (Parent B) by integrating the Pheromone-based 
social interaction model of Parent A with the NLMS (Normalized Least Mean Squares) adaptive filter of Parent B.

The mathematical bridge between the two parents lies in the use of signal processing and decay principles.
In Parent A, pheromone signals decay over time according to their half-life. Similarly, in Parent B, the NLMS 
algorithm adapts to changing signals by updating its weights based on the error between the predicted and actual 
outputs. By fusing these two concepts, we can create a hybrid system that adapts to changing social interactions 
and pheromone signals.

The hybrid system works as follows:

1.  The social interaction model of Parent A generates pheromone signals based on the frequency of words in a given text.
2.  The NLMS algorithm of Parent B is used to adapt to these pheromone signals, predicting the next signal value based 
    on past values and updating its weights accordingly.
3.  The adapted weights are then used to filter the pheromone signals, effectively denoising them and improving their 
    accuracy.

This fusion enables the creation of a more robust and adaptive system that can better model complex social interactions 
and pheromone signals.

"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter, deque
from datetime import datetime, timezone
from dataclasses import dataclass

Vector = list[float]

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(random.getrandbits(128))
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    _entries: dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> list[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> list[dict]:
        rows = []
        for entry in cls.get_by_surface(surface_key):
            before = entry.signal_value
            entry.apply_decay()
            rows.append({"surface_key": surface_key, "before": before, "after": entry.signal_value})
        return rows


def social_interaction_and_pheromone(text: str, g_best: Vector, k: int = 1, r1: float | None = None, seed: int | str | None = None) -> list[float]:
    words_list = text.split()
    word_counts = Counter(words_list)
    word_frequencies = [word_counts[word] / len(words_list) for word in word_counts]
    pheromone_value = np.mean(word_frequencies)
    return [pheromone_value]


def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """
    Perform one NLMS adaptation step.

    Args:
        weights: Current weight vector (shape (n,)).
        x: Input feature vector (shape (n,)).
        target: Desired scalar output.
        mu: Step‑size (0 < mu ≤ 1).
        eps: Small constant to avoid division by zero.

    Returns:
        (new_weights, error) where error = target - y.
    """
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    new_weights = weights + (mu * error / power) * x
    return new_weights, error


def hybrid_pheromone_nlms(text: str, weights: np.ndarray, mu: float = 0.5) -> tuple[np.ndarray, float]:
    pheromone_value = social_interaction_and_pheromone(text, None)[0]
    x = np.array([pheromone_value])
    new_weights, error = nlms_update(weights, x, pheromone_value, mu)
    return new_weights, error


def char_frequency_vector(text: str) -> np.ndarray:
    vec = np.zeros(26, dtype=float)
    for ch in text.lower():
        if 'a' <= ch <= 'z':
            vec[ord(ch) - ord('a')] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def hybrid_pheroomone_char_nlms(text: str, weights: np.ndarray, mu: float = 0.5) -> tuple[np.ndarray, float]:
    char_vec = char_frequency_vector(text)
    new_weights, error = nlms_update(weights, char_vec, 1.0, mu)
    return new_weights, error


if __name__ == "__main__":
    np.random.seed(42)
    text = "This is a test sentence for pheromone and NLMS hybrid"
    weights = np.random.rand(1)
    new_weights, error = hybrid_pheromone_nlms(text, weights)
    print(f"New Weights: {new_weights}, Error: {error}")
    char_weights = np.random.rand(26)
    new_char_weights, char_error = hybrid_pheroomone_char_nlms(text, char_weights)
    print(f"New Char Weights: {new_char_weights}, Char Error: {char_error}")