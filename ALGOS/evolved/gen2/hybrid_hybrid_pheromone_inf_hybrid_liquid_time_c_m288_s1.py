# DARWIN HAMMER — match 288, survivor 1
# gen: 2
# parent_a: hybrid_pheromone_infotaxis_m3_s1.py (gen1)
# parent_b: hybrid_liquid_time_constant_minhash_m10_s2.py (gen1)
# born: 2026-05-29T23:28:11Z

import argparse
import json
import math
import os
import pathlib
import random
import sys
import hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, Any

import numpy as np


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def deterministic_hash(token: str, seed: int) -> int:
    """Return a deterministic 64‑bit integer hash for a token + seed."""
    h = hashlib.sha256(f"{token}:{seed}".encode("utf-8")).digest()
    # Use first 8 bytes as unsigned integer
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def minhash_signature(tokens: List[str], num_hash_functions: int) -> List[int]:
    """
    Compute a deterministic MinHash signature for a token list.
    """
    signature = []
    for i in range(num_hash_functions):
        min_hash = (1 << 64) - 1  # max unsigned 64‑bit int
        for token in tokens:
            h = deterministic_hash(token, i)
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return signature


def minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Jaccard‑like similarity based on identical hash positions."""
    if len(sig1) != len(sig2) or not sig1:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def calculate_entropy(probs: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probs)
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = np.array(probs) / total
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log(probs)))


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """Weighted expected entropy for a hit/miss scenario."""
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError("p_hit must be in [0, 1]")
    return p_hit * calculate_entropy(hit_state) + (1.0 - p_hit) * calculate_entropy(miss_state)


def best_action(action_dict: Dict[Any, Tuple[float, List[float], List[float]]]) -> Any:
    """
    Choose the action with the lowest expected entropy.
    ``action_dict`` maps an identifier to a tuple ``(p_hit, hit_state, miss_state)``.
    """
    if not action_dict:
        raise ValueError("actions required")
    return min(
        action_dict.items(),
        key=lambda item: expected_entropy(*item[1])
    )[0]


# ----------------------------------------------------------------------
# Pheromone model with realistic decay
# ----------------------------------------------------------------------
class PheromoneBank:
    """
    Stores pheromone values per (surface_key, signal_kind) and updates them
    according to exponential decay with a configurable half‑life.
    """

    def __init__(self, half_life_seconds: float):
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive")
        self.half_life = half_life_seconds
        self._store: Dict[Tuple[str, str], Tuple[float, datetime]] = {}

    def _decay_factor(self, elapsed: float) -> float:
        """Exponential decay factor for a given elapsed time (seconds)."""
        return 0.5 ** (elapsed / self.half_life)

    def add_signal(self, surface_key: str, signal_kind: str, value: float, timestamp: datetime) -> None:
        """Add a raw signal; existing value is decayed then increased."""
        key = (surface_key, signal_kind)
        if key in self._store:
            old_val, old_ts = self._store[key]
            elapsed = (timestamp - old_ts).total_seconds()
            decayed = old_val * self._decay_factor(elapsed)
            new_val = decayed + value
        else:
            new_val = value
        self._store[key] = (new_val, timestamp)

    def get_signal(self, surface_key: str, signal_kind: str, timestamp: datetime) -> float:
        """Retrieve the current signal strength after applying decay up to ``timestamp``."""
        key = (surface_key, signal_kind)
        if key not in self._store:
            return 0.0
        val, ts = self._store[key]
        elapsed = (timestamp - ts).total_seconds()
        return val * self._decay_factor(elapsed)


# ----------------------------------------------------------------------
# Liquid‑Time‑Constant (LTC) network utilities
# ----------------------------------------------------------------------
def ltc_gate(x: float, I: float, theta: float) -> float:
    """Learned gating function – a simple tanh non‑linearity."""
    return math.tanh(x + I + theta)


def ltc_step(x: float, I: float, theta: float, tau: float, alpha: float, s_t: float) -> float:
    """
    One integration step of the hybrid LTC dynamics.
    The effective time constant is modulated by the gating output and the
    similarity signal ``s_t`` (coming from MinHash).
    """
    f = ltc_gate(x, I, theta)
    # Prevent division by zero and ensure positivity
    tau_eff = max(tau / (1.0 + tau * (f + alpha * s_t)), 1e-6)
    return -x / tau_eff + f * I


# ----------------------------------------------------------------------
# Core hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_forward(
    sequence: List[str],
    num_hash_functions: int,
    tau: float,
    alpha: float,
    theta: float,
    pheromone_bank: PheromoneBank,
    start_time: datetime,
    time_step: float = 1.0,
) -> float:
    """
    Run the hybrid dynamics over a list of textual inputs.

    Parameters
    ----------
    sequence: list of strings – the textual observations.
    num_hash_functions: int – size of MinHash signatures.
    tau, alpha, theta: float – LTC hyper‑parameters.
    pheromone_bank: PheromoneBank – shared pheromone state.
    start_time: datetime – logical start of the simulation.
    time_step: float – seconds between consecutive observations.
    """
    x = 0.0
    I = 0.0
    prev_signature: List[int] | None = None
    current_time = start_time

    for text in sequence:
        # ----- MinHash processing -----
        tokens = text.split()
        signature = minhash_signature(tokens, num_hash_functions)
        s_t = (
            minhash_similarity(prev_signature, signature)
            if prev_signature is not None
            else 0.0
        )
        prev_signature = signature

        # ----- LTC dynamics -----
        x = ltc_step(x, I, theta, tau, alpha, s_t)

        # ----- Pheromone update -----
        # Emit a unit signal each step; decay is handled internally.
        pheromone_bank.add_signal("surface", "kind", 1.0, current_time)

        # Read the decayed signal to feed back as external input I.
        I = pheromone_bank.get_signal("surface", "kind", current_time)

        # Advance logical clock
        current_time += timedelta(seconds=time_step)

    return x


# ----------------------------------------------------------------------
# Command‑line interface (optional)
# ----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid pheromone‑LTC‑MinHash algorithm.")
    parser.add_argument(
        "--sequence",
        nargs="+",
        default=["hello world", "hello again", "goodbye world"],
        help="Space‑separated list of textual observations.",
    )
    parser.add_argument("--hashes", type=int, default=10, help="Number of MinHash functions.")
    parser.add_argument("--tau", type=float, default=1.0, help="Base time constant τ.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Similarity coupling coefficient.")
    parser.add_argument("--theta", type=float, default=0.1, help="Gating bias.")
    parser.add_argument(
        "--half_life", type=float, default=10.0, help="Pheromone half‑life in seconds."
    )
    parser.add_argument(
        "--time_step", type=float, default=1.0, help="Logical seconds between observations."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bank = PheromoneBank(half_life_seconds=args.half_life)
    start = datetime.now(timezone.utc)

    result = hybrid_forward(
        sequence=args.sequence,
        num_hash_functions=args.hashes,
        tau=args.tau,
        alpha=args.alpha,
        theta=args.theta,
        pheromone_bank=bank,
        start_time=start,
        time_step=args.time_step,
    )
    print(f"Final LTC state: {result:.6f}")


if __name__ == "__main__":
    main()