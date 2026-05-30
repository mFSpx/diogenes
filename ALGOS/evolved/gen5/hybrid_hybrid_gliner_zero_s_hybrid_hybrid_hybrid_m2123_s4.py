# DARWIN HAMMER — match 2123, survivor 4
# gen: 5
# parent_a: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m376_s0.py (gen4)
# born: 2026-05-29T23:40:59Z

import argparse
import json
import math
import sys
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def text_to_feature(text: str, dim: int = 64) -> np.ndarray:
    """
    Produce a deterministic dense feature vector from a string.
    The implementation hashes the UTF‑8 bytes and spreads the bits
    across ``dim`` dimensions using a simple xor‑folding scheme.
    """
    raw = sha256_bytes(text.encode("utf-8", errors="replace"))
    # repeat the hash to have enough bytes
    repeated = (raw * ((dim // len(raw)) + 1))[:dim]
    # map bytes to floats in [0,1)
    return np.frombuffer(repeated, dtype=np.uint8).astype(np.float32) / 255.0


def parse_labels(raw: str | None) -> List[str]:
    if not raw:
        return ["Operator", "Rainmaker", "Paladin / God-Mode", "Psyche / State-Collapse"]
    p = Path(raw)
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            labels = data.get("required_exact_labels") or data.get("labels") or []
        else:
            labels = data
        return [str(x).strip() for x in labels if str(x).strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8", errors="replace")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str
    feature: np.ndarray  # dense representation for the bandit


@dataclass
class Endpoint:
    """
    Represents a contextual bandit arm (i.e. a possible endpoint).
    Implements the LinUCB algorithm.
    """
    d: int                     # dimensionality of feature space
    A: np.ndarray              # d×d matrix (covariance)
    b: np.ndarray              # d×1 vector (reward accumulator)
    alpha: float = 1.0         # exploration parameter

    @classmethod
    def new(cls, d: int, alpha: float = 1.0) -> "Endpoint":
        return cls(d=d, A=np.identity(d), b=np.zeros(d), alpha=alpha)

    def p(self, x: np.ndarray) -> float:
        """
        Upper confidence bound for a given context vector x.
        """
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        mean = float(theta @ x)
        var = float(x @ A_inv @ x)
        return mean + self.alpha * math.sqrt(var)

    def update(self, x: np.ndarray, reward: float) -> None:
        """
        Standard LinUCB update.
        """
        self.A += np.outer(x, x)
        self.b += reward * x


# ----------------------------------------------------------------------
# Core algorithms
# ----------------------------------------------------------------------
def gliner_zero_shot_extractor(text: str, max_len: int = 30, step: int = 5) -> List[Span]:
    """
    Very lightweight placeholder extractor.
    Generates overlapping substrings up to ``max_len`` characters,
    stepping by ``step`` characters to keep the number of spans
    tractable (O(len(text) / step)).
    """
    spans: List[Span] = []
    n = len(text)
    for start in range(0, n, step):
        for end in range(start + 1, min(start + max_len, n) + 1):
            snippet = text[start:end]
            feature = text_to_feature(snippet)
            span = Span(
                start=start,
                end=end,
                text=snippet,
                label="extracted",
                score=1.0,          # placeholder confidence
                backend="gliner",
                feature=feature,
            )
            spans.append(span)
    return spans


def hoeffding_confidence(r: float, delta: float, n: int) -> float:
    """
    Returns the half‑width of the Hoeffding confidence interval for a bounded
    random variable in [0, r] after ``n`` i.i.d. observations.
    """
    if n <= 0:
        return float("inf")
    return r * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def run_hybrid_bandit(
    spans: List[Span],
    delta: float = 0.05,
    horizon: int = 10,
) -> Tuple[Span, List[Endpoint]]:
    """
    Executes a contextual bandit over the supplied spans.
    The algorithm uses LinUCB with a Hoeffding‑based confidence scaling.
    """
    if not spans:
        raise ValueError("No spans to process")

    d = spans[0].feature.shape[0]
    # One endpoint per span (conceptually each span could be an arm)
    endpoints: List[Endpoint] = [Endpoint.new(d) for _ in spans]

    # Simulated reward function: higher reward for longer spans (just for demo)
    def reward_fn(span: Span) -> float:
        return len(span.text) / 100.0  # normalised length

    chosen: Span = spans[0]

    for t in range(1, horizon + 1):
        # Compute UCB for each arm
        ucbs = np.array([ep.p(span.feature) for ep, span in zip(endpoints, spans)])
        idx = int(np.argmax(ucbs))
        chosen = spans[idx]
        r = reward_fn(chosen)

        # Update the selected endpoint
        endpoints[idx].update(chosen.feature, r)

        # Optional: shrink alpha using Hoeffding to tighten exploration over time
        for ep in endpoints:
            ep.alpha = hoeffding_confidence(r=1.0, delta=delta, n=t)

    return chosen, endpoints


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Improved Hybrid Bandit over extracted spans")
    parser.add_argument("--text", help="Text to process")
    parser.add_argument("--file", help="Path to a file containing the text")
    parser.add_argument("--horizon", type=int, default=20, help="Number of bandit iterations")
    parser.add_argument("--delta", type=float, default=0.05, help="Hoeffding failure probability")
    args = parser.parse_args()

    text = load_text(args)
    if not text:
        sys.exit("No input text provided.")

    spans = gliner_zero_shot_extractor(text)
    chosen_span, _ = run_hybrid_bandit(spans, delta=args.delta, horizon=args.horizon)

    print(f"Chosen span (length {len(chosen_span.text)}): {chosen_span.text}")


if __name__ == "__main__":
    main()