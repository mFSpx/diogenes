# DARWIN HAMMER — match 38, survivor 3
# gen: 4
# parent_a: indy_learning_vector.py (gen0)
# parent_b: hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s1.py (gen3)
# born: 2026-05-29T23:26:29Z

"""Hybrid INDY‑Bandit Fusion

Parent A: ``indy_learning_vector.py`` – provides deterministic tokenisation,
chunking and ontology‑aware identifiers for text passages.

Parent B: ``hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s1.py`` – supplies a
fold‑change detection differential update (log‑count based) and a contextual
multi‑armed bandit with UCB‑style confidence bounds.

Mathematical bridge
-------------------
The bridge is the *log‑count* statistic that both parents already use:

* A yields token frequency vectors  f ∈ ℝⁿ for each chunk.
* B’s fold‑change detector updates a control signal *u* from the log‑ratio
  Δ = log(1+fᵢ) − log(1+fᵢ₋₁).

We feed Δ into the bandit’s propensity update:


propensity ← propensity + gain·Δ·dt


Thus the evolving token distribution directly modulates action selection,
creating a unified hybrid system.

The module implements:
* text tokenisation / chunking (A)
* fold‑change step (B)
* hybrid update that maps token‑frequency changes onto bandit propensities
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – tokenisation & chunking utilities
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"\S+")
DEFAULT_TERMS = (
    "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "ACTION", "EVENT", "TIME", "EVIDENCE",
    "CLAIM", "HYPOTHESIS", "SIGNAL", "PATTERN", "TOOL", "ALGORITHM", "BOOK",
    "SOURCE", "LEAD", "LOCATION", "LAW", "RULE",
)


def sha256_json(value: Any) -> str:
    """Deterministic hash of a JSON‑serialisable value."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def load_go_terms(root: Path = ROOT) -> List[str]:
    """Load ontology terms from ``OFFICIAL_ONTOLOGY.json`` – fall back to defaults."""
    p = root / "OFFICIAL_ONTOLOGY.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        terms = data.get("active_terms") or []
        return [str(t).upper() for t in terms if str(t).strip()]
    except Exception:
        return list(DEFAULT_TERMS)


def tokenize(text: str) -> List[Dict[str, Any]]:
    """Return a list of token dicts with text and character offsets."""
    return [
        {"token": m.group(0), "start": m.start(), "end": m.end()}
        for m in WORD_RE.finditer(text)
    ]


def chunk_text_tokens(
    text: str,
    *,
    max_tokens: int = 500,
    overlap_tokens: int = 0,
    source_ref: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Split *text* into overlapping token windows."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be >=0 and < max_tokens")
    toks = tokenize(text)
    source_ref = dict(source_ref or {})
    if not toks:
        cid = "chunk:" + sha256_json({"source_ref": source_ref, "empty": True})[:24]
        return [
            {
                "chunk_id": cid,
                "chunk_index": 0,
                "token_start": 0,
                "token_end": 0,
                "char_start": 0,
                "char_end": 0,
                "token_count": 0,
                "text": "",
                "source_ref": source_ref,
            }
        ]

    chunks: List[Dict[str, Any]] = []
    token_start = 0
    idx = 0
    while token_start < len(toks):
        token_end = min(len(toks), token_start + max_tokens)
        char_start = toks[token_start]["start"]
        char_end = toks[token_end - 1]["end"]
        chunk_text = text[char_start:char_end]
        cid = "chunk:" + sha256_json(
            {
                "source_ref": source_ref,
                "token_start": token_start,
                "token_end": token_end,
                "text": chunk_text,
            }
        )[:24]
        chunks.append(
            {
                "chunk_id": cid,
                "chunk_index": idx,
                "token_start": token_start,
                "token_end": token_end,
                "char_start": char_start,
                "char_end": char_end,
                "token_count": token_end - token_start,
                "text": chunk_text,
                "source_ref": source_ref,
            }
        )
        idx += 1
        token_start = token_end - overlap_tokens
    return chunks


def token_frequency(chunk: Dict[str, Any]) -> Counter:
    """Count token occurrences inside a chunk."""
    return Counter(tokenize(chunk["text"])).most_common()


# ----------------------------------------------------------------------
# Parent B – fold‑change detection & bandit core
# ----------------------------------------------------------------------
class BanditAction:
    def __init__(
        self,
        action_id: str,
        propensity: float = 0.0,
        expected_reward: float = 0.0,
        confidence_bound: float = 0.0,
        algorithm: str = "hybrid",
    ):
        self.action_id = action_id
        self.propensity = propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm

    def ucb_score(self) -> float:
        """Upper‑confidence‑bound score used for selection."""
        return self.expected_reward + self.confidence_bound


class BanditUpdate:
    def __init__(self, context_id: str, action_id: str, reward: float, propensity: float):
        self.context_id = context_id
        self.action_id = action_id
        self.reward = reward
        self.propensity = propensity


_POLICY: Dict[str, List[float]] = {}  # action_id → [cumulative_reward, count]


def reset_policy() -> None:
    _POLICY.clear()


def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]


def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        total, n = _POLICY.get(u.action_id, [0.0, 0.0])
        _POLICY[u.action_id] = [total + u.reward, n + 1]


def fold_change_step(
    u: float,
    x: float,
    y: float,
    *,
    dt: float = 1.0,
    gain: float = 1.0,
    decay_x: float = 0.1,
) -> Tuple[float, float]:
    """
    Discrete fold‑change dynamics.

    x  – internal state (e.g. previous log‑count)
    y  – current observation (log‑count)
    u  – external control (here derived from bandit propensity)

    The update mimics the classic integral‑feedback motif:

        du/dt = gain * (log(1+y) - log(1+x)) - decay_x * u
        dx/dt = -decay_x * x + u

    The function returns the new (x, u) after a single step of size *dt*.
    """
    # log‑ratio term (fold‑change)
    delta = math.log1p(y) - math.log1p(x)
    du = gain * delta - decay_x * u
    dx = -decay_x * x + u
    u_new = u + du * dt
    x_new = x + dx * dt
    return x_new, u_new


# ----------------------------------------------------------------------
# Hybrid layer – marrying token statistics to bandit propensities
# ----------------------------------------------------------------------
class HybridEngine:
    """Core object that holds actions and runs the hybrid update."""

    def __init__(self, action_ids: Iterable[str]):
        self.actions: Dict[str, BanditAction] = {
            aid: BanditAction(action_id=aid, propensity=0.0, expected_reward=0.0, confidence_bound=0.0)
            for aid in action_ids
        }
        # internal state per action for fold‑change detection
        self._state_x: Dict[str, float] = {aid: 0.0 for aid in action_ids}
        self._state_u: Dict[str, float] = {aid: 0.0 for aid in action_ids}

    def _log_token_count(self, freq: Counter) -> float:
        """Compress a frequency Counter into a scalar log‑count."""
        total = sum(count for _, count in freq)
        return math.log1p(total)

    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        1. Compute token log‑count for the chunk.
        2. For each action, run a fold‑change step using the current log‑count.
        3. Update propensities and confidence bounds.
        4. Emit a synthetic BanditUpdate (reward = log‑count) to the global policy.
        """
        freq = token_frequency(chunk)
        logcnt = self._log_token_count(freq)

        updates: List[BanditUpdate] = []

        for aid, action in self.actions.items():
            x = self._state_x[aid]
            u = self._state_u[aid]

            # Fold‑change dynamics driven by current propensity (as external control)
            x_new, u_new = fold_change_step(u, x, logcnt, dt=1.0, gain=0.5, decay_x=0.05)

            # Propensity is nudged by the internal control signal
            action.propensity = max(0.0, action.propensity + u_new)

            # Confidence bound follows classic UCB: sqrt(2*log(t)/n)
            t = sum(_count(a) for a in self.actions) + 1.0
            n = max(1.0, _count(aid))
            action.confidence_bound = math.sqrt(2.0 * math.log(t) / n)

            # Expected reward pulls from the global policy
            action.expected_reward = _reward(aid)

            # Store updated internal states
            self._state_x[aid] = x_new
            self._state_u[aid] = u_new

            # Create a synthetic update (reward proportional to log‑count)
            updates.append(BanditUpdate(context_id=chunk["chunk_id"], action_id=aid, reward=logcnt, propensity=action.propensity))

        update_policy(updates)

    def select_action(self) -> BanditAction:
        """Return the action with the highest UCB score."""
        return max(self.actions.values(), key=lambda a: a.ucb_score())

    def current_policy_snapshot(self) -> Dict[str, Tuple[float, float]]:
        """Utility for inspection – returns propensity and expected reward per action."""
        return {
            aid: (act.propensity, act.expected_reward) for aid, act in self.actions.items()
        }


# ----------------------------------------------------------------------
# Convenience functions exposing the hybrid workflow
# ----------------------------------------------------------------------
def hybrid_chunkify(texts: Iterable[str], max_tokens: int = 200, overlap: int = 20) -> List[Dict[str, Any]]:
    """Flatten a collection of raw texts into a list of token chunks."""
    all_chunks: List[Dict[str, Any]] = []
    for i, txt in enumerate(texts):
        src = {"source_id": f"text_{i}"}
        chunks = chunk_text_tokens(txt, max_tokens=max_tokens, overlap_tokens=overlap, source_ref=src)
        all_chunks.extend(chunks)
    return all_chunks


def run_hybrid_engine(
    texts: Iterable[str],
    action_ids: Iterable[str],
    max_tokens: int = 200,
    overlap: int = 20,
) -> HybridEngine:
    """
    End‑to‑end driver:
    * tokenises & chunks the input texts,
    * feeds each chunk through the HybridEngine,
    * returns the populated engine for inspection.
    """
    engine = HybridEngine(action_ids)
    chunks = hybrid_chunkify(texts, max_tokens=max_tokens, overlap=overlap)
    for ch in chunks:
        engine.process_chunk(ch)
    return engine


def sample_action_selection(engine: HybridEngine, trials: int = 5) -> List[str]:
    """Perform *trials* selections using the current policy and return the chosen IDs."""
    chosen: List[str] = []
    for _ in range(trials):
        act = engine.select_action()
        chosen.append(act.action_id)
        # Simulate a tiny reward to keep the policy alive
        update_policy([BanditUpdate(context_id="sim", action_id=act.action_id, reward=0.1, propensity=act.propensity)])
    return chosen


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 5,
        "Artificial intelligence and machine learning are rapidly evolving fields. " * 4,
        "Quantum mechanics reveals the probabilistic nature of reality. " * 3,
    ]

    actions = ["summarize", "translate", "classify"]
    reset_policy()
    engine = run_hybrid_engine(sample_texts, actions, max_tokens=50, overlap=10)

    print("=== Policy snapshot after processing ===")
    for aid, (prop, rew) in engine.current_policy_snapshot().items():
        print(f"Action: {aid:10s} | Propensity: {prop: .3f} | Expected Reward: {rew: .3f}")

    print("\n=== Sample selections ===")
    selections = sample_action_selection(engine, trials=7)
    print("Chosen actions:", selections)