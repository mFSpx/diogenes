# DARWIN HAMMER — match 1734, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_bandit_m53_s0.py (gen3)
# parent_b: hybrid_indy_learning_vector_hybrid_hybrid_geomet_m113_s2.py (gen3)
# born: 2026-05-29T23:38:35Z

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Dict

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

@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    chunk_index: int
    tokens: List[str]

_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}
_HISTORY: List[List[float]] = []

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()
    _HISTORY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0
        s[2] = max(s[2], u.propensity)

def _reward(a: str) -> float:
    total, n, _ = _POLICY.get(a, [0.0, 0.0, 0.0])
    return total / n if n else 0.0

def _propensity(a: str) -> float:
    _, _, propensity = _POLICY.get(a, [0.0, 0.0, 0.0])
    return propensity

def tokenize(text: str) -> List[str]:
    return text.split()

def chunk_text_tokens(text: str, max_tokens: int = 200, overlap_tokens: int = 0) -> List[TextChunk]:
    tokens = tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap_tokens):
        chunk_id = f"chunk:{i}"
        chunk_index = i
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(TextChunk(chunk_id, chunk_index, chunk_tokens))
    return chunks

def select_action(context: Dict[str, float], actions: List[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> BanditAction:
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)
    if algorithm == 'epsilon-greedy':
        if rng.random() < epsilon:
            return BanditAction(random.choice(actions), 0.0, 0.0, 0.0, algorithm)
        else:
            best_action = max(actions, key=lambda a: _reward(a) + _propensity(a))
            return BanditAction(best_action, _propensity(best_action), _reward(best_action), 0.0, algorithm)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def update_confidence_bounds(actions: List[str], chunks: List[TextChunk]) -> None:
    for action in actions:
        propensity = 0.0
        for chunk in chunks:
            if action in chunk.tokens:
                propensity += 1.0 / len(chunk.tokens)
        _STORE[action] = propensity

def run_hybrid_algorithm(text: str, actions: List[str]) -> List[BanditAction]:
    chunks = chunk_text_tokens(text)
    updates = []
    for chunk in chunks:
        for token in chunk.tokens:
            if token in actions:
                updates.append(BanditUpdate(chunk.chunk_id, token, 1.0, _STORE.get(token, 0.0)))
    update_policy(updates)
    update_confidence_bounds(actions, chunks)
    selected_actions = []
    for _ in range(len(actions)):
        selected_action = select_action({}, actions)
        selected_actions.append(selected_action)
    return selected_actions

def main():
    text = "This is a sample text with some actions like action1 and action2"
    actions = ["action1", "action2"]
    selected_actions = run_hybrid_algorithm(text, actions)
    for action in selected_actions:
        print(action)

if __name__ == "__main__":
    main()