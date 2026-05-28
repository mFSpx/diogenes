from __future__ import annotations
from dataclasses import dataclass
import random, math

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    expected_reward: float
    confidence_bound: float
    propensity: float

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: dict[str, list[float]] = {}
_RNG = random.Random(7)

def reset_policy() -> None:
    _POLICY.clear()
    _RNG.seed(7)

def update_policy(updates: list[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def _mean(action_id: str) -> float:
    reward, n = _POLICY.get(action_id, [0.0, 0.0])
    return reward / n if n else 0.0

def select_action(context: dict[str, float], actions: list[str], algorithm: str = "linucb", epsilon: float = 0.1, seed: int | str | None = None) -> BanditAction:
    if not actions:
        raise ValueError("actions must be non-empty")
    if len(actions) == 1:
        a = actions[0]
        return BanditAction(a, _mean(a), 0.0, 1.0)
    alg = algorithm.lower()
    if alg == "epsilon_greedy":
        if _RNG.random() < epsilon:
            chosen = _RNG.choice(actions)
        else:
            chosen = max(actions, key=_mean)
        return BanditAction(chosen, _mean(chosen), 0.0, (epsilon / len(actions)) + (0 if chosen != max(actions, key=_mean) else 1-epsilon))
    if alg == "thompson":
        def sample(a: str) -> float:
            reward, n = _POLICY.get(a, [0.0, 0.0])
            success = max(0.0, reward)
            fail = max(0.0, n - reward)
            return _RNG.betavariate(1.0 + success, 1.0 + fail)
        chosen = max(actions, key=sample)
        return BanditAction(chosen, _mean(chosen), 0.0, 1.0 / len(actions))
    # LinUCB-lite: mean + confidence bonus; deterministic tie by input order.
    total = sum(v[1] for v in _POLICY.values()) + 1.0
    def score(a: str) -> float:
        n = _POLICY.get(a, [0.0, 0.0])[1]
        bonus = math.sqrt(math.log(total + 1.0) / (n + 1.0))
        return _mean(a) + bonus
    chosen = max(actions, key=score)
    n = _POLICY.get(chosen, [0.0, 0.0])[1]
    cb = math.sqrt(math.log(total + 1.0) / (n + 1.0))
    return BanditAction(chosen, _mean(chosen), cb, 1.0 / len(actions))
