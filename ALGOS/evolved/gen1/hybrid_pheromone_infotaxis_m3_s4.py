# DARWIN HAMMER — match 3, survivor 4
# gen: 1
# parent_a: pheromone.py (gen0)
# parent_b: infotaxis.py (gen0)
# born: 2026-05-29T23:14:18Z

import argparse
import json
import math
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Pheromone handling – a lightweight in‑memory store with proper timestamps
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
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
        """Return the multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """Singleton‑like in‑memory store for demo purposes."""
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> List[Dict]:
        rows = []
        for entry in cls.get_by_surface(surface_key):
            before = entry.signal_value
            entry.apply_decay()
            rows.append({
                "pheromone_uuid": entry.uuid,
                "surface_key": entry.surface_key,
                "signal_kind": entry.signal_kind,
                "signal_value_before": before,
                "signal_value_after": entry.signal_value,
                "half_life_seconds": entry.half_life_seconds,
            })
        return rows

    @classmethod
    def snapshot(cls, surface_key: str) -> np.ndarray:
        """Return a vector of current signal values for a surface."""
        entries = cls.get_by_surface(surface_key)
        if not entries:
            return np.array([])
        values = np.array([e.signal_value for e in entries], dtype=float)
        # Normalise to a probability distribution (adds a tiny epsilon to avoid zeros)
        eps = 1e-12
        total = values.sum() + eps
        return values / total


# ----------------------------------------------------------------------
# Entropy utilities – vectorised with NumPy
# ----------------------------------------------------------------------
def entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability vector."""
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log(probs)))


def expected_entropy(p_hit: float,
                     hit_dist: np.ndarray,
                     miss_dist: np.ndarray) -> float:
    """Expected entropy after a binary observation."""
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError("p_hit must lie in [0, 1]")
    return p_hit * entropy(hit_dist) + (1.0 - p_hit) * entropy(miss_dist)


# ----------------------------------------------------------------------
# Infotaxis – choose action that maximises expected information gain
# ----------------------------------------------------------------------
def best_action(actions: Dict[str, Tuple[float, np.ndarray, np.ndarray]]) -> str:
    """
    actions: mapping name -> (p_hit, hit_distribution, miss_distribution)
    Returns the action with the lowest expected entropy (i.e. highest info gain).
    """
    if not actions:
        raise ValueError("no actions supplied")
    scores = {name: expected_entropy(*data) for name, data in actions.items()}
    # tie‑break by lexical order for determinism
    return min(scores, key=lambda n: (scores[n], n))


# ----------------------------------------------------------------------
# CLI commands – signal, decay, infotaxis-step
# ----------------------------------------------------------------------
def cmd_signal(args: argparse.Namespace) -> Dict:
    entry = PheromoneEntry(
        surface_key=args.surface_key,
        signal_kind=args.signal_kind,
        signal_value=args.signal_value,
        half_life_seconds=args.half_life_seconds,
    )
    if args.execute:
        PheromoneStore.add(entry)
    report = {
        "action": "signal",
        "execute_performed": args.execute,
        "pheromone_uuid": entry.uuid if args.execute else None,
        "surface_key": entry.surface_key,
        "signal_kind": entry.signal_kind,
        "signal_value": entry.signal_value,
        "half_life_seconds": entry.half_life_seconds,
        "status": "PASS",
    }
    return report


def cmd_decay(args: argparse.Namespace) -> Dict:
    rows = PheromoneStore.decay_surface(args.surface_key) if args.execute else []
    report = {
        "action": "decay",
        "execute_performed": args.execute,
        "surface_key": args.surface_key,
        "rows_updated": len(rows),
        "rows": rows[: args.limit],
        "status": "PASS",
    }
    return report


def cmd_infotaxis(args: argparse.Namespace) -> Dict:
    """
    Simulate a single infotaxis decision.
    For demonstration we generate dummy actions with synthetic hit probabilities.
    """
    # Current belief over pheromone sources on the surface
    belief = PheromoneStore.snapshot(args.surface_key)
    if belief.size == 0:
        raise RuntimeError(f"No pheromone entries for surface '{args.surface_key}'")

    # Dummy action generator – in a real system actions would be spatial moves
    np.random.seed(0)  # deterministic for reproducibility in the demo
    actions = {}
    for i in range(args.num_actions):
        name = f"action_{i}"
        # synthetic hit probability proportional to a random weight
        p_hit = np.clip(np.random.rand(), 0.05, 0.95)
        # assume hit concentrates belief around a random subset of entries
        mask = np.random.choice([0, 1], size=belief.shape, p=[0.7, 0.3])
        hit_dist = belief * mask
        miss_dist = belief * (1 - mask)
        # renormalise
        hit_dist = hit_dist / (hit_dist.sum() + 1e-12)
        miss_dist = miss_dist / (miss_dist.sum() + 1e-12)
        actions[name] = (p_hit, hit_dist, miss_dist)

    chosen = best_action(actions)
    report = {
        "action": "infotaxis",
        "surface_key": args.surface_key,
        "chosen_action": chosen,
        "expected_entropy": expected_entropy(*actions[chosen]),
        "status": "PASS",
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid pheromone‑entropy engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # signal ------------------------------------------------------------
    sp = sub.add_parser("signal", help="record a pheromone signal")
    sp.add_argument("--surface-key", required=True)
    sp.add_argument(
        "--signal-kind",
        choices=["generated", "used", "promoted", "forked", "decayed", "archived", "operator_selected"],
        required=True,
    )
    sp.add_argument("--signal-value", type=float, default=1.0)
    sp.add_argument("--half-life-seconds", type=int, default=604800)
    sp.add_argument("--execute", action="store_true", help="persist the signal")
    sp.set_defaults(func=cmd_signal)

    # decay -------------------------------------------------------------
    dp = sub.add_parser("decay", help="apply decay to a surface")
    dp.add_argument("--surface-key", required=True)
    dp.add_argument("--limit", type=int, default=20, help="max rows shown in report")
    dp.add_argument("--execute", action="store_true", help="actually decay stored entries")
    dp.set_defaults(func=cmd_decay)

    # infotaxis ---------------------------------------------------------
    ip = sub.add_parser("infotaxis", help="run a single infotaxis decision step")
    ip.add_argument("--surface-key", required=True)
    ip.add_argument("--num-actions", type=int, default=5, help="how many synthetic actions to evaluate")
    ip.set_defaults(func=cmd_infotaxis)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = args.func(args)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()