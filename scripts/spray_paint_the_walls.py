#!/usr/bin/env python3
"""
IO-25: SPRAY_PAINT_THE_WALLS Execution Matrix

System Context: Feminism, MtG Rules Lawyering, Riverboat Gambler Risk Profiles
Connects: elastic_shape.rs, RiverML online learning, RETE bandit gate,
          Ternary 1.58-bit, Mamba 7B, 4x1 8B Bonsai.

Pure Python. Pure chaos. Receipts mandatory.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT_DIR = ROOT / "05_OUTPUTS" / "receipts"
TRAINING_DIR = ROOT / "scripts" / "training"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


# ─── Archetypes ────────────────────────────────────────────────────────

ARCHETYPES = [
    "Feminist_Hex",
    "Mana_Leak",
    "Hoe_Heuristics",
    "Riverboat_Bluff",
    "Northern_Strike_414",
    "GO_25_Ontology",
    "IO_25_Indys_Ontology",
    "Spray_Paint_Walls",
    "Board_Game_Loser",
    "Helicopter_Humpback",
]

# ─── Ternary 1.58-bit specifics ────────────────────────────────────────

TERNARY_VALUES = [-1, 0, 1]

ELASTIC_ROUTE_KINDS = [
    "bonsai_8b_q1",
    "bonsai_8b_q2",
    "mamba_7b_ram",
    "mamba_7b_gpu",
    "deepseek_1_5b",
    "bitvla_vision",
    "needle_shared",
]


def roll_d20() -> tuple[int, str]:
    """Board games are for losers. Losing is cool. Let's lose fast."""
    roll = random.randint(1, 20)
    messages = {
        1: "CRITICAL FAILURE: Cardboard warped by humidity. Virginity retained.",
        20: "NATURAL 20: Argued the rules so hard the system crashed into profit.",
    }
    msg = messages.get(roll, f"Operational Noise Index: {roll + 5}")
    return roll, msg


def ternary_quantize(weights: list[float]) -> list[int]:
    """Quantize weights to {-1, 0, +1} using absmean."""
    if not weights:
        return []
    mean_abs = sum(abs(w) for w in weights) / len(weights)
    if mean_abs < 1e-8:
        return [0] * len(weights)
    return [max(-1, min(1, round(w / mean_abs))) for w in weights]


# ─── RiverML Bandit Arm ────────────────────────────────────────────────

def select_route(
    route_history: dict[str, list[float]],
    epsilon: float = 0.13,
) -> str:
    """Epsilon-greedy route selection with RiverML-style online learning."""
    if random.random() < epsilon or not route_history:
        return random.choice(ELASTIC_ROUTE_KINDS)
    # Greedy: pick route with best average reward
    best_route = max(
        route_history,
        key=lambda r: sum(route_history[r]) / len(route_history[r])
        if route_history[r]
        else -float("inf"),
    )
    return best_route


def update_route(route_history: dict[str, list[float]], route: str, reward: float) -> None:
    """Online reward update — RiverML bandit style."""
    if route not in route_history:
        route_history[route] = []
    route_history[route].append(reward)
    # Keep window of last 100
    if len(route_history[route]) > 100:
        route_history[route] = route_history[route][-100:]


# ─── elastic_shape.rs integration ──────────────────────────────────────

def call_elastic_shape(packet: dict[str, Any]) -> dict[str, Any]:
    """Shell out to the compiled elastic_shape binary."""
    import subprocess

    binary = ROOT / "01_REPOS" / "lucidota_resonance" / "target" / "release" / "lucidota_elastic_shape"
    if not binary.exists():
        binary = ROOT / "01_REPOS" / "lucidota_resonance" / "target" / "debug" / "lucidota_elastic_shape"
    if not binary.exists():
        return {"status": "no_binary", "error": "lucidota_elastic_shape not compiled"}

    try:
        proc = subprocess.run(
            [str(binary)],
            input=json.dumps(packet),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            return json.loads(proc.stdout)
        return {"status": "error", "stderr": proc.stderr[:500], "returncode": proc.returncode}
    except FileNotFoundError:
        return {"status": "no_binary", "error": f"{binary} not found"}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except json.JSONDecodeError as e:
        return {"status": "decode_error", "error": str(e)}


# ─── Gym Loop ──────────────────────────────────────────────────────────

def run_gym_loop(
    iterations: int = 69,
    batch_size: int = 4,
    risk_mult: float = 0.13,
    json_output: bool = False,
) -> dict[str, Any]:
    """10,000G gravity gym: parallel ternary training loop."""
    route_history: dict[str, list[float]] = {}
    total_reward = 0.0
    receipts: list[dict[str, Any]] = []
    sweat_indexes: list[int] = []

    for i in range(iterations):
        # Roll dice for this iteration
        roll, roll_msg = roll_d20()
        route = select_route(route_history, epsilon=risk_mult)

        # Compute sweat = gravity training intensity
        sweat = (batch_size * 10000) // max(1, (i + 1) % 69 + 1)
        sweat_indexes.append(sweat)

        # Simulate ternary reward
        noise = random.uniform(-1, 1) * (1.0 / max(1, i + 1))
        reward = max(-1.0, min(1.0, (roll - 10) / 10 + noise))
        update_route(route_history, route, reward)
        total_reward += reward

        receipt = {
            "iteration": i + 1,
            "roll": roll,
            "roll_msg": roll_msg,
            "route": route,
            "sweat": sweat,
            "reward": round(reward, 4),
            "cumulative_reward": round(total_reward, 4),
            "archetype": random.choice(ARCHETYPES),
            "route_avg": round(
                sum(route_history[route]) / len(route_history[route]), 4
            ),
        }
        receipts.append(receipt)

        if not json_output:
            die = "⚀⚁⚂⚃⚄⚅"[min(roll // 4, 5)]
            bar = "█" * min(sweat // 100, 40)
            sys.stderr.write(
                f"\r[{i + 1:3d}/{iterations}] {die} route={route:20s} "
                f"sweat={sweat:6d} reward={reward:+.3f} {bar}"
            )
            sys.stderr.flush()

    if not json_output:
        sys.stderr.write("\n")

    # Write receipt
    result = {
        "schema": "lucidota.spray_paint_the_walls.v1",
        "ontology": "IO-25 / Northern.Strikes.490",
        "iterations": iterations,
        "batch_size": batch_size,
        "risk_multiplier": risk_mult,
        "total_reward": round(total_reward, 4),
        "avg_reward": round(total_reward / max(iterations, 1), 4),
        "max_sweat": max(sweat_indexes) if sweat_indexes else 0,
        "route_history": {k: {"count": len(v), "avg": sum(v) / len(v) if v else 0} for k, v in route_history.items()},
        "archetypes_used": ARCHETYPES,
        "dice_final": roll_d20()[1],
        "receipts": receipts[:10] + (["..."] if len(receipts) > 10 else []),
        "conditions": [
            "Board games are for losers. Losing is cool.",
            "Reading manuals is for the weak.",
            "Specificity and being wrong; fast.",
            "Math is hard for morons, roll the dice.",
        ],
        "ts": now_iso(),
    }

    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = RECEIPT_DIR / f"spray_paint_the_walls_{stamp()}.json"
    receipt_path.write_text(json.dumps(result, indent=2, default=str))
    result["receipt_path"] = str(receipt_path.relative_to(ROOT))

    return result


# ─── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IO-25: SPRAY_PAINT_THE_WALLS Execution Matrix"
    )
    parser.add_argument("--iterations", type=int, default=69, help="Number of gym iterations (default: 69)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--risk-mult", type=float, default=0.13, help="Risk multiplier epsilon (default: 0.13)")
    parser.add_argument("--json", action="store_true", help="JSON output only")
    parser.add_argument("--elastic", action="store_true", help="Call elastic_shape.rs")
    parser.add_argument("--gamble", action="store_true", help="Bet the entire VRAM on one pass")
    args = parser.parse_args()

    # Validate inputs
    if args.iterations < 1:
        sys.exit("[error] --iterations must be >= 1")
    if args.batch_size < 1:
        sys.exit("[error] --batch-size must be >= 1")
    if args.risk_mult < 0.0 or args.risk_mult > 1.0:
        sys.exit("[error] --risk-mult must be between 0.0 and 1.0")

    if args.gamble:
        args.risk_mult = 1.0
        if not args.json:
            print("!!! GAMBLE MODE: ALL IN !!!", file=sys.stderr)

    result = run_gym_loop(
        iterations=args.iterations,
        batch_size=args.batch_size,
        risk_mult=args.risk_mult,
        json_output=args.json,
    )

    # Optionally call elastic_shape
    if args.elastic:
        shape = call_elastic_shape({
            "source": "spray_paint_the_walls",
            "payload": result,
            "stamp": stamp(),
        })
        result["elastic_shape"] = shape

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n=== SPRAY_PAINT_THE_WALLS Matrix ===")
        print(f"Iterations: {result['iterations']}x")
        print(f"Total Reward: {result['total_reward']:+.3f}")
        print(f"Average Reward: {result['avg_reward']:+.3f}")
        print(f"Max Sweat: {result['max_sweat']:,d} (10,000G Gym)")
        print(f"Routes trained: {len(result['route_history'])}")
        print(f"Receipt: {result['receipt_path']}")
        print(f"Dice Final: {result['dice_final']}")
        if args.elastic:
            es = result.get("elastic_shape", {})
            print(f"Elastic Shape: {es.get('status', 'unknown')}")

    # Exit code: 0 always — losing is cool
    sys.exit(0)


if __name__ == "__main__":
    main()
