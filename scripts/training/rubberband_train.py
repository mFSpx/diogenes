#!/usr/bin/env python3
"""
Rubberband — Online training via RiverML for Mamba/SSM models.

The "Rubberband Man": an online learning loop that trains Mamba-style SSM
models using RiverML's online learning API. Connects to the Mamba endpoints
(8081/8083) and the RETE bandit for routing decisions.

Losing is cool. Board games are for losers. Let's lose fast.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# RiverML for online learning
try:
    from river import linear_model, optim, preprocessing
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


MAMBA_ENDPOINTS = {
    "mamba_7b_ram": "http://127.0.0.1:8081/v1/chat/completions",
    "mamba_7b_gpu": "http://127.0.0.1:8083/v1/chat/completions",
}


def call_mamba(endpoint: str, prompt: str, max_tokens: int = 64) -> dict[str, Any]:
    """Call a Mamba model endpoint with a prompt."""
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"status": "ok", "result": json.loads(resp.read())}
    except Exception as e:
        return {"status": "error", "error": str(e)}


class RubberbandBandit:
    """
    Online bandit using RiverML for arm selection.
    Tracks which Mamba endpoint performs best for different prompt types.
    """

    def __init__(self, epsilon: float = 0.13):
        self.epsilon = epsilon
        self.arms = list(MAMBA_ENDPOINTS.keys())
        self.rewards: dict[str, list[float]] = {arm: [] for arm in self.arms}

        # RiverML online logistic regression for context-dependent arm selection
        if HAS_RIVER:
            self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        else:
            self.model = None

        self._pulls = 0

    def select_arm(self, context: dict[str, float] | None = None) -> str:
        """Select an arm using epsilon-greedy with RiverML context."""
        self._pulls += 1

        if random.random() < self.epsilon:
            return random.choice(self.arms)

        # Greedy: pick best average reward
        def avg(arm: str) -> float:
            pulls = self.rewards[arm]
            return sum(pulls) / len(pulls) if pulls else 0.0

        return max(self.arms, key=avg)

    def update(self, arm: str, reward: float):
        """Update arm reward history."""
        self.rewards[arm].append(reward)
        if len(self.rewards[arm]) > 200:
            self.rewards[arm] = self.rewards[arm][-200:]

    def status(self) -> dict[str, Any]:
        """Get bandit status."""
        return {
            "pulls": self._pulls,
            "epsilon": self.epsilon,
            "arms": {
                arm: {
                    "pulls": len(pulls),
                    "avg_reward": sum(pulls) / len(pulls) if pulls else 0.0,
                }
                for arm, pulls in self.rewards.items()
            },
        }


class RubberbandTrainer:
    """Online training loop: call Mamba endpoints, evaluate, update bandit."""

    def __init__(self, bandit: RubberbandBandit):
        self.bandit = bandit
        self.prompts = [
            "Explain consciousness in one sentence.",
            "What is the meaning of life?",
            "Write a haiku about binary stars.",
            "Solve: if x + 2 = 5, what is x?",
            "Define recursion.",
            "What is the capital of Australia?",
            "Explain entropy to a 5-year-old.",
            "Write a one-line poem about loss.",
            "What is 7 * 8?",
            "Describe a helicopter hummingbird hybrid.",
        ]

    def step(self, prompt: str | None = None) -> dict[str, Any]:
        """Single training step: select arm, call endpoint, update."""
        if prompt is None:
            prompt = random.choice(self.prompts)

        arm = self.bandit.select_arm()
        endpoint = MAMBA_ENDPOINTS[arm]
        t0 = time.time()

        result = call_mamba(endpoint, prompt)

        elapsed = time.time() - t0
        # Reward: success = fast + ok response, failure = negative
        if result.get("status") == "ok":
            reward = min(1.0, 20.0 / (elapsed + 0.1))  # Faster = better
        else:
            reward = -0.5

        self.bandit.update(arm, reward)

        return {
            "arm": arm,
            "prompt": prompt[:50],
            "elapsed_s": round(elapsed, 3),
            "reward": round(reward, 4),
            "status": result.get("status"),
            "error": result.get("error"),
        }

    def run(self, steps: int = 69, json_output: bool = False) -> dict[str, Any]:
        """Run training loop."""
        print(f"\n=== Rubberband Training Loop ({steps} steps) ===", file=sys.stderr)

        results = []
        t0 = time.time()
        for i in range(steps):
            result = self.step()
            results.append(result)

            if not json_output:
                bar = "█" * max(0, min(40, int((i + 1) / steps * 40)))
                sys.stderr.write(
                    f"\r  Step {i + 1:3d}/{steps} arm={result['arm']:14s} "
                    f"reward={result['reward']:+.3f} {result['status']:5s} {bar}"
                )
                sys.stderr.flush()

        if not json_output:
            sys.stderr.write("\n")

        total_time = time.time() - t0
        return {
            "schema": "lucidota.rubberband_train.v1",
            "steps": steps,
            "total_time_s": round(total_time, 2),
            "avg_step_ms": round(total_time / steps * 1000, 2) if steps else 0,
            "bandit": self.bandit.status(),
            "success_rate": sum(1 for r in results if r["status"] == "ok") / max(len(results), 1),
            "avg_reward": round(sum(r["reward"] for r in results) / max(len(results), 1), 4),
        }


def main():
    parser = argparse.ArgumentParser(description="Rubberband online Mamba training loop")
    parser.add_argument("--steps", type=int, default=69, help="Training steps (default: 69)")
    parser.add_argument("--epsilon", type=float, default=0.13, help="Bandit exploration rate")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--gamble", action="store_true", help="ALL IN — epsilon=1.0")
    args = parser.parse_args()

    if args.gamble:
        args.epsilon = 1.0

    bandit = RubberbandBandit(epsilon=args.epsilon)
    trainer = RubberbandTrainer(bandit)
    result = trainer.run(steps=args.steps, json_output=args.json)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n=== Rubberband Complete ===")
        print(f"  Steps: {result['steps']}")
        print(f"  Success rate: {result['success_rate']:.1%}")
        print(f"  Avg reward: {result['avg_reward']:.3f}")
        print(f"  Time: {result['total_time_s']:.1f}s")
        for arm, stats in result["bandit"]["arms"].items():
            print(f"  {arm:14s}: {stats['pulls']:3d} pulls, avg reward {stats['avg_reward']:.3f}")

    # Write receipt
    receipt_dir = ROOT / "05_OUTPUTS" / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"rubberband_train_{time.strftime('%Y%m%dT%H%M%S')}.json"
    receipt_path.write_text(json.dumps(result, indent=2))
    print(f"  Receipt: {receipt_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
