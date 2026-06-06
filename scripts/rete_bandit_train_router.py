#!/usr/bin/env python3
"""
RETE Bandit Training Router — wires ALGOS/rete_bandit_gate.py into
the training pipeline for model routing between Bonsai/Mamba/Vision arms.

Uses the existing RETE forward-chaining rules + bandit regret for
adaptive model selection during training.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ALGOS.rete_bandit_gate import apply_rete_bandit  # noqa: E402
from ALGOS.bandit_router import select_action, update_policy  # noqa: E402


TRAINING_ARMS = {
    "bonsai_8b_q1": {
        "engine": "prismml_llama.cpp",
        "cost": 0.4,
        "vram": 0.0,
        "context": 1024,
        "description": "Bonsai 8B Q1 (ternary, CPU)",
    },
    "mamba_7b_ram": {
        "engine": "llama.cpp mamba",
        "cost": 0.3,
        "vram": 0.0,
        "context": 256,
        "description": "Mamba 7B RAM (CPU)",
    },
    "mamba_7b_gpu": {
        "engine": "llama.cpp mamba",
        "cost": 0.25,
        "vram": 1.2,
        "context": 128,
        "description": "Mamba 7B GPU (partial offload)",
    },
    "deepseek_1_5b": {
        "engine": "llama.cpp",
        "cost": 0.1,
        "vram": 0.8,
        "context": 2048,
        "description": "DeepSeek R1 1.5B Q4 (GPU)",
    },
    "bitvla_vision": {
        "engine": "siglip + bitnet 1.58",
        "cost": 0.15,
        "vram": 0.0,
        "context": 256,
        "description": "BitVLA vision (SigLIP CPU)",
    },
    "needle_shared": {
        "engine": "needle jax",
        "cost": 0.2,
        "vram": 0.5,
        "context": 512,
        "description": "Needle shared (6 slots)",
    },
}


class ReteBanditTrainRouter:
    """Wires RETE bandit gate into training pipeline."""

    def __init__(self, arms: dict[str, dict[str, Any]] | None = None):
        self.arms = arms or TRAINING_ARMS
        self.arm_rewards: dict[str, list[float]] = {arm: [] for arm in self.arms}
        self.rule_hits: list[str] = []
        self.total_decisions = 0

    def select_arm_for_task(
        self,
        task_type: str,
        packet: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Use RETE bandit gate to select the best arm for a task."""
        self.total_decisions += 1

        # Build context packet for RETE
        context_packet = {
            "task_type": task_type,
            "source": f"rete_bandit_train_router_{self.total_decisions}",
            "payload": {
                "available_arms": list(self.arms.keys()),
                "task_type": task_type,
                "total_decisions": self.total_decisions,
            },
        }
        if packet:
            context_packet["payload"].update(packet)

        # Call the RETE bandit gate
        try:
            rete_result = apply_rete_bandit(context_packet)
        except Exception as e:
            rete_result = {
                "selected_algorithm": random.choice(list(self.arms.keys())),
                "rule_hits": [f"fallback: {e}"],
            }

        selected = rete_result.get("selected_algorithm", random.choice(list(self.arms.keys())))
        # Map algorithm to arm
        if selected not in self.arms:
            selected = random.choice(list(self.arms.keys()))

        self.rule_hits.extend(rete_result.get("rule_hits", []))

        # Penalty: prefer cheaper arms when context is short
        arm_config = self.arms[selected]
        penalty = 0.0
        if task_type == "fast" and arm_config.get("cost", 0) > 0.2:
            penalty = -0.3

        return {
            "selected_arm": selected,
            "arm_config": arm_config,
            "rule_hits": rete_result.get("rule_hits", []),
            "total_decisions": self.total_decisions,
            "penalty": penalty,
        }

    def record_reward(self, arm: str, reward: float):
        """Record a reward for an arm."""
        if arm in self.arm_rewards:
            self.arm_rewards[arm].append(reward)
            if len(self.arm_rewards[arm]) > 200:
                self.arm_rewards[arm] = self.arm_rewards[arm][-200:]

    def status(self) -> dict[str, Any]:
        """Get full router status."""
        return {
            "total_decisions": self.total_decisions,
            "rule_hits_count": len(self.rule_hits),
            "arms": {
                arm: {
                    "pulls": len(pulls),
                    "avg_reward": sum(pulls) / len(pulls) if pulls else 0.0,
                    "config": self.arms[arm],
                }
                for arm, pulls in self.arm_rewards.items()
            },
        }


def simulate_training_run(
    router: ReteBanditTrainRouter,
    steps: int = 69,
    json_output: bool = False,
) -> dict[str, Any]:
    """Simulate a training run with RETE bandit routing."""
    task_types = ["chat", "code", "vision", "fast"]
    prompts = [
        "Write a poem about neural networks.",
        "Explain the transformer attention mechanism.",
        "Describe this image of a cat.",
        "Quick: what is 2+2?",
        "Generate a training loop for a ternary model.",
        "What's the capital of France?",
        "Debug this: x = [1,2,3]; print(x[3])",
        "Summarize the concept of entropy.",
    ]

    t0 = time.time()

    for i in range(steps):
        task = random.choice(task_types)
        prompt = random.choice(prompts)

        decision = router.select_arm_for_task(task)
        arm = decision["selected_arm"]

        # Simulated reward
        base_reward = random.uniform(-0.5, 1.0)
        penalty = decision.get("penalty", 0.0)
        reward = max(-1.0, min(1.0, base_reward + penalty))
        router.record_reward(arm, reward)

        if not json_output and (i + 1) % 10 == 0:
            sys.stderr.write(f"  Step {i + 1:3d}/{steps} arm={arm:16s} task={task:6s} reward={reward:+.3f}\n")

    total_time = time.time() - t0

    return {
        "schema": "lucidota.rete_bandit_train_router.v1",
        "steps": steps,
        "total_time_s": round(total_time, 2),
        "total_decisions": router.total_decisions,
        "rule_hits_sample": router.rule_hits[:10],
        "arms": router.status()["arms"],
    }


def main():
    parser = argparse.ArgumentParser(description="RETE Bandit Training Router")
    parser.add_argument("--steps", type=int, default=69, help="Number of training steps (default: 69)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--train", action="store_true", help="Simulate training run")
    parser.add_argument("--status", action="store_true", help="Show current router status")
    args = parser.parse_args()

    # Validate inputs
    if args.steps < 1:
        sys.exit("[error] --steps must be >= 1")

    router = ReteBanditTrainRouter()

    if args.train or not args.status:
        result = simulate_training_run(router, steps=args.steps, json_output=args.json)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n=== RETE Bandit Training Router ===")
            print(f"  Steps: {result['steps']}")
            print(f"  Decisions: {result['total_decisions']}")
            print(f"  Time: {result['total_time_s']:.1f}s")
            print(f"\n  Arm status:")
            for arm, stats in result["arms"].items():
                print(f"    {arm:18s}: {stats['pulls']:3d} pulls, avg reward {stats['avg_reward']:.3f}")

    if args.status:
        status = router.status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"\n=== Router Status ===")
            print(f"  Total decisions: {status['total_decisions']}")
            print(f"  Rule hits: {status['rule_hits_count']}")

    # Receipt
    receipt = router.status()
    receipt_dir = ROOT / "05_OUTPUTS" / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"rete_bandit_router_{time.strftime('%Y%m%dT%H%M%S')}.json"
    try:
        receipt_path.write_text(json.dumps(receipt, indent=2))
        if not args.json:
            print(f"\n  Receipt: {receipt_path.relative_to(ROOT)}")
    except OSError as e:
        print(f"\n  [warn] Failed to write receipt: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
