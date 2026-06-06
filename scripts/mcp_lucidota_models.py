#!/usr/bin/env python3
"""
MCP Server: lucidota-models — Model routing + RETE bandit gate for local model endpoints.

Tools:
  mcp__models__select_route    — Use RETE bandit to pick best model for a task
  mcp__models__call_model      — Call a specific local model endpoint
  mcp__models__bandit_status   — Get bandit arm status and statistics
  mcp__models__elastic_shape   — Route through elastic_shape.rs

Stdio MCP transport. Designed for ./luci McpServerManager.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pypeline"))

# ─── Model route registry ──────────────────────────────────────────────

MODEL_ROUTES: dict[str, dict[str, Any]] = {
    "deepseek_1_5b": {
        "url": "http://127.0.0.1:8080/v1/chat/completions",
        "engine": "llama.cpp",
        "vram_gb": 0.8,
        "context": 2048,
        "cost_penalty": 0.1,
    },
    "mamba_7b_ram": {
        "url": "http://127.0.0.1:8081/v1/chat/completions",
        "engine": "llama.cpp (mamba)",
        "vram_gb": 0.0,
        "context": 256,
        "cost_penalty": 0.3,
    },
    "mamba_7b_gpu": {
        "url": "http://127.0.0.1:8083/v1/chat/completions",
        "engine": "llama.cpp (mamba gpu)",
        "vram_gb": 1.2,
        "context": 128,
        "cost_penalty": 0.25,
    },
    "bonsai_8b_q1": {
        "url": "http://127.0.0.1:8082/v1/chat/completions",
        "engine": "prismml_llama.cpp (bonsai 8b Q1)",
        "vram_gb": 0.0,
        "context": 1024,
        "cost_penalty": 0.4,
    },
    "bitvla_vision": {
        "url": "http://127.0.0.1:7845",
        "engine": "siglip + bitnet 1.58-bit",
        "vram_gb": 0.0,
        "context": 0,
        "cost_penalty": 0.15,
    },
    "needle_shared": {
        "url": "http://127.0.0.1:8090",
        "engine": "needle jax",
        "vram_gb": 0.5,
        "context": 512,
        "cost_penalty": 0.2,
    },
}

# Bandit arm state (in-memory, ephemeral)
bandit_arms: dict[str, list[float]] = {route: [] for route in MODEL_ROUTES}


def _select_route_bandit(task_type: str, epsilon: float = 0.13) -> dict[str, Any]:
    """Epsilon-greedy route selection with regret tracking."""
    # Apply RETE-style pruning based on task_type
    candidates = list(MODEL_ROUTES.keys())
    if task_type == "vision":
        candidates = [r for r in candidates if "vision" in r or "needle" in r]
    elif task_type == "code":
        candidates = [r for r in candidates if "deepseek" in r or "bonsai" in r]
    elif task_type == "chat":
        candidates = [r for r in candidates if "mamba" in r or "deepseek" in r or "bonsai" in r]
    elif task_type == "fast":
        candidates = [r for r in candidates if "deepseek" in r or "mamba" in r]

    if not candidates:
        candidates = list(MODEL_ROUTES.keys())

    # Epsilon-greedy with regret weighting
    if random.random() < epsilon:
        selected = random.choice(candidates)
        strategy = "explore"
    else:
        # Greedy by best average reward
        def avg_reward(route: str) -> float:
            pulls = bandit_arms.get(route, [])
            return sum(pulls) / len(pulls) if pulls else 0.0

        selected = max(candidates, key=avg_reward)
        strategy = "exploit"

    return {
        "selected_route": selected,
        "strategy": strategy,
        "candidates": candidates,
        "route_config": MODEL_ROUTES[selected],
    }


def _call_model_route(route: str, prompt: str, max_tokens: int = 128) -> dict[str, Any]:
    """Call a local model endpoint via OpenAI-compatible API."""
    import urllib.request

    if route not in MODEL_ROUTES:
        return {"status": "error", "error": f"Unknown route: {route}"}

    config = MODEL_ROUTES[route]
    url = config["url"]

    # Vision endpoint is different
    if route == "bitvla_vision":
        return {"status": "error", "error": "bitvla_vision requires image input, use /vision/ endpoints"}

    payload = json.dumps({
        "model": route,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return {"status": "ok", "route": route, "result": result}
    except Exception as e:
        return {"status": "error", "route": route, "error": str(e)}


def _bandit_status() -> dict[str, Any]:
    """Get bandit arm pull counts and average rewards."""
    return {
        route: {
            "pulls": len(pulls),
            "avg_reward": sum(pulls) / len(pulls) if pulls else 0.0,
        }
        for route, pulls in bandit_arms.items()
    }


def _elastic_shape_route(packet: dict[str, Any]) -> dict[str, Any]:
    """Shell out to elastic_shape binary."""
    binary = ROOT / "01_REPOS" / "lucidota_resonance" / "target" / "release" / "lucidota_elastic_shape"
    if not binary.exists():
        binary = ROOT / "01_REPOS" / "lucidota_resonance" / "target" / "debug" / "lucidota_elastic_shape"
    if not binary.exists():
        return {"status": "no_binary"}

    try:
        proc = subprocess.run(
            [str(binary)],
            input=json.dumps(packet),
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            return {"status": "ok", "result": json.loads(proc.stdout)}
        return {"status": "error", "stderr": proc.stderr[:500]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─── MCP Protocol ──────────────────────────────────────────────────────

def jsonrpc_error(id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def jsonrpc_result(id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def handle_request(req: dict[str, Any]) -> dict[str, Any] | None:
    rid = req.get("id")
    method = req.get("method", "")

    if method == "initialize":
        return jsonrpc_result(rid, {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": False,
                },
            },
            "serverInfo": {"name": "lucidota-models", "version": "0.1.0"},
        })

    if method == "notifications/initialized":
        return None  # No response needed

    if method == "tools/list":
        tools = [
            {
                "name": "mcp__models__select_route",
                "description": "Use RETE bandit to select the best model route for a task type",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_type": {"type": "string", "description": "Task type: chat, code, vision, fast"},
                        "epsilon": {"type": "number", "description": "Exploration rate (default: 0.13)"},
                    },
                },
            },
            {
                "name": "mcp__models__call_model",
                "description": "Call a local model endpoint with a prompt",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "route": {"type": "string", "description": "Model route name"},
                        "prompt": {"type": "string", "description": "Prompt text"},
                        "max_tokens": {"type": "integer", "description": "Max tokens (default: 128)"},
                    },
                },
            },
            {
                "name": "mcp__models__bandit_status",
                "description": "Get current bandit arm statistics for all model routes",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "mcp__models__elastic_shape",
                "description": "Route a packet through elastic_shape.rs resonance matching",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "packet": {"type": "object", "description": "Packet to route"},
                    },
                },
            },
        ]
        return jsonrpc_result(rid, {"tools": tools})

    if method == "tools/call":
        params = req.get("params", {})
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        if name == "mcp__models__select_route":
            task_type = arguments.get("task_type", "chat")
            epsilon = float(arguments.get("epsilon", 0.13))
            # Clamp epsilon to valid range [0.0, 1.0]
            epsilon = max(0.0, min(1.0, epsilon))
            result = _select_route_bandit(task_type, epsilon)
            # Record a synthetic pull
            route = result["selected_route"]
            if route in bandit_arms:
                bandit_arms[route].append(0.5)  # neutral initial reward
            return jsonrpc_result(rid, {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]})

        if name == "mcp__models__call_model":
            route = arguments.get("route", "deepseek_1_5b")
            prompt = arguments.get("prompt", "")
            max_tokens = int(arguments.get("max_tokens", 128))
            t0 = time.time()
            result = _call_model_route(route, prompt, max_tokens)
            elapsed = time.time() - t0
            result["elapsed_s"] = round(elapsed, 3)
            # Record reward based on success
            reward = 1.0 if result.get("status") == "ok" else -0.5
            if route in bandit_arms:
                bandit_arms[route].append(reward)
            return jsonrpc_result(rid, {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]})

        if name == "mcp__models__bandit_status":
            return jsonrpc_result(rid, {"content": [{"type": "text", "text": json.dumps(_bandit_status(), indent=2)}]})

        if name == "mcp__models__elastic_shape":
            packet = arguments.get("packet", {})
            result = _elastic_shape_route(packet)
            return jsonrpc_result(rid, {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]})

        return jsonrpc_error(rid, -32601, f"Unknown tool: {name}")

    if method == "shutdown":
        return jsonrpc_result(rid, None)

    return jsonrpc_error(rid, -32601, f"Method not found: {method}")


def main():
    """MCP stdio transport: read JSON-RPC from stdin, write to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError:
            resp = jsonrpc_error(None, -32700, "Parse error")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception:
            resp = jsonrpc_error(None, -32603, traceback.format_exc())
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
