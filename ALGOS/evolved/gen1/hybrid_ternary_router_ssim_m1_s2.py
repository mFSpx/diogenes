# DARWIN HAMMER — match 1, survivor 2
# gen: 1
# parent_a: ternary_router.py (gen0)
# parent_b: ssim.py (gen0)
# born: 2026-05-29T23:13:26Z

"""Hybrid router integrating FairyFuse ternary routing with Structural Similarity Index (SSIM).

Parent A (ternary_router.py) provides a packet‑routing skeleton that selects an
engine channel based on textual intent.  Parent B (ssim.py) offers a statistical
measure of similarity between two equal‑length numeric sequences:

    SSIM(x, y) = ((2·μₓ·μ_y + C₁)·(2·σₓy + C₂)) / ((μₓ²+μ_y²+C₁)·(σₓ²+σ_y²+C₂))

where μ is the mean, σ² the variance and σₓy the covariance.

The hybrid algorithm treats the payload of a packet as a numeric feature vector.
It computes the SSIM between this payload and a stored prototype vector.  The
resulting similarity score becomes a quantitative “routing metric” that
augments the original intent‑based decision.  High SSIM drives the packet to
the ternary engine; low SSIM diverts it to an alternative (binary) engine.
Thus the statistical topology of SSIM is fused into the routing topology of
the original daemon.

The module supplies:
* `compute_ssim` – NumPy‑accelerated SSIM implementation.
* `hybrid_score` – extracts a numeric payload and returns its SSIM against a
  prototype.
* `route_packet_hybrid` – builds a routing dictionary enriched with the SSIM
  score and selects an engine channel accordingly.
* `daemon_hybrid` – an always‑on loop that logs heart‑beats together with the
  latest SSIM score for a synthetic test packet.

All code is pure Python 3 and depends only on the standard library and NumPy."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Constants and simple placeholders (stand‑ins for the real FairyFuse backend)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[0]
RUNTIME_DIR = ROOT / "runtime"
DEFAULT_HEARTBEAT = RUNTIME_DIR / "hybrid_router_heartbeat.jsonl"

# A mock prototype vector against which payloads are compared.
# In a real system this would be loaded from a model or a database.
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)


# ----------------------------------------------------------------------
# SSIM implementation (Parent B)
# ----------------------------------------------------------------------
def compute_ssim(
    x: Sequence[float],
    y: Sequence[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Return the Structural Similarity Index between two equal‑length sequences.

    The implementation mirrors the original `ssim` function but uses NumPy for
    vectorised arithmetic, yielding the same mathematical result.
    """
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Hybrid routing utilities (fusion of Parent A and Parent B)
# ----------------------------------------------------------------------
def hybrid_score(packet: Dict[str, Any]) -> float:
    """Compute an SSIM‑based similarity score for a packet.

    The function expects the packet to contain a numeric payload under the key
    ``payload``.  If the payload is missing or not a sequence of numbers, a
    score of 0.0 is returned.
    """
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        return 0.0
    try:
        # Truncate or pad the payload to match the prototype length.
        payload_vec = np.asarray(payload, dtype=np.float64)
        if payload_vec.size < PROTOTYPE_VECTOR.size:
            # Pad with zeros
            payload_vec = np.pad(payload_vec, (0, PROTOTYPE_VECTOR.size - payload_vec.size))
        elif payload_vec.size > PROTOTYPE_VECTOR.size:
            # Truncate
            payload_vec = payload_vec[: PROTOTYPE_VECTOR.size]
        return compute_ssim(payload_vec, PROTOTYPE_VECTOR, dynamic_range=1.0)
    except Exception:
        return 0.0


def route_packet_hybrid(packet: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a routing dictionary enriched with an SSIM similarity metric.

    The original intent extraction mirrors Parent A.  The SSIM score influences
    the selected ``engine_channel`` and adds a ``similarity`` field to the
    output.
    """
    # Extract textual intent (fallback to a default)
    text = str(
        packet.get("text_surface")
        or packet.get("raw_command")
        or packet.get("text")
        or ""
    )
    intent = str(
        packet.get("normalized_intent")
        or packet.get("intent")
        or "bytewax_rete_bandit"
    )

    # Compute similarity score
    similarity = hybrid_score(packet)

    # Simple decision rule: high similarity → ternary engine, else binary engine
    engine_channel = (
        "cpu_fairyfuse_ternary" if similarity >= 0.5 else "cpu_fairyfuse_binary"
    )
    outbound_state = "draft_only" if similarity >= 0.5 else "queued"

    # Build the routing payload (a minimal stand‑in for the real FairyFuse route)
    route = {
        "text": text[:4096],
        "intent": intent,
        "engine_channel": engine_channel,
        "outbound_state": outbound_state,
        "similarity": similarity,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return route


def daemon_hybrid(
    idle_sleep: float = 2.0,
    max_cycles: int = 0,
    heartbeat_path: Path = DEFAULT_HEARTBEAT,
) -> int:
    """Run an always‑on daemon that writes heart‑beats with SSIM scores.

    The daemon creates a synthetic test packet each cycle, computes its hybrid
    route, and logs the full receipt (including the similarity metric) to the
    specified heartbeat file.
    """
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

    stop = {"flag": False}

    def _handle_signal(signum: int, frame):  # noqa: ANN001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    cycles = 0
    while not stop["flag"]:
        cycles += 1

        # Synthetic packet: random payload of length 5 in [0,1]
        test_packet = {
            "payload": np.random.rand(5).tolist(),
            "text_surface": "heartbeat test",
            "normalized_intent": "heartbeat_intent",
        }

        route = route_packet_hybrid(test_packet)

        receipt = {
            "schema": "lucidota.fairyfuse.hybrid_router_heartbeat.v1",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "cycle": cycles,
            "pid": os.getpid(),
            "always_on": True,
            "engine_channel": route["engine_channel"],
            "similarity": route["similarity"],
            "status": {
                "intent": route["intent"],
                "outbound_state": route["outbound_state"],
            },
        }

        with heartbeat_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")

        if max_cycles and cycles >= max_cycles:
            break
        time.sleep(max(0.1, float(idle_sleep)))
    return 0


# ----------------------------------------------------------------------
# Command‑line interface (mirrors Parent A but adds the hybrid command)
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid FairyFuse router with SSIM")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # status (placeholder)
    p_status = sub.add_parser("status")
    p_status.set_defaults(func=lambda _: print(json.dumps({"status": "ok"})))

    # route
    p_route = sub.add_parser("route")
    p_route.add_argument("--raw-command", default="")
    p_route.add_argument("--normalized-intent", default="bytewax_rete_bandit")
    p_route.add_argument("--payload", default="[]")
    p_route.set_defaults(func=_cmd_route)

    # daemon
    p_daemon = sub.add_parser("daemon")
    p_daemon.add_argument("--idle-sleep", type=float, default=2.0)
    p_daemon.add_argument("--max-cycles", type=int, default=0)
    p_daemon.add_argument(
        "--heartbeat",
        default=str(DEFAULT_HEARTBEAT.relative_to(ROOT)),
    )
    p_daemon.set_defaults(func=_cmd_daemon)

    parser.set_defaults(func=lambda _: parser.print_help())
    return parser


def _cmd_route(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(args.payload)
        if not isinstance(payload, list):
            raise ValueError
    except Exception:
        print(json.dumps({"error": "payload must be a JSON array of numbers"}))
        return 1

    packet = {
        "raw_command": args.raw_command,
        "normalized_intent": args.normalized_intent,
        "payload": payload,
    }
    route = route_packet_hybrid(packet)
    print(json.dumps(route, ensure_ascii=False, sort_keys=True))
    return 0


def _cmd_daemon(args: argparse.Namespace) -> int:
    hb_path = Path(args.heartbeat)
    if not hb_path.is_absolute():
        hb_path = ROOT / hb_path
    return daemon_hybrid(
        idle_sleep=args.idle_sleep,
        max_cycles=args.max_cycles,
        heartbeat_path=hb_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


# ----------------------------------------------------------------------
# Smoke test executed when the module is run directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple smoke test: route a handcrafted packet and print the result.
    sample_packet = {
        "text_surface": "example command",
        "normalized_intent": "example_intent",
        "payload": [0.2, 0.5, 0.3, 0.7, 0.1],
    }
    print("Hybrid route output:")
    print(json.dumps(route_packet_hybrid(sample_packet), indent=2, sort_keys=True))

    # Run a short daemon (2 cycles) to verify heart‑beat creation.
    print("\nRunning short daemon (2 cycles)...")
    daemon_hybrid(idle_sleep=0.1, max_cycles=2, heartbeat_path=DEFAULT_HEARTBEAT)
    print(f"Heartbeat written to {DEFAULT_HEARTBEAT}")