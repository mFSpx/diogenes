# DARWIN HAMMER — match 2131, survivor 0
# gen: 5
# parent_a: ternary_router.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_geomet_m429_s2.py (gen4)
# born: 2026-05-29T23:40:56Z

"""HybridRouter Module
Combines:
- Parent A: ternary_router.py – command routing, context parsing, and engine channel tagging.
- Parent B: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_geomet_m429_s2.py – Multivector algebra and Voronoi‑based region assignment.

Mathematical Bridge:
The packet payload from Parent A is embedded into a *Multivector* (Parent B) where each
field contributes a scalar coefficient.  A 2‑dimensional projection of this multivector
is treated as a point in Euclidean space; Voronoi partitioning (the `assign`/`nearest`
functions from Parent B) then selects the most appropriate processing engine
(e.g. “cpu_fairyfuse_ternary” or “cpu_fairyfuse_geometric”).  This creates a unified
routing decision that simultaneously respects the original command logic and the
geometric resource‑allocation model.
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def parse_context(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value

def emit_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))

# ----------------------------------------------------------------------
# Stub for Parent A's route_command (the real implementation lives in
# services.fairyfuse.fairyfuse_backend).  For the purpose of this hybrid
# module we provide a minimal stand‑in that mimics the required interface.
# ----------------------------------------------------------------------
class _RouteResult:
    def __init__(self, route: str):
        self._route = route

    def to_dict(self) -> dict:
        return {"route": self._route, "timestamp": now_z()}

def route_command(text: str, intent: str, context: dict) -> _RouteResult:
    # Very simple heuristic: combine text length and intent hash.
    score = len(text) + (hash(intent) & 0xFFFF)
    return _RouteResult(route=f"score_{score}")

# ----------------------------------------------------------------------
# Multivector & Voronoi utilities from Parent B
# ----------------------------------------------------------------------
class Multivector:
    def __init__(self, components: Dict[Tuple[int, ...], float], n: int):
        # Store only non‑zero components; blades are represented as sorted tuples.
        self.components = {tuple(sorted(k)): float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector({blade: coef for blade, coef in self.components.items() if len(blade) == k}, self.n)

    def scalar_part(self) -> float:
        return self.components.get(tuple(), 0.0)

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)
        elif lst[i] == lst[i + 1]:
            # cancel duplicate index (square to zero)
            lst.pop(i + 1)
            lst.pop(i)
            sign *= 0
            n -= 2
            i = max(i - 1, 0)
        else:
            i += 1
    return lst, sign

def _multiply_blades(blade_a: Tuple[int, ...], blade_b: Tuple[int, ...]) -> Tuple[frozenset, int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    regions: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Hybrid Functions (meeting the “at least three functions” requirement)
# ----------------------------------------------------------------------
def packet_to_multivector(packet: Dict[str, Any]) -> Multivector:
    """
    Encode selected packet fields into a multivector.
    - scalar (grade‑0) component: length of the raw command.
    - grade‑1 components: hash‑derived floats of intent and source.
    - grade‑2 component: numeric payload sum (if any).
    """
    components: Dict[Tuple[int, ...], float] = {}
    raw = str(packet.get("raw_command") or "")
    intent = str(packet.get("normalized_intent") or packet.get("intent") or "")
    source = str(packet.get("source") or "")
    payload = packet.get("payload") or {}

    # Grade‑0 (scalar)
    components[tuple()] = float(len(raw))

    # Grade‑1 (vectors)
    components[(1,)] = float(hash(intent) & 0xFFFFFFFF) / 1e9
    components[(2,)] = float(hash(source) & 0xFFFFFFFF) / 1e9

    # Grade‑2 (bivector) – sum of numeric payload values
    if isinstance(payload, dict):
        numeric_sum = sum(float(v) for v in payload.values() if isinstance(v, (int, float)))
        if numeric_sum != 0.0:
            components[(1, 2)] = numeric_sum

    return Multivector(components, n=3)

def multivector_to_point(mv: Multivector) -> Tuple[float, float]:
    """
    Project a multivector onto a 2‑D Euclidean plane.
    X coordinate = scalar part + first grade‑1 component.
    Y coordinate = second grade‑1 component + magnitude of grade‑2 part.
    """
    x = mv.scalar_part()
    grade1 = mv.grade(1).components
    x += grade1.get((1,), 0.0)
    y = grade1.get((2,), 0.0)
    grade2 = mv.grade(2).components
    y += math.sqrt(sum(v * v for v in grade2.values()))
    return (x, y)

def hybrid_route_packet(packet: Dict[str, Any],
                       engine_seeds: List[Tuple[float, float]]) -> Dict[str, Any]:
    """
    Perform a fused routing decision:
    1. Use the original `route_command` logic (Parent A) to obtain a base route.
    2. Encode the packet as a multivector and map it to a 2‑D point.
    3. Apply Voronoi assignment (Parent B) to select an engine channel.
    4. Return a combined dictionary with both route information and the chosen
       engine channel.
    """
    # Step 1 – base route
    base_route = route_command(
        packet.get("raw_command", ""),
        packet.get("normalized_intent", ""),
        parse_context(packet.get("context"))
    ).to_dict()

    # Step 2 – multivector projection
    mv = packet_to_multivector(packet)
    point = multivector_to_point(mv)

    # Step 3 – Voronoi engine selection
    idx = nearest(point, engine_seeds)
    engine_map = {
        0: "cpu_fairyfuse_ternary",
        1: "cpu_fairyfuse_geometric",
        2: "cpu_fairyfuse_hybrid"
    }
    engine_channel = engine_map.get(idx, "cpu_fairyfuse_unknown")

    # Step 4 – combine results
    result = {
        **base_route,
        "engine_channel": engine_channel,
        "outbound_state": "draft_only",
        "mv_point": {"x": point[0], "y": point[1]},
        "timestamp_utc": now_z()
    }
    return result

def compute_voronoi_regions(packets: List[Dict[str, Any]],
                            engine_seeds: List[Tuple[float, float]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Convert a list of packets into Voronoi regions based on their multivector
    projections.  Returns a mapping from seed index to the list of original
    packets that fall into that region.
    """
    points = [multivector_to_point(packet_to_multivector(p)) for p in packets]
    assignment = assign(points, engine_seeds)
    regions: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(engine_seeds))}
    for i, pkt in enumerate(packets):
        seed_idx = nearest(points[i], engine_seeds)
        regions[seed_idx].append(pkt)
    return regions

# ----------------------------------------------------------------------
# Command‑line interface (mirrors Parent A's minimal CLI)
# ----------------------------------------------------------------------
def status_cmd(_args: argparse.Namespace) -> int:
    # Placeholder status – in a real system this would query the backend.
    status = {"service": "HybridRouter", "state": "ready", "timestamp": now_z()}
    emit_json(status)
    return 0

def route_cmd(args: argparse.Namespace) -> int:
    packet = {
        "raw_command": args.raw_command,
        "normalized_intent": args.normalized_intent,
        "context": args.context,
        "payload": json.loads(args.payload) if args.payload else {}
    }
    # Fixed three seeds representing three engine channels.
    seeds = [(0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0)]
    result = hybrid_route_packet(packet, seeds)
    emit_json(result)
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid router combining ternary and geometric logic.")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Print service status")
    status.set_defaults(func=status_cmd)

    route = sub.add_parser("route", help="Route a single packet")
    route.add_argument("--raw_command", required=True, help="Raw command string")
    route.add_argument("--normalized_intent", default="", help="Normalized intent")
    route.add_argument("--context", default=None, help="JSON‑encoded context object")
    route.add_argument("--payload", default=None, help="JSON‑encoded payload dictionary")
    route.set_defaults(func=route_cmd)

    return parser

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple smoke test without invoking the CLI.
    sample_packet = {
        "raw_command": "activate shield",
        "normalized_intent": "defense_activate",
        "source": "sensor_alpha",
        "payload": {"temperature": 23.5, "pressure": 101.3}
    }
    engine_seeds = [(0.0, 0.0), (500.0, 500.0), (1000.0, 0.0)]
    print("Hybrid routing result:")
    print(json.dumps(hybrid_route_packet(sample_packet, engine_seeds), indent=2, ensure_ascii=False))

    # Demonstrate Voronoi region computation on a batch of packets.
    batch = [
        sample_packet,
        {"raw_command": "launch missile", "normalized_intent": "offense_launch", "source": "operator_bravo", "payload": {"ammo": 5}},
        {"raw_command": "scan area", "normalized_intent": "recon_scan", "source": "drone_zeta", "payload": {"range": 300}}
    ]
    regions = compute_voronoi_regions(batch, engine_seeds)
    print("\nVoronoi region assignment (seed index -> packet count):")
    print({k: len(v) for k, v in regions.items()})

    # If arguments are provided, run the CLI.
    if len(sys.argv) > 1:
        parser = build_parser()
        args = parser.parse_args()
        sys.exit(args.func(args))