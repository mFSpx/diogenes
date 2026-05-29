#!/usr/bin/env python3
"""CLI wrapper for the existing RETE/bandit fastlane-slowlane router.

This copies no algorithm code and mutates no sovereign ALGOS artifact. It makes
ALGOS.rete_bandit_gate runnable from the operator shell and writes a receipt.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ALGOS.rete_bandit_gate import apply_rete_bandit  # noqa: E402

OUT = ROOT / "05_OUTPUTS" / "routing"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_json_arg(value: str) -> dict[str, Any]:
    data = json.loads(value)
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("packet JSON must decode to an object")
    return data


def load_packet(args: argparse.Namespace) -> dict[str, Any]:
    if args.packet_json is not None:
        packet = load_json_arg(args.packet_json)
    elif args.packet_file is not None:
        path = Path(args.packet_file)
        if not path.is_absolute():
            path = ROOT / path
        packet = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(packet, dict):
            raise SystemExit("packet file must contain a JSON object")
        packet.setdefault("source_ref", rel(path))
    else:
        packet = {
            "source": "operator_cli",
            "source_ref": args.source_ref or "inline_text",
            "text_surface": args.text,
            "ontology_terms": args.ontology_term or [],
            "payload": {},
        }
    if args.source_ref:
        packet["source_ref"] = args.source_ref
    if args.ontology_term:
        merged = list(packet.get("ontology_terms") or [])
        for term in args.ontology_term:
            if term not in merged:
                merged.append(term)
        packet["ontology_terms"] = merged
    return packet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route one local packet through deterministic RETE pruning + bandit selection."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--packet-json", help="Full packet JSON object.")
    source.add_argument("--packet-file", help="Path to a packet JSON file.")
    source.add_argument("--text", help="Inline text surface for a small operator packet.")
    parser.add_argument("--source-ref", help="Optional source reference stamped into the packet.")
    parser.add_argument("--ontology-term", action="append", default=[], help="GO/ontology term hint; repeatable.")
    parser.add_argument("--strategy", default="linucb", choices=["linucb", "epsilon_greedy", "thompson"])
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--no-receipt", action="store_true", help="Do not write 05_OUTPUTS/routing receipt.")
    parser.add_argument("--json", action="store_true", help="Print the full decision JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    packet = load_packet(args)
    decision = apply_rete_bandit(packet, strategy=args.strategy, epsilon=args.epsilon, seed=args.seed)
    receipt = {
        "schema": "lucidota.rete_bandit_gate_cli.receipt.v1",
        "generated_at": now_iso(),
        "execute_performed": True,
        "canonical_graph_writes_performed": False,
        "packet_source_ref": packet.get("source_ref"),
        "strategy": args.strategy,
        "decision": decision,
    }
    if not args.no_receipt:
        OUT.mkdir(parents=True, exist_ok=True)
        out = OUT / f"rete_bandit_gate_cli_{stamp()}.json"
        receipt["report_path"] = rel(out)
        out.write_text(json.dumps(receipt, indent=2, sort_keys=False), encoding="utf-8")
        print("REPORT_PATH=" + rel(out))
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    print("RETE_BANDIT_GATE=PASS" if decision.get("execution_status") == "succeeded" else "RETE_BANDIT_GATE=FAIL")
    print("SELECTED_ALGORITHM=" + str(decision.get("selected_algorithm")))
    print("SELECTED_ENGINE=" + str(decision.get("selected_engine")))
    return 0 if decision.get("execution_status") == "succeeded" else 4


if __name__ == "__main__":
    raise SystemExit(main())
