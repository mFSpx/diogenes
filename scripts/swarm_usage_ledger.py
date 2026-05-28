#!/usr/bin/env python3
"""Aggregate Groq/local/main-agent token receipts into the requested swarm ledger."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "goals"
TARGET_POLICY = {
    "main_agent": {"target_share": 0.10, "note": "operator requested Codex/main-window kept near 10% when authoritative token receipts exist"},
    "groq": {"target_share": 0.35, "note": "fast cloud model lane"},
    "cohere": {"target_share": 0.15, "note": "cloud model lane and cross-audit lane"},
    "local": {"target_share": 0.40, "note": "cheap local execution lane; maximize when capable and receipt-backed"},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_model_invocation_receipt(path: Path) -> bool:
    try:
        payload = load(path)
    except Exception:
        return False
    return str(payload.get("schema") or "").startswith("lucidota.model_invocation.")


def discover_receipts(*, root: Path = ROOT) -> list[Path]:
    receipt_dir = Path(root) / "05_OUTPUTS" / "model_invocations"
    if not receipt_dir.exists():
        return []
    return [path for path in sorted(receipt_dir.glob("*.json")) if is_model_invocation_receipt(path)]


def provider_of(payload: dict[str, Any]) -> str:
    provider = str(payload.get("provider") or "")
    schema = str(payload.get("schema") or "")
    if provider == "groq" or "groq_chat" in schema:
        return "groq"
    if provider == "cohere" or "cohere_chat" in schema:
        return "cohere"
    if provider == "local" or "local_chat" in schema:
        return "local"
    return provider or "unknown"


def token_count_from_usage(usage: Any) -> tuple[int, str] | None:
    if not isinstance(usage, dict):
        return None
    if isinstance(usage.get("total_tokens"), int):
        return int(usage["total_tokens"]), "provider_usage"
    if isinstance(usage.get("tokens"), dict):
        tokens = usage["tokens"]
        total = sum(int(tokens.get(k) or 0) for k in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"))
        if total:
            return total, "provider_usage.tokens"
    if isinstance(usage.get("billed_units"), dict):
        billed = usage["billed_units"]
        total = sum(int(billed.get(k) or 0) for k in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"))
        if total:
            return total, "provider_usage.billed_units"
    return None


def token_count(payload: dict[str, Any]) -> tuple[int, str]:
    acct = payload.get("token_accounting")
    if isinstance(acct, dict) and isinstance(acct.get("total_tokens"), int):
        return int(acct["total_tokens"]), str(acct.get("source") or "token_accounting")
    root_usage = token_count_from_usage(payload.get("usage"))
    if root_usage:
        return root_usage
    raw_response = payload.get("raw_response")
    if isinstance(raw_response, dict):
        nested_usage = token_count_from_usage(raw_response.get("usage"))
        if nested_usage:
            tokens, source = nested_usage
            return tokens, f"raw_response.{source}"
    return 0, "missing"


def share(tokens: int, total: int) -> float:
    return (tokens / total) if total else 0.0


def discovery_mode_for(receipt_paths: list[Path]) -> str:
    if receipt_paths and all("model_invocations" in path.parts for path in receipt_paths):
        return "auto_model_invocations"
    return "explicit_receipts"


def build_ledger(receipt_paths: list[Path], *, main_tokens: int = 0, discovery_mode: str | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    totals = {k: {"tokens": 0, "receipt_count": 0, "share": 0.0} for k in TARGET_POLICY}
    totals["main_agent"]["tokens"] = int(main_tokens)
    if main_tokens:
        totals["main_agent"]["receipt_count"] = 1
    for path in receipt_paths:
        payload = load(Path(path))
        provider = provider_of(payload)
        tokens, source = token_count(payload)
        row = {"path": rel(path), "provider": provider, "lane": payload.get("lane"), "model": payload.get("model"), "status": payload.get("status"), "tokens": tokens, "token_source": source}
        rows.append(row)
        totals.setdefault(provider, {"tokens": 0, "receipt_count": 0, "share": 0.0})
        totals[provider]["tokens"] += tokens
        totals[provider]["receipt_count"] += 1
        if payload.get("blockers"):
            blockers.append({"path": rel(path), "provider": provider, "blockers": payload.get("blockers")})
    total = sum(int(v["tokens"]) for v in totals.values())
    for item in totals.values():
        item["share"] = share(int(item["tokens"]), total)
    mode = discovery_mode or discovery_mode_for([Path(p) for p in receipt_paths])
    return {
        "schema": "lucidota.goals.swarm_usage_ledger.v1",
        "generated_at": now(),
        "target_policy": TARGET_POLICY,
        "discovery": {"mode": mode, "receipt_count": len(receipt_paths)},
        "totals": {**totals, "all_accounted_tokens": total},
        "receipts": rows,
        "blockers": blockers,
        "main_agent_note": "Main-window tokens are external to repo receipts; pass --main-tokens when an authoritative account figure is available.",
    }


def write_outputs(ledger: dict[str, Any], md_path: str | None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"swarm_usage_ledger_{stamp()}.json"
    ledger["report_path"] = rel(path)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    if md_path:
        write_markdown(ledger, ROOT / md_path)
    return path


def write_markdown(ledger: dict[str, Any], path: Path) -> None:
    lines = ["# Swarm Wiring Usage Ledger", "", f"Generated: `{ledger['generated_at']}`", "", "| lane | tokens | share | policy |", "|---|---:|---:|---|"]
    for lane in ["main_agent", "local", "groq", "cohere"]:
        item = ledger["totals"][lane]
        policy = ledger.get("target_policy", {}).get(lane, {})
        policy_text = "; ".join(f"{k}={v}" for k, v in policy.items())
        lines.append(f"| {lane} | {item['tokens']} | {item['share']:.3f} | {policy_text} |")
    lines += ["", "## Receipts", "", "| provider | lane | tokens | source | status | path |", "|---|---|---:|---|---|---|"]
    for row in ledger["receipts"]:
        lines.append(f"| {row['provider']} | {row.get('lane') or ''} | {row['tokens']} | {row['token_source']} | {row.get('status') or ''} | `{row['path']}` |")
    if ledger["blockers"]:
        lines += ["", "## Blockers", ""]
        for b in ledger["blockers"]:
            lines.append(f"- `{b['path']}`: {', '.join(map(str, b['blockers']))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a receipt-backed token ledger for the LUCIDOTA swarm target split.")
    ap.add_argument("--receipt", action="append", default=[])
    ap.add_argument("--main-tokens", type=int, default=0)
    ap.add_argument("--write-md")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    receipt_paths = [Path(p) for p in args.receipt] if args.receipt else discover_receipts()
    ledger = build_ledger(receipt_paths, main_tokens=args.main_tokens, discovery_mode="explicit_receipts" if args.receipt else "auto_model_invocations")
    path = write_outputs(ledger, args.write_md)
    if args.json:
        print(json.dumps(ledger, sort_keys=True))
    print("REPORT_PATH=" + rel(path))
    print("SWARM_USAGE_LEDGER=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
