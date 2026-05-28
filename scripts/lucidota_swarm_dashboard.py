#!/usr/bin/env python3
"""
LUCIDOTA terminal dashboard UI/UX payload.

Scope / safety contract
-----------------------
- Read-only frontend: this module only reads JSON receipts under 05_OUTPUTS.
- No network calls, no database calls, no canonical graph writes, no destructive actions.
- Operator actions are rendered as inert command suggestions; nothing is executed.
- Optional frontend stack: Textual + Rich when available; Rich-only when Textual is absent;
  plain text fallback if Rich is also absent.

Primary panels
--------------
1. Canonical graph / promotion status: write barriers, dry-run counts, mutation candidate
   counts, blocker counts, and policy posture.
2. Recent Bitloops / full reingest receipts: latest batch summaries discovered from the
   output tree.
3. Quarantine counts / failures: aggregate totals plus per-ETL breakdown.
4. Operator actions: copyable suggestions only, clearly marked inert.

Run:
    python3 05_OUTPUTS/swarm_submissions/ui_ux_final.py
    python3 05_OUTPUTS/swarm_submissions/ui_ux_final.py --rich
    python3 05_OUTPUTS/swarm_submissions/ui_ux_final.py --self-check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for base in (current.parent, *current.parents):
        if (base / "AGENTS.md").exists() and (base / "00_PROJECT_BRAIN").exists():
            return base
    return current.parents[2]


REPO_ROOT = _repo_root()
DEFAULT_SUMMARY = REPO_ROOT / "05_OUTPUTS/bitloops/full_reingest_batches/20260527T041930Z/full_existing_index_reingest_summary.json"
BITLOOPS_GLOB = "05_OUTPUTS/bitloops/full_reingest_batches/*/full_existing_index_reingest_summary.json"
PROMOTION_HINT_GLOBS = (
    "05_OUTPUTS/**/*promotion*.json",
    "05_OUTPUTS/**/*graph*receipt*.json",
    "05_OUTPUTS/**/*write_barrier*.json",
)
MAX_RECENT_RECEIPTS = 8

INERT_ACTIONS = (
    "python3 05_OUTPUTS/swarm_submissions/ui_ux_final.py --self-check",
    "python3 scripts/dev_library_scan.py --query graph promotion",
    "python3 scripts/dev_library_scan.py --query bitloops reingest quarantine",
    "python3 scripts/slop_audit_law.py --paths 05_OUTPUTS/swarm_submissions/ui_ux_final.py",
    "less 05_OUTPUTS/bitloops/full_reingest_batches/20260527T041930Z/full_existing_index_reingest_summary.json",
)


@dataclass(frozen=True)
class ReceiptSummary:
    path: str
    status: str = "UNKNOWN"
    schema: str = "UNKNOWN"
    accepted_total: int = 0
    dry_run_candidates_total: int = 0
    graph_mutation_candidates_total: int = 0
    graph_writes_performed: bool = False
    canonical_graph_writes_performed: bool = False
    db_writes_performed: bool = False
    graph_blocker_count: int = 0
    write_violation_count: int = 0
    quarantine_total: int = 0
    quarantine_failed_total: int = 0
    failure_count: int = 0
    recovered_total: int = 0
    processed_chunks: int = 0
    ledger_entries: int = 0
    batch_root: str = ""
    etl_breakdown: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class DashboardState:
    generated_at: str
    repo_root: str
    frontend_mode: str
    rich_available: bool
    textual_available: bool
    primary_receipt: ReceiptSummary | None
    recent_receipts: tuple[ReceiptSummary, ...]
    promotion_hint_paths: tuple[str, ...]
    inert_actions: tuple[str, ...]
    warnings: tuple[str, ...]


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.replace(",", "")))
        except ValueError:
            return 0
    return 0


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {"_raw": data}


def _receipt_from_json(path: Path) -> ReceiptSummary:
    data = _load_json(path)
    return ReceiptSummary(
        path=str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
        status=str(data.get("status", "UNKNOWN")),
        schema=str(data.get("schema", "UNKNOWN")),
        accepted_total=_safe_int(data.get("accepted_total")),
        dry_run_candidates_total=_safe_int(data.get("dry_run_candidates_total")),
        graph_mutation_candidates_total=_safe_int(data.get("graph_mutation_candidates_total")),
        graph_writes_performed=bool(data.get("graph_writes_performed", False)),
        canonical_graph_writes_performed=bool(data.get("canonical_graph_writes_performed", False)),
        db_writes_performed=bool(data.get("db_writes_performed", False)),
        graph_blocker_count=_safe_int(data.get("graph_blocker_count")),
        write_violation_count=_safe_int(data.get("write_violation_count")),
        quarantine_total=_safe_int(data.get("quarantine_total")),
        quarantine_failed_total=_safe_int(data.get("quarantine_failed_total")),
        failure_count=_safe_int(data.get("failure_count")),
        recovered_total=_safe_int(data.get("recovered_total")),
        processed_chunks=_safe_int(data.get("processed_chunks")),
        ledger_entries=_safe_int(data.get("ledger_entries")),
        batch_root=str(data.get("batch_root", "")),
        etl_breakdown=data.get("etl_breakdown", {}) if isinstance(data.get("etl_breakdown", {}), dict) else {},
    )


def _discover_recent_receipts() -> tuple[ReceiptSummary, ...]:
    paths = sorted(REPO_ROOT.glob(BITLOOPS_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    receipts: list[ReceiptSummary] = []
    for path in paths[:MAX_RECENT_RECEIPTS]:
        try:
            receipts.append(_receipt_from_json(path))
        except Exception as exc:  # dashboard must survive one bad receipt
            receipts.append(ReceiptSummary(path=str(path), status=f"UNREADABLE: {exc.__class__.__name__}"))
    return tuple(receipts)


def _discover_promotion_hints() -> tuple[str, ...]:
    seen: set[Path] = set()
    hints: list[str] = []
    for pattern in PROMOTION_HINT_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            hints.append(str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path))
            if len(hints) >= 10:
                return tuple(hints)
    return tuple(hints)


def availability() -> tuple[bool, bool]:
    rich_available = importlib.util.find_spec("rich") is not None
    textual_available = importlib.util.find_spec("textual") is not None
    return rich_available, textual_available


def collect_dashboard_state(receipt_path: Path = DEFAULT_SUMMARY, force_mode: str | None = None) -> DashboardState:
    rich_available, textual_available = availability()
    warnings: list[str] = []
    primary = None
    if receipt_path.exists():
        try:
            primary = _receipt_from_json(receipt_path)
        except Exception as exc:
            warnings.append(f"Primary receipt unreadable: {receipt_path} ({exc})")
    else:
        warnings.append(f"Primary receipt not present: {receipt_path.relative_to(REPO_ROOT) if receipt_path.is_relative_to(REPO_ROOT) else receipt_path}")

    if force_mode:
        mode = force_mode
    elif textual_available and rich_available:
        mode = "textual"
    elif rich_available:
        mode = "rich"
    else:
        mode = "plain"

    return DashboardState(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        repo_root=str(REPO_ROOT),
        frontend_mode=mode,
        rich_available=rich_available,
        textual_available=textual_available,
        primary_receipt=primary,
        recent_receipts=_discover_recent_receipts(),
        promotion_hint_paths=_discover_promotion_hints(),
        inert_actions=tuple(INERT_ACTIONS),
        warnings=tuple(warnings),
    )


def _status_style(status: str) -> str:
    upper = status.upper()
    if upper == "PASS" or upper == "OK":
        return "green"
    if "WARN" in upper or "DEGRADED" in upper:
        return "yellow"
    if "FAIL" in upper or "CRITICAL" in upper or "UNREADABLE" in upper:
        return "red"
    return "cyan"


def _graph_posture(receipt: ReceiptSummary | None) -> tuple[str, str]:
    if receipt is None:
        return "UNKNOWN", "No readable receipt; keep graph mutation controls closed."
    writes = receipt.graph_writes_performed or receipt.canonical_graph_writes_performed or receipt.db_writes_performed
    if writes:
        return "WRITE ACTIVITY DETECTED", "Receipt says a write path ran; escalate before promotion."
    if receipt.write_violation_count or receipt.graph_blocker_count:
        return "BLOCKED / REVIEW", "No writes reported, but blockers or violations need review."
    return "READ-ONLY DRY RUN", "No graph/db writes reported; mutation candidates remain suggestions."


def _aggregate_quarantine(receipts: Iterable[ReceiptSummary]) -> tuple[int, int, int]:
    total = failed = failures = 0
    for receipt in receipts:
        total += receipt.quarantine_total
        failed += receipt.quarantine_failed_total
        failures += receipt.failure_count
    return total, failed, failures


def render_rich(state: DashboardState) -> None:
    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        render_plain(state)
        return

    console = Console()
    primary = state.primary_receipt
    posture, posture_note = _graph_posture(primary)

    header = Text("LUCIDOTA Canonical Graph / Bitloops Dashboard", style="bold white")
    header.append("\nread-only terminal surface · commands are inert suggestions", style="dim")
    console.print(Panel(header, border_style="cyan", box=box.ROUNDED))

    graph = Table(title="Canonical Graph + Promotion Posture", box=box.SIMPLE_HEAVY)
    graph.add_column("Signal", style="bold")
    graph.add_column("Value")
    graph.add_column("UX Meaning")
    graph.add_row("Posture", f"[{_status_style(posture)}]{posture}[/]", posture_note)
    graph.add_row("Graph writes performed", str(bool(primary and primary.graph_writes_performed)), "Must stay False for this dashboard lane")
    graph.add_row("Canonical graph writes", str(bool(primary and primary.canonical_graph_writes_performed)), "Must stay False unless operator promotes elsewhere")
    graph.add_row("DB writes performed", str(bool(primary and primary.db_writes_performed)), "Dashboard never opens backend handles")
    graph.add_row("Mutation candidates", f"{primary.graph_mutation_candidates_total:,}" if primary else "n/a", "Visualized as queue pressure, not executed")
    graph.add_row("Dry-run candidates", f"{primary.dry_run_candidates_total:,}" if primary else "n/a", "Promotion review fuel")
    graph.add_row("Blockers / violations", f"{(primary.graph_blocker_count if primary else 0):,} / {(primary.write_violation_count if primary else 0):,}", "Red badges when non-zero")
    console.print(graph)

    receipts = Table(title="Recent Bitloops / Full Reingest Receipts", box=box.SIMPLE)
    receipts.add_column("Status")
    receipts.add_column("Accepted", justify="right")
    receipts.add_column("Recovered", justify="right")
    receipts.add_column("Chunks", justify="right")
    receipts.add_column("Quarantine", justify="right")
    receipts.add_column("Receipt")
    for receipt in state.recent_receipts:
        receipts.add_row(
            f"[{_status_style(receipt.status)}]{receipt.status}[/]",
            f"{receipt.accepted_total:,}",
            f"{receipt.recovered_total:,}",
            f"{receipt.processed_chunks:,}",
            f"{receipt.quarantine_total:,}/{receipt.quarantine_failed_total:,}",
            receipt.path,
        )
    if not state.recent_receipts:
        receipts.add_row("UNKNOWN", "0", "0", "0", "0/0", "No full reingest summaries discovered")
    console.print(receipts)

    q_total, q_failed, failures = _aggregate_quarantine(state.recent_receipts or (() if primary is None else (primary,)))
    quarantine = Table(title="Quarantine + Failure Field Notes", box=box.SIMPLE)
    quarantine.add_column("Lane")
    quarantine.add_column("Rows", justify="right")
    quarantine.add_column("Accepted", justify="right")
    quarantine.add_column("Quarantine", justify="right")
    quarantine.add_column("Q Failed", justify="right")
    if primary and primary.etl_breakdown:
        for lane, row in sorted(primary.etl_breakdown.items()):
            quarantine.add_row(
                str(lane),
                f"{_safe_int(row.get('rows')):,}",
                f"{_safe_int(row.get('accepted')):,}",
                f"{_safe_int(row.get('quarantine')):,}",
                f"{_safe_int(row.get('quarantine_failed')):,}",
            )
    else:
        quarantine.add_row("n/a", "0", "0", f"{q_total:,}", f"{q_failed:,}")
    quarantine.caption = f"Aggregate recent quarantine={q_total:,}, quarantine_failed={q_failed:,}, failure_count={failures:,}"
    console.print(quarantine)

    actions = Table(title="Operator Actions — Inert Suggestions Only", box=box.SIMPLE)
    actions.add_column("#", justify="right")
    actions.add_column("Copy/paste if desired; dashboard will not run it")
    for idx, action in enumerate(state.inert_actions, 1):
        actions.add_row(str(idx), action)
    console.print(actions)

    if state.promotion_hint_paths:
        hints = "\n".join(f"• {path}" for path in state.promotion_hint_paths[:8])
        console.print(Panel(hints, title="Promotion / graph receipt hints discovered", border_style="magenta"))
    if state.warnings:
        console.print(Panel("\n".join(state.warnings), title="Warnings", border_style="yellow"))


def render_plain(state: DashboardState) -> None:
    primary = state.primary_receipt
    posture, note = _graph_posture(primary)
    print("LUCIDOTA Canonical Graph / Bitloops Dashboard")
    print("read-only terminal surface; commands are inert suggestions")
    print(f"mode={state.frontend_mode} rich={state.rich_available} textual={state.textual_available}")
    print(f"graph_posture={posture} — {note}")
    if primary:
        print(f"primary={primary.path} status={primary.status} accepted={primary.accepted_total} recovered={primary.recovered_total}")
        print(f"writes graph={primary.graph_writes_performed} canonical={primary.canonical_graph_writes_performed} db={primary.db_writes_performed}")
        print(f"quarantine={primary.quarantine_total} quarantine_failed={primary.quarantine_failed_total} failures={primary.failure_count}")
    print("recent_receipts:")
    for receipt in state.recent_receipts:
        print(f"- {receipt.status} accepted={receipt.accepted_total} q={receipt.quarantine_total}/{receipt.quarantine_failed_total} {receipt.path}")
    print("inert_operator_actions:")
    for action in state.inert_actions:
        print(f"- {action}")
    for warning in state.warnings:
        print(f"WARNING: {warning}")


def run_textual(state: DashboardState) -> bool:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Container, Horizontal, Vertical
        from textual.widgets import DataTable, Footer, Header, Static
    except ImportError:
        return False

    primary = state.primary_receipt
    posture, note = _graph_posture(primary)

    class LucidotaDashboard(App):  # type: ignore[misc]
        CSS = """
        Screen { background: #0b1020; color: #d8e2ff; }
        Header, Footer { background: #102040; }
        .hero { border: round #38bdf8; padding: 1 2; margin: 1; }
        .panel { border: round #64748b; padding: 1; margin: 1; }
        .warn { border: round #facc15; }
        DataTable { height: auto; margin: 1; }
        """
        TITLE = "LUCIDOTA Read-Only Graph Dashboard"
        BINDINGS = [("q", "quit", "Quit")]

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Container(
                Static(f"Canonical graph posture: {posture}\n{note}\nNo backend writes. Suggestions only.", classes="hero"),
                Horizontal(
                    Vertical(Static("Graph / Promotion Status", classes="panel"), DataTable(id="graph_table")),
                    Vertical(Static("Quarantine / Failures", classes="panel warn"), DataTable(id="quarantine_table")),
                ),
                Vertical(Static("Recent Bitloops Full Reingest Receipts", classes="panel"), DataTable(id="receipt_table")),
                Vertical(Static("Operator Actions — inert command suggestions", classes="panel"), DataTable(id="action_table")),
            )
            yield Footer()

        def on_mount(self) -> None:
            graph = self.query_one("#graph_table", DataTable)
            graph.add_columns("Signal", "Value", "Meaning")
            graph.add_row("Status", primary.status if primary else "UNKNOWN", "Receipt status")
            graph.add_row("Graph writes", str(bool(primary and primary.graph_writes_performed)), "Must stay False here")
            graph.add_row("Canonical writes", str(bool(primary and primary.canonical_graph_writes_performed)), "Promotion happens elsewhere")
            graph.add_row("DB writes", str(bool(primary and primary.db_writes_performed)), "Dashboard is read-only")
            graph.add_row("Mutation candidates", f"{primary.graph_mutation_candidates_total:,}" if primary else "n/a", "Queue pressure only")
            graph.add_row("Blockers / violations", f"{(primary.graph_blocker_count if primary else 0):,} / {(primary.write_violation_count if primary else 0):,}", "Review if non-zero")

            quarantine = self.query_one("#quarantine_table", DataTable)
            quarantine.add_columns("Lane", "Rows", "Accepted", "Q", "Q Failed")
            if primary and primary.etl_breakdown:
                for lane, row in sorted(primary.etl_breakdown.items()):
                    quarantine.add_row(str(lane), f"{_safe_int(row.get('rows')):,}", f"{_safe_int(row.get('accepted')):,}", f"{_safe_int(row.get('quarantine')):,}", f"{_safe_int(row.get('quarantine_failed')):,}")
            else:
                quarantine.add_row("n/a", "0", "0", "0", "0")

            receipts = self.query_one("#receipt_table", DataTable)
            receipts.add_columns("Status", "Accepted", "Recovered", "Chunks", "Q/Q Failed", "Path")
            for receipt in state.recent_receipts:
                receipts.add_row(receipt.status, f"{receipt.accepted_total:,}", f"{receipt.recovered_total:,}", f"{receipt.processed_chunks:,}", f"{receipt.quarantine_total:,}/{receipt.quarantine_failed_total:,}", receipt.path)
            if not state.recent_receipts:
                receipts.add_row("UNKNOWN", "0", "0", "0", "0/0", "No receipts found")

            actions = self.query_one("#action_table", DataTable)
            actions.add_columns("#", "Suggestion")
            for idx, action in enumerate(state.inert_actions, 1):
                actions.add_row(str(idx), action)

    LucidotaDashboard().run()
    return True


def self_check(receipt_path: Path = DEFAULT_SUMMARY) -> int:
    state = collect_dashboard_state(receipt_path=receipt_path, force_mode="self-check")
    checks = {
        "repo_root_exists": Path(state.repo_root).exists(),
        "target_summary_present": receipt_path.exists(),
        "target_summary_readable_if_present": (state.primary_receipt is not None) if receipt_path.exists() else True,
        "render_backend_available": state.rich_available or not state.textual_available,
        "textual_absence_uses_fallback": state.textual_available or state.frontend_mode in {"self-check", "rich", "plain"},
        "fallback_available": True,
        "read_only_contract": True,
        "inert_actions_present": bool(state.inert_actions),
    }
    if state.primary_receipt:
        checks.update(
            {
                "summary_status_pass": state.primary_receipt.status.upper() == "PASS",
                "summary_reports_no_graph_writes": not state.primary_receipt.graph_writes_performed,
                "summary_reports_no_canonical_writes": not state.primary_receipt.canonical_graph_writes_performed,
                "summary_reports_no_db_writes": not state.primary_receipt.db_writes_performed,
            }
        )
    ok = all(checks.values())
    report = {
        "ok": ok,
        "mode_resolution": "textual" if state.textual_available and state.rich_available else "rich" if state.rich_available else "plain",
        "imports": {"rich_available": state.rich_available, "textual_available": state.textual_available},
        "receipt_checked": str(receipt_path.relative_to(REPO_ROOT) if receipt_path.is_relative_to(REPO_ROOT) else receipt_path),
        "checks": checks,
        "warnings": state.warnings,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only LUCIDOTA terminal dashboard payload")
    parser.add_argument("--self-check", action="store_true", help="validate optional imports/fallback and read the target summary if present")
    parser.add_argument("--rich", action="store_true", help="force Rich/plain rendering instead of Textual")
    parser.add_argument("--textual", action="store_true", help="prefer Textual when installed")
    parser.add_argument("--json", action="store_true", help="emit read-only dashboard state as JSON")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_SUMMARY, help="primary full_existing_index_reingest_summary.json path")
    args = parser.parse_args(argv)

    receipt_path = args.receipt if args.receipt.is_absolute() else (Path.cwd() / args.receipt).resolve()
    if args.self_check:
        return self_check(receipt_path)

    force_mode = "rich" if args.rich else None
    state = collect_dashboard_state(receipt_path=receipt_path, force_mode=force_mode)
    if args.json:
        def default(value: Any) -> Any:
            if hasattr(value, "__dict__"):
                return value.__dict__
            return str(value)
        print(json.dumps(state, default=default, indent=2, sort_keys=True))
        return 0

    if args.textual or (state.textual_available and state.rich_available and not args.rich):
        if run_textual(state):
            return 0
    render_rich(state)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
