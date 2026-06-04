#!/usr/bin/env python3
"""INDY_READs readiness manifest checker.

Reads configuration-only artifacts and reports whether startup/comms/readiness
preconditions are set. No model calls and no outgoing messages are performed.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVICE_MANIFEST = ROOT / "04_RUNTIME" / "INDY_READS" / "indy_reads_service_manifest.json"
DEFAULT_STARTUP_MANIFEST = ROOT / "04_RUNTIME" / "indy_reads_startup_comms_manifest.json"
DEFAULT_BOOK_INVENTORY = ROOT / "05_OUTPUTS" / "runtime" / "indy_reads_book_inventory_status_latest.json"
DEFAULT_RECEIPT = ROOT / "05_OUTPUTS" / "runtime" / "indy_readiness_status_latest.json"

SCHEMA = "lucidota.indy_reads.readiness_status.v1"


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} is not a JSON object")
    return data


def _first_positive_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int) and value > 0:
            return value
    return None


def check_auto_start(
    manifest: dict[str, Any], startup_manifest: dict[str, Any], startup_manifest_path: Path
) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    notes: list[str] = []

    boot = manifest.get("boot", {}) if isinstance(manifest, dict) else {}
    startup = startup_manifest.get("startup", {}) if isinstance(startup_manifest, dict) else {}

    service_ref = boot.get("service_file") or startup.get("service_file")
    startup_ref = boot.get("startup_manifest")

    if not service_ref:
        blockers.append("auto-start service_file is missing")
        notes.append("using startup manifest startup.service_file would enable auto-start checks")

    if not startup_ref:
        blockers.append("boot.startup_manifest is missing")
        notes.append(f"using CLI startup manifest argument for check: {rel(startup_manifest_path)}")

    service_path = Path(service_ref) if service_ref else None
    startup_path = Path(startup_ref) if startup_ref else None

    service_exists = bool(service_path and service_path.exists())
    startup_exists = bool(startup_path and startup_path.exists())

    if service_path is not None and service_path.exists() is False:
        blockers.append(f"startup service file does not exist: {service_ref}")

    if startup_path is not None and startup_path.exists() is False:
        blockers.append(f"startup manifest file does not exist: {startup_ref}")

    service_contents_ok = False
    if service_path and service_path.exists():
        text = service_path.read_text(encoding="utf-8", errors="ignore")
        service_contents_ok = "[Service]" in text and "Type=" in text
        if "ExecStart" not in text:
            blockers.append(f"service file missing ExecStart: {service_ref}")

    if (
        startup_path is not None
        and startup_path.exists()
        and startup_path.resolve() != startup_manifest_path.resolve()
    ):
        notes.append("startup manifest path was overridden")

    check = {
        "status": "PASS" if not blockers else "FAIL",
        "boot_target": boot.get("boot_target"),
        "startup_boot_target": startup.get("boot_target"),
        "service_file": rel(service_ref),
        "service_file_exists": service_exists,
        "startup_manifest": rel(startup_ref),
        "startup_manifest_exists": startup_exists,
        "service_contents_has_service_and_type": service_contents_ok,
        "service_origin": "manifest_boot_service_file" if boot.get("service_file") else "manifest_startup_service_file",
        "requires_boot_file": rel(boot.get("service_file"))
        if boot.get("service_file")
        else "",
        "next_action_if_missing": "Create/update INDY_READs startup service entry and manifest paths in 04_RUNTIME/INDY_READS/indy_reads_service_manifest.json",
    }
    if service_path is None:
        check["requires_boot_file"] = ""
    return check, blockers, notes


def check_book_count(manifest: dict[str, Any], inventory_path: Path | None) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    notes: list[str] = []

    inventory = manifest.get("inventory", {}) if isinstance(manifest, dict) else {}
    staged = manifest.get("book_training_plans", {}) if isinstance(manifest, dict) else {}

    service_book_count = inventory.get("book_count")
    actual_file_count = inventory.get("actual_book_file_count")
    staged_book_count = inventory.get("staged_book_count")
    training_plan_book_count = staged.get("book_count")

    known_count = _first_positive_int(
        service_book_count,
        staged_book_count,
        actual_file_count,
        training_plan_book_count,
    )

    if known_count is None:
        blockers.append("service manifest does not provide a positive inventory.book_count")

    status = "PASS" if not blockers else "FAIL"

    if inventory_path and inventory_path.exists():
        try:
            inventory_report = load_json(inventory_path)
            staged_payload = inventory_report.get("staged_book_lora", {})
            actual_from_inventory = staged_payload.get("actual_book_file_count")
            staged_status_count = staged_payload.get("book_count")
            if staged_status_count is not None and actual_from_inventory is not None:
                notes.append(
                    "intentional staging mismatch observed: "
                    f"book_count={staged_status_count}, actual_book_file_count={actual_from_inventory}"
                    if staged_status_count != actual_from_inventory
                    else "book staging counts are internally consistent"
                )
                if service_book_count is not None and staged_status_count is not None and service_book_count != staged_status_count:
                    notes.append(
                        f"service manifest book_count={service_book_count} differs from inventory book_count={staged_status_count}"
                    )
                if actual_file_count is not None and actual_file_count != actual_from_inventory:
                    notes.append(
                        f"service manifest actual_book_file_count={actual_file_count} differs from inventory actual_book_file_count={actual_from_inventory}"
                    )
        except Exception as exc:
            notes.append(f"could not read book inventory status at {rel(inventory_path)}: {type(exc).__name__}: {exc}")

    check = {
        "status": status,
        "book_count_known": known_count is not None,
        "book_count": service_book_count or staged_book_count or training_plan_book_count,
        "actual_book_file_count": actual_file_count,
        "training_plan_book_count": training_plan_book_count,
        "staged_targets": staged.get("targets"),
        "inventory_report_path": rel(inventory_path) if inventory_path else "",
    }
    return check, blockers, notes


def check_comms(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    notes: list[str] = []

    email = manifest.get("email", {}) if isinstance(manifest, dict) else {}
    signal = manifest.get("signal", {}) if isinstance(manifest, dict) else {}
    direct_chat = manifest.get("direct_chat", {}) if isinstance(manifest, dict) else {}

    email_recipients = email.get("recipients", [])
    email_mode = email.get("mode")
    signal_mode = signal.get("mode")
    direct_preferred = direct_chat.get("preferred")

    email_ok = isinstance(email_recipients, list) and len(email_recipients) > 0 and isinstance(email_mode, str)
    signal_ok = isinstance(signal_mode, str) and signal_mode != ""
    direct_ok = isinstance(direct_preferred, str) and direct_preferred != ""

    if not email_ok:
        blockers.append("email config missing recipients or mode")
    if not signal_ok:
        blockers.append("signal config missing mode")
    if not direct_ok:
        blockers.append("direct chat config missing preferred transport")

    check = {
        "status": "PASS" if not blockers else "FAIL",
        "email_mode": email_mode,
        "email_recipients": email_recipients,
        "signal_mode": signal_mode,
        "direct_chat_preferred": direct_preferred,
        "direct_chat_socket_path": direct_chat.get("socket_path", ""),
        "direct_chat_loopback_port": direct_chat.get("loopback_port", ""),
    }
    return check, blockers, notes


def check_response_helper(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    notes: list[str] = []

    app = manifest.get("luci_app", {}) if isinstance(manifest, dict) else {}
    enabled = bool(app.get("indy_response_helper"))
    if not enabled:
        blockers.append("luci_app.indy_response_helper is false or missing")

    check = {
        "status": "PASS" if enabled else "FAIL",
        "indy_response_helper": enabled,
        "response_policy": app.get("response_policy", ""),
        "speed_slo_ms": app.get("speed_slo_ms", {}),
    }
    return check, blockers, notes


def build_readiness(*, service_manifest_path: Path, startup_manifest_path: Path, book_inventory_path: Path | None) -> dict[str, Any]:
    blockers: list[str] = []
    notes: list[str] = []

    service_manifest = load_json(service_manifest_path)
    startup_manifest = load_json(startup_manifest_path)

    auto_start_check, auto_blockers, auto_notes = check_auto_start(service_manifest, startup_manifest, startup_manifest_path)
    blockers.extend(auto_blockers)
    notes.extend(auto_notes)

    book_check, book_blockers, book_notes = check_book_count(service_manifest, book_inventory_path)
    blockers.extend(book_blockers)
    notes.extend(book_notes)

    startup_ref = service_manifest.get("boot", {}).get("startup_manifest")
    if startup_ref and Path(startup_ref).resolve() != startup_manifest_path.resolve():
        notes.append(
            "service manifest startup_manifest differs from CLI manifest argument;"
            f" using {rel(startup_manifest_path)} for this check"
        )

    comms_check, comms_blockers, comms_notes = check_comms(startup_manifest)
    blockers.extend(comms_blockers)
    notes.extend(comms_notes)

    helper_check, helper_blockers, helper_notes = check_response_helper(startup_manifest)
    blockers.extend(helper_blockers)
    notes.extend(helper_notes)

    checks = {
        "auto_start": auto_start_check,
        "book_count": book_check,
        "comms": comms_check,
        "response_helper": helper_check,
    }

    status = "PASS" if not blockers else "BLOCKED"
    next_actions = [
        b
        for b in blockers
        if b and not b.startswith("service file")
    ]
    if "intentional staging mismatch observed" in " ".join(notes):
        next_actions.append("Reconcile staged book_count vs actual_book_file_count with a fresh staging receipt.")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": now_z(),
        "status": status,
        "service_manifest": rel(service_manifest_path),
        "startup_manifest": rel(startup_manifest_path),
        "book_inventory_manifest": rel(book_inventory_path) if book_inventory_path else "",
        "checks": checks,
        "blockers": blockers,
        "notes": notes,
        "next_actions": next_actions,
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
        "db_writes_performed": False,
    }

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate INDY_READs readiness manifest/controls status receipt.")
    parser.add_argument("--service-manifest", default=str(DEFAULT_SERVICE_MANIFEST))
    parser.add_argument("--startup-manifest", default=str(DEFAULT_STARTUP_MANIFEST))
    parser.add_argument("--book-inventory", default=str(DEFAULT_BOOK_INVENTORY), help="Optional book-lora inventory receipt")
    parser.add_argument("--receipt", default=str(DEFAULT_RECEIPT), help="Output receipt path")
    parser.add_argument("--json", action="store_true", help="Print receipt JSON only")
    parser.add_argument("--include-missing-book-inventory", action="store_true", help="Require book inventory manifest existence")
    args = parser.parse_args()

    service_manifest = Path(args.service_manifest)
    startup_manifest = Path(args.startup_manifest)
    if not service_manifest.exists():
        raise SystemExit(f"service manifest not found: {service_manifest}")
    if not startup_manifest.exists():
        raise SystemExit(f"startup manifest not found: {startup_manifest}")

    inv_path = Path(args.book_inventory)
    if args.include_missing_book_inventory:
        if not inv_path.exists():
            raise SystemExit(f"book inventory manifest not found: {inv_path}")
    elif not inv_path.exists():
        inv_path = None

    payload = build_readiness(
        service_manifest_path=service_manifest,
        startup_manifest_path=startup_manifest,
        book_inventory_path=inv_path,
    )

    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["receipt_path"] = rel(args.receipt)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    print("REPORT_PATH=" + rel(args.receipt))
    print("INDY_READS_READINESS=" + payload["status"])
    return 0 if payload["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
