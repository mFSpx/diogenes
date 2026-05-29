#!/usr/bin/env python3
"""Isolated document parser bakeoff for the LUCIDOTA tech bench.

Bench-only: no production mutation, no parser output treated as truth.
Checks MinerU, Docling, and SmolDocling installability/import state and, when
--execute is used, runs only safe local/import or tiny text-derived smoke paths.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "05_OUTPUTS" / "tech_bench"
PARSERS = [
    {
        "name": "MinerU",
        "modules": ["magic_pdf", "mineru"],
        "pip_hint": "pip install mineru or magic-pdf in an isolated bench environment",
        "role": "PDF/OCR/layout extraction candidate; preserve page/block/span provenance.",
    },
    {
        "name": "Docling",
        "modules": ["docling"],
        "pip_hint": "pip install docling in an isolated bench environment",
        "role": "PDF/DOCX/PPTX/XLSX-to-structured Markdown/JSON candidate.",
    },
    {
        "name": "SmolDocling",
        "modules": ["smoldocling", "docling"],
        "pip_hint": "install SmolDocling-compatible Docling extras/model only in isolated bench; no model download in this script",
        "role": "lightweight VLM-style structured document parsing candidate.",
    },
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def module_status(names: list[str]) -> dict[str, Any]:
    statuses = {}
    present = False
    for name in names:
        found = importlib.util.find_spec(name) is not None
        statuses[name] = "present" if found else "missing"
        present = present or found
    return {"present": present, "modules": statuses}


def sample_contract(path: Path | None, parser_name: str) -> dict[str, Any]:
    data = b""
    if path and path.exists() and path.is_file():
        data = path.read_bytes()[:4096]
    else:
        data = b"LUCIDOTA parser bakeoff sample. Operator, KORPUS, Chrono-Ledger.\n"
    return {
        "parser": parser_name,
        "truth_status": "parser_output_not_truth_bench_only",
        "source": str(path) if path else "synthetic_inline_sample",
        "source_sha256_prefix": sha256_bytes(data)[:16],
        "mime_guess": mimetypes.guess_type(str(path))[0] if path else "text/plain",
        "provenance_shape": {
            "pages": [],
            "blocks": [],
            "spans": [
                {"start_char": 0, "end_char": min(len(data), 32), "text_preview_sha256": sha256_bytes(data[:32])}
            ],
            "tables": [],
            "ocr": [],
        },
        "output_contract": "structured Markdown/JSON with page/block/span/layout/table/OCR provenance when parser is available",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--input")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    mode = "execute" if args.execute else "dry_run"
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_ROOT / stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input) if args.input else None
    entries = []
    for parser in PARSERS:
        st = module_status(parser["modules"])
        if parser["name"] == "SmolDocling":
            # Docling may be installed while the SmolDocling-specific module/model path is not.
            # Keep the distinction explicit: SmolDocling compatibility is not proven by base Docling alone.
            smol_present = st["modules"].get("smoldocling") == "present"
            install_status = "present" if smol_present else "missing_dependency"
            smoke = "not_run_dry_run" if not args.execute else ("import_present_contract_only" if smol_present else "skipped_missing_smoldocling_module_or_model_path")
        else:
            install_status = "present" if st["present"] else "missing_dependency"
            smoke = "not_run_dry_run" if not args.execute else ("import_present_contract_only" if st["present"] else "skipped_missing_dependency")
        entries.append({
            "name": parser["name"],
            "role": parser["role"],
            "install_status": install_status,
            "module_status": st,
            "pip_hint": parser["pip_hint"],
            "smoke_test_result": smoke,
            "sample_contract": sample_contract(input_path, parser["name"]),
        })
    report = {
        "schema": "lucidota.tech_bench.document_parse_bakeoff.v1",
        "generated_at": now_iso(),
        "mode": mode,
        "input": str(input_path) if input_path else None,
        "parser_output_truth_status": "not_truth_bench_only",
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "production_schema_mutation": False,
        "entries": entries,
        "blockers": [f"{e['name']}_missing_dependency" for e in entries if e["install_status"] == "missing_dependency"],
    }
    out = out_dir / "document_parse_bakeoff_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    print(f"REPORT_PATH={out.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
