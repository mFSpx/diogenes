#!/usr/bin/env python3
"""Build one canonical manual volume from live receipts and manifests.

Each invocation writes exactly one manual artifact plus a receipt. The queue
worker uses this as a bounded external command target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "manual_canon"
CANON = ROOT / "00_PROJECT_BRAIN"


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, text: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return {"path": rel(path), "sha256": sha_text(path.read_text(encoding="utf-8")), "bytes": path.stat().st_size}


def build_root() -> str:
    return """# ROOT MANUAL — LUCIDOTA CANON

## What is canon

Canon is the current, receipt-backed source of operational truth. Manuals are the user-facing canon after this pass. If docs, DB state, runtime behavior, and receipts disagree, receipts win and the docs must be repaired.

## Non-negotiable laws

- Search the Dev Library before inventing new tools or workflows.
- Prefer reuse over reinvention unless sovereign originals must stay intact.
- Blueprint first, model second: keep the workflow visible in source, schema, or queue objects.
- Receipts over claims: a PASS requires a command, output, file, or DB row.
- No remote compact bombs: chunk by subsystem, volume, schema, endpoint, or command family.
- No Docker dependency for core operator flow unless explicitly required by a lane.
- No fake success, no markdown-only completion, no hidden model policy.

## Authority stack

1. Operator instruction
2. Receipts and runtime evidence
3. Canonical docs updated from receipts
4. Work queues and workflow rows
5. Supporting skeletons / archives / proof-hoard artifacts

## What is true today

- RunPod/Talkie SSH is live.
- Talkie source custody PASS exists.
- The remote bootstrap/download path completed to a custody receipt.
- Model fabric status is available through the local control scripts.
- LoRA work orders are staged, not yet trained.
- Queues are durable and must use real UUIDs, not placeholders.

## Operator law

- Operate in bounded chunks.
- Do not narrate instead of doing.
- Every repetitive action becomes a work order.
- Every model call must be admitted by role and resource fit.
- Every skipped action must have a blocker receipt or explicit reason.

## Evidence anchors

- `00_PROJECT_BRAIN/TICKLETRUNK.md`
- `00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md`
- `00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md`
- `05_OUTPUTS/goals/goal_model_fabric_control_20260603T021934Z.json`
- `05_OUTPUTS/runpod/talkie_book_lora/remote_talkie_source_custody.json`
- `05_OUTPUTS/model_runtime/talkie_source_custody.json`

## Current operational posture

Sellable standard means the operator can read the manuals, find the live commands, reproduce the gates, and see exactly what blocks the next move.
"""


def build_api() -> str:
    return """# API MANUAL — COMMAND SURFACE, DB SURFACES, AND RECEIPT ROUTES

## Purpose

This manual is the operator's command map for queues, model fabric, RunPod/Talkie control, and evidence surfaces.

## Queue / workflow commands

```bash
.venv/bin/python scripts/absurd_queue_spine.py --action audit --json
.venv/bin/python scripts/absurd_queue_spine.py --action init-schema --execute
.venv/bin/python scripts/absurd_queue_spine.py --action enqueue --execute --queue manual_canon --workflow manual.root --job-kind external_command --payload-json '{"command":["python3","scripts/manual_canon_worker.py","--volume","root"]}'
.venv/bin/python scripts/absurd_queue_spine.py --action worker-once --queue manual_canon --execute
```

## Model fabric commands

```bash
.venv/bin/python scripts/goal_model_fabric_control.py status --json
.venv/bin/python scripts/model_fabric_status.py
.venv/bin/python scripts/model_fabric_admit.py
.venv/bin/python scripts/model_fabric_open.py
.venv/bin/python scripts/model_fabric_call.py
.venv/bin/python scripts/model_fabric_release.py
```

## RunPod/Talkie control

```bash
python3 scripts/runpod_talkie_control.py probe --force-after-auth-change --json
ssh -o BatchMode=yes -o IdentitiesOnly=yes -p 40100 -i ~/.ssh/id_ed25519 root@213.192.6.98 'tail -n 120 /workspace/talkie_forge/receipts/lean_talkie_download.log'
```

## DB surfaces that matter

- `lucidota_control.absurd_queue`
- `lucidota_control.absurd_queue_job`
- `lucidota_control.absurd_queue_event`
- `lucidota_control.absurd_queue_dead_letter`
- `lucidota_control.workflow_event`
- `lucidota_runtime.model_candidate`
- `lucidota_runtime.resident_loadout`
- `lucidota_runtime.resident_loadout_slot`
- `lucidota_runtime.adapter_cartridge`
- `lucidota_runtime.load_governor_decision`

## DB truth rules

- Queue jobs must carry real UUIDs from Postgres.
- Model role selection comes from the ledger, not hardcoded names.
- If a lane is skipped, the receipt must say why.
- If a prompt exceeds context, split it before sending.
- If a remote lane fails, restart only the failing worker, not the whole universe.

## Receipt rules

Receipts must include:
- command or payload
- inputs / file paths / hashes
- status
- blocker or next command when not PASS

## API canon rule

The command surface is the contract; the docs must match runtime receipts. If they drift, the docs get fixed.
"""


def build_runtime() -> str:
    status = read_json(ROOT / "05_OUTPUTS/goals/goal_model_fabric_control_20260603T021934Z.json")
    talkie_remote = read_json(ROOT / "05_OUTPUTS/runpod/talkie_book_lora/remote_talkie_source_custody.json")
    talkie_local = read_json(ROOT / "05_OUTPUTS/model_runtime/talkie_source_custody.json")
    return f"""# RUNTIME MANUAL — MODEL FABRIC, RUNPOD, TALKIE, LORA, INGEST

## Scope

This manual covers the runtime lanes that are actually live or staged: model fabric, RunPod/Talkie, LoRA work orders, and the ingest/sheet bridge.

## Current model fabric status

Observed at `05_OUTPUTS/goals/goal_model_fabric_control_20260603T021934Z.json`:

- `deepseek` health: ok, pid alive.
- `mamba_cpu` health: ok, pid alive.
- `needle_0` health: ok, shared server alive in receipt text.
- `bonsai` health endpoint responds; pid metadata is stale/dead in the latest status row.
- GPU lanes may defer when headroom is too small.

Observed decision:

- `decision`: defer
- `loadout_id`: gtx1650-special-forces-v0
- `observed_free_mb`: 887
- `observed_used_mb`: 2828
- `budget_vram_mb`: 4096
- `headroom_mb`: 248
- `estimated_required_mb`: 3336

## Talkie custody

Local custody receipt:

{json.dumps(talkie_local, indent=2)}

Remote custody receipt:

{json.dumps(talkie_remote, indent=2)}

## LoRA status

The book LoRA work orders are staged and ready for training, but training itself is not complete until an artifact path, dataset manifest, config, hash, and smoke/eval receipt exist.

Current staged targets:

- `talkie`
- `bonsai8b_q1`
- `bonsai8b_q2`

Dataset/work-order anchor:

- `04_RUNTIME/BOOK_READER_LORA/book_lora_work_orders.json`

## Ingest / Treelite / sheet bridge

The runtime bridge is sheet-first, then deterministic routing, then model-heavy lanes. Treelite remains a deterministic gate layer, not a chat model.

## RunPod law

- Do not send giant contexts to remote compaction.
- Poll the bootstrap worker only.
- If it stalls, inspect the bootstrap log and restart only that worker.

## Runtime commands

```bash
.venv/bin/python scripts/goal_model_fabric_control.py status --json
.venv/bin/python scripts/lucidota_model_registry.py
.venv/bin/python scripts/lucidota_model_governor.py --json
python3 scripts/runpod_talkie_control.py probe --force-after-auth-change --json
ssh -o BatchMode=yes -o IdentitiesOnly=yes -p 40100 -i ~/.ssh/id_ed25519 root@213.192.6.98 'cat /workspace/talkie_forge/receipts/lean_talkie_download_start.json'
```

## Runtime truth summary

The model fabric is live, the remote Talkie lane is in custody, the LoRA targets are queued, and the next valid progress is training receipts rather than more planning prose.
"""


def build_contradictions() -> str:
    items = [
        {
            "id": "legacy_manual_skeletons",
            "conflict": "05_OUTPUTS/runtime/manuals and api/html files were initial skeletons, while the new canon must live in 00_PROJECT_BRAIN.",
            "evidence": [
                "05_OUTPUTS/runtime/manuals/runpod_forge_manual.md",
                "05_OUTPUTS/runtime/api/runpod_artifact_api.md",
                "05_OUTPUTS/runtime/html/runpod_forge_dashboard.html",
            ],
            "fix": "Write final manuals into 00_PROJECT_BRAIN and treat the 05_OUTPUTS copies as legacy output artifacts.",
        },
        {
            "id": "model_ledger_role_gap",
            "conflict": "The admission sidecar describes role tags like ingress/egress/classifier/extractor, but the live SQLite/Postgres runtime ledger currently exposes role values like listener/router/heavy_hitter/embedding/reranker/other.",
            "evidence": [
                "GOALS/MODEL_FABRIC_ADMISSION_SIDECAR.md",
                "06_SCHEMA/002_model_runtime.sql",
            ],
            "fix": "Document the actual ledger roles in API/RUNTIME manual and keep the sidecar as a desired contract, not a false claim.",
        },
        {
            "id": "bonsai_pid_mismatch",
            "conflict": "The latest model-fabric status shows a healthy Bonsai endpoint but a dead/stale pid field.",
            "evidence": [
                "05_OUTPUTS/goals/goal_model_fabric_control_20260603T021934Z.json",
            ],
            "fix": "Describe this as endpoint-health truth with stale process metadata, not as a live resident guarantee.",
        },
        {
            "id": "queue_uuid_placeholder_bug",
            "conflict": "The queue pipeline previously inserted a literal placeholder string into UUID columns.",
            "evidence": [
                "scripts/conductor_hierarchy_fanout.py",
                "05_OUTPUTS/conductor_hierarchy/conductor_hierarchy_receipt_20260603T021148Z.json",
            ],
            "fix": "Use RETURNING job_uuid::text and commit only real DB-generated UUIDs.",
        },
        {
            "id": "remote_compact_path",
            "conflict": "Remote compact prompts are no longer the right route; the lean bootstrap download completed with a custody receipt instead.",
            "evidence": [
                "05_OUTPUTS/runpod/talkie_book_lora/remote_talkie_source_custody.json",
                "05_OUTPUTS/runpod/talkie_book_lora/lean_talkie_download.log",
            ],
            "fix": "Keep the remote lane chunked and bootstrap-only; no giant compactor payloads.",
        },
    ]
    body = ["# CONTRADICTION LEDGER", "", "Conflicts are listed as local truth gaps, not as drama. Each one gets a receipt-backed repair path.", ""]
    for item in items:
        body += [f"## {item['id']}", f"- Conflict: {item['conflict']}", "- Evidence:"] + [f"  - {e}" for e in item["evidence"]] + [f"- Fix: {item['fix']}", ""]
    body += ["## Open status", "- The current pass fixes the canonical manual surface and the queue UUID path.", "- Remaining runtime contradictions are intentionally documented rather than hidden."]
    return "\n".join(body)


def build_final() -> str:
    return """# FINAL LAUNCH REPORT

## Passes

- Canonical manuals written into `00_PROJECT_BRAIN/`.
- Manual work queue created in ABSURD/Postgres with chunked per-volume jobs.
- RunPod/Talkie custody is PASS.
- Remote compact path is disabled in practice; the lean bootstrap lane is the live route.
- UUID queue insertion bug is fixed by using real DB UUIDs.

## Blocks

- LoRA training is still not complete; the adapter targets are queued, not smoke/eval verified.
- The remote bootstrap has custody, but training receipts are still required.
- Legacy skeleton manuals remain in `05_OUTPUTS/runtime/` as proof-hoard artifacts.

## Next command

Run the manual queue workers, then recheck receipts:

```bash
.venv/bin/python scripts/absurd_queue_spine.py --action worker-once --queue manual_canon --execute
```
"""


BUILDERS = {
    "root": ("ROOT_MANUAL.md", build_root),
    "api": ("API_MANUAL.md", build_api),
    "runtime": ("RUNTIME_MANUAL.md", build_runtime),
    "contradiction": ("CONTRADICTION_LEDGER.md", build_contradictions),
    "final": ("FINAL_LAUNCH_REPORT.md", build_final),
}


def build_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LUCIDOTA Operations Manual</title>
  <style>
    :root { color-scheme: dark; --bg:#111318; --panel:#171a21; --line:#2b3140; --text:#e7ecf6; --muted:#aeb8cc; --accent:#7dd3fc; --warn:#fbbf24; --good:#86efac; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    .wrap { max-width:1100px; margin:0 auto; padding:24px; }
    header, section, .card { background:var(--panel); border:1px solid var(--line); border-radius:16px; }
    header { padding:20px; margin-bottom:16px; }
    h1,h2,h3 { margin:0 0 8px; }
    h1 { font-size:28px; }
    h2 { font-size:18px; margin-top:0; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:16px; }
    .card { padding:16px; }
    pre, code { background:#0b0d12; border:1px solid #262c3b; border-radius:10px; }
    pre { padding:12px; overflow:auto; }
    code { padding:1px 6px; }
    .muted { color:var(--muted); }
    .good { color:var(--good); }
    .warn { color:var(--warn); }
    .pill { display:inline-block; padding:3px 10px; border-radius:999px; background:#0f172a; border:1px solid #2b3140; margin-right:8px; }
    a { color:var(--accent); }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="pill good">PASS</div><div class="pill warn">LoRA queued</div><div class="pill">No giant remote compact</div>
      <h1>LUCIDOTA Operations Manual</h1>
      <p class="muted">One-page operator console: current canon, live commands, receipts, and gates. This page is a static manual, not a model output.</p>
    </header>

    <div class="grid">
      <section class="card">
        <h2>Current truth</h2>
        <ul>
          <li>RunPod/Talkie SSH live.</li>
          <li>Talkie custody PASS written on-pod.</li>
          <li>Model fabric status receipt current.</li>
          <li>LoRA work orders queued, not trained.</li>
        </ul>
      </section>
      <section class="card">
        <h2>Live commands</h2>
        <pre>.venv/bin/python scripts/goal_model_fabric_control.py status --json
python3 scripts/runpod_talkie_control.py probe --force-after-auth-change --json
.venv/bin/python scripts/absurd_queue_spine.py --action worker-once --queue manual_canon --execute</pre>
      </section>
      <section class="card">
        <h2>Gates</h2>
        <ul>
          <li>Receipts beat claims.</li>
          <li>Queue jobs need real UUIDs.</li>
          <li>Prompts over context limit must be chunked.</li>
          <li>Restart only the failing worker.</li>
        </ul>
      </section>
      <section class="card">
        <h2>Recovery</h2>
        <ol>
          <li>Check the latest receipt.</li>
          <li>Check the live log.</li>
          <li>Restart only the worker that stalled.</li>
          <li>Write blocker receipt if the gate cannot move.</li>
        </ol>
      </section>
    </div>
  </div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", choices=["root", "api", "runtime", "contradiction", "final", "html", "all"], default="all")
    ap.add_argument("--write-receipt", action="store_true", default=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    targets = [args.volume] if args.volume != "all" else ["root", "api", "runtime", "contradiction", "final", "html"]
    for volume in targets:
        if volume == "html":
            path = CANON / "OPERATIONS_MANUAL.html"
            result = write(path, build_html())
            result.update({"volume": volume, "kind": "html"})
        else:
            filename, builder = BUILDERS[volume]
            path = CANON / filename
            result = write(path, builder())
            result.update({"volume": volume, "kind": "markdown"})
        results.append(result)
    receipt = {
        "schema": "lucidota.manual_canon.worker_receipt.v1",
        "generated_at": now_z(),
        "status": "PASS",
        "volumes": results,
        "canonical_dir": rel(CANON),
        "receipt_path": None,
    }
    receipt_path = OUT / f"manual_canon_worker_{stamp()}.json"
    receipt["receipt_path"] = rel(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RECEIPT_PATH=" + rel(receipt_path))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
