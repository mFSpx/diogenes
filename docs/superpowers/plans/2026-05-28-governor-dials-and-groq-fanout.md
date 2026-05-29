# Governor Dials and Groq Fanout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat worker cap with a live ECU-style governor that reads editable dials, separates local vs cloud telemetry, and fans out Groq work orders in bounded parallel batches.

**Architecture:** Keep the local governor focused on laptop safety and DB pressure, but move cloud-worker scaling to an AIMD loop driven by Groq/API telemetry. Store dials in one live JSON file, let the governor re-read them each loop, and expose a tiny CLI that updates the same file. Add a small work-order compiler that slices the existing gap/workflow receipt into Groq-safe batches and emits receipts for each launch.

**Tech Stack:** Python 3.12, JSON, pathlib, subprocess, asyncio/httpx, psycopg, existing Groq bridge scripts, existing GOALS / 05_OUTPUTS receipts.

---

### Task 1: Add live governor dials and dual telemetry

**Files:**
- Modify: `scripts/resource_governor.py`
- Create: `05_OUTPUTS/runtime/governor_dials.json`
- Test: `tests/test_resource_governor_dials.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

def test_default_dials_file_is_created(tmp_path, monkeypatch):
    from scripts import resource_governor as rg
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(rg, "ROOT", root)
    dials = rg.load_dials(root)
    assert dials["GLOBAL_MODE"] == "BALANCED"
    assert dials["MAX_CLOUD_WORKERS"] >= 10
    assert dials["MAX_DB_CONNECTIONS"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resource_governor_dials.py -v`
Expected: FAIL because `load_dials` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
DEFAULT_DIALS = {
    "GLOBAL_MODE": "BALANCED",
    "MAX_CLOUD_WORKERS": 24,
    "MAX_DB_CONNECTIONS": 12,
    "TARGET_API_LATENCY_MS": 1200,
    "MAX_LOCAL_WORKERS": 4,
    "KILL_SWITCH": False,
}

def load_dials(root=ROOT):
    path = root / "05_OUTPUTS" / "runtime" / "governor_dials.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        save_dials(dict(DEFAULT_DIALS), root=root)
        return dict(DEFAULT_DIALS)
    data = json.loads(path.read_text(encoding="utf-8"))
    merged = dict(DEFAULT_DIALS)
    merged.update(data)
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resource_governor_dials.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/resource_governor.py tests/test_resource_governor_dials.py 05_OUTPUTS/runtime/governor_dials.json
git commit -m "feat: live governor dials and dual telemetry"
```

### Task 2: Add governor tune CLI for manual overrides

**Files:**
- Modify: `scripts/resource_governor.py`
- Test: `tests/test_resource_governor_tune_cli.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_tune_cli_updates_dials(tmp_path):
    proc = subprocess.run(
        [sys.executable, "scripts/resource_governor.py", "--root", str(tmp_path), "tune", "--mode", "AGGRESSIVE", "--max-cloud-workers", "80"],
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0
    dials = json.loads((tmp_path / "05_OUTPUTS/runtime/governor_dials.json").read_text())
    assert dials["GLOBAL_MODE"] == "AGGRESSIVE"
    assert dials["MAX_CLOUD_WORKERS"] == 80
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resource_governor_tune_cli.py -v`
Expected: FAIL because `tune` subcommand does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def cmd_tune(args):
    dials = load_dials()
    if args.mode:
        dials["GLOBAL_MODE"] = args.mode
    if args.max_cloud_workers is not None:
        dials["MAX_CLOUD_WORKERS"] = int(args.max_cloud_workers)
    if args.max_db_connections is not None:
        dials["MAX_DB_CONNECTIONS"] = int(args.max_db_connections)
    if args.target_api_latency_ms is not None:
        dials["TARGET_API_LATENCY_MS"] = int(args.target_api_latency_ms)
    if args.max_local_workers is not None:
        dials["MAX_LOCAL_WORKERS"] = int(args.max_local_workers)
    if args.kill_switch is not None:
        dials["KILL_SWITCH"] = bool(args.kill_switch)
    save_dials(dials)
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resource_governor_tune_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/resource_governor.py tests/test_resource_governor_tune_cli.py
git commit -m "feat: add governor tune CLI"
```

### Task 3: Split cloud and local telemetry paths in the governor

**Files:**
- Modify: `scripts/resource_governor.py`
- Test: `tests/test_resource_governor_telemetry_split.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cloud_telemetry_ignores_local_cpu(monkeypatch):
    from scripts import resource_governor as rg
    telemetry = {
        "cpu": {"loadavg_1m": 99.0},
        "memory": {"available_mb": 128.0},
        "postgres": {"available": True, "connection_count": 2},
        "cloud": {"http_429_rate": 0.0, "latency_ms": 200},
    }
    dials = {"GLOBAL_MODE": "AGGRESSIVE", "MAX_CLOUD_WORKERS": 50, "TARGET_API_LATENCY_MS": 1200}
    decision = rg.decide_cloud_workers(telemetry, dials)
    assert decision["safe_workers"] > 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resource_governor_telemetry_split.py -v`
Expected: FAIL because `decide_cloud_workers` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def decide_cloud_workers(telemetry, dials, requested_workers=None):
    mode = _dial_mode(dials)
    max_workers = max(1, _dial_int(dials, "MAX_CLOUD_WORKERS", 24))
    target_latency = float(dials.get("TARGET_API_LATENCY_MS", 1200))
    cloud = telemetry.get("cloud", {})
    base = requested_workers if requested_workers is not None else (10 if mode == "AGGRESSIVE" else 6 if mode == "BALANCED" else 3)
    safe = max(1, min(int(base), max_workers))
    if cloud.get("http_429_rate", 0) > 0 or float(cloud.get("latency_ms", 0) or 0) > target_latency:
        safe = max(1, safe // 2)
    else:
        safe = min(max_workers, safe + (5 if mode == "AGGRESSIVE" else 2 if mode == "BALANCED" else 1))
    return {"requested_workers": int(base), "safe_workers": safe, "throttle": False}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resource_governor_telemetry_split.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/resource_governor.py tests/test_resource_governor_telemetry_split.py
git commit -m "feat: split cloud and local governor telemetry"
```

### Task 4: Compile remaining work into Groq-safe batches

**Files:**
- Create: `scripts/groq_workorder_compiler.py`
- Modify: `scripts/lucidota_gap_workflow_compiler.py` if needed for a stable JSON source
- Test: `tests/test_groq_workorder_compiler.py`

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

def test_compiler_groups_workflows_into_batches(tmp_path):
    from scripts import groq_workorder_compiler as gwc
    src = tmp_path / "workflows.json"
    src.write_text(json.dumps({"workflows": [{"workflow_id": "a"}, {"workflow_id": "b"}, {"workflow_id": "c"}]}))
    report = gwc.compile_workorders(src, batch_size=2)
    assert len(report["batches"]) == 2
    assert report["batches"][0][0]["workflow_id"] == "a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_groq_workorder_compiler.py -v`
Expected: FAIL because `groq_workorder_compiler.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def compile_workorders(path, batch_size=8):
    data = json.loads(Path(path).read_text())
    workflows = list(data.get("workflows") or [])
    batches = [workflows[i:i+batch_size] for i in range(0, len(workflows), batch_size)]
    return {"workflows": workflows, "batches": batches, "batch_size": batch_size}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_groq_workorder_compiler.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/groq_workorder_compiler.py tests/test_groq_workorder_compiler.py scripts/lucidota_gap_workflow_compiler.py
git commit -m "feat: compile remaining work into Groq batches"
```

### Task 5: Add Groq batch launcher with receipts and AIMD scaling hooks

**Files:**
- Create: `scripts/groq_batch_launcher.py`
- Test: `tests/test_groq_batch_launcher.py`

- [ ] **Step 1: Write the failing test**

```python
import asyncio

def test_launcher_splits_batches_and_emits_receipts(monkeypatch):
    from scripts import groq_batch_launcher as gbl
    batches = [[{"workflow_id": "a"}], [{"workflow_id": "b"}]]
    report = gbl.plan_launches(batches, max_cloud_workers=2)
    assert report["launch_count"] == 2
    assert report["selected_workers"] == 2

    async def fake_launch(batch, **kwargs):
        return {"returncode": 0, "batch_size": len(batch), "report_path": "fake.json", "blockers": []}

    monkeypatch.setattr(gbl, "_launch_one_batch", fake_launch)
    out = asyncio.run(gbl.launch_batches(batches, execute=False, max_cloud_workers=2))
    assert out["selected_workers"] == 2
    assert len(out["launches"]) == 2
    assert all(item["returncode"] == 0 for item in out["launches"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_groq_batch_launcher.py -v`
Expected: FAIL because launcher module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
async def launch_batches(batches, *, execute=False, model="llama-3.1-8b-instant", max_tokens=256, timeout_sec=120.0, max_cloud_workers=24):
    plan = plan_launches(batches, max_cloud_workers=max_cloud_workers)
    semaphore = asyncio.Semaphore(plan["selected_workers"])
    async def run(batch):
        async with semaphore:
            return await _launch_one_batch(batch, execute=execute, model=model, max_tokens=max_tokens, timeout_sec=timeout_sec)
    launches = await asyncio.gather(*(run(batch) for batch in batches))
    return {"launch_count": len(batches), "selected_workers": plan["selected_workers"], "execute_performed": bool(execute), "launches": launches}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_groq_batch_launcher.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/groq_batch_launcher.py tests/test_groq_batch_launcher.py
git commit -m "feat: add Groq batch launcher"
```

### Task 6: Wire receipts and handoff updates

**Files:**
- Modify: `GOALS/CURRENT_HANDOFF.md`
- Modify: `GOALS/GOAL_LOG.md`
- Test: `tests/test_goal_handoff.py` or existing handoff tests if they cover step entries

- [ ] **Step 1: Write the failing test**

```python
def test_handoff_mentions_governor_dials_and_groq_batches():
    text = Path("GOALS/CURRENT_HANDOFF.md").read_text()
    assert "governor dials" in text.lower()
    assert "groq" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_goal_handoff.py -v`
Expected: FAIL until the new handoff language is added.

- [ ] **Step 3: Update handoff text**

```markdown
- Next action: run the live governor with editable dials, then compile the remaining work into Groq-safe batches and launch them with receipts.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_goal_handoff.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add GOALS/CURRENT_HANDOFF.md GOALS/GOAL_LOG.md tests/test_goal_handoff.py
git commit -m "docs: record governor dials and Groq batch fanout"
```
