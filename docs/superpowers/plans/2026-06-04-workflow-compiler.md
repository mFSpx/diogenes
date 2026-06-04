# Workflow Compiler and Basic Workflows as Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn repeated operator text and repeated script-shaped work into DB-visible workflow batches/items, while keeping simple/basic workflows first-class as workflows in the DB graph and manual surface.

**Architecture:** Reuse the existing `scripts/ontology_work_compiler.py` compiler and the live workflow registry/manual surfaces instead of inventing a new execution plane. The compiler should start from ontology and schema truth first, then normalize operator text into batch/item rows, classify each item with subsystem + GO/CO/IO-style tags + risk/parallelism + executor recommendation, and persist the batch so `/todo_current` and `/manual_current` can show active work. Existing simple workflows such as `basic-workflows` remain first-class entries in `lucidota_control.workflow_registry`; the compiler should emit workflow-shaped output that preserves those simple paths instead of flattening them into ad hoc script notes, and only invent new terms when the ontology/schema cannot already express the work.

**Tech Stack:** Python 3.11+, psycopg, PostgREST, existing `lucidota_control` / `lucidota_canon` schemas, receipt-gated pytest, live manual/status routes.

---

### Task 1: Lock the workflow compiler shape and keep basic workflows first-class

**Files:**
- Modify: `scripts/ontology_work_compiler.py`
- Modify: `tests/test_ontology_work_compiler.py`

- [ ] **Step 1: Write the failing test**

```python
from scripts import ontology_work_compiler


def test_basic_workflows_stay_workflow_shaped(monkeypatch):
    monkeypatch.setattr(
        ontology_work_compiler.indy_runtime_broker,
        "registry_snapshot",
        lambda **kwargs: {
            "local_model_roles": {
                "router": {"model_id": "needle-26m", "role": "router", "provider_key": "local_model"},
                "classifier": None,
                "summarizer": None,
                "embedder": None,
                "reranker": None,
                "thinker": None,
                "watcher": None,
                "treelite_gate": None,
            }
        },
    )
    payload = ontology_work_compiler.compile_work_batch(
        """
        1. Keep basic workflows as workflows in the DB graph.
        2. Preserve the workflow_registry entry for basic-workflows.
        3. Show the active batch in manual_current and todo_current.
        """.strip()
    )

    assert payload["schema"] == "lucidota.ontology_work_compiler.v1"
    assert payload["batch"]["batch_kind"] == "workflow_batch"
    assert payload["batch"]["workflows_preserved"] is True
    assert any(item["work_kind"] == "workflow" for item in payload["items"])
    assert any(item["workflow_name"] == "basic-workflows" for item in payload["items"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_ontology_work_compiler.py::test_basic_workflows_stay_workflow_shaped -v`
Expected: FAIL because `batch_kind`, `workflows_preserved`, or `workflow_name` are not yet emitted.

- [ ] **Step 3: Write minimal implementation**

```python
# in scripts/ontology_work_compiler.py

def infer_work_kind(title: str, body: str) -> str:
    low = f"{title}\n{body}".lower()
    if any(token in low for token in ("workflow", "workflow_registry", "basic workflow", "basic-workflows")):
        return "workflow"
    ...


def build_item(...):
    item = {
        ...,
        "work_kind": kind,
        "workflow_name": "basic-workflows" if kind == "workflow" else "",
        ...,
    }
    return item


def compile_work_batch(...):
    batch = {
        ...,
        "batch_kind": "workflow_batch" if any(item["work_kind"] == "workflow" for item in items) else "ontology_batch",
        "workflows_preserved": any(item["workflow_name"] == "basic-workflows" for item in items),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_ontology_work_compiler.py::test_basic_workflows_stay_workflow_shaped -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/ontology_work_compiler.py tests/test_ontology_work_compiler.py
git commit -m "feat: keep basic workflows workflow-shaped"
```

---

### Task 2: Persist compiler batches so the manual can show them as live workflow work

**Files:**
- Modify: `06_SCHEMA/153_ontology_work_batch.sql`
- Modify: `scripts/ontology_work_compiler.py`
- Modify: `tests/test_ontology_work_compiler.py`
- Modify: `tests/test_manual_current_surface.py`
- Modify: `tests/test_orchestrator_registry_routes.py`

- [ ] **Step 1: Write the failing test**

```python
from scripts import ontology_work_compiler


def test_compile_and_persist_exposes_todo_and_manual_batch(monkeypatch):
    monkeypatch.setattr(
        ontology_work_compiler.indy_runtime_broker,
        "choose_local_model",
        lambda role, base_url=None: {"model_id": "needle-26m", "role": role, "provider_key": "local_model"} if role == "router" else None,
    )
    result = ontology_work_compiler.compile_and_persist(
        """
        1. Audit the live route surface.
        2. Preserve the workflow_registry entry for basic-workflows.
        3. Serialize schema edits and keep them receipt-backed.
        """.strip(),
        base_url="http://127.0.0.1:3000",
    )
    assert result["batch"]["batch_uuid"]
    assert result["batch"]["workflow_count"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_ontology_work_compiler.py::test_compile_and_persist_exposes_todo_and_manual_batch -v`
Expected: FAIL until the batch row carries workflow-aware fields into `todo_current`.

- [ ] **Step 3: Write minimal implementation**

```python
# in scripts/ontology_work_compiler.py
# ensure the persisted batch row includes workflow_count, batch_kind, workflows_preserved,
# and a compact list of workflow-shaped items.
```

```sql
-- in 06_SCHEMA/153_ontology_work_batch.sql
ALTER TABLE lucidota_control.ontology_work_batch
  ADD COLUMN IF NOT EXISTS workflow_count integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS workflows_preserved boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS batch_kind text NOT NULL DEFAULT 'ontology_batch';
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_ontology_work_compiler.py::test_compile_and_persist_exposes_todo_and_manual_batch -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 06_SCHEMA/153_ontology_work_batch.sql scripts/ontology_work_compiler.py tests/test_ontology_work_compiler.py tests/test_manual_current_surface.py tests/test_orchestrator_registry_routes.py
git commit -m "feat: surface workflow batches in todo/manual"
```

---

### Task 3: Keep the learning loop fed by workflow outcomes instead of ad hoc script noise

**Files:**
- Modify: `scripts/bytewax_abductive_blender.py`
- Modify: `scripts/indy_reads.py`
- Modify: `tests/test_bytewax_compact_windows.py`
- Modify: `tests/test_indy_runtime_broker.py`

- [ ] **Step 1: Write the failing test**

```python
from scripts import bytewax_abductive_blender


def test_workflow_outcome_rows_include_learning_fields(monkeypatch):
    payload = bytewax_abductive_blender.build_compact_window_row(
        work_order_uuid="00000000-0000-0000-0000-000000000123",
        source="workflow_event",
        topic="basic-workflows",
        object_type="workflow",
        event_ids=["evt-1"],
        source_hashes=["sha256:deadbeef"],
        summary="basic workflows executed and receipts written",
        dropped_raw_bodies=1,
        needs_cloud_reasoning=False,
    )
    assert payload["features"]["workflow_family"] == "basic-workflows"
    assert payload["scores"]["learning_signal"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_bytewax_compact_windows.py::test_workflow_outcome_rows_include_learning_fields -v`
Expected: FAIL until workflow-family fields are added.

- [ ] **Step 3: Write minimal implementation**

```python
# in scripts/bytewax_abductive_blender.py
# add a workflow_family field to compact-window features/scores when source/object_type
# clearly identify workflow outcome rows, especially basic workflows.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_bytewax_compact_windows.py::test_workflow_outcome_rows_include_learning_fields -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bytewax_abductive_blender.py scripts/indy_reads.py tests/test_bytewax_compact_windows.py tests/test_indy_runtime_broker.py
git commit -m "feat: feed workflow outcomes into learning loop"
```

---

### Task 4: Make Indy journal/wiki write-capable workflows

**Files:**
- Modify: `scripts/lucidota_wiki_query.py`
- Modify: `scripts/indy_reads.py`
- Create: `tests/test_lucidota_journal_wiki_workflow.py`
- Modify: `tests/test_orchestrator_registry_routes.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import subprocess
import urllib.request
from pathlib import Path


def test_indy_journal_wiki_workflow_is_registered():
    with urllib.request.urlopen("http://127.0.0.1:3000/workflow_registry?workflow_name=eq.indy-journal-wiki", timeout=5) as resp:
        rows = json.loads(resp.read().decode("utf-8"))
    assert rows and rows[0]["status"] == "active"
    assert "wiki" in rows[0]["notes"].lower() or "journal" in rows[0]["notes"].lower()

    wiki_query = Path("scripts/lucidota_wiki_query.py").read_text()
    assert "BOOKS/.indy_reads/wiki" in wiki_query
    assert "BOOKS/.indy_reads/private_journal" in wiki_query

    indy_reads = Path("scripts/indy_reads.py").read_text()
    assert "def write_journal_entry" in indy_reads
    assert "journal" in indy_reads.lower()


def test_wiki_query_includes_indy_journal_dirs():
    out = subprocess.run(
        [".venv/bin/python", "scripts/lucidota_wiki_query.py", "Indy journal wiki", "--json", "--limit", "1"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_lucidota_journal_wiki_workflow.py::test_indy_journal_wiki_workflow_is_registered -v`
Expected: FAIL until the journal/wiki workflow is registered and the wiki query knows about Indy journal/wiki dirs.

- [ ] **Step 3: Write minimal implementation**

```python
# in scripts/indy_reads.py
def write_journal_entry(*, title: str, body: str, kind: str = "note") -> dict[str, str]:
    path = ROOT / "BOOKS" / ".indy_reads" / "private_journal" / f"{stamp()}_{slugify(title)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return {"ok": "true", "path": rel(path)}

# in scripts/lucidota_wiki_query.py
# add BOOKS/.indy_reads/wiki and BOOKS/.indy_reads/private_journal when they exist,
# while keeping the search bounded and read-only.

# in 06_SCHEMA/006_workflow_registry.sql
('indy-journal-wiki', 'indy+journal+wiki', '015', 'active', 'scripts/lucidota_wiki_query.py',
 '{"surface":"private journal/wiki search","authority":"indy-only-write"}',
 '{"receipt":"workflow_registry"}',
 'Indy’s personal journal/wiki workflow; operator-triggered or on-demand, not the nightly backup job.')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_lucidota_journal_wiki_workflow.py::test_indy_journal_wiki_workflow_is_registered -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 06_SCHEMA/006_workflow_registry.sql scripts/lucidota_wiki_query.py scripts/indy_reads.py tests/test_lucidota_journal_wiki_workflow.py tests/test_orchestrator_registry_routes.py
git commit -m "feat: register indy journal wiki workflow"
```

---

### Task 5: Make nightly GitHub backup a separate scheduled workflow

**Files:**
- Modify: `06_SCHEMA/006_workflow_registry.sql`
- Modify: `scripts/lucidota_daily_backup.sh`
- Create: `tests/test_lucidota_daily_backup.py`
- Modify: `tests/test_orchestrator_registry_routes.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import pathlib
import urllib.request


def test_nightly_backup_workflow_is_registered_and_scripted():
    with urllib.request.urlopen("http://127.0.0.1:3000/workflow_registry?workflow_name=eq.indy-daily-backup", timeout=5) as resp:
        rows = json.loads(resp.read().decode("utf-8"))
    assert rows and rows[0]["status"] == "active"
    assert rows[0]["command"] == "scripts/lucidota_daily_backup.sh"

    script = pathlib.Path("scripts/lucidota_daily_backup.sh").read_text()
    assert "git -C \"$ROOT\"" in script
    assert "dolt" in script
    assert "0 2 * * *" in script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_lucidota_daily_backup.py::test_nightly_backup_workflow_is_registered_and_scripted -v`
Expected: FAIL until the nightly backup workflow is registered explicitly.

- [ ] **Step 3: Write minimal implementation**

```python
# in 06_SCHEMA/006_workflow_registry.sql
('indy-daily-backup', 'indy+backup', '015', 'active', 'scripts/lucidota_daily_backup.sh',
 '{"schedule":"cron 0 2 * * *","includes":"repo backup and schema snapshot"}',
 '{"receipt":"05_OUTPUTS/receipts/daily_backup_*.json"}',
 'Nightly repo backup workflow; separate from Indy journal/wiki usage.')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_lucidota_daily_backup.py::test_nightly_backup_workflow_is_registered_and_scripted -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 06_SCHEMA/006_workflow_registry.sql scripts/lucidota_daily_backup.sh tests/test_lucidota_daily_backup.py tests/test_orchestrator_registry_routes.py
git commit -m "feat: register nightly backup workflow"
```

---

### Task 6: Verify live manual and route surfaces reflect the workflow compiler and basic workflow lane

**Files:**
- Modify: `tests/test_manual_current_surface.py`
- Modify: `tests/test_root_orchestrator_current_surface.py`
- Modify: `tests/test_luci_shell_help_surface.py`
- Modify: `GOALS/CURRENT_HANDOFF.md`
- Modify: `GOALS/GOAL_LOG.md`

- [ ] **Step 1: Write the failing test**

```python
import json
import urllib.request


def test_manual_mentions_basic_workflows_and_ontology_compiler():
    with urllib.request.urlopen("http://127.0.0.1:3000/manual_current?limit=1", timeout=5) as resp:
        rows = json.loads(resp.read().decode("utf-8"))
    manual = rows[0]
    assert "ontology_work_batch" in {route["route_id"] for route in manual["route_list"]}
    assert "basic-workflows" in json.dumps(manual)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest -q tests/test_manual_current_surface.py::test_manual_mentions_basic_workflows_and_ontology_compiler -v`
Expected: FAIL if the manual still lacks workflow-compiler context.

- [ ] **Step 3: Write minimal implementation**

```python
# add the workflow-compiler / basic-workflows line to manual_current and the
# goal handoff so the live operator surface tells the truth.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest -q tests/test_manual_current_surface.py::test_manual_mentions_basic_workflows_and_ontology_compiler -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_manual_current_surface.py tests/test_root_orchestrator_current_surface.py tests/test_luci_shell_help_surface.py GOALS/CURRENT_HANDOFF.md GOALS/GOAL_LOG.md
git commit -m "feat: document workflow compiler in manual"
```

---

### Task 7: Receipt-gated verification batch

**Files:**
- Test: `tests/test_ontology_work_compiler.py`
- Test: `tests/test_bytewax_compact_windows.py`
- Test: `tests/test_manual_current_surface.py`
- Test: `tests/test_root_orchestrator_current_surface.py`

- [ ] **Step 1: Run the receipt-gated batch**

Run: `./.venv/bin/python scripts/test_receipt_gate.py run --scope workflow_compiler --watch scripts/ontology_work_compiler.py --watch scripts/bytewax_abductive_blender.py --watch scripts/indy_reads.py --watch scripts/lucidota_daily_backup.sh --watch scripts/lucidota_wiki_query.py --watch 06_SCHEMA/153_ontology_work_batch.sql --watch 06_SCHEMA/006_workflow_registry.sql --watch tests/test_ontology_work_compiler.py --watch tests/test_bytewax_compact_windows.py --watch tests/test_lucidota_daily_backup.py --watch tests/test_lucidota_journal_wiki_workflow.py --watch tests/test_manual_current_surface.py --watch tests/test_root_orchestrator_current_surface.py -- ./.venv/bin/python -m pytest -q tests/test_ontology_work_compiler.py tests/test_bytewax_compact_windows.py tests/test_lucidota_daily_backup.py tests/test_lucidota_journal_wiki_workflow.py tests/test_manual_current_surface.py tests/test_root_orchestrator_current_surface.py`
Expected: PASS with a DB receipt UUID and zero failures.

- [ ] **Step 2: Read the receipt**

Confirm the receipt shows:
- workflow batches are live
- basic workflows remain first-class workflows
- workflow outcomes feed the learning loop
- `/manual_current` still reflects the live workflow surface

- [ ] **Step 3: Update the goal handoff**

Write the new step summary into `GOALS/CURRENT_HANDOFF.md` using the exact prefix `Save This Prompt, Pass on this Handoff:` and append the same entry to `GOALS/GOAL_LOG.md`.

- [ ] **Step 4: Commit**

```bash
git add scripts/ontology_work_compiler.py scripts/bytewax_abductive_blender.py scripts/indy_reads.py 06_SCHEMA/153_ontology_work_batch.sql tests/test_ontology_work_compiler.py tests/test_bytewax_compact_windows.py tests/test_manual_current_surface.py tests/test_root_orchestrator_current_surface.py GOALS/CURRENT_HANDOFF.md GOALS/GOAL_LOG.md
git commit -m "feat: workflow compiler and workflow learning loop"
```
