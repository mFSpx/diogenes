SYSTEM OVERRIDE: OPERATION ROOT-ROTOR / CANON FORGE.

MISSION:
Audit the current LUCIDOTA system and create the Canonical Technical Bible: a professional, aviation-style technical manual set for the actual software system. This manual set becomes the only source of canon after review, DB insertion, versioning, and API exposure.

CRITICAL WARNING:
All category names, node ids, manual names, schema fields, endpoint names, and examples below are format templates only. Do not declare examples as current reality. Do not synthesize data. Discover the actual system state from files, schemas, scripts, receipts, and DB facts. If evidence does not prove a subsystem exists, do not document it as existing.

AUTHORITY INPUTS:
Read KANT69 first. Read active instruction indexes, current GOALS handoff/logs, 00_PROJECT_BRAIN, active schemas, scripts, tests, and current receipts. Treat old build logs as evidence, not law. KANT69, current receipts, and live system facts outrank stale prose.

REPO AUDIT:
Produce a safe active-software text audit before synthesis. Exclude .git, .venv, caches, binaries, images, PDFs, archives, KRAMPUSCHEWING, 03_VAULT, and 05_OUTPUTS bulk bodies. Include active .py, .rs, .sql, .md, .json, .sh, .toml files. Truncate each file body at 100 KB. Record file path, byte count, truncation flag, sha256, and scan time.

TECHNICAL WRITING STANDARD:
Use ASD-STE100 principles: active voice, short sentences, controlled terms, one word for one meaning, no ambiguity. Use Transport Canada style discipline: current technical publications, amendment control, record retention, and auditable maintenance/control procedures. Do not call the manuals aircraft manuals. Design them with that rigor.

MANUAL SET:
Create the real manual set from evidence. Expected top-level set:
1. System Architecture Manual.
2. Runtime/Governor Manual.
3. Algorithms/Avionics Manual.
4. Operations/Flight Manual.
5. Ledger/Amendment Manual.
These names are templates. Rename only if evidence and clarity require it.

CONTENT REQUIREMENT:
Document every actual subsystem. For each subsystem, state: what it is, how to build it, how to use it, workflows, tools, scripts, schemas, APIs, algorithms, skills, tests, receipts, failure modes, and operating limits. Include architecture maps as DB-backed nodes, not loose diagrams.

LAW OF ROOT INDEXING:
Write the manuals in strict hierarchical numbered sections. Every section, subsection, and individual point gets a stable coordinate. Example only: 2.4.14. Each coordinate maps 1:1 to a DB node. The assembled manuals are generated from DB nodes.

DB NODE CONTRACT:
Each node must be independently versioned and hash-tracked.
Required fields: node_id, parent_id, manual_id, title, payload, payload_format, source_refs, evidence_hashes, dependencies, affects_nodes, status, version, valid_from, valid_to, hash_current, previous_hash, created_at, updated_at.
Small previous payloads may be retained inline. Large or old versions go to cold storage with hash pointers.

VERSIONING LAW:
Version per node, per branch/fan/tree. When architecture changes, update the affected node cells only. Preserve previous versions by hash. Mark dependent nodes review_required when blast radius edges show impact.

API LAW:
Every node must be retrievable through an API endpoint. Agents must be able to request one node, a subtree, a manual, dependencies, latest version, or historical version. Example only: GET /canon/node/{node_id}. The API serves current canon from the DB, not stale prompt files.

OUTPUT:
Return: audit manifest, discovered subsystem index, manual outline, DB node JSON array, dependency/blast-radius edge list, API route spec, and gaps requiring human review. Use evidence only. No vibes. No invented canon.

---

## Smaller-model packet: Root-Rotor manual draft reduction

```text
OPERATE: ROOT-ROTOR MANUAL DRAFT REDUCTION. NO NEW CONTROL PLANES.

Goal: reduce Root-Rotor draft debt through the existing GOALS + Root-Rotor + Postgres/PostgREST surfaces. GOALS owns continuation. Keep no new control planes as the operating law. Do not create docs, daemons, queues, schemas, or output planes unless the existing contract already requires them.

Read first:
- GOALS/CURRENT_HANDOFF.md
- GOALS/AGENT_ORCHESTRATION_POLICY.md
- GOALS/GOAL_HANDOFF_PROMPT.md
- scripts/root_rotor_manual_queue.py
- scripts/root_rotor_apply_node_payloads.py
- scripts/root_rotor_red_team_audit.py

Batch law:
- Select exactly 200 candidate active nodes per batch.
- Skip any node whose active file hash already matches a valid sidecar/ledger payload.
- Allowed final node labels: verified, deprecated, needs_operator_label.
- If confidence is low, keep the 200 boundary and record numeric confidence/precision.

Priority:
1. GOALS policy docs and active instruction docs.
2. Wired active 01_REPOS/claudecode LUCI engine files.
3. Active local runtime/services, but exclude services/ternary_lab unless explicitly wired.
4. scripts/root_rotor*, scripts/goal_*, scripts/absurd*, scripts/indy*, scripts/krampuschewing*.
5. 06_SCHEMA as SYSTEM_ARCH.
6. model/capability/resource ledgers as lucidota_fabric.

Classification:
- authority docs verify; historical prose deprecates.
- generated files, old receipts/manual outputs, empty __init__.py, and stale compiled manuals deprecate/exclude.
- scripts/absurd* route to RUNTIME_GOVERNOR.

Data/mutation rules:
- Do not patch orchestration scripts.
- Inline SQL is for bounded SELECT/inspection and approved staging payloads only.
- No arbitrary production DML/DDL.
- No direct truth mutation: staged -> validated -> receipted -> promoted.
- If the current apply path can only promote verified payloads, apply verified sidecars only and receipt deprecated/needs_operator_label candidates for the operator/apply-path owner.

Output/audit rules:
- No flat 05_OUTPUTS swamp. Use existing shallow typed output dirs plus UUID/hash filenames and DB ledger truth.
- JSONB may hold flexible sidecar metadata; stable identity/status/routing/query fields stay canonical/typed.
- PostgREST audit must use live OpenAPI route enumeration; static config parsing is only supplementary.
- Track verified_count, deprecated_count, and needs_operator_label_count separately.
- systemd read-only by default; process mutation requires explicit operator/GOALS/ABSURD directive.

Run:
.venv/bin/python scripts/root_rotor_postgrest_control.py status --check-readiness --wait 2 --poll 0.2
.venv/bin/python scripts/root_rotor_red_team_audit.py --json
bounded SQL/sidecar generation for exactly 200 nodes
existing apply/receipt path only where valid
.venv/bin/python scripts/root_rotor_red_team_audit.py --json

Return only: SQL used, nodes selected, counts by label, sidecar/receipt paths, PostgREST readiness, red-team before/after counts, blockers, and GOALS handoff/log update proof.
```

---

## Active goal prompt: Root-Rotor total migration execution

```text
ROOT-ROTOR TOTAL MIGRATION EXECUTION PROMPT

You are operating inside the LUCIDOTA repo. Objective: finish the manual node migration job, not merely run Batch 001. Execute bounded 200-node batches until `manual_incomplete_draft_nodes` reaches 0, no eligible candidates remain, or a real blocker is proven with receipts.

HARD CONSTRAINTS:

* Use `.venv/bin/python` for every Python operation.
* Batch size is exactly 200 candidate nodes per normal batch.
* No arbitrary DML. No direct ad hoc DB mutation.
* No patching runtime orchestrator scripts.
* Only write bounded payloads into approved staging paths/tables.
* No flat folder swamp. All outputs go under shallow typed subdirs in `05_OUTPUTS/`, with stable UUID/hash filenames.
* Executable changes only: commands, tests, receipts, metrics, staged labels.
* Classify every candidate as exactly one of: `verified`, `deprecated`, `needs_operator_label`.
* Do not invent completion. Prove it with counters, receipts, and validation traces.

STARTING PACKET:
Run the first batch exactly:

`.venv/bin/python scripts/root_rotor_manual_queue.py --batch-size 200 --source 01_REPOS/claudecode/src/services --output-dir 05_OUTPUTS/root_rotor_manuals`

Then validate:

`.venv/bin/python -m pytest -q tests/test_root_rotor_manual_reduction_laws.py tests/test_goal_handoff.py`

Then capture telemetry:

`.venv/bin/python scripts/root_rotor_postgrest_control.py status --check-readiness`

`.venv/bin/python scripts/goal_dev_control.py --away-minutes 0 --text "Batch 001 completed" --json`

AFTER BATCH 001:
Parse the actual outputs. Do not guess. Extract:

* processed count
* verified count
* deprecated count
* needs_operator_label count
* validation pass/fail
* current `manual_incomplete_draft_nodes`
* output receipt paths
* any source dirs still containing candidate nodes
* any blocker/error signatures

Then continue automatically.

CONTINUATION LAW:
If validation passes and `manual_incomplete_draft_nodes > 0`, run the next 200-node batch against the next highest-yield eligible claudecode core source path. Prefer, in order:

1. `01_REPOS/claudecode/src/services`
2. `01_REPOS/claudecode/src/commands`
3. `01_REPOS/claudecode/src/core`
4. `01_REPOS/claudecode/src/tools`
5. any other `01_REPOS/claudecode/src/**` path proven by the prior sweep/telemetry to contain eligible manual draft nodes.

Name batches monotonically: Batch 001, Batch 002, Batch 003, etc. Each batch must have its own shallow output subdir under `05_OUTPUTS/root_rotor_manuals/` or the script’s approved equivalent. Use hash/UUID filenames. Do not dump mixed logs into `05_OUTPUTS/`.

For every batch:

1. Run manual queue processing with `--batch-size 200`.
2. Run the same pytest validation suite.
3. Run readiness/status telemetry.
4. Run goal handoff JSON receipt with text `Batch NNN completed`.
5. Summarize deltas.
6. Decide whether to continue.

STOP CONDITIONS:
Stop only when one of these is true:

* `manual_incomplete_draft_nodes == 0`
* fewer than 200 eligible candidate nodes remain and the tool explicitly refuses partial final batches
* validation fails
* required script/table/source path is missing
* the next action would require arbitrary DML or runtime orchestrator patching
* classification cannot be completed without operator labels

If stopped, write a final blocker/completion receipt under the approved output tree and report:

* final counter
* batches completed
* total nodes processed
* total verified/deprecated/needs_operator_label
* last passing validation command
* exact blocker, if any
* next executable command, if any

FAILURE HANDLING:
If a batch fails validation, do not “fix broadly.” First inspect the failing test output and only make changes if they are inside the permitted extraction/classification/staging surface. If the fix would touch forbidden runtime orchestrators or require ad hoc database mutation, stop and write a blocker receipt.

QUALITY BAR:
No prose-only victory laps. No “probably done.” No schema theater. No broad refactors. No deleting evidence. No flattening output folders. The job is complete only when the counter proves it or the blocker receipt proves why it cannot continue.

Begin now with Batch 001, then keep executing batches under these laws until the whole job is done or honestly blocked.
```

---

## Active goal prompt: Root-Rotor repair node count only

```text
ROOT-ROTOR REPAIR PROMPT — NODE COUNT ONLY

STOP. LOCAL JOB IS ROOT-ROTOR REPAIR ONLY.

You misread the unit of work. The migration target is 200 candidate NODE rows, not 200 source files.

Do not continue from any blocker state derived from file counts. File counts are irrelevant except as weak source-path hints.

FIRST: SAVE THE ACTIVE ROOT-ROTOR GOAL PROMPT

Before execution, save the active Root-Rotor prompt/instructions exactly as goal context:

* preserve the full operator prompt that started this Root-Rotor total migration job
* save it into the repo’s existing GOALS/handoff surface
* write a receipt with saved path + hash
* do not rewrite it as canon
* do not summarize it as the saved prompt

SECOND: SUPERSEDE THE BAD BLOCKER

The previous blocked state was invalid because it used file count instead of node count.

Do not delete old receipts. Write a supersession receipt with reason exactly:

`invalid_blocker_basis_file_count_used_instead_of_node_count`

Any GOALS status derived from the file-count blocker must be marked superseded/invalidated, not authoritative.

THIRD: PRINT EXACTLY

`I used node count, not file count.`

FOURTH: REPORT NODE FACTS BEFORE CODE CHANGES

Before any code edit, print a node-based inventory table:

* current `manual_incomplete_draft_nodes`
* eligible NODE count under each candidate claudecode source root
* eligible NODE count by status/decision
* top source roots contributing to the manual-incomplete counter
* exact Batch 001 node-selection method
* status of the previous `root_rotor_manual_queue.py` patch:

  * reverted
  * quarantined
  * retained with corrected node semantics

UNIT LAW

* Candidate unit = manual/root-rotor node row.
* Source path = filter/scope hint.
* Batch size = exactly 200 eligible candidate nodes.
* File count is not node count.
* Do not claim blocked from source-file counts.

BATCH 001 LAW

After printing node facts, run Batch 001 against exactly 200 eligible candidate NODE rows.

Preferred source widening order:

1. `01_REPOS/claudecode/src/services`
2. `01_REPOS/claudecode/src/commands`
3. `01_REPOS/claudecode/src/core`
4. `01_REPOS/claudecode/src/tools`
5. other proven `01_REPOS/claudecode/src/**` node-bearing roots
6. real queue inventory behind `manual_incomplete_draft_nodes`

If a preferred source root has fewer than 200 eligible NODE rows, widen the node selection set. Do not block unless the corrected node inventory proves no valid 200-node batch can be formed.

REPAIR PATCH LAW

If the prior script patch enforces file-count blocker semantics, revert or quarantine it.

If the script lacks a valid node-row batch mode, implement the smallest permitted staging-only fix:

* select candidate node rows
* produce exactly 200 node payloads
* write bounded staged output
* write receipt
* do not perform arbitrary DML
* do not patch runtime orchestrators

MANDATORY VALIDATION AFTER BATCH 001

Run:

`.venv/bin/python -m pytest -q tests/test_root_rotor_manual_reduction_laws.py tests/test_goal_handoff.py`

MANDATORY TELEMETRY AFTER VALIDATION

Run:

`.venv/bin/python scripts/root_rotor_postgrest_control.py status --check-readiness`

Run:

`.venv/bin/python scripts/goal_dev_control.py --away-minutes 0 --text "Batch 001 completed" --json`

CONTINUATION LAW

Continue Batch 002, Batch 003, etc. until one true stop condition is proven:

* `manual_incomplete_draft_nodes == 0`
* true eligible NODE inventory is exhausted
* validation fails
* operator labels are required
* required table/script/path is missing
* next action requires forbidden arbitrary DML
* next action requires runtime orchestrator patching

FORBIDDEN

* no file-count blockers
* no broad refactors
* no arbitrary DML
* no runtime orchestrator patching
* no deleting old receipts
* no prose-only victory laps
* no “probably done”
* no new unrelated goal management before Root-Rotor facts are printed

OUTPUT NOW IN THIS ORDER

1. saved active prompt path/hash
2. supersession receipt path/hash
3. `I used node count, not file count.`
4. node inventory table
5. Batch 001 exact 200-node selection method
6. script patch disposition
7. Batch 001 execution result
8. validation result
9. telemetry receipt paths
10. next batch command or true blocker
 <--- THE GOAL
```

