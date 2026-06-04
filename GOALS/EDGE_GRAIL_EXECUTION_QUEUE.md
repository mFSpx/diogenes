# Edge Grail Execution Queue

Generated: `2026-06-02T22:30:06.832969Z`

Ordering law: SQL/sheet if exact → Treelite/router if cheap → algorithm if uncertainty remains → Bonsai/Mamba if language needed → deep model only when admitted → receipt or it did not happen.

## Active parallel agents

### eg-001-runpod-talkie-lora-forge — RUNPOD
- Owner: Worker A / Archimedes
- Status: IN_PROGRESS_PARALLEL_AGENT
- Goal: Use Jupyter API, not SSH, to launch or prove blocker for Talkie LoRA readiness/training.
- Acceptance:
  - no secret token in stdout/receipts
  - dry-run supported
  - live command receipt or exact blocker receipt
- Blockers:
  - SSH auth ignored; use Jupyter API only
  - do not download books

### eg-002-sheet-workflow-spine — LOCAL_DB_SHEET_FASTLANE
- Owner: Worker B / Russell
- Status: IN_PROGRESS_PARALLEL_AGENT
- Goal: Make executable spreadsheet-first workflow spine for ingest/evidence/graph/network/forms before algos/models.
- Acceptance:
  - dry-run/no-DB smoke works
  - SQL routing order explicit
  - pytest targeted pass

### eg-003-indy-ironclaw-readiness — LOCAL_SERVICE_READINESS
- Owner: Worker C / Bernoulli
- Status: IN_PROGRESS_PARALLEL_AGENT
- Goal: Check IronClaw/Indy_READs startup, book count, comms config, response-helper hook without sending messages.
- Acceptance:
  - no email or Signal side effects
  - reports exact missing env/config
  - pytest targeted pass
- Blockers:
  - do not send mail/messages from automation without explicit send confirmation

### eg-004-book-lora-status-and-work-orders — LOCAL_BOOK_LORA_STAGE
- Owner: Worker D / Leibniz
- Status: IN_PROGRESS_PARALLEL_AGENT
- Goal: Count every local Indy/book asset and report 3x LoRA target readiness, chunks/cards/embeddings, missing work orders.
- Acceptance:
  - local files only
  - 3x adapter count clear
  - 500-token chunk/cards/embedding counts clear
  - pytest targeted pass
- Blockers:
  - no Anna/API piracy acquisition automation

### eg-005-needle-shared-prefix-runner — LOCAL_VRAM_NEEDLE
- Owner: Main orchestrator after current workers return or next slice
- Status: READY_NOT_STARTED
- Goal: Refactor Needle runner to reuse encoder_out/enc_mask for identical immutable 500-token prefix across lane tasks, with receipt proving tensor reuse boundary.
- Acceptance:
  - exact pointer/tensor reuse or explicit architecture-impossible receipt
  - no mutation of upstream sovereign repo unless copied/adapted
- Blockers:
  - current exact KV pointer sharing is unproven

### eg-006-graph-materialization-fenced — LOCAL_GRAPH_SLOWLANE
- Owner: Main or later worker after sheet spine green
- Status: BLOCKED_ON_SHEET_AND_OPERATOR_GATE
- Goal: Materialize deferred 1128 book/embedding packets into graph only after sheet projection and write barrier checks pass.
- Acceptance:
  - write barrier pass
  - row counts and hashes match
  - rollback path documented
- Blockers:
  - canonical graph writes fenced until sheet/write-barrier proof
