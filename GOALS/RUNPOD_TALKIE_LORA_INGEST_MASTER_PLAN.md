# RunPod Talkie / LoRA / Ingest Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or execute one bounded task at a time. Workers are not alone in the codebase; do not revert others. Heavy artifacts go on RunPod `/workspace`; local keeps manifests, receipts, import scripts, docs, and small tests only.

**Goal:** Use RunPod as the heavy compute worker to train/stage Talkie + book/ontology LoRAs, accelerate embedding/OCR/extraction ingestion, then wire outputs into LUCI’s sheet-first → algo → model runtime.

**Architecture:** RunPod performs heavyweight GPU/CPU batch work and emits bounded artifacts + receipts. Local LUCIDOTA remains the truth spine: Postgres/Absurd/sheets ingest pulled artifacts through COPY/UPSERT and route via Tokio/rig.rs/tool manifests. Direct RunPod→DB writes are optional only after a DB secret/tunnel is explicitly configured.

**Tech Stack:** RunPod Jupyter API, PyTorch/Talkie, PEFT/QLoRA where compatible, JSONL/Parquet artifacts, Postgres/DuckDB sheet layer, Tokio bounded refs, Treelite/FIL receipts, local `GOALS/` handoff receipts.

---

## Current Proven State

- RunPod Jupyter API works; SSH remains noncritical.
- Talkie checkpoint is on RunPod: `/workspace/talkie_forge/models/talkie-lm__talkie-1930-13b-it/rl-refined.pt`.
- Talkie vocab is on RunPod: `/workspace/talkie_forge/models/talkie-lm__talkie-1930-13b-it/vocab.txt`.
- GPU capacity receipt: RTX PRO 6000 Blackwell, ~97.9GiB VRAM.
- RAM capacity: ~1.5TiB RAM available; load smoke is safe to attempt.
- Talkie deep inspection proves 13,280,257,721 bf16 params, 40 layers, custom Talkie modules.
- Talkie LoRA targets are **not** Llama names. Use `attn_query/attn_key/attn_value/attn_resid` and `mlp_gate/mlp_linear/mlp_resid`.
- Book pack is on RunPod with chunks/cards/embeddings already unpacked.
- Treelite manifest exists: `04_RUNTIME/treelite_vram_readiness_work_orders.json`; GPU/FIL residency is not proven yet.

---

## Phase 0 — Orchestration Budget Law

**Files:**
- Modify: `GOALS/CURRENT_HANDOFF.md`
- Modify: `GOALS/GOAL_LOG.md`
- Create/modify: `GOALS/RUNPOD_TALKIE_LORA_INGEST_MASTER_PLAN.md`

- [x] Save this plan in `GOALS/` so weekly-token budget does not evaporate in chat.
- [ ] Keep the main session as thin orchestrator: dispatch bounded workers, integrate receipts, avoid rewriting huge code inline.
- [ ] Prefer RunPod/Jupyter execution for heavyweight jobs.
- [ ] Do not move giant model artifacts back to the laptop.

Acceptance check:
```bash
sed -n '1,220p' GOALS/RUNPOD_TALKIE_LORA_INGEST_MASTER_PLAN.md
```

---

## Phase 1 — RunPod Talkie Readiness Gate

**Files / artifacts:**
- Remote: `/workspace/talkie_forge/receipts/talkie_load_smoke.json`
- Remote: `/workspace/talkie_forge/receipts/talkie_moe_readiness_manifest.json`
- Local: `05_OUTPUTS/runpod/talkie_book_lora/talkie_load_smoke.json`
- Local: `04_RUNTIME/TALKIE_MOE/talkie_moe_readiness_manifest.json`

- [x] Download checkpoint.
- [x] Download vocab.
- [x] Clone Talkie source to RunPod.
- [x] Inspect checkpoint deeply.
- [ ] Run load/forward smoke on GPU.
- [ ] If smoke fails due package import, bypass package `__init__` or install minimal deps; do not download checkpoint again.
- [ ] Copy smoke receipt local.

Acceptance check:
```bash
python3 -m json.tool 05_OUTPUTS/runpod/talkie_book_lora/talkie_load_smoke.json | grep '"status": "PASS"'
```

---

## Phase 2 — RunPod Ingest/Embedding Accelerator

**Files / artifacts:**
- Create remote: `/workspace/lucidota_ingest_accel/run_embed_batch.py`
- Create remote: `/workspace/lucidota_ingest_accel/receipts/embed_batch_receipt.json`
- Create local: `scripts/runpod_artifact_pull_import.py`
- Create/modify local: `04_RUNTIME/RUNPOD_ACCEL/runpod_ingest_embedding_accelerator.json`

- [x] Record feasibility manifest: artifact-pull mode now; direct DB writes conditional.
- [ ] Build remote worker that reads bounded chunk JSONL and emits embeddings as JSONL/Parquet + SHA256 receipt.
- [ ] Build local pull/import command that fetches remote artifact through Jupyter API and COPY/UPSERTs into Postgres.
- [ ] Keep direct DB write mode disabled unless `DATABASE_URL` secret/tunnel exists.

Acceptance checks:
```bash
python3 -m json.tool 04_RUNTIME/RUNPOD_ACCEL/runpod_ingest_embedding_accelerator.json
python3 scripts/runpod_artifact_pull_import.py --help
```

---

## Phase 3 — Book LoRA Training Matrix

**Files / artifacts:**
- Existing local: `04_RUNTIME/BOOK_READER_LORA/book_lora_work_orders.json`
- Remote input: `/workspace/talkie_book_lora/talkie_book_lora_runpod_pack/book_lora/cards/reading_cards.train.jsonl`
- Remote output: `/workspace/talkie_forge/adapters/*`
- Remote receipts: `/workspace/talkie_forge/receipts/book_lora_train_*.json`

- [ ] Generate target map for `talkie`, `bonsai8b_q1`, `bonsai8b_q2`.
- [ ] Talkie target map uses custom module names from Phase 1.
- [ ] Bonsai target maps are only marked trainable after compatible base architecture is identified.
- [ ] Train/evaluate one tiny smoke adapter first; do not launch all 21 until smoke receipt passes.
- [ ] Scale to 3x adapters per actual book after smoke.

Acceptance check:
```bash
python3 -m json.tool 04_RUNTIME/BOOK_READER_LORA/book_lora_work_orders.json | head -80
```

---

## Phase 4 — Talkie MoE Upcycle

**Files / artifacts:**
- Create remote: `/workspace/talkie_forge/src/talkie_moe_upcycle.py`
- Create remote: `/workspace/talkie_forge/receipts/talkie_moe_disk_guard.json`
- Create remote: `/workspace/talkie_forge/receipts/talkie_moe_router_smoke.json`

- [ ] Implement MLP-only 4-expert wrapper: expertize `mlp_gate`, `mlp_linear`, `mlp_resid`; share attention/embed/lm_head.
- [ ] Add router `Linear(5120 -> 4)` per layer or shared policy; start top-1.
- [ ] Initialize experts from dense MLP weights without writing a full 4x bf16 duplicate.
- [ ] Enforce disk guard: fail if projected write exceeds safe volume budget.
- [ ] Train router/LoRA first; do not do full finetune.

Acceptance check:
```bash
python3 -m json.tool /workspace/talkie_forge/receipts/talkie_moe_router_smoke.json
```

---

## Phase 5 — Sheet-First Runtime Wiring

**Files:**
- Existing: `06_SCHEMA/147_lucidota_sheet_layer.sql`
- Existing: `04_RUNTIME/lucidota_workflow_registry.json`
- Modify: CLI sheet/workflow commands as needed

- [ ] Ensure work types route: evidence ingest, corpus ingest, graph ops, network analysis, pivot search, docs/forms, workflow automation.
- [ ] SQL/sheet resolves first; algos second; model last.
- [ ] Add RunPod artifact-import as a workflow task.
- [ ] Receipts must include source hash, row count, duration, output hash.

Acceptance check:
```bash
./luci ingest seed-workflow-tasks --json
./luci sheet list --json || true
```

---

## Phase 6 — Treelite / VRAM Lane Proof

**Files:**
- Existing: `04_RUNTIME/treelite_vram_readiness_work_orders.json`
- Remote/local output: FIL/GPU residency receipt

- [ ] Keep existing 301 Treelite artifacts indexed.
- [ ] Prove FIL/GPU residency separately; do not claim VRAM residency from local `.tl` loadability alone.
- [ ] Wire truth flags so route consumers fail safe when GPU residency is unproven.

Acceptance check:
```bash
python3 -m json.tool 04_RUNTIME/treelite_vram_readiness_work_orders.json | grep -E 'gpu_residency_proven|fil_gpu_residency_proven'
```

---

## Phase 7 — Manuals / API / HTML Pages

**Files:**
- Create: `05_OUTPUTS/runtime/manuals/*.md`
- Create: `05_OUTPUTS/runtime/api/*.md`
- Create: `05_OUTPUTS/runtime/html/*.html`

- [ ] Generate four ops manuals: RunPod Forge, Local Edge Runtime, Ingest/Sheet Layer, Indy_READs/Ironclaw.
- [ ] Generate API specs for artifact pull/import, sheet tasks, model/adapter admission, receipts.
- [ ] Generate simple HTML dashboard pages that read static receipt JSON.

Acceptance check:
```bash
find 05_OUTPUTS/runtime -maxdepth 3 -type f | sort
```

---

## Phase 8 — Indy_READs / Ironclaw Online

**Files / artifacts:**
- Modify/create runtime service manifests under `04_RUNTIME/`
- Write receipts under `05_OUTPUTS/indy_reads/`

- [ ] Indy starts with system; uses existing book corpus and ROOT414/manuals as reading queue.
- [ ] Email/Signal/direct-chat are separate tool manifests; no secret hardcoding.
- [ ] Indy assists LUCI app responses through bounded adapter/RAG route, not raw resident sprawl.

Acceptance check:
```bash
find 05_OUTPUTS/indy_reads -maxdepth 1 -type f | sort | tail -40
```

---

## Phase 9 — Hypertimeline / RiverML Training Points

**Files / artifacts:**
- Create: `04_RUNTIME/RIVERML/hypertimeline_training_point_manifest.json`
- Create local/remote workers after graph data is canonical

- [ ] Wait until graph/corpus data is actually in canonical/sheet projections.
- [ ] Extract deltas as training examples: state_before, operator_move, evidence, strategy, state_after, receipt.
- [ ] Priority order: Rickshaw, Nordby, Crave Church, 4265, Kristin, Scarlet, research.
- [ ] Train behavior on deltas; exact facts remain RAG/Postgres.

Acceptance check:
```bash
python3 -m json.tool 04_RUNTIME/RIVERML/hypertimeline_training_point_manifest.json
```

---

## Immediate Parallel Agent Work Orders

1. **RunPod Forge Worker:** finish Talkie smoke, target-map, MoE disk guard.
2. **Ingest Accelerator Worker:** remote embed worker + local artifact pull/import.
3. **Sheet Runtime Worker:** add RunPod artifact-import workflow rows and sheet receipts.
4. **Manuals Worker:** generate four manuals/API/HTML skeleton from live manifests.
5. **Indy Worker:** wire reading queue/status receipts from actual book inventory.

## Self-Review

- Spec coverage: covers RunPod Talkie, 3x LoRAs, model return/wiring, manuals/API/HTML, spreadsheet layer, Indy_READs, ingestion, hypertimeline/RiverML training points, Treelite lane.
- Placeholder scan: no `TBD`; all phases include concrete files/artifacts and acceptance checks.
- Type consistency: RunPod is heavy worker; local is truth spine; direct DB writes stay conditional unless tunnel/secret exists.
