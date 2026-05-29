# LUCIDOTA GAPS → ACTIONABLE WORKFLOWS

Generated: `2026-05-28T08:10:57Z`

Total queued gaps: **205**

## real_backend_not_wired

- Severity: `medium`
- Source: `status_ledger.open_blockers`
- Gap: Ternary/FairyFuse backend remains STUB_BACKEND_NOT_WIRED; no real backend, model download, or bitnet.cpp build has been performed.
- Action: Wire and benchmark a real low-bit backend before marking Ternary runtime verified.
- Closure gate: blocker removed or downgraded with evidence

## SMOLDOCLING_DIRECT_PACKAGE_OR_MODEL_PATH_NOT_CONFIRMED

- Severity: `low`
- Source: `status_ledger.open_blockers`
- Gap: Docling is installed, but no standalone SmolDocling Python module/model path is confirmed locally.
- Action: Provide or approve a SmolDocling-compatible local model/package path before using SmolDocling as an active parser.
- Closure gate: blocker removed or downgraded with evidence

## MODERNBERT_LOCAL_MODEL_PATH_NOT_CONFIGURED

- Severity: `low`
- Source: `status_ledger.open_blockers`
- Gap: Transformers runtime is installed, but no ModernBERT weights/local path have been downloaded or configured.
- Action: Choose and approve a ModernBERT model path/download before running local classifier smoke.
- Closure gate: blocker removed or downgraded with evidence

## BGE_MODEL_WEIGHTS_NOT_DOWNLOADED

- Severity: `low`
- Source: `status_ledger.open_blockers`
- Gap: FlagEmbedding package is installed, but BGE/M3 reranker weights are not downloaded.
- Action: Choose and approve exact BGE/M3 model before reranker execution.
- Closure gate: blocker removed or downgraded with evidence

## OLLAMA_MODELS_NOT_PULLED

- Severity: `low`
- Source: `status_ledger.open_blockers`
- Gap: Ollama binary/service is installed and active, but no Ollama models are pulled.
- Action: Run an explicit ollama pull for a chosen model when operator selects target.
- Closure gate: blocker removed or downgraded with evidence

## KRAMPUSCHEWING

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: Production ingestion/OCR/kernel-routed lanes incomplete; health/custody evidence only.
- Action: Migrate actual intake execution under DBOS only after explicit queue-worker contract.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## DBOS workflow spine

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: DBOS tables exist; remaining handlers/authorization/supervision incomplete.
- Action: Migrate custom daemon internals after wrappers stay stable.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Model CPU scheduler

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: GPU/model registry exists but governed invocation not wired.
- Action: Connect model inventory to job fair allocator.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## FairyFuse

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: Real backend not wired.
- Action: Benchmark/replace symbolic backend with real BitNet path only when local artifacts are ready.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Fidelity Guard

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: not launch-complete
- Action: Materialize schema and checker.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Phase 0.5 Brain Archaeology

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: not launch-complete
- Action: Create schema/workflow scaffolds.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Job Fair Allocator

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: file missing
- Action: Create allocator scaffold.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Biometric Stream

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: file missing
- Action: Create stream scaffold.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## LUCIDOTA CLI / CLAW fork

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: Fork modified; not CEP/kernel/DBOS command surface yet.
- Action: Record CLAW runtime contract and tests.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Ternary Lens Lab

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: Research/scaffold; BitNet/FairyFuse backend not wired.
- Action: Keep BitNet/FairyFuse model backend separate; symbolic Command Envelope Router is wired.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Phase 0.5B Feature & Telemetry Contract Formalization

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: Queued behind Chrono daemon supervision priority.
- Action: After Chrono daemon stability window, formalize Sticker v1, streaming/batch ML outputs, topology findings, Cruelty Protocol outputs, and authority-class mapping.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## System Genesis Actions

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: not launch-complete
- Action: Close genesis window when conversation_command exists, runtime can reach it, first non-genesis command envelope is written, and ledger records it.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Command Envelope Protocol boundary

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: conversation_command_runtime_not_active
- Action: Implement lucidota_control.conversation_command before load-bearing surfaces.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Dead-Letter Review Runbook

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: not launch-complete
- Action: Review unresolved DLQ rows without mutating forced-smoke row unless explicitly commanded.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Dead-letter forced smoke

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: OPERATOR_APPROVAL_REQUIRED_TO_MUTATE_DLQ
- Action: Run explicit DLQ classify/resolve command after operator approval.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## KRAMPUSCHEWING DBOS wrapper

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: Verified health wrapper; not production ingestion worker.
- Action: Use wrapper as DBOS observation path; implement River/Bytewax wrapper next.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Mem0 / Behavioral Memory Engine

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: repeated_behavior_not_preference_review_required
- Action: Define behavior review/crucifixion workflow before any persistent behavior memory.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Workflow Foundry

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: operator_review_required_before_workflow_canon
- Action: Mine workflow variants only into drafts; no auto-canonization.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Fungible Semantic Handles

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: semantic_handle_namespace_review_pending
- Action: Define operator-owned namespace rules and alias revision flow.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## HippoRAG Locked Gate

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: HIPPO_RAG_LOCKED_UNTIL_GRAPH_CLEAN
- Action: Clean/prune graph and produce graph readiness report before any HippoRAG sandbox walk.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## ncnn edge inference candidate

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: ncnn Python module not installed; VULKAN_SDK/vulkaninfo absent; no ncnn model artifact selected.
- Action: Use claw ncnn-probe before any build/install; only integrate ncnn behind explicit model artifacts and benchmark receipts.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Security Secret Quarantine

- Severity: `medium`
- Source: `status_ledger.software`
- Gap: Security scan still detects masked secret-like material in archived local knowledge/vault/log artifacts.
- Action: Redact/copy, rotate any live credentials, and quarantine archived secret-bearing logs without deleting originals.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## ≤4GB VRAM profile

- Severity: `medium`
- Source: `status_ledger.hardware_runtime_targets`
- Gap: not launch-complete
- Action: Create explicit FairyFuse/llama config.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## CPU-only fallback

- Severity: `medium`
- Source: `status_ledger.hardware_runtime_targets`
- Gap: not launch-complete
- Action: Add model CPU fallback policy.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Samsung S22 Ultra+ class mobile profile

- Severity: `medium`
- Source: `status_ledger.hardware_runtime_targets`
- Gap: device runtime profile missing
- Action: Define storage/model constraints.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Phase 0.5: Brain Archaeology prep

- Severity: `medium`
- Source: `status_ledger.phases`
- Gap: not launch-complete
- Action: Create SPEC-001.5 schema and workflows.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## Phase 3: governed model CPUs

- Severity: `medium`
- Source: `status_ledger.phases`
- Gap: not launch-complete
- Action: Implement scheduler and model policy.
- Closure gate: ledger entry reaches executed/verified or blocker is named harder

## scripts/abductive_db_os_ledger.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 140 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/abductive_db_os_ledger.py
```

## scripts/absurd_abductive_ledger.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 140 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/absurd_abductive_ledger.py
```

## scripts/absurd_chrono_worker.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 359 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/absurd_chrono_worker.py
```

## scripts/absurd_consume_one.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 271 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/absurd_consume_one.py
```

## scripts/absurd_flows.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 402 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/absurd_flows.py
```

## scripts/absurd_graph_promotion_worker.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 239 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/absurd_graph_promotion_worker.py
```

## scripts/absurd_queue_spine.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 397 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/absurd_queue_spine.py
```

## scripts/absurd_remaining_worker_contract_alignment_check.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 138 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/absurd_remaining_worker_contract_alignment_check.py
```

## scripts/absurd_river_worker.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 691 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/absurd_river_worker.py
```

## scripts/absurd_worker_contracts.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 279 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/absurd_worker_contracts.py
```

## scripts/activity_tree_ingest_dry_run.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 110 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/activity_tree_ingest_dry_run.py
```

## scripts/authority_class_mapper.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 146 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/authority_class_mapper.py
```

## scripts/bitloops_airlock_audit.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 154 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/bitloops_airlock_audit.py
```

## scripts/bitloops_automation_loop.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 400 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/bitloops_automation_loop.py
```

## scripts/bitloops_full_reingest_manifest.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 200 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/bitloops_full_reingest_manifest.py
```

## scripts/bitloops_red_team_suite.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 502 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/bitloops_red_team_suite.py
```

## scripts/board_effect_doctrine.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 180 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/board_effect_doctrine.py
```

## scripts/boring_beast.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 894 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/boring_beast.py
```

## scripts/boring_beast_full_e2e.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 121 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/boring_beast_full_e2e.py
```

## scripts/bytewax_abductive_blender.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 1244 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/bytewax_abductive_blender.py
```

## scripts/bytewax_chrono_lane_normalizer.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 165 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/bytewax_chrono_lane_normalizer.py
```

## scripts/canonical_graph_write_scanner.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 128 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/canonical_graph_write_scanner.py
```

## scripts/capability_pack_registry.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 154 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/capability_pack_registry.py
```

## scripts/case_workspace.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 110 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/case_workspace.py
```

## scripts/cep_full_e2e.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 327 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/cep_full_e2e.py
```

## scripts/chroma_gliner_bounded_probe.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 231 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/chroma_gliner_bounded_probe.py
```

## scripts/chrono_conservation_verify.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 259 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/chrono_conservation_verify.py
```

## scripts/chrono_freeze_mtime.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 183 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/chrono_freeze_mtime.py
```

## scripts/chrono_lane_split_projection_gate.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 622 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/chrono_lane_split_projection_gate.py
```

## scripts/chrono_phase_c_conservation_report.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 296 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/chrono_phase_c_conservation_report.py
```

## scripts/chrono_queue_event_bridge.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 231 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/chrono_queue_event_bridge.py
```

## scripts/chrono_replay_cursor_validator.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 107 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/chrono_replay_cursor_validator.py
```

## scripts/chrono_snapshot_line_slicer.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 159 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/chrono_snapshot_line_slicer.py
```

## scripts/cohere_chat_cli.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 208 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/cohere_chat_cli.py
```

## scripts/conversation_command_accept_worker.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 259 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/conversation_command_accept_worker.py
```

## scripts/dev_journey_decision_points.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 371 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/dev_journey_decision_points.py
```

## scripts/dev_order_gate.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 113 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/dev_order_gate.py
```

## scripts/dev_order_matrix_wrapper.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 203 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/dev_order_matrix_wrapper.py
```

## scripts/diogenes_memory_gate.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 211 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/diogenes_memory_gate.py
```

## scripts/diogenes_mirror.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 284 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/diogenes_mirror.py
```

## scripts/document_claim_packet_worker.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 261 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/document_claim_packet_worker.py
```

## scripts/document_parse_bakeoff.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 124 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/document_parse_bakeoff.py
```

## scripts/durable_workflow_decision_check.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 158 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/durable_workflow_decision_check.py
```

## scripts/fast_slow_lane_gate.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 358 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/fast_slow_lane_gate.py
```

## scripts/full_system_soak_audit.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 208 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/full_system_soak_audit.py
```

## scripts/gauntlet_state_promoter.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 207 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/gauntlet_state_promoter.py
```

## scripts/generate_bucket_deep_passes.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 730 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/generate_bucket_deep_passes.py
```

## scripts/goal_model_fabric_control.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 101 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/goal_model_fabric_control.py
```

## scripts/goal_scenario_batch.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 310 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/goal_scenario_batch.py
```

## scripts/goal_scenario_compare.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 130 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/goal_scenario_compare.py
```

## scripts/goal_swarm_brief.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 175 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/goal_swarm_brief.py
```

## scripts/golden_path_regression_gate.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 163 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/golden_path_regression_gate.py
```

## scripts/graph_journal_completeness_check.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 110 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/graph_journal_completeness_check.py
```

## scripts/graph_materialization_helper.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 539 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/graph_materialization_helper.py
```

## scripts/graph_promotion_approval_worker.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 326 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/graph_promotion_approval_worker.py
```

## scripts/graph_promotion_dry_run.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 130 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/graph_promotion_dry_run.py
```

## scripts/graph_promotion_execute.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 227 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/graph_promotion_execute.py
```

## scripts/graph_promotion_gate.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 115 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/graph_promotion_gate.py
```

## scripts/graph_promotion_materialize.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 324 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/graph_promotion_materialize.py
```

## scripts/graph_symbol_compare.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 123 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/graph_symbol_compare.py
```

## scripts/graph_symbol_condensation.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 169 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/graph_symbol_condensation.py
```

## scripts/graph_write_blocker_probe.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 122 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/graph_write_blocker_probe.py
```

## scripts/groq_chat_cli.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 218 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/groq_chat_cli.py
```

## scripts/hunch_hypertimeline_audit.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 354 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/hunch_hypertimeline_audit.py
```

## scripts/hunch_postgres_ingest.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 293 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/hunch_postgres_ingest.py
```

## scripts/hypertimeline_engine.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 569 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/hypertimeline_engine.py
```

## scripts/indy_book_learning_pipeline.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 191 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/indy_book_learning_pipeline.py
```

## scripts/indy_reads.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 568 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/indy_reads.py
```

## scripts/indy_reads_safe_books_batch.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 150 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/indy_reads_safe_books_batch.py
```

## scripts/instruction_conflict_scanner.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 114 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/instruction_conflict_scanner.py
```

## scripts/kernel_control_packet.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 144 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/kernel_control_packet.py
```

## scripts/korpus_derived_compute_worker.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 381 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/korpus_derived_compute_worker.py
```

## scripts/korpus_krampii.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 1415 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/korpus_krampii.py
```

## scripts/krampus_rechronologize.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 456 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampus_rechronologize.py
```

## scripts/krampus_time_machine.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 138 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampus_time_machine.py
```

## scripts/krampuschewing_archive_unpack.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 411 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampuschewing_archive_unpack.py
```

## scripts/krampuschewing_chrono_graph.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 251 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampuschewing_chrono_graph.py
```

## scripts/krampuschewing_derived_recovery_gate.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 191 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampuschewing_derived_recovery_gate.py
```

## scripts/krampuschewing_graph_materialize.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 155 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampuschewing_graph_materialize.py
```

## scripts/krampuschewing_graph_stage.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 204 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampuschewing_graph_stage.py
```

## scripts/krampuschewing_large_file_validate.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 263 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampuschewing_large_file_validate.py
```

## scripts/krampuschewing_master_index.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 307 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampuschewing_master_index.py
```

## scripts/krampuschewing_reclassify_index.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 244 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampuschewing_reclassify_index.py
```

## scripts/krampuschewing_recovery_gate.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 154 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampuschewing_recovery_gate.py
```

## scripts/krampuschewing_river_rows.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 212 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampuschewing_river_rows.py
```

## scripts/krampuschewing_role_discovery.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 190 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampuschewing_role_discovery.py
```

## scripts/krampuschewing_watcher.sh

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 180 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/krampuschewing_watcher.sh
```

## scripts/llxprt_project2501.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 337 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/llxprt_project2501.py
```

## scripts/local_model_chat_cli.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 189 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/local_model_chat_cli.py
```

## scripts/lucidota_algos_smoke.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 155 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_algos_smoke.py
```

## scripts/lucidota_anthropic_llama_proxy.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 121 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_anthropic_llama_proxy.py
```

## scripts/lucidota_body_capture_evidence.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 202 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_body_capture_evidence.py
```

## scripts/lucidota_bonsai_ternary_handler.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 201 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_bonsai_ternary_handler.py
```

## scripts/lucidota_brain_ingest.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 270 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_brain_ingest.py
```

## scripts/lucidota_cas_gc.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 108 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_cas_gc.py
```

## scripts/lucidota_cockpit.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 162 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_cockpit.py
```

## scripts/lucidota_current_system_docs.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 286 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_current_system_docs.py
```

## scripts/lucidota_decision_delta.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 147 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_decision_delta.py
```

## scripts/lucidota_drive_import_manifest.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 164 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_drive_import_manifest.py
```

## scripts/lucidota_go_ingest.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 554 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_go_ingest.py
```

## scripts/lucidota_goal_audit.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 245 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_goal_audit.py
```

## scripts/lucidota_graph_nav.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 445 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_graph_nav.py
```

## scripts/lucidota_indy_corpus.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 191 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_indy_corpus.py
```

## scripts/lucidota_indy_lora_train.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 115 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_indy_lora_train.py
```

## scripts/lucidota_ingest_watchdog.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 1134 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_ingest_watchdog.py
```

## scripts/lucidota_mega_gate.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 280 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_mega_gate.py
```

## scripts/lucidota_model_governor.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 264 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_model_governor.py
```

## scripts/lucidota_omni_front_sprint_orchestrator.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 431 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_omni_front_sprint_orchestrator.py
```

## scripts/lucidota_ouroboros_loop.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 261 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_ouroboros_loop.py
```

## scripts/lucidota_pipeline_synthesis.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 323 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_pipeline_synthesis.py
```

## scripts/lucidota_runtime_smoke.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 120 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_runtime_smoke.py
```

## scripts/lucidota_security_quarantine_gate.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 200 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_security_quarantine_gate.py
```

## scripts/lucidota_status_ledger.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 373 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_status_ledger.py
```

## scripts/lucidota_strict_model_stack_admission.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 280 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_strict_model_stack_admission.py
```

## scripts/lucidota_swarm_dashboard.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 423 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_swarm_dashboard.py
```

## scripts/lucidota_swarm_router.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 504 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_swarm_router.py
```

## scripts/lucidota_synthesis_pass.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 345 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/lucidota_synthesis_pass.py
```

## scripts/master_eye_review_worker.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 239 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/master_eye_review_worker.py
```

## scripts/matrix_stream_executor.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 546 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/matrix_stream_executor.py
```

## scripts/matrix_trace_checker.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 212 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/matrix_trace_checker.py
```

## scripts/model_generation_event_bridge.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 306 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/model_generation_event_bridge.py
```

## scripts/model_invocation_audit.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 315 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/model_invocation_audit.py
```

## scripts/model_output_contract_audit.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 134 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/model_output_contract_audit.py
```

## scripts/model_runner_cli.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 169 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/model_runner_cli.py
```

## scripts/mudcrab_merchant_tui.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 149 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/mudcrab_merchant_tui.py
```

## scripts/ncnn_edge_runtime_probe.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 105 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/ncnn_edge_runtime_probe.py
```

## scripts/openai_codex_prompt_guide_ingest.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 143 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/openai_codex_prompt_guide_ingest.py
```

## scripts/operator_ontology_fidelity_guard.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 127 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/operator_ontology_fidelity_guard.py
```

## scripts/percyphon_kernel_bridge.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 116 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/percyphon_kernel_bridge.py
```

## scripts/phase05_design_atom_extractor.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 217 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/phase05_design_atom_extractor.py
```

## scripts/phase05_workflow_blueprint_synthesizer.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 346 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/phase05_workflow_blueprint_synthesizer.py
```

## scripts/project2501_board_move.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 455 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/project2501_board_move.py
```

## scripts/project2501_board_worker.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 578 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/project2501_board_worker.py
```

## scripts/project2501_bytewax_board_stream.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 401 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/project2501_bytewax_board_stream.py
```

## scripts/project2501_bytewax_board_stream_service.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 138 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/project2501_bytewax_board_stream_service.py
```

## scripts/project2501_script_audit_worker.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 468 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/project2501_script_audit_worker.py
```

## scripts/project2501_watch_server.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 766 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/project2501_watch_server.py
```

## scripts/project_brain_doc_authority_check.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 146 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/project_brain_doc_authority_check.py
```

## scripts/proof_kernel.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 338 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/proof_kernel.py
```

## scripts/real_stress_test.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 115 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/real_stress_test.py
```

## scripts/resource_governor.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 459 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/resource_governor.py
```

## scripts/rete_bandit_gate_cli.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 102 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/rete_bandit_gate_cli.py
```

## scripts/rfc_program_check.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 171 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/rfc_program_check.py
```

## scripts/rickshaw_go25_receipt_audit.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 149 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/rickshaw_go25_receipt_audit.py
```

## scripts/run_golden_path_hardening_checks.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 108 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/run_golden_path_hardening_checks.py
```

## scripts/run_instruction_hygiene.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 299 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/run_instruction_hygiene.py
```

## scripts/safe_stress_test.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 231 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/safe_stress_test.py
```

## scripts/same_lineage_validator.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 322 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/same_lineage_validator.py
```

## scripts/script_bucket_manifest.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 215 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/script_bucket_manifest.py
```

## scripts/script_survival_audit.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 245 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/script_survival_audit.py
```

## scripts/signal_aggregator.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 415 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/signal_aggregator.py
```

## scripts/simplemem_candidate_index.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 163 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/simplemem_candidate_index.py
```

## scripts/slop_audit_law.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 263 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/slop_audit_law.py
```

## scripts/slop_jsonl_to_parquet.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 211 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/slop_jsonl_to_parquet.py
```

## scripts/snapshot_slicer.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 184 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/snapshot_slicer.py
```

## scripts/spine_authority_checker.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 240 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/spine_authority_checker.py
```

## scripts/spine_document_parse_worker.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 245 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/spine_document_parse_worker.py
```

## scripts/spine_kernel_authorization.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 180 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/spine_kernel_authorization.py
```

## scripts/spine_krampus_worker.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 301 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/spine_krampus_worker.py
```

## scripts/spine_surface_cep_worker.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 124 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/spine_surface_cep_worker.py
```

## scripts/status_ledger_evidence_gate.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 214 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/status_ledger_evidence_gate.py
```

## scripts/sticker_feature_extractor_v1.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 183 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/sticker_feature_extractor_v1.py
```

## scripts/subsystem_abcd_sweeper.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 253 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/subsystem_abcd_sweeper.py
```

## scripts/subsystem_quality_audit.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 103 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/subsystem_quality_audit.py
```

## scripts/surface_instruction_compile_dry_run.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 279 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/surface_instruction_compile_dry_run.py
```

## scripts/surface_reuse_registry_validator.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 135 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/surface_reuse_registry_validator.py
```

## scripts/surface_sidecar_validator.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 123 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/surface_sidecar_validator.py
```

## scripts/swarm_usage_ledger.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 156 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/swarm_usage_ledger.py
```

## scripts/tickletrunk_scan.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 597 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/tickletrunk_scan.py
```

## scripts/tool_function_bucket_manifest.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 674 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/tool_function_bucket_manifest.py
```

## scripts/updated_abcd_sequence_runner.py

- Severity: `high`
- Source: `active_root_loc_audit`
- Gap: 523 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/updated_abcd_sequence_runner.py
```

## scripts/villager_status.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 175 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/villager_status.py
```

## scripts/working_reality_record.py

- Severity: `medium`
- Source: `active_root_loc_audit`
- Gap: 181 nonblank LOC in active root; >100 LOC must justify itself
- Action: split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper
- Closure gate: active root file <=100 LOC or has written exception with proof

```bash
python3 scripts/slop_audit_law.py --paths scripts/working_reality_record.py
```
