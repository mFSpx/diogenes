# Multi-Model Algorithmic Routing Ledger — Edge + Talkie

"Save This Prompt, Pass on this Handoff:"

## Decision

Yes: make the models work together through **algorithmic routing**, not by pretending every model is always resident in local RAM/VRAM.

Local edge box keeps the fast pressure-hull stack:

- Bonsai / Ternary-Bonsai 8B small quant lanes.
- Needle x6 shared server lane.
- Mamba streaming lane.
- Deterministic algo lattice.
- zram daily swap cushion.
- cgroup memory cages.

Talkie is the deep vintage critic / forge model:

- Source: `talkie-lm/talkie-1930-13b-it`.
- Local file being staged: `03_VAULT/models/talkie-lm/talkie-1930-13b-it/rl-refined.pt`.
- Expected checkpoint size: `26,560,686,563` bytes.
- Truth: this is not an 8 GB local always-resident RAM model. It is a RunPod / cloud / offline forge lane until transformed, quantized, and proven by memory receipts.

## Routing truth

The system can coordinate all lanes if the orchestrator moves **refs, receipts, hashes, paths, and small challenge packets**, not full bodies.

Do not route full transcripts, raw books, raw JSON blobs, full model outputs, or DB result sets through MPSC/HTTP payloads.

Route shape:

```text
operator/event/audio/file
  -> bounded ingress receipt
  -> Mamba/Needle/Bonsai fast lanes
  -> deterministic algo lattice
  -> small GO challenge packet
  -> route decision
       FAST: Bonsai final
       CHECK: Bonsai proposer + Bonsai critic
       STREAM: Mamba/Needle only
       DEEP: Talkie RunPod/cloud critic
       PANIC: deterministic algos only
  -> small receipt + body artifact hash
```

## Local resident profile

The local daily profile may keep:

- zram on.
- disk swap off or tiny emergency only.
- `MemoryHigh` as throttle.
- `MemoryMax` as final cage.
- Bonsai context capped.
- KV cache quantized.
- Needle one-process/six-slot server.
- Mamba as streaming/chunk lane.
- Python algos either safe-inline or subprocess-caged.

The local daily profile must not keep:

- Talkie full checkpoint loaded into RAM.
- Dolphin/Mixtral lane loaded or downloaded.
- all Python algos imported into one sticky daemon.
- unbounded prompt history.
- unbounded DB fetches.

## Talkie lane admission

Talkie can participate now as:

1. Download/staged source custody.
2. RunPod deep critic / model surgery lane.
3. Offline artifact forge for MoE/ReLU/sparse experiments.
4. Future local admission only after quantized/exported artifact proves:
   - RSS,
   - cgroup memory,
   - page faults/sec,
   - tokens/sec,
   - P95 latency,
   - VRAM,
   - thermal state,
   - clean kill behavior.


## Treelite deterministic gate layer

All Treelites are admitted as deterministic routing/scoring organs, not chat models.

Current inventory receipt:

```text
05_OUTPUTS/model_runtime/TREELITE_WHERE_ARE_THEY.md
05_OUTPUTS/model_runtime/treelite_where_are_they_latest.json
```

Current found count: **301** Treelite-ish artifacts.

Current kind counts: `{'shared_object': 71, 'json_receipt': 80, 'other': 47, 'tl': 103}`.

Main burrows:

- AHOY lab: `/home/mfspx/BOARD_GAMES/AHOY/05_OUTPUTS/ahoy/`
- Canonical router: `03_VAULT/router/treelite_router_v0.tl`
- Dev Journey route family: `05_OUTPUTS/dev_journey_decision_points/models/*.tl`
- Scripts/schemas/receipts: `scripts/polycareer_treelite_gate.py`, `scripts/legacy/lucidota_treelite_router.py`, `06_SCHEMA/009_treelite_router.sql`, `05_OUTPUTS/treelite/`


### Loadability receipt — 2026-06-02

Latest proof artifacts:

```text
05_OUTPUTS/model_runtime/TREELITE_LOADABILITY_SUMMARY.md
05_OUTPUTS/model_runtime/treelite_loadability_latest.json
```

Measured local footprint:

- `.tl` total: **2.059 MiB**.
- `.so` total: **6.625 MiB**.
- all Treelite-ish artifacts together: **11.777 MiB**.

Local load proof:

- `.tl` Treelite models deserialized: **103 / 103**.
- AHOY `.so` ELF-like files: **43 / 71**.
- AHOY `.so` files loaded by `ctypes.CDLL`: **43 / 71**.

VRAM truth: these are tiny enough to keep around. The exact path to actual GPU/VRAM execution is a FIL/GPU backend integration step; local proof currently demonstrates Python Treelite deserialization and ELF loadability, not automatic FIL residency.

Routing role:

```text
Needle/Bonsai/Mamba outputs
  -> GO/JSON feature packet
  -> Treelite/FIL-style deterministic gates
  -> certainty / contradiction / topology / route labels
  -> choose FAST | CHECK | STREAM | DEEP | PANIC
```

Truth note: `.tl` artifacts are directly loadable through the Python Treelite runtime already present in `.venv`; some AHOY `.so` artifacts need ELF/loadability verification and should be routed through their receipts until proven loadable.

## Algorithmic routing policy sketch

Route to Talkie only when one of these is true:

- Bonsai proposer/critic disagree.
- Algo lattice returns high uncertainty or contradiction.
- Operator explicitly asks for deep vintage critic.
- The task is historical/style/persona work where Talkie is the right instrument.
- A final answer needs a second-pass deep critique and RunPod is available.

Otherwise keep Talkie asleep.

## Memory doctrine

Swap is a parachute, not a scheduler.

- Daily: zram yes; tiny/emergency disk swap optional.
- Torture: `swapoff -a` only to prove failure modes.
- HTTP/subprocess now; FFI later after receipts.
- mmap is not free RAM.
- Long context is live cache debt.
- Receipts stay small; bodies go to CAS/temp artifacts.

## Technical Summary Review and Dev Notes

The choir can sing together if the conductor passes sheet-music cards, not whole libraries. Talkie is the cathedral organ on RunPod for now; the 8 GB local cryptid carries whistles, needles, Bonsai teeth, Mamba stream, and a very strict pressure gauge.


## Edge Grail Treelite route interface

Runnable bounded route switchyard:

```bash
scripts/edge_grail_treelite_router.py --packet '{"event_id":"demo","input_hash":"...","operator_deep":true}'
```

Contract:

- Input: small GO/JSON feature packet or receipt pointer packet.
- Output: `FAST | CHECK | STREAM | DEEP | PANIC`.
- Body handling: raw body keys are stripped from output; route receipts keep hashes/paths/previews only.
- Treelite evidence: output embeds `treelite_loadability_latest.json` summary.
- Talkie rule: `DEEP` permits Talkie only when operator/policy asks or Treelite external route earns it; `PANIC` suppresses Talkie.

Implementation: `scripts/edge_grail_treelite_router.py`.
Tests: `tests/test_edge_grail_treelite_router.py`.

## Exact topology decision — VRAM/CPU lanes, auxiliary models, and style/audio controls

Operator decision, 2026-06-02: build this as a **spinnable lane topology**, not a fantasy always-loaded pile.

### VRAM daily fast path

Target resident VRAM organs:

1. **Bonsai 8B Q1_0, dual slot/pass mode**
   - Artifact: `03_VAULT/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf`.
   - Runtime truth: one loaded Q1_0 weight file with `--parallel 2`, `--kv-unified`, `--kv-offload`, and quantized KV. This is the correct implementation of “2x 1-bit 8B Bonsai” on a 4 GB card: two proposer/critic slots sharing the resident weights/KV policy, not two duplicate weight copies.

2. **Mamba 7B one-bit-class stream lane, with Q2 fallback**
   - Preferred artifact: `03_VAULT/models/mradermacher/Falcon3-Mamba-7B-Instruct-i1-GGUF/Falcon3-Mamba-7B-Instruct.i1-IQ1_S.gguf`.
   - Fallback artifact: `03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf`.
   - Truth: no explicit public Q1_0 Mamba artifact is currently proven here. IQ1_S is the one-bit-class attempt; Q2_K is the survival fallback if IQ1_S behaves badly.

3. **Needle x6 shared server**
   - Runtime: `scripts/lucidota_needle_worker.py --slots 6`.
   - Current truth: one shared process/weight lane with batched identical-prefix handling and rolling 500-token chunk policy. Exact tensor-level KV pointer sharing remains a future low-level backend surgery.

4. **Treelites in the deterministic gate layer**
   - Inventory: 301 Treelite-ish artifacts; 103/103 `.tl` deserialize; measured total Treelite-ish footprint is under 12 MiB.
   - VRAM truth: tiny enough to keep near the hot path. Actual GPU-resident FIL execution is an integration step, not something to claim until measured.

5. **Four VRAM-candidate algos**
   - Treelite/Hoeffding-Gini route gate.
   - Epistemic certainty receipt gate.
   - Tri-algo conduit ingress gate.
   - Diffusion/LTC/HDC-style scratch lane for short bounded tensors.
   - Truth: the hardened Python/NumPy versions are memory-capped, but “on VRAM” requires CUDA/CuPy/Torch/FIL kernels or C++ bindings per algo. Until ported, they are VRAM-candidate organs with deterministic CPU fallback.

### CPU / deep forge lanes

1. **Ternary Bonsai 8B Q2_0 CPU lane**
   - Artifact: `03_VAULT/models/prism-ml/Ternary-Bonsai-8B-gguf/Ternary-Bonsai-8B-Q2_0.gguf`.
   - Role: CPU fallback / extra critic / cold worker when VRAM is busy or OCR/embed/extract temporarily owns the card.

2. **Talkie double-sparse forge lane**
   - Source: `talkie-lm/talkie-1930-13b-it`.
   - Role: RunPod/deep forge model surgery lane and optional deep critic after proof receipts.
   - Truth: Talkie is not admitted as local always-resident 8 GB RAM until quantized/transformed/exported and measured.

### On-demand auxiliary model lanes

OCR, embedding, extraction, vision, Whisper, and Piper are **wired but not always resident**.

Policy:

- Load only on earned route or operator request.
- Spin down Bonsai slot 2 first if VRAM pressure rises.
- Spin down Mamba next for OCR/vision bursts.
- Keep Needle/Treelite/deterministic gates as long as possible because they are cheap.
- CPU Ternary Bonsai can cover lightweight critique while VRAM is borrowed.
- Audio uses bounded ring buffers and partial transcript caps; never growing strings.

### Whisper + Piper + accent/style lane

Doable and admitted as a bounded output-control lane:

- **Whisper**: ears, on-demand or wake/VAD bounded stream.
- **Piper**: mouth, on-demand TTS worker.
- **Australian accent**: choose/install an Australian English Piper voice when available; otherwise mark voice as missing and fall back explicitly rather than pretending.
- **Style mix**: use algorithmic style controls plus a Jinja template over the final response packet, e.g. `modern_ratio=0.60`, `old_timey_ratio=0.40`, `accent_voice=en_AU_*`.

Truth: “60% modern / 40% old-timey” is a local style-control target, not a mathematically guaranteed linguistic ratio. The enforceable implementation is a bounded style profile with lexical constraints, examples, forbidden excesses, and an evaluator that can retry or downshift when the output drifts.

Suggested style packet:

```json
{
  "style_profile": "modern_60_oldtimey_40",
  "modern_ratio_target": 0.60,
  "old_timey_ratio_target": 0.40,
  "voice": "piper_en_AU_if_installed",
  "max_response_chars": 2000,
  "retry_on_style_drift": true
}
```

### Rust orchestration choice

Yes: **Tokio + Rig.rs is a sane control-plane stack**.

- Tokio handles subprocesses, bounded channels, timers, cancellation, watchdogs, sockets, and backpressure.
- Rig.rs can sit at the LLM/agent/provider abstraction layer for OpenAI-compatible/Groq/Mistral/local HTTP style calls and prompt pipelines.
- Local model engines should remain isolated subprocesses or local HTTP/UDS services first; FFI comes later after survival receipts.
- MPSC events must move refs/receipts/hashes/paths/previews, not raw bodies.

### Spindown law

Yes: models may always be spun down as necessary to run OCR, embedding, extraction, vision, or audio models.

Priority order under VRAM pressure:

1. Drop/stall optional output styling retries.
2. Stop second Bonsai slot / critic pass.
3. Pause Mamba streaming lane.
4. Borrow VRAM for OCR/embed/vision/extract.
5. Fall back to CPU Ternary Bonsai or deterministic algos.
6. Resume VRAM fast path when pressure clears.

Technical Summary Review and Dev Notes: This is a cryptid switchboard, not a monster heap. The little trees stay cheap, the voice gets a hat and an accent, Talkie goes to the deep forge, and Tokio/Rig.rs conduct by passing receipts instead of hauling corpses through channels.

## Sheet Layer, auxiliary admission, Indy boot/comms, and one-loop proof — 2026-06-02

Operator decision: most tasks are **sheet tasks before algorithms**.

### Sheet Layer now exists

Durable artifacts:

```text
06_SCHEMA/147_lucidota_sheet_layer.sql
04_RUNTIME/lucidota_sheet_manifest.json
scripts/luci_sheet.py
```

Applied to `lucidota_state`.

Schemas:

- `lucidota_sheet` — formulas/views/functions/task registry.
- `lucidota_scratch` — unlogged disposable scratch sheets.
- `lucidota_projection` — materialized/cached dashboard tabs.

First sheet objects:

- `lucidota_sheet.sheet_task` with generated formula columns `priority_band`, `route_band`, and `needs_operator`.
- `lucidota_sheet.active_work` live status sheet.
- `lucidota_sheet.next_work_batch` score/action sheet.
- `lucidota_projection.case_pressure_sheet` materialized pivot/projection.
- `lucidota_scratch.route_score_scratch` unlogged route-score scratch table.
- `lucidota_sheet.sheet_refresh_receipt` receipt table.

CLI:

```bash
./luci sheet list --json
./luci sheet show active_work --json
./luci sheet explain next_work_batch --json
./luci sheet refresh case_pressure_sheet --json
./luci sheet export next_work_batch --format csv --json
```

Live DB proof:

```text
psql lucidota_state -f 06_SCHEMA/147_lucidota_sheet_layer.sql
SELECT lucidota_sheet.refresh_case_pressure_sheet(false);
```

Result: refresh returned `PASS`, row_count `0`, duration_ms `10`, and wrote one refresh receipt.

### Auxiliary model admission controller now exists

Durable artifacts:

```text
04_RUNTIME/aux_model_admission_manifest.json
scripts/aux_model_admission.py
```

Policy: embedders, OCR, GLiNER, Whisper, Piper, vision/layout, and heavy Python algos are scheduled tools with cages and budgets, not resident roommates.

Key rules:

- MPSC moves refs, not bodies.
- `SSD_DEEP` and `AUX_MODEL_BURST` are mutually exclusive on the 8 GB box.
- Auxiliary tools default to non-resident, receipt-required, input/output/time/cgroup bounded.
- Talkie/deep forge remains batch/deep/offline unless an earned route admits it.

### Indy_READs boot/comms/speed surfaces now exist

Durable artifacts:

```text
04_RUNTIME/indy_reads_startup_comms_manifest.json
services/lucidota-indy-reads-watcher.service
scripts/indy_reads_comms.py
scripts/luci_speed_probe.py
```

Policy:

- Indy_READs should start on laptop user login via systemd user service.
- Existing watcher script remains `scripts/lucidota_start_indy_reads_watcher.sh`.
- Email to `MaroonedPilot@gmail.com` and `mfspx@proton.me` is **queue-only until operator send approval**.
- Signal is optional and requires `signal-cli` plus operator approval; current local status shows no `signal-cli` binary.
- Preferred lightweight direct chat is local Unix socket or loopback SSE, not external chat by default.
- Indy_READs may help LUCI responses after deterministic route/sheet/model receipt; she does not override receipts.

### One-command whole-loop proof now exists

Durable artifact:

```text
scripts/luci_edge_loop_smoke.py
05_OUTPUTS/runtime/edge_loop_smoke_latest.json
```

Command:

```bash
./luci edge-loop-smoke --receipt 05_OUTPUTS/runtime/edge_loop_smoke_latest.json --json
```

Verified loop:

1. Sheet Layer manifest and routing order.
2. Aux admission for `embedder_onnx_cpu` under budget.
3. Treelite route gate over a refs-only packet.
4. Indy comms status with no external send.
5. LUCI speed probe for `./luci sheet list --json`.

Latest receipt status: `PASS`.

Latest speed sample: `./luci sheet list --json` p95 about `116 ms` against a `2500 ms` budget.

Technical Summary Review and Dev Notes: The seam is now a little operating-system reflex: sheet first, admission second, route third, Indy only with receipts, and one smoke command that proves the tiny beast moves without dragging a whole corpse through the pipe.

## Talkie RunPod readiness + INDY_READs BOOK_READER_LORA staging — 2026-06-02

Operator decision: prioritize Talkie being RunPod-ready and stage **BOOK_READER_LORA** adapters for every Indy_READs book.

### Talkie source custody

Official source cloned:

```text
01_REPOS/talkie
https://github.com/talkie-lm/talkie
```

Custody receipt:

```text
05_OUTPUTS/model_runtime/talkie_source_custody.json
```

Recorded architecture truth from source:

- decoder-only GPT family.
- `n_layer=40`.
- `n_head=40`.
- `n_embd=5120`.
- `head_dim=128`.
- MLP uses SiLU/SwiGLU-style gate.
- Official README says bf16 inference wants CUDA GPU with at least 28 GB VRAM, so local 8 GB hot runtime is false; RunPod/deep forge lane is true.

### Indy_READs book count/chart

Current Indy library count from `scripts.indy_reads.library()`:

- **7 total visible library items**.
- **6 actual book files**.
- **1 markdown context pack**.

By extension:

- `.epub`: 3.
- `.mobi`: 2.
- `.pdf`: 1.
- `.md`: 1.

Chart:

```text
04_RUNTIME/BOOK_READER_LORA/book_chart.json
```

### BOOK_READER_LORA staged dataset

Durable staging root:

```text
04_RUNTIME/BOOK_READER_LORA/
```

Generated files:

```text
04_RUNTIME/BOOK_READER_LORA/book_chart.json
04_RUNTIME/BOOK_READER_LORA/cards/reading_cards.train.jsonl
04_RUNTIME/BOOK_READER_LORA/cards/reading_cards.val.jsonl
04_RUNTIME/BOOK_READER_LORA/adapter/adapter_manifest.json
04_RUNTIME/BOOK_READER_LORA/receipts/stage_receipt.json
scripts/book_reader_lora_stage.py
```

Latest staging receipt:

- cards written: **55**.
- train cards: **44**.
- validation cards: **11**.
- status: `PASS`.

Training doctrine encoded:

```text
LoRA teaches reading behavior, extraction style, chapter mapping, motif detection,
GO-25 packetization, old/modern voice blend, and answer discipline.
Postgres/RAG/CAS keeps exact book text, quotes, page refs, evidence, and citations.
```

The generated card tasks include:

- `go25_packet`.
- `chapter_map`.
- `entity_claim_extract`.
- `motif_questions`.
- `style_voice`.

Truth note: this is staged data, not a trained adapter yet. The adapter manifest status is `STAGED_DATASET_NOT_TRAINED`.

### RunPod pack

Portable RunPod pack built:

```text
05_OUTPUTS/runpod/talkie_book_lora/talkie_book_lora_runpod_pack.tar.gz
05_OUTPUTS/runpod/talkie_book_lora/pack_receipt.json
```

Contains:

- Talkie source custody.
- Talkie-only RunPod bootstrap.
- BOOK_READER_LORA cards.
- adapter manifest.
- training stub.
- next-step instructions.

Dolphin touched: **false**.

### Current Talkie checkpoint status

Local Talkie checkpoint is still downloading:

```text
03_VAULT/models/talkie-lm/talkie-1930-13b-it/rl-refined.pt
```

Latest checked size: about **7.32 GB / 26.56 GB** (**27.576%**). Keep curl running; final checksum/source receipt still pending.

Technical Summary Review and Dev Notes: The RunPod suitcase is packed: Talkie papers, book-reader cards, adapter map, no Dolphin slime. The big checkpoint is still walking in from the rain.
