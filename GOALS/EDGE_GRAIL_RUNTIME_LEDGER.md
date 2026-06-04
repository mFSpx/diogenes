# Edge Grail Runtime Ledger — Q1 Bonsai, Needle x6, Treelites

"Save This Prompt, Pass on this Handoff:"

## Current verified runtime state

- **Bonsai Q1_0**: one `llama-server` process on `127.0.0.1:8082`.
  - Model: `03_VAULT/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf`.
  - Alias: `bonsai8b-q1-shared2`.
  - Slots: `2`.
  - KV: `--kv-unified`, `--kv-offload`, `q8_0` K/V cache.
  - GPU: all 37/37 layers offloaded on GTX 1650 in the live proof.

- **Needle x6**: one Python server process on `127.0.0.1:8090`.
  - Model: `needle-26m` checkpoint at `03_VAULT/models/needle/needle.pkl`.
  - Instance: `needle-shared-6`.
  - Slots: `6`.
  - Endpoint: `/generate_batch` accepts up to six lanes in one process.
  - Truth: current implementation shares one process and one weight load, and batches identical-prefix work. Exact tensor-level K/V de-duplication is a runner-extension target because upstream Needle exposes encoder output batching, not an external KV cache pointer API.

- **Treelites found**:
  - AHOY external lab: `/home/mfspx/BOARD_GAMES/AHOY` has 71 compiled Treelite `.so` artifacts, 1 `.tl` artifact, and receipts.
  - LUCIDOTA router: `03_VAULT/router/treelite_router_v0.tl`.
  - Dev journey route family: `05_OUTPUTS/dev_journey_decision_points/models/*.tl`.
  - RunPod FIL proof: `05_OUTPUTS/model_runtime/treelite_fil_residency_all_tl_latest.json` proves cuML FIL/GPU inference for all 103 uploaded `.tl` artifacts (`103/103 PASS`) on RTX PRO 6000. Local GTX 1650 admission remains a separate proof gate.

- **Mamba one-bit-class weight**:
  - Literal public `Q1_0` / 900 MiB target artifact is **not present**.
  - Downloaded closest one-bit-class artifact: `03_VAULT/models/mradermacher/Falcon3-Mamba-7B-Instruct-i1-GGUF/Falcon3-Mamba-7B-Instruct.i1-IQ1_S.gguf`.
  - Size: ~1560 MiB; SHA256 recorded in `03_VAULT/models/mradermacher/Falcon3-Mamba-7B-Instruct-i1-GGUF/model_source.json`.
  - Truth: this is IQ1_S, not literal Q1_0. Current verified ledger fits with less headroom than the original 900 MiB target math.

## KV/cache management policy

Use a **rolling sliding-window chunk ring**, not full flushes every 500 tokens.

Policy:

1. Mamba/stream lane emits overlapping chunk packets, default target chunk width `500` tokens.
2. Maintain a ring of recent chunk IDs with stable content hashes.
3. Needle shared server receives the same chunk payload for all six lanes in one `/generate_batch` call.
4. If the chunk hash matches the previous hot prefix, reuse cached/batched prefix state where the runner supports it.
5. If the chunk hash changes but overlaps the previous window, keep overlap metadata and evict only expired cells/representations.
6. Full flush happens only on document boundary, schema/toolset boundary, corrupted packet, or explicit operator reset.

Why not full flush every 500 tokens:

- It wastes prefill work.
- It erases overlap continuity.
- It turns streaming into repeated cold starts.

Current implementation truth:

- Bonsai already has real llama.cpp unified KV slots.
- Needle now has the correct one-process/six-lane server shape and batched prefix processing.
- Needle still needs a runner-level optimization to expose and reuse identical-prefix encoder/cross-attention K/V tensors directly.
- Treelite/FIL GPU inference is proven on RunPod for all 103 `.tl` artifacts; local 1650 residency is not yet proven.

## Current verified VRAM ledger adjustment

The original target math assumed a ~900 MiB literal Mamba `Q1_0` artifact. Current filesystem truth is IQ1_S at ~1560 MiB.

Current verified one-bit-class ledger:

- Mamba IQ1_S weights: ~1560 MiB.
- Mamba SSM state: ~20 MiB.
- Bonsai Q1 shared weights: ~1000 MiB.
- Bonsai shared KV Q8: ~80 MiB.
- Needle weights total: ~80 MiB.
- Needle shared-prefix KV target: ~5 MiB.
- Treelite FIL array target: ~15 MiB.
- VRAM algo scratchpad: ~150 MiB.
- Active LoRAs: ~50 MiB.
- CUDA context: ~250 MiB.

Total current verified target allocation: ~3210 MiB / 3714 MiB.
Remaining target headroom: ~504 MiB.

The old ~2550 MiB / ~1164 MiB headroom ledger remains a target only if a literal ~900 MiB Q1_0 Mamba artifact is found and proven.

## Technical Summary Review and Dev Notes

The topology is now shaped like the intended creature: Bonsai is a two-headed Q1 VRAM goblin; Needle is one six-mouthed worker instead of six little processes. The remaining cryptid footprint is exact Needle tensor-KV pointer reuse, which needs runner surgery rather than more shell flags.

## Local GTX 1650 admission receipt

Receipt: `05_OUTPUTS/runtime/edge_grail_local_admission_latest.json`.

Current GPU telemetry at receipt time:

- GPU: NVIDIA GeForce GTX 1650.
- Total: 4096 MiB.
- Used: 1264 MiB.
- Free: 2451 MiB.
- Temperature: 56 C.

Admission result for the current verified IQ1_S ledger:

- Needed free VRAM including 450 MiB reserve: 3660 MiB.
- Current free VRAM: 2451 MiB.
- Status: `ADMISSIBLE_AFTER_SPINDOWN`.

Meaning: the corrected Edge Grail ledger fits the GTX 1650 budget after unrelated/resident VRAM users are spun down. It is not currently admitted while 1264 MiB is already occupied.

## Needle KV/source probe

Receipt: `05_OUTPUTS/runtime/needle_kv_probe_latest.json`.

Current Needle source truth:

- Architecture: JAX encoder-decoder transformer.
- Current reusable tensor is `encoder_out` / `enc_mask`, not a llama.cpp-style external KV pointer API.
- Worker source proves one process, one checkpoint load, six slots, and `/generate_batch` batched execution.
- Exact tensor pointer sharing is not yet proven by the current worker.
- Reuse is possible for identical full encoder input because upstream exposes separate encode/decode methods.
- Reuse across different lane tools/tasks requires a runner patch that separates immutable 500-token chunk-prefix encoding from lane-specific task/tool conditioning.

Next runner patch shape:

1. Add `shared_prefix` / `chunk_ref` fields to `/generate_batch`.
2. Tokenize and encode the immutable 500-token chunk once.
3. Condition lane-specific task/tool data outside the shared encoder prefix, or prove identical full encoder input.
4. Broadcast/reuse `encoder_out` / `enc_mask` for exact identical full encoder inputs.
5. Write telemetry: `encode_calls_saved`, `prefix_hash`, `lane_count`, `output_tail_tokens`, `peak_memory`.
