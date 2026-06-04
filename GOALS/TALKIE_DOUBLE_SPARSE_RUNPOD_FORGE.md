# Talkie Double-Sparse RunPod Forge — Pressure-Hull First

"Save This Prompt, Pass on this Handoff:"

## Decision

Build the experimental double-sparse lane on **Talkie**, specifically:

- Source model: `talkie-lm/talkie-1930-13b-it`
- Official project/chat surface: `https://talkie-lm.com/chat`
- Official library: `https://github.com/talkie-lm/talkie`

**Dolphin stays untouched.** No Dolphin, no Mixtral giant pull, no Dolphin-Mixtral lane in this forge pass.

## Current source truth

Official Talkie materials describe Talkie as a 13B vintage model trained on pre-1931 English-language text, with `talkie-1930-13b-it` as the instruction-tuned chat/checkpoint. The GitHub README states the Python library can download and run Talkie models, and warns bfloat16 inference wants CUDA GPU VRAM and roughly 26-50 GB disk per model. This is why model surgery belongs on RunPod, not the 8 GB local survival box.

Local staging currently downloads:

```text
03_VAULT/models/talkie-lm/talkie-1930-13b-it/rl-refined.pt
```

Expected source size: `26,560,686,563` bytes.

## Architecture stance

HTTP/subprocess first; FFI later.

Reason: on an 8 GB survival box, blast-radius control beats theoretical nanosecond wins. FFI only happens after the subprocess/HTTP engine survives under cgroup pressure and receipts prove the hot path is worth binding.

## Pressure-hull laws to preserve

1. Daily profile uses `zram`; do not fully kill swap except torture profile.
2. `MemoryHigh` is the main throttle; `MemoryMax` is the last-resort cage.
3. MPSC moves refs, not bodies: event packets carry hashes/paths/byte lengths/previews, not full transcripts.
4. Every edge has byte caps: stdout, stderr, prompt, JSON, audio, rows, files, response tokens.
5. mmap is not free RAM; record major faults, RSS, cgroup memory, page-cache pressure, and tokens/sec.
6. KV cache is live debt; context stays capped and KV quantized.
7. No SELECT *; DB reads must use LIMIT, LEFT(text, N), statement_timeout, low work_mem, and streaming rows.
8. Heavy Python/NumPy/model algos run caged as subprocesses, not imported into one sticky daemon.
9. Receipts are small; bodies live in CAS/temp artifacts.
10. Dolphin/Mixtral giant path remains asleep and out of scope.

## RunPod forge phases

### Phase 0 — Source custody

- Download `talkie-lm/talkie-1930-13b-it`.
- Record repo SHA, selected file, byte size, SHA256, license, and source URLs.
- Do not modify the source checkpoint in place.

### Phase 1 — Inspection

- Identify architecture modules and activation implementation.
- Confirm whether the shipped checkpoint uses SwiGLU or another FFN activation.
- Measure baseline memory with the official Talkie library before any surgery.

### Phase 2 — MoE upcycling prototype

- Use `mergekit` or a small local conversion harness to create an experimental Talkie 4-expert MoE scaffold.
- Experts start as cloned Talkie weights only for proof-of-shape.
- Train/tune only a router adapter first; do not pretend this is a quality model until eval receipts exist.

### Phase 3 — ReLU/sparsity experiment

- Replace/patch FFN activation only in a forked experiment checkpoint.
- Expect quality collapse initially.
- Fine-tune with small vintage/pre-1931 instruction data or Talkie-native data for a bounded step budget.
- Record activation sparsity histograms before claiming microscopic sparsity.

### Phase 4 — Quantize/export

- Export a GGUF or backend-compatible artifact only after phase receipts pass.
- Quantization target is experimental; local 8 GB runtime admission requires cgroup/page-fault receipts.

### Phase 5 — Local admission gate

The local box may only run the forged artifact if a receipt records:

- model file and SHA256
- quantization/backend
- context size
- active RAM window
- major faults/sec
- prefill tokens/sec
- decode tokens/sec
- P95 latency
- RSS and cgroup `memory.current`
- VRAM usage
- thermal state
- kill reason if killed

No receipt, no belief.

## Non-goals

- No Dolphin pull.
- No Mixtral pull.
- No default FFI integration.
- No claim that mmap/PowerInfer makes model weights free.
- No local 8 GB always-on giant model.

## Technical Summary Review and Dev Notes

Talkie is the creature for the RunPod altar. The local box gets the pressure hull; RunPod gets the weird model surgery. No dolphin teeth in this spell circle.
