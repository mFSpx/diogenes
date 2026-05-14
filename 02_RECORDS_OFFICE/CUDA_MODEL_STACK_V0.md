# CUDA Model Stack V0

## Status

Working architecture note for the 4GB GTX 1650 local model stack. This records intent and verified hardware facts, and separates them from claims that still need benchmark proof.

## Verified Local Hardware

- GPU: `NVIDIA GeForce GTX 1650`
- VRAM: `4096 MiB`
- Driver: `580.126.18`
- Compute capability: `7.5`
- CUDA toolkit: `12.0.140`
- PCI device: `TU117M [GeForce GTX 1650 Mobile / Max-Q]`
- Display/iGPU: Intel CometLake-H UHD Graphics present.
- `nvidia-cuda-mps-control`: installed at `/usr/bin/nvidia-cuda-mps-control`.
- `nvcc`: not currently installed.

Important correction: this GTX 1650 is Turing-generation compute capability 7.5, but the local device should not be assumed to have useful tensor-core acceleration until directly benchmarked. Design for memory bandwidth, small batches, and low overhead first.

## Target Resident Stack

Current preferred shape:

- 1x Mamba/Mamba2-class listener: long-running stream listener / anomaly compressor.
- 6x Needle-class routers: function-call / JSON routing layer.
- 2x roughly 500M models: mid-tier tacticians / formatters / local reasoning hops.
- 1x roughly 1.5B model: Indy_Reads resident heavy hitter.
- Treelite / River / classical algorithms: decision math and streaming learning.
- CKDOG1 kernel: proof, policy, state transition, gRPC control surface.
- SQLite/Postgres graph/state substrate: memory/provenance, not raw context-window stuffing.

Do not optimize around 35 always-on routers until benchmarks prove a real need. Six fast routers plus richer resident models is the better current hypothesis.

## Needle Intake

Needle source:

- repo: `https://github.com/cactus-compute/needle`
- local source shelf: `01_REPOS/needle`
- cloned revision at intake: `b4a4d5b`
- license: MIT per GitHub repo metadata.
- README claim: 26M parameter function-call model, 6000 tok/s prefill and 1200 decode speed in Cactus production.
- architecture claim: Simple Attention Network, no FFN, encoder-decoder, cross-attention, gated residuals.
- docs claim: tool calling is retrieval-and-assembly; no FFN is viable when external tools/knowledge carry the memory load.

LUCIDOTA interpretation:

- Needle is a router/switchboard candidate.
- It is not the resident book brain.
- It should emit tool calls, DeltaApplyRequest payloads, retrieval intents, and LoRA-load requests.
- It should not be asked to ingest whole books or act as factual long-term memory.

## Software Stack Hypothesis

Bare-metal layer:

- Pop!_OS / Linux.
- NVIDIA driver.
- NVIDIA MPS for CUDA multi-process overlap and reduced context switching, if it behaves on this laptop GPU.
- cgroups/systemd slices for browser/scraper containment.

Inference engines:

- `llama.cpp` / `llama-server` for GGUF-friendly resident models where possible.
- Needle's own Python/JAX/PyTorch path first, then compile/harden only after a local smoke.
- Torch compile / CUDA Graphs only after Needle baseline works and profiling shows Python overhead is the bottleneck.

Orchestration:

- Rust Clawd command/control.
- CKDOG1 Python gRPC kernel.
- DBOS workflow control.
- River for online learning.
- Treelite for compiled tree inference.

Storage:

- short context windows.
- graph/state databases hold memory.
- source cartridges and LoRA metadata are content-addressed.

## LoRA Cartridge Reality

Correct split:

- Needle routes LoRA requests.
- A model with FFNs or suitable adapter insertion points carries LoRA behavior.
- The resident 1.5B-ish model is the likely adapter host.

Do not stream whole books through Needle. Books belong in the ingestion pipeline:

1. hash/provenance.
2. chunk.
3. annotate.
4. embed.
5. graph-link.
6. train/fine-tune/adapt only where lawful and useful.
7. register adapter or retrieval cartridge.

Hot-swapping estimates must be measured. Initial benchmark targets:

- load 1 adapter from warm filesystem cache.
- load 1 adapter from cold disk.
- load 6 adapters sequentially.
- load 6 adapters with prefetch.
- unload and verify VRAM returns.
- measure generation latency before and after adapter activation.

## Predictive Fetching

The play is predictive LoRA/source-cartridge prefetch:

- River watches stream state.
- Needle emits retrieval/adapter intent.
- CKDOG1 records which cartridge was actually used.
- online learner predicts likely next cartridge.
- prefetch warms RAM or VRAM airlock before Indy_Reads needs it.

Do not claim convergence by wall-clock time yet. Record events first, then evaluate:

- number of events observed.
- prediction top-1/top-3 hit rate.
- false prefetch cost.
- missed-prefetch latency.
- memory pressure.
- user-visible benefit.

## Anti-Slop Checks

- Do not trust tok/s claims from remote hardware as local GTX 1650 numbers.
- `llama.cpp` confirms the GTX 1650 lacks tensor cores and may need an MMQ/Pascal-style CUDA build path for better performance.
- Do not assume LoRA training speed; benchmark with a tiny lawful dataset first.
- Do not assume MPS gives hard percentage partitioning on this GPU.
- Do not assume no-KV-cache is free; every backend has its own memory behavior.
- Do not train "book LoRAs" without provenance and rights.
- Do not let model lore override measured local telemetry.

## First Benchmarks To Build

1. Needle install smoke in isolated local path. Done: package imports in LUCIDOTA venv.
2. `needle run` with one simple tool schema.
3. measure CPU-only and CUDA path if available.
4. benchmark MPS on/off with two tiny CUDA workloads.
5. test one GGUF 500M/1.5B resident model through `llama.cpp`.
6. record VRAM before/after each process. Started via `scripts/lucidota_record_runtime_inventory.py`.
7. build a `lucidota model-status` report once numbers exist.

## llama.cpp Local Build

Source shelf:

- `01_REPOS/llama.cpp`
- intake commit: `7f3f843c3`
- vendored LUCIDOTA commit: `fc9160d`

Build helper:

- `scripts/build_llama_cuda.sh`

Current smoke:

- `llama-cli --version`: works.
- `llama-server --version`: works.
- CUDA backend detects the GTX 1650 at compute capability 7.5.
- Warning captured: no tensor cores; investigate `CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual` and `GGML_CUDA_FORCE_MMQ` for this Turing/1650 target.
