# Knowledge Card: flywheel1412 ncnn

- ID: `flywheel1412_ncnn`
- Authority class: `research_reference`
- Source: https://gitlab.com/flywheel1412/ncnn
- Local clone: `01_REPOS/flywheel1412_ncnn/`
- Cloned commit: `023dc1821aa7097e0131cca8c1287aa271da16f4`
- License: BSD 3-Clause (`01_REPOS/flywheel1412_ncnn/LICENSE.txt`)

## What it is

This clone is in the Tencent ncnn family: a high-performance neural-network inference framework optimized for mobile and edge deployment. Its main value to LUCIDOTA is not replacing the current GGUF/llama.cpp path blindly; it is an edge/mobile inference candidate with strong CPU/Vulkan, model conversion, benchmarking, and Python-wrapper hooks.

## Learned pattern

1. Deployment-first inference runtime.
   - ncnn is designed for phone/edge deployment, cross-platform builds, no third-party dependencies in the core C++ runtime, and CPU/Vulkan acceleration.

2. Explicit model files and bounded execution.
   - The core example path is param/bin model loading, input `Mat` construction, `Extractor` execution, and named output extraction.
   - This fits LUCIDOTA's bounded-tool rule: explicit model artifact in, deterministic inference call out, receipt around it.

3. Benchmark before belief.
   - `benchmark/benchncnn.cpp` and `benchmark/benchncnn_llm.cpp` are evidence hooks.
   - Any claim that ncnn is useful for a local LUCIDOTA model must be backed by a benchmark receipt, not vibes.

4. Useful edge examples exist.
   - `examples/whisper.cpp` and `examples/piper.cpp` matter for local audio/voice lanes.
   - YOLO/OCR/object examples may matter for document/image intake lanes.

5. Python wrapper is possible but not assumed.
   - `python/README.md` documents pip/source install and zero-copy NumPy interop.
   - Current LUCIDOTA posture should probe importability before depending on it.

## No-brainer LUCIDOTA commitments

- Keep ncnn indexed as an edge/mobile inference candidate in knowledge library + TICKLETRUNK.
- Add a read-only capability probe before any build/install adapter work: `python3 scripts/ncnn_edge_runtime_probe.py --json`.
- Do not make ncnn the core LLM runtime by assumption; llama.cpp/GGUF remains the existing local LLM baseline until concrete ncnn model artifacts and benchmarks exist.
- If used later, wrap ncnn as a deterministic bounded tool with explicit model files, fixed input/output names, and proof receipts.
- Treat Vulkan/CPU/mobile benchmarks as the first real integration target, not a giant adapter.

## Verification / blockers

Passed:

- Clone exists at `01_REPOS/flywheel1412_ncnn`.
- Read-only probe: `scripts/ncnn_edge_runtime_probe.py`.
- Expected reference files include README, LICENSE, CMakeLists, Python README, benchmark docs, LLM benchmark source, Whisper/Piper examples, and PNNX tooling tree.

Not done:

- No submodules initialized.
- No build performed.
- No install performed.
- No model inference performed.
