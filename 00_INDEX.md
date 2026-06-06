# 00_INDEX.md -- llama.cpp (diogenes fork)

**Repository:** `/home/mfspx/LUCIDOTA/01_REPOS/llama.cpp`
**Remote:** `https://github.com/mFSpx/diogenes.git` (OUR fork)
**Git status:** NOT a git repository (plain source tree, no `.git/` directory present)
**File count:** ~2,645 files (estimate, including build-cuda artifacts)
**Size:** ~556MB

---

## Top-Level Files

| File | Description |
|------|-------------|
| `AGENTS.md` | AI contribution policy (upstream llama.cpp) |
| `AUTHORS` | Project authors list |
| `CLAUDE.md` | Claude Code guidance (LUCIDOTA onboarding + DSPy reference) |
| `CMakeLists.txt` | CMake build definition |
| `CMakePresets.json` | CMake preset configurations |
| `CODEOWNERS` | GitHub code ownership rules |
| `CONTRIBUTING.md` | Contribution guidelines |
| `LICENSE` | License file |
| `Makefile` | GNU Make build definition |
| `README.md` | Project README |
| `SECURITY.md` | Security policy |
| `flake.nix` | Nix flake build definition |
| `mypy.ini` | mypy type checker config |
| `pyproject.toml` | Python project metadata |
| `pyrightconfig.json` | Pyright type checker config |
| `requirements.txt` | Python dependencies |
| `ty.toml` | Type enforcement config |
| `.clang-format` | C/C++ code formatting rules |
| `.clang-tidy` | C/C++ linting rules |
| `.editorconfig` | Editor configuration |
| `.flake8` | Python linter config |
| `.gitattributes` | Git attributes |
| `.gitignore` | Git ignore rules |
| `.gitmodules` | Git submodules (empty) |
| `.dockerignore` | Docker ignore rules |
| `.ecrc` | EditorConfig checker config |
| `build-xcframework.sh` | XCFramework build script |

---

## Directory Tree

### `benches/` -- Benchmarks
- `dgx-spark/` -- DGX Spark benchmark results
- `mac-m2-ultra/` -- Mac M2 Ultra benchmark results
- `nemotron/` -- Nemotron benchmark results

### `build-cuda/` -- CUDA build artifacts (NOT source, ~450MB)

### `ci/` -- CI scripts
- `run.sh`, `README.md`, `README-MUSA.md`

### `cmake/` -- CMake support modules
- `arm64-apple-clang.cmake`, `arm64-linux-clang.cmake`, `arm64-windows-llvm.cmake`
- `x64-windows-llvm.cmake`, `riscv64-spacemit-linux-gnu-gcc.cmake`
- `build-info.cmake`, `common.cmake`, `download-models.cmake`, `git-vars.cmake`, `license.cmake`

### `common/` -- Shared library code
- Core: `common.cpp/h`, `log.cpp/h`, `sampling.cpp/h`, `console.cpp/h`, `debug.cpp/h`
- Chat: `chat.cpp/h`, `chat-auto-parser.cpp/h`, `chat-peg-parser.cpp/h`, `chat-diff-analyzer.cpp`
- Grammar/JSON: `json-schema-to-grammar.cpp/h`, `json-partial.cpp/h`, `regex-partial.cpp/h`
- Jinja: `jinja/` -- Jinja templating engine (lexer, parser, runtime, caps)
- PEG: `peg-parser.cpp/h`, `chat-auto-parser-generator.cpp`
- Other: `arg.cpp/h`, `base64.hpp`, `build-info.h`, `download.cpp/h`, `fit.cpp/h`, `hf-cache.cpp/h`, `http.h`, `llguidance.cpp`, `ngram-cache.cpp/h`, `ngram-map.cpp/h`, `ngram-mod.cpp/h`, `preset.cpp/h`, `reasoning-budget.cpp/h`, `speculative.cpp/h`, `unicode.cpp/h`

### `.devops/` -- Docker and packaging
- Dockerfiles: `cann.Dockerfile`, `cpu.Dockerfile`, `cuda.Dockerfile`, `intel.Dockerfile`, `musa.Dockerfile`, `openvino.Dockerfile`, `rocm.Dockerfile`, `s390x.Dockerfile`, `vulkan.Dockerfile`, `llama-cli-cann.Dockerfile`
- SRPM specs: `llama-cpp.srpm.spec`, `llama-cpp-cuda.srpm.spec`
- Nix: `nix/` (apps, devshells, docker, package, scope, sif, etc.)
- `tools.sh`

### `docs/` -- Documentation
- `android.md`, `autoparser.md`
- `backend/` -- Backend-specific docs (BLIS, OPENVINO, snapdragon, VirtGPU)
- `development/` -- Dev docs (parsing.md, HOWTO-add-model.md, llama-star/)
- `multimodal/`, `ops/`

### `examples/` -- Usage examples
- Core: `batched/`, `embedding/`, `parallel/`, `simple/`, `simple-chat/`, `speculative/`, `speculative-simple/`
- Platforms: `llama.android/` (Android), `llama.swiftui/` (iOS)
- Model: `convert-llama2c-to-ggml/`, `diffusion/`, `model-conversion/`, `training/`
- Tools: `debug/`, `deprecation-warning/`, `eval-callback/`, `gen-docs/`, `gguf/`, `gguf-hash/`, `idle/`, `llama-eval/`, `lookahead/`, `lookup/`, `passkey/`, `retrieval/`, `save-load-state/`, `simple-cmake-pkg/`, `sycl/`

### `.gemini/` -- Gemini configuration

### `ggml/` -- GGML tensor library (core ML backend)
- `cmake/` -- Build support
- `include/` -- Public headers
- `src/` -- Backend implementations:
  - `ggml-blas/`, `ggml-cann/`, `ggml-cpu/`, `ggml-cuda/`, `ggml-hexagon/`, `ggml-hip/`, `ggml-metal/`, `ggml-musa/`, `ggml-opencl/`, `ggml-openvino/`, `ggml-rpc/`, `ggml-sycl/`, `ggml-virtgpu/`, `ggml-vulkan/`, `ggml-webgpu/`, `ggml-zdnn/`, `ggml-zendnn/`

### `gguf-py/` -- GGUF Python library
- `gguf/` -- Main package + `scripts/`
- `examples/`, `tests/`

### `.github/` -- GitHub configuration
- `actions/` -- CI actions (get-tag-name, install-exe, linux-setup-*, unarchive-tar, windows-setup-*)
- `ISSUE_TEMPLATE/` -- Issue form templates
- `workflows/` -- CI workflows (build, test, release, docker, lint, etc.)
- `labeler.yml`, `pull_request_template.md`

### `grammars/` -- Grammar files for constrained generation

### `include/` -- Public C/C++ headers (`llama.h`, `ggml.h`, etc.)

### `licenses/` -- Third-party license files

### `media/` -- Media assets

### `models/` -- Model templates and download scripts
- `templates/`

### `.pi/gg/` -- Pi generation system prompt

### `pocs/` -- Proofs of concept
- `vdot/`

### `requirements/` -- Dependency requirement files (split by platform)

### `scripts/` -- Utility scripts
- `apple/`, `hip/`, `jinja/`, `snapdragon/` (adb, qdc, windows)

### `src/` -- Core llama.cpp source
- `models/` -- Model architecture implementations

### `tests/` -- Test suite
- `peg-parser/`, `snapshots/`

### `tools/` -- Command-line tools
- `batched-bench/`, `cli/`, `completion/`, `cvector-generator/`, `export-lora/`, `fit-params/`, `gguf-split/`, `imatrix/`, `llama-bench/`, `parser/`, `perplexity/`, `quantize/`, `results/`, `rpc/`, `tokenize/`, `tts/`
- `mtmd/` -- Multi-modal tool (debug, legacy-models, models, tests)
- `server/` -- HTTP server (bench, public/static, tests, webui)

### `vendor/` -- Third-party vendored dependencies
- `cpp-httplib/`, `miniaudio/`, `nlohmann/` (JSON), `sheredom/`, `stb/`

---

## Build System

- **Primary:** CMake (`CMakeLists.txt` + `CMakePresets.json`)
- **Alternative:** GNU Make (`Makefile`)
- **Nix:** `flake.nix`
- **Docker:** `.devops/*.Dockerfile`
- **CUDA build:** `build-cuda/` directory (pre-existing build artifacts)
- **XCFramework:** `build-xcframework.sh`
