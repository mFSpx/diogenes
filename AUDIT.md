# AUDIT.md -- llama.cpp (diogenes fork)

**Audit Date:** 2026-06-06

---

## Repository Identity

- **Path:** `/home/mfspx/LUCIDOTA/01_REPOS/llama.cpp`
- **Designated remote:** `https://github.com/mFSpx/diogenes.git` (OUR fork)
- **Status:** NOT a git repository -- no `.git/` directory present
- **Observation:** The source tree is a plain copy/clone with the `.git/` directory removed. This means git history, remote tracking, and branch information are unavailable at this location. Only loose `.git*` files remain (`.gitattributes`, `.gitignore`, `.gitmodules`). The `.gitmodules` file is empty.

---

## Disk Usage & File Count

- **Size:** ~556MB (per user estimate)
- **CUDA build artifacts** (`build-cuda/`): accounts for a significant portion of the size (~450MB)
- **Source code (excl. build-cuda):** ~300MB estimated

---

## Language Breakdown (estimated from file extensions)

| Language | File Extension | Estimated Count |
|----------|---------------|-----------------|
| C | `.c` | ~200 |
| C++ | `.cpp` | ~400 |
| C/C++ Headers | `.h`, `.hpp` | ~300 |
| Python | `.py` | ~30 |
| Shell | `.sh` | ~25 |
| CMake | `CMakeLists.txt`, `.cmake` | ~40 |
| Other | (build artifacts, data, docs, configs) | remainder |

---

## Build System Notes

- **Primary build:** CMake (`CMakeLists.txt`, `CMakePresets.json`)
- **Alternative build:** GNU Make (`Makefile`)
- **Nix flake:** `flake.nix` (no `flake.lock`)
- **CUDA build:** Pre-existing `build-cuda/` directory with compiled artifacts
- **No Poetry:** `poetry.lock` is absent (unlike prismml fork)
- **Docker:** Multiple `*.Dockerfile` in `.devops/` for various backends

---

## Key Differences from Upstream

Compared to the upstream `ggml-org/llama.cpp` baseline, this fork at `diogenes`:

- Has a `build-cuda/` directory with pre-built artifacts
- Has `.pi/gg/SYSTEM.md` -- a Pi generation system prompt file
- Missing `.git/` directory (not a live git repository)

---

## Notable Files

- `CLAUDE.md` -- LUCIDOTA onboarding guidance (updated 2026-06-06 with DSPy reference)
- `AGENTS.md` -- Upstream llama.cpp AI contribution policy (updated 2026-06-06 with DSPy protocol)
- `00_INDEX.md` -- This directory index (created 2026-06-06)
- `.pi/gg/SYSTEM.md` -- Pi generation system prompt (non-upstream addition)
