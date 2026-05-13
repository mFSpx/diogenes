# DIOGENES Status

## Current State

- Umbrella workspace renamed to `/home/mfspx/LUCIDOTA`; `/home/mfspx/DIOGENES` remains a compatibility symlink.
- `LUCIDOTA` is the private umbrella project/repo. `doggystyle` remains the kernel-only repo.
- `claudecode` is now treated as the LUCIDOTA-owned interface fork, not an external dependency.
- Repo-level ignore policy covers secrets, credentials, runtime state, vault data, databases, model weights, logs, and build outputs.
- Karpathy/GBRAIN/LLMWiki setup now lives in the repo as project-brain source, not build code.
- Project brain initialized.
- `doggystyle` cloned into `01_REPOS`.
- `claudecode` cloned into `01_REPOS`.
- GitLab CLI installed system-wide.
- Codex global config includes project working directives.
- `THEPLAN.md` saved in project brain.
- Anti-slop audit doctrine added.
- Linux dev toolkit installed from Pop/Ubuntu repos.
- NVIDIA GTX 1650 is visible, idle, and not driving display workload (`Disp.A Off`, 2 MiB VRAM observed).
- CKDOG1 Python package installs in local venv.
- CKDOG1 test suite passes: 45 tests.
- CKDOG1 CLI smoke passes: doctor, genesis, ontology init, soul create, trait set, semantic route, route.
- CKDOG1 gRPC bridge exists and passes live smoke on `127.0.0.1:50051`.
- Rust/Cargo installed system-wide.
- Clawd release build succeeds.
- Clawd exposes `diogenes-smoke`, which starts CKDOG1 locally and calls it through generated Rust tonic/prost gRPC bindings.
- Rust workspace tests pass, including doc-tests.
- Combined LUCIDOTA/DIOGENES harness passes after the Rust tonic bridge and workspace rename.
- Rustdoc dynamic linker path repaired system-wide for the distro Rust package.
- Clawd plugin hook runner now tolerates hook scripts that exit without reading stdin.
- GitLab CLI API auth is not active yet (`glab auth status` reports no token), so private remote creation/push is pending user auth.

## Next Verification

- Create private GitLab repo `LUCIDOTA` after GitLab API auth is available.
- Begin DBOS/Postgres storage design after the kernel/interface bridge stays green.
