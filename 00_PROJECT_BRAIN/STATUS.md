# DIOGENES Status

## Current State

- Umbrella workspace renamed to `/home/mfspx/LUCIDOTA`; `/home/mfspx/DIOGENES` remains a compatibility symlink.
- `LUCIDOTA` is the private umbrella project/repo. `doggystyle` remains the kernel-only repo.
- `claudecode` is now treated as the LUCIDOTA-owned interface fork, not an external dependency.
- Active long-term goal recorded: get the working sovereign local system done, with verified code over vibes.
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
- Private GitLab remote created and pushed: `https://gitlab.com/mfspx/LUCIDOTA`.
- Google account is connected for Drive/Contacts as `maroonedpilot@gmail.com`; Gmail and Calendar tools are not exposed in this session yet.
- DBOS/Postgres topology note created.
- PostgreSQL 18.3 installed from PGDG; `lucidota_state` and `lucidota_graph` databases created.
- `lucidota_graph` has verified `vector` 0.8.2 and `age` 1.7.0 extension behavior.
- Local Postgres role `mfspx` is a dev superuser so AGE can be loaded from normal dev commands; this is local dev posture, not release policy.
- DBOS 2.21.0 smoke workflow runs against `lucidota_state` and initializes DBOS system schema.
- Private Morrowind-inspired Clawd UI track recorded; private game-derived assets stay local/ignored and are not public release material.
- UI/image tooling installed for private theme work: `chafa`, ImageMagick, `optipng`, `pngquant`, `caca-utils`.

## Next Verification

- Begin DBOS/Postgres storage design after the kernel/interface bridge stays green.
- Add operational Gmail/Calendar tooling; current Google access is not enough for email/calendar assistant work.
