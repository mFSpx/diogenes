# DIOGENES Sources

## Repositories

- LUCIDOTA private umbrella repo: https://gitlab.com/mfspx/LUCIDOTA
- doggystyle / CKDOG1 kernel: https://gitlab.com/mfspx/doggystyle
- claudecode / Clawd interface: https://github.com/soongenwong/claudecode
- needle / Cactus Compute function-call router: https://github.com/cactus-compute/needle
- llama.cpp / ggml local inference backend: https://github.com/ggml-org/llama.cpp

## Key Tools / Systems

- DBOS: workflow control layer.
- PostgreSQL 18: database substrate, installed from official PGDG apt repository.
- pgvector: vector storage/search; PGDG package target is `postgresql-18-pgvector`.
- Apache AGE: graph extension for Postgres; PGDG package target is `postgresql-18-age`.
- Bytewax: live dataflow layer.
- River: online machine learning.
- Treelite: tree model deployment/inference.
- Himalaya: CLI email client.
- Apache UIMA / GATE: semantic annotation patterns.
- JSON-LD / RDF / OWL / SKOS / SHACL: semantic web and validation stack.
- Core Linux dev tools installed: `ripgrep`, `fd`, `jq`, `yq`, `fzf`, `bat`, `eza`, `tree`, `tmux`, `direnv`, `shellcheck`, `shfmt`, `hyperfine`, `entr`, `btop`, `nvtop`, `gh`, `git-lfs`, `protobuf-compiler`, `postgresql-client`, `sqlite3`, `cmake`, `ninja`.
- Rust/Cargo installed from Pop/Ubuntu repos; `rustdoc` required `/usr/lib/rustlib/x86_64-unknown-linux-gnu/lib` in `ld.so` config.
- LUCIDOTA/DIOGENES local verifier: `/home/mfspx/LUCIDOTA/check_diogenes.sh`.
- Current Google app surface: Drive and Contacts connectors only. Gmail/Calendar need separate connector or local CLI integration.
- UI asset/theme tooling installed: `chafa`, ImageMagick, `optipng`, `pngquant`, `caca-utils`.

## UI References

- Morrowind menu model: stats, inventory, magic, and map windows are the relevant interaction reference.
- Private-dev boundary: Morrowind-style functionality/code/layouts may live in this repo; literal Bethesda art/font/audio/game assets must stay local/ignored and must not be pushed or shipped.
- OpenMW: FOSS GPL engine reference for Morrowind-style implementation boundaries and MWUI templates.
- Local OpenMW mechanics map: `02_RECORDS_OFFICE/OPENMW_UI_SOURCE_MAP.md`.

## Model Notes

- Needle 26M: Cactus Compute Needle tool/function-calling model.
- Mamba 1.3B: target state-space sequence model family.
- Model source intelligence map: `02_RECORDS_OFFICE/MODEL_SOURCE_INTEL_MAP.md`.
- CUDA/model stack working note: `02_RECORDS_OFFICE/CUDA_MODEL_STACK_V0.md`.
- Missing links architecture note: `02_RECORDS_OFFICE/MISSING_LINKS_COMMAND_CENTER_GOVERNANCE_ORCHESTRATION.md`.

## Current Verified Interfaces

- `ckdog1-grpc`: Python gRPC server entrypoint backed by doggystyle.
- `scripts/diogenes_grpc_smoke.py`: in-process CKDOG1 gRPC smoke harness.
- `claw diogenes-smoke`: Clawd-side command that starts CKDOG1 gRPC locally and calls it through generated Rust tonic/prost client bindings.
- `claw lucidota-status`: Clawd-side build status bar command.
- `scripts/lucidota_cuda_inventory.sh`: local CUDA/model software inventory helper.
- `06_SCHEMA/001_lucidota_control.sql`: control-plane workflow/governance/runtime inventory schema.
- `06_SCHEMA/002_model_runtime.sql`: model candidate, resident loadout, and adapter cartridge registry.
- `scripts/apply_lucidota_control_schema.sh`: applies the control-plane schema to `lucidota_state`.
- `scripts/lucidota_runtime_smoke.py`: verifies runtime imports, CUDA visibility, and control schema.
- `scripts/lucidota_record_runtime_inventory.py`: writes CUDA/Python runtime inventory snapshots to Postgres.
- `scripts/lucidota_model_registry.py`: prints active model loadout registry from Postgres.
- `scripts/lucidota_mps_start.sh`, `scripts/lucidota_mps_stop.sh`: local NVIDIA MPS lifecycle helpers.
- `scripts/build_llama_cuda.sh`: reproducible CUDA build for local `llama-cli` and `llama-server`.
