# DIOGENES Sources

## Repositories

- LUCIDOTA private umbrella repo: https://gitlab.com/mfspx/LUCIDOTA
- doggystyle / CKDOG1 kernel: https://gitlab.com/mfspx/doggystyle
- claudecode / Clawd interface: https://github.com/soongenwong/claudecode

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

## Model Notes

- Needle 26M: Cactus Compute Needle tool/function-calling model.
- Mamba 1.3B: target state-space sequence model family.

## Current Verified Interfaces

- `ckdog1-grpc`: Python gRPC server entrypoint backed by doggystyle.
- `scripts/diogenes_grpc_smoke.py`: in-process CKDOG1 gRPC smoke harness.
- `claw diogenes-smoke`: Clawd-side command that starts CKDOG1 gRPC locally and calls it through generated Rust tonic/prost client bindings.
