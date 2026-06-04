#!/usr/bin/env bash
# Talkie-only RunPod forge bootstrap. Does not download Dolphin/Mixtral.
set -euo pipefail

MODEL_ID="${MODEL_ID:-talkie-lm/talkie-1930-13b-it}"
WORKDIR="${WORKDIR:-/workspace/talkie_forge}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$WORKDIR" "$WORKDIR/models" "$WORKDIR/receipts"
cd "$WORKDIR"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

$PYTHON_BIN -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Core custody/download + planned forge tools. mergekit is for MoE upcycling experiments.
python -m pip install \
  huggingface_hub hf_transfer safetensors numpy torch transformers accelerate sentencepiece \
  mergekit

python - <<'PY'
import hashlib, json, os, time
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

model_id = os.environ.get("MODEL_ID", "talkie-lm/talkie-1930-13b-it")
workdir = Path(os.environ.get("WORKDIR", "/workspace/talkie_forge"))
model_dir = workdir / "models" / model_id.replace("/", "__")
model_dir.mkdir(parents=True, exist_ok=True)
api = HfApi()
info = api.model_info(model_id, files_metadata=True)
filename = "rl-refined.pt"
path = Path(hf_hub_download(repo_id=model_id, filename=filename, local_dir=str(model_dir)))
h = hashlib.sha256()
with path.open("rb") as f:
    for chunk in iter(lambda: f.read(64 * 1024 * 1024), b""):
        h.update(chunk)
sha = h.hexdigest()
receipt = {
    "schema": "lucidota.runpod.talkie_source_custody.v1",
    "status": "PASS",
    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model_id": model_id,
    "repo_sha": info.sha,
    "selected_file": filename,
    "path": str(path),
    "size_bytes": path.stat().st_size,
    "sha256": sha,
    "constraint": "Talkie only; Dolphin untouched",
}
(workdir / "receipts" / "talkie_source_custody.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
PY

cat > "$WORKDIR/NEXT_STEPS.md" <<'MD'
# Next steps after source custody

1. Inspect Talkie architecture and activation modules.
2. Record baseline inference memory before surgery.
3. Prototype MoE upcycling scaffold with mergekit or a tiny controlled harness.
4. Patch activation only in an experimental fork/checkpoint.
5. Fine-tune/evaluate with bounded receipts.

Dolphin stays untouched.
MD
