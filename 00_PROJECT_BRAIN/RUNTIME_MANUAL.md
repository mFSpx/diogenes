# RUNTIME MANUAL — MODEL FABRIC, RUNPOD, TALKIE, LORA, INGEST

## Scope

This manual covers the runtime lanes that are actually live or staged: model fabric, RunPod/Talkie, LoRA work orders, and the ingest/sheet bridge.

## Current model fabric status

Observed at `05_OUTPUTS/goals/goal_model_fabric_control_20260603T021934Z.json`:

- `deepseek` health: ok, pid alive.
- `mamba_cpu` health: ok, pid alive.
- `needle_0` health: ok, shared server alive in receipt text.
- `bonsai` health endpoint responds; pid metadata is stale/dead in the latest status row.
- GPU lanes may defer when headroom is too small.

Observed decision:

- `decision`: defer
- `loadout_id`: gtx1650-special-forces-v0
- `observed_free_mb`: 887
- `observed_used_mb`: 2828
- `budget_vram_mb`: 4096
- `headroom_mb`: 248
- `estimated_required_mb`: 3336

## Talkie custody

Local custody receipt:

{
  "architecture": {
    "activation": "SwiGLU/Silu gate per source model.py",
    "class": "decoder_only_gpt",
    "head_dim": 128,
    "n_embd": 5120,
    "n_head": 40,
    "n_layer": 40,
    "source_file": "01_REPOS/talkie/src/talkie/model.py",
    "source_sha256": "82df955e44a3a2e96f5b215434dbe7ee9b6a952a60382cf46eda5348b7873255"
  },
  "git_head": "35317ba3a84861a84c84065bd73faf88ad19329c",
  "github_url": "https://github.com/talkie-lm/talkie",
  "license": "                                 Apache License",
  "local_repo": "01_REPOS/talkie",
  "models": {
    "checkpoint_filename": "rl-refined.pt",
    "instruction_tuned": "talkie-lm/talkie-1930-13b-it",
    "local_checkpoint": "03_VAULT/models/talkie-lm/talkie-1930-13b-it/rl-refined.pt"
  },
  "recorded_at": "2026-06-02T19:54:00Z",
  "runtime_truth": {
    "bf16_vram_requirement_gb": 28,
    "local_8gb_hot_runtime": false,
    "note": "Official README requires CUDA GPU with >=28GB VRAM for bf16 inference; local path is source custody/staging, not hot runtime.",
    "runpod_or_forge_lane": true
  },
  "schema": "lucidota.talkie.source_custody.v1"
}

Remote custody receipt:

{
  "constraint": "Talkie only; Dolphin untouched",
  "download_mode": "lean_huggingface_only",
  "model_id": "talkie-lm/talkie-1930-13b-it",
  "path": "/workspace/talkie_forge/models/talkie-lm__talkie-1930-13b-it/rl-refined.pt",
  "recorded_at": "2026-06-02T20:56:05Z",
  "repo_sha": "8033675be6360ae0127fa75f941c12d52064f1dc",
  "schema": "lucidota.runpod.talkie_source_custody.v1",
  "selected_file": "rl-refined.pt",
  "sha256": "c6fd903b67ad312c7537095fa982b0e77735ccb689c8cfc774ca492694e70064",
  "size_bytes": 26560686563,
  "status": "PASS"
}

## LoRA status

The book LoRA work orders are staged and ready for training, but training itself is not complete until an artifact path, dataset manifest, config, hash, and smoke/eval receipt exist.

Current staged targets:

- `talkie`
- `bonsai8b_q1`
- `bonsai8b_q2`

Dataset/work-order anchor:

- `04_RUNTIME/BOOK_READER_LORA/book_lora_work_orders.json`

## Ingest / Treelite / sheet bridge

The runtime bridge is sheet-first, then deterministic routing, then model-heavy lanes. Treelite remains a deterministic gate layer, not a chat model.

## RunPod law

- Do not send giant contexts to remote compaction.
- Poll the bootstrap worker only.
- If it stalls, inspect the bootstrap log and restart only that worker.

## Runtime commands

```bash
.venv/bin/python scripts/goal_model_fabric_control.py status --json
.venv/bin/python scripts/lucidota_model_registry.py
.venv/bin/python scripts/lucidota_model_governor.py --json
python3 scripts/runpod_talkie_control.py probe --force-after-auth-change --json
ssh -o BatchMode=yes -o IdentitiesOnly=yes -p 40100 -i ~/.ssh/id_ed25519 root@213.192.6.98 'cat /workspace/talkie_forge/receipts/lean_talkie_download_start.json'
```

## Runtime truth summary

The model fabric is live, the remote Talkie lane is in custody, the LoRA targets are queued, and the next valid progress is training receipts rather than more planning prose.
