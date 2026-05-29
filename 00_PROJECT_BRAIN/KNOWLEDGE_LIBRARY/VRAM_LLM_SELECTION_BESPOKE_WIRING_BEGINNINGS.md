# Knowledge Card: VRAM LLM Selection — Bespoke and Wiring Beginnings

Authority class: `research_reference` / staged runtime design stub.
Status: staged, not applied.

## Purpose

Capture the operator's bespoke GTX 1650 / 4GB headless VRAM loadout idea as a durable design stub and staging ledger, without silently activating the runtime configuration.

This card is about **model selection and residency discipline**, not truth promotion and not runtime activation.

## Source anchors

- Falcon3-Mamba-7B-Instruct model card: https://huggingface.co/tiiuae/Falcon3-Mamba-7B-Instruct
- Falcon3-Mamba-7B-Instruct GGUF staged fallback: https://huggingface.co/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF
- Ternary Bonsai 4B GGUF: https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf
- DeepSeek-R1-Distill-Qwen-1.5B GGUF source used locally: https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
- Needle router source recorded historically: https://huggingface.co/Cactus-Compute/needle
- Local staging manifest: `04_RUNTIME/inference_os/vram_llm_selection_bespoke_staging.json`
- Staging receipt: latest under `05_OUTPUTS/model_staging/vram_llm_selection_bespoke_staging_*.json`

## Operator design intent

Target conceptual resident stack:

| Lane | Intended role | Target footprint | Residency intent |
|---|---|---:|---|
| Big Mamba 7B | supreme stream watcher / semantic route brain | ~1.45GB if native 1.58-bit ternary exists | always locked |
| Ternary Bonsai 4B | compact logic/model organ | ~0.86-1.07GB observed GGUF candidate | always locked if governor allows |
| Needle swarm x6 | tiny tool/function routers | ~180MB conceptual / ~50MB checkpoint observed | always locked |
| DeepSeek R1 Qwen 1.5B Q4_K_M | small reasoning lane | ~1.1GB weights | always locked if KV capped |
| CUDA/context/KV/LoRA reserve | runtime overhead | strict explicit reserve | never hand-waved |

## Critical correction from staging

The design math is excellent as a **target**, but the exact public artifact matters:

- A true native packed 1.58-bit 7B Mamba at ~1.38-1.45GB was **not found** during this staging pass.
- The closest staged Falcon3-Mamba-7B-Instruct GGUF candidate is `Q2_K`, not native ternary, and is **2.565GB**.
- Therefore the fully resident 4GB stack is **not proven viable with the staged Big Mamba file**.
- If a real ~1.45GB 1.58-bit Mamba artifact and compatible BitNet/Triton kernels are identified, the design should be re-evaluated by `lucidota_model_governor.py` before activation.

## Staged artifacts

| Artifact | Local path | Status |
|---|---|---|
| Falcon3-Mamba-7B-Instruct Q2_K fallback | `03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf` | downloaded, hash-verified, **not exact ternary** |
| Ternary Bonsai 4B Q2_0 | `03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-Q2_0.gguf` | already present, hash-verified |
| DeepSeek R1 Distill Qwen 1.5B Q4_K_M | `03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` | already present, hash-verified to bartowski source |
| Needle checkpoint | `03_VAULT/models/needle/needle.pkl` | already present, hash-recorded |

## Runtime guardrails before activation

1. Do not activate this config from prose.
2. Require a model-governor receipt with observed `nvidia-smi` state, exact model paths, hashes, estimated resident MB, reserve MB, and decision.
3. Treat Falcon3-Mamba Q2_K as a **staged candidate**, not as proof of the 1.45GB ternary Big Mamba lane.
4. Cap transformer attention contexts and KV cache explicitly for DeepSeek/Bonsai lanes.
5. Keep LoRA/adapters below 15-20MB until measured.
6. One model loadout cannot claim full residency unless measured VRAM, CUDA context, KV/cache, and adapter overhead fit with reserve.
7. If the exact native ternary 7B Mamba artifact appears, add a `model_source.json`, hash receipt, and governor dry-run before launch.

## Adoption stance

Adopt the **design pressure** and **residency discipline** immediately. Do not adopt the full simultaneous residency claim until the missing native 1.58-bit Big Mamba artifact and runtime kernels are real and measured.

