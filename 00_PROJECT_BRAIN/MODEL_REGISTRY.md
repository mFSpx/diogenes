# MODEL REGISTRY — CANONICAL

All model files. Proof-only. Last verified: 2026-05-29.

## GGUF MODELS (llama.cpp runtime)

| Model | File | Size | Status | Role |
|---|---|---|---|---|
| Falcon3-Mamba-7B Q2_K | `03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf` | 2.4G | ✅ | Mamba RAM+GPU |
| Falcon3-Mamba-7B Q3_K_M | `03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q3_K_M.gguf` | 3.1G | ✅ | Mamba GPU (higher quality) |
| Bonsai 4B Q2 | `03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-Q2_0.gguf` | 1.1G | ✅ | Always-on generalist |
| DeepSeek R1 Q4_K_M | `03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` | 1.1G | ✅ | VRAM reasoning (lighter) |
| DeepSeek R1 Q8 | `03_VAULT/models/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf` | 1.8G | ✅ | VRAM reasoning (quality) |
| BGE-M3 Q8 | `04_RUNTIME/models/bge-m3-q8_0.gguf` | 606M | ✅ | Embedding/reranking |

## COMPILED GATES

| Model | File | Size | Status | Role |
|---|---|---|---|---|
| Treelite router v0 | `03_VAULT/router/treelite_router_v0.tl` | 487B | ✅ | Routing gate |

## PKL / NATIVE

| Model | File | Size | Status | Role |
|---|---|---|---|---|
| Needle | `03_VAULT/models/needle/needle.pkl` | 51M | ✅ | Embedding swarm ×6 |
| GLiNER small v2.1 | `03_VAULT/models/gliner/urchade_gliner_small-v2.1/pytorch_model.bin` | 583M | ✅ | NER extraction |

## SAFETENSORS / ONNX (HuggingFace runtime)

| Model | File | Size | Status | Role |
|---|---|---|---|---|
| SmolDocling 256M | `04_RUNTIME/models/smoldocling-256m-preview/model.safetensors` | 490M | ✅ | Doc parsing/OCR |
| SmolDocling ONNX q4 | `04_RUNTIME/models/smoldocling-256m-preview/onnx/decoder_model_merged_q4.onnx` | 83M | ✅ | Doc parsing fast path |
| ModernBERT-base | `04_RUNTIME/models/modernbert-base/model.safetensors` | 571M | ✅ | NLI/classification |

## TERNARY MODELS — VERIFIED
Falcon3-Mamba-7B Q2_K: GGUF magic ✅
Falcon3-Mamba-7B Q3_K_M: GGUF magic ✅
Bonsai 4B Q2: GGUF magic ✅

## llama.cpp COMPATIBILITY
All GGUF files verified valid magic bytes (47475546).
Mamba architecture support: requires llama.cpp build with GGML_MAMBA support.
Verify: `./llama.cpp/build/bin/llama-cli -m <model> --help`
