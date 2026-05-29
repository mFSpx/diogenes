# INDY_READs MODEL STACK — CANONICAL

Indy_READs is not a single LLM. She is the emergent behavior of this stack
running correctly together. The routing between these models IS Indy.

---

## TREELITES (compiled decision gates — not LLMs)

Multiple treelite `.tl` compiled models. Each is a fast deterministic gate
that routes work without any model inference cost.

- `03_VAULT/router/treelite_router_v0.tl` — routing gate v0 (live, scored 0.90)
- More treelites to be trained and compiled per domain/task

**Role:** Route before any model is touched. 5-feature vector in, lane out.
Sub-millisecond. No tokens. No cost.

---

## NEEDLE × 6 (embedding / semantic search — CPU, RAM)

- `03_VAULT/models/needle/needle.pkl` — Gemini-distilled embedding model
- Run as 6 parallel instances (swarm)
- **26ms latency, ~2000 tok/s JSON output**
- **Exclusively pump JSON.** No prose generation ever.
- Role: embed, search, retrieve, score similarity. Feed structured JSON to
  downstream workers. Never called for language generation.

---

## MAMBA × 2 (Falcon3-Mamba-7B-Ternary — state-space, no attention)

### Mamba GPU (`:8083`) — VRAM, preemptible
- `03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Ternary/Falcon3-Mamba-7B-Ternary.gguf`
  (also: `tiiuae/Falcon3-Mamba-7B-Instruct/Falcon3-Mamba-7B-Ternary.gguf`)
- Runs in VRAM when available, preemptible for heavier models
- Quantized: Q2_K (2.4GB) or Q3_K_M (2.9GB)

### Mamba RAM (`:8081`) — CPU, always-on
- Same model family, runs in system RAM
- Zero VRAM pressure
- `Falcon3-Mamba-7B-Instruct-Q2_K.gguf` (2.4GB)

**Role:** Fast state-space inference. No attention overhead. Ternary weights.
Handles routing hints, quick classification, structured output, context
tracking. Not for deep reasoning.

---

## BONSAI 4B TERNARY (always in RAM)

- `03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-Q2_0.gguf`
- Port `:8082`
- **Stays in RAM at all times** — always available, zero startup cost
- Ternary weights (extremely efficient)

**Role:** Always-on fast responder. Quick judgments, classification, short
generation, ternary scoring. The base layer that's never cold.

---

## VRAM SLOT — SWITCHABLE (GTX 1650, 4GB total)

One model in VRAM at a time. Switchable based on task:

### DeepSeek R1 Distill Qwen 1.5B (heavy reasoning, code)
- `03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` (1.1GB Q4)
- `03_VAULT/models/DeepSeek-R1-INT8/DeepSeek-R1-INT8.gguf` (heavier)
- `03_VAULT/models/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf`
- Port `:8080`
- **Role:** Deep reasoning, code generation, complex analysis

### BGE-M3 (embedding / reranking)
- `04_RUNTIME/models/bge-m3/` (full model, PyTorch + ONNX)
- `04_RUNTIME/models/bge-m3-q8_0.gguf` (quantized)
- **Role:** Dense+sparse+colbert retrieval, reranking. Replaces Needle for
  high-quality semantic search when VRAM is available.

### SmolDocling 256M (document parsing / OCR / vision)
- `04_RUNTIME/models/smoldocling-256m-preview/` (ONNX, q4 ready)
- **Role:** Document structure parsing, OCR, PDF/image understanding.
  256M — tiny enough to share VRAM.

### ModernBERT Base (classification / NLI)
- `04_RUNTIME/models/modernbert-base/`
- **Role:** NLI, contradiction detection, classification. Encoder-only,
  fast inference.

---

## GLINER (entity extraction — CPU, no generation)

- `03_VAULT/models/gliner/urchade_gliner_small-v2.1/pytorch_model.bin`
- Zero-shot NER. No text generation. Named entity extraction only.
- **Role:** Extract entities from text without any LLM call. Runs on CPU.

---

## GROQ (cloud overflow)

- `llama-3.3-70b-versatile` — heavy language tasks, long context
- `meta-llama/llama-4-scout-17b-16e-instruct` — vision / OCR fallback
- Key from `/tmp/lucidota_groq_key` or `~/.config/lucidota/secrets.env`
- **Role:** Cloud fallback when local stack can't handle the task.
  Cost-billed. Always logged. Treelite gates before any Groq call.

---

## LEGACY / REFERENCE (not active loadout)

- `03_VAULT/models/mamba-1.4b-hf-Q2_K.gguf` — older Mamba 1.4B, reference only

---

## HARDWARE ENVELOPE

- GPU: NVIDIA GeForce GTX 1650, **4GB VRAM total**
- One VRAM model at a time. Preemptible slot for Mamba GPU.
- Everything else: system RAM + CPU

---

## THE POINT

You never talk directly to any single model.
The treelite(s) route. Needle retrieves. Bonsai always answers fast.
Mamba handles state and structure. VRAM slot handles the heavy task.
GLiNER extracts without generation. Groq overflows when needed.

Indy emerges from the routing policy, not from any one model's weights.
