# RFC-080: Local RAM / VRAM / LLM Fabric

Status: DRAFT  
Subject ID: `local_llm_fabric`  
Normative role: defines the sovereign local inference fabric as hardware-truthful, advisory-gated, receipt-backed, and draft-only unless a later authority gate promotes an artifact.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

LUCIDOTA's local model fabric is not “run a huge model and hope.” It is a hardware-aware local inference operating system: small quantized models, explicit RAM/VRAM budgets, advisory governors, provenance requirements, and draft-only output lanes. The point is sovereignty without fantasy.

The local truth in the current registry is a 4GB GTX 1650 environment with llama.cpp and Ollama availability, GGUF inventory, and explicit rejection of unbounded 4GB inference claims. Therefore the correct system behavior is to punch above the hardware by routing work, quantizing expectations, swapping/advising slots, and preserving receipts — not by pretending the hardware is a datacenter.

## 2. Sources

### Local sources

- `00_PROJECT_BRAIN/gpu_model_runtime_registry.json` — detected GPU, 4GB VRAM rule, available GGUF files, feasible-now/later/fantasy lists, provenance requirements.
- `04_RUNTIME/inference_os/` — runtime receipt zone for model/governor plans.
- `scripts/lucidota_model_governor.py` — advisory model governor that observes hardware/model registry and writes decisions without importing weights.
- `pypeline/math/model_vram_scheduler.py` — advisory VRAM/LoRA preemption planner, dual-engine residency planner, and runtime receipt writer.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md` — maps model fabric authority as advisory/extraction/draft authority only.

### External Source Anchors

- llama.cpp's project goal is local/cloud inference with minimal setup across hardware; that supports using llama.cpp as a sovereign local backend rather than a remote service dependency: <https://github.com/ggml-org/llama.cpp/blob/master/README.md>.
- Hugging Face documents GGUF use with llama.cpp; this supports local quantized model files as first-class runtime artifacts: <https://huggingface.co/docs/hub/en/gguf-llamacpp>.
- W3C PROV-O supports recording which agent/activity/model generated which entity; LUCIDOTA maps that to command envelopes, model hashes, prompt/input hashes, and output hashes: <https://www.w3.org/TR/prov-o/>.
- Blueprint-first workflow design supports keeping routing and authority in code/schema instead of hidden model cognition: <https://arxiv.org/abs/2508.02721>.

## 3. Fabric Law

The model fabric MUST obey four laws:

1. **Hardware truth first.** GPU existence is not proof of feasible resident inference. VRAM budget, model size, context, batch, and adapter headroom must be explicit.
2. **Advisory planning before allocation.** Governors and schedulers may plan, score, defer, and receipt; they must not silently import/load weights as a side effect of policy checks.
3. **Local-first sovereignty.** Local GGUF/CPU/GPU paths are preferred. Remote downloads or services require explicit operator acceptance and provenance.
4. **Draft-only output.** Model text is candidate synthesis unless a downstream authority gate promotes an artifact.

## 4. Runtime Shape

Current local shape from the registry/scheduler:

```text
operator/workflow request
  -> model fabric policy check
  -> GPU memory / file receipt / model registry inspection
  -> slot/adaptation plan
  -> model call only when a runner is explicitly invoked
  -> prompt/input hash + model path/hash/device/context profile
  -> draft output hash + receipt
  -> language membrane / template / review gate
```

The scheduler's dual-engine plan is especially important as a design argument: CPU FairyFuse-style conceptual smoothing and GPU Q4 DeepSeek/Mamba-style synthesis/cache lanes can be reasoned about separately. The exact backends may change; the required property is that residency, memory pressure, adapter swapping, and outbound status are explicit.

## 5. Whole-System Interaction

- **Slop Laws:** rejects fantasy model claims and requires receipts for any model-backed artifact.
- **Main Spine:** model output enters as candidate/draft data, not canonical truth.
- **Full ETL/KORPUS:** models consume bounded components, not mystery piles.
- **Diogenes:** model-heavy actions should be command-envelope controlled when they mutate state or spend significant resources.
- **PercyphonAI:** supplies zero-VRAM procedural scaffolding so not every routing/persona problem burns model memory.
- **Language membrane:** weaves model synthesis with deterministic templates and exact quotes, marking outbound text `draft_only`.
- **Indy_READs:** can use local base/adapters for reading style and margin notes while preserving page locks.
- **Constant learning:** LoRA/SONA/River candidates must be planned as adapters or learning records, not untracked mutations.
- **Filesystem law:** models belong in vault/runtime registries, not scattered into active source directories.

## 6. Benefit to the Whole System

This fabric gives LUCIDOTA sovereign computation without self-deception. It can use local models for extraction, draft synthesis, routing, and teammate behavior while still being honest about a 4GB GPU. That honesty creates more capability, not less, because it forces the system to:

- chunk inputs through KORPUS,
- use deterministic lanes where possible,
- call small models when sufficient,
- schedule adapters instead of loading everything,
- preserve prompt/output hashes,
- never confuse model fluency with proof.

The result is an “insanely punch up off the hardware we have” architecture: use every local watt deliberately, keep data sovereign, and make model influence inspectable.

## 7. Correctness Argument

I believe this RFC is correct because the local registry explicitly separates feasible, later, and fantasy states. The model governor and VRAM scheduler are advisory-only in code comments and behavior; they observe GPU/model files and write receipts rather than importing weights. The registry's provenance requirements name exactly the facts needed for replayability: command envelope, model path/hash, device, VRAM profile, input hash, and output hash.

The correctness claim is therefore narrow: LUCIDOTA can safely use local LLMs only if the fabric treats inference as a bounded, receipted runtime event. It is not correct to say the current hardware can host arbitrary large models; it is correct to say the architecture can exploit quantized local models and deterministic non-model organs in a sovereign way.

## 8. Falsifiers

This RFC is wrong if:

- model policy checks load weights or allocate VRAM as hidden side effects,
- runtime outputs lack model/input/output provenance,
- remote model calls happen without explicit policy and receipts,
- the system treats GPU detection as proof of model feasibility,
- model output can directly mutate canonical graph truth,
- deterministic alternatives are bypassed for tasks that do not require a model.

## 9. Filesystem / Runtime Consequences

- Keep model files under vault/runtime model paths, not source-code folders.
- Keep runtime plans/receipts under `04_RUNTIME/inference_os/` or scoped outputs.
- Keep model registry files updated when hardware/model inventory changes.
- Any runner that performs inference must emit model/input/output provenance.
- Governors and schedulers remain advisory unless a separate RFC explicitly grants stronger authority.
- External downloads/service calls require explicit operator policy.
