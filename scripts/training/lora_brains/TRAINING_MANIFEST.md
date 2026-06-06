# LoRA Brain Training Manifest

**5 brains, 18 books, 9.9M chars, 1 neural ecosystem.**

---

## Quick Start

```bash
# Train Speed Demon (Mamba 1.4B) — fast tactical routing
python3 scripts/lucidota_indy_lora_train.py --config scripts/training/lora_brains/01_speed_demon.json

# Train Philosopher (Bonsai 8B Q2) — deep dialectical reasoning
python3 scripts/lucidota_indy_lora_train.py --config scripts/training/lora_brains/02_philosopher.json

# Train Narrative Mind (Bonsai 8B Q1) — story/satire understanding
python3 scripts/lucidota_indy_lora_train.py --config scripts/training/lora_brains/03_narrative_mind.json

# Train Observer (BitVLA) — visual-textual grounding
python3 scripts/lucidota_indy_lora_train.py --config scripts/training/lora_brains/04_observer.json

# Train Pattern Seer (RWKV 500M) — authorial voice + rhythm
# First: download RWKV model
# Then: python3 scripts/lucidota_indy_lora_train.py --config scripts/training/lora_brains/05_pattern_seer.json
```

## Model Weights Required

| Brain | Model | Status |
|-------|-------|--------|
| Speed Demon | `mamba-1.4b-hf-Q2_K.gguf` | ✅ Downloaded (130MB) |
| Philosopher | `Ternary-Bonsai-8B-Q2_0.gguf` | ✅ Already on disk |
| Narrative Mind | `Bonsai-8B-Q1_0.gguf` | ✅ Already running on :8082 |
| Observer | `bitvla-ternary` | ✅ Downloaded + quantized |
| Pattern Seer | `RWKV-5-world-500M` | ❌ Need to download |

## Training Data

All 18 books pass through Subtle Knife pipeline. Each brain gets specific books:

| Brain | Books | Training Objective |
|-------|-------|-------------------|
| Speed Demon | Art of War, Common Sense, Manifesto | ABBA³ vector prediction |
| Philosopher | Republic, On Liberty, Social Contract, Discourse | Argument structure extraction |
| Narrative Mind | Gulliver, Candide, Herland, Big Boy Did It | Narrative graph + irony detection |
| Observer | Out of Darkness, Blood in Machine, Small & Mighty | Text→visual embedding alignment |
| Pattern Seer | The Prince, Vindication, One Day Everyone, Death in Malta | Author attribution + style embedding |

## Architecture

```
Training Data (book texts)
  → Per-brain filtering (which chapters/passages)
  → Label generation (ABBA³, argument maps, narrative graphs, etc.)
  → PEFT/LoRA training
  → GGUF conversion (for Bonsai/Mamba/llama.cpp brains)
  → Receipt + adapter save
  → Deployment (adapter loaded alongside base model)
```

## Conversions

- PEFT → GGUF LoRA via `01_REPOS/prismml_llama.cpp/convert_lora_to_gguf.py`
- Mamba LoRAs may need custom export (SSM architecture)
- BitVLA LoRA stays in PyTorch (it's a transformers model)
- RWKV LoRA stays in PyTorch (no GGUF support for RWKV yet)
