# TRM 7M — Tiny Recursive Mamba-2 Attention Hybrid Implementation Plan

Saved 2026-06-05. This is the build plan for training a 7M-parameter recursive Mamba-2 model
on LUCIDOTA project data, following Wang & Reid (2026) "Tiny Recursive Reasoning with Mamba-2
Attention Hybrid" (arxiv 2602.12078v2).

## Architecture Spec (from paper, confirmed against implementation)

```
TR-mamba2attn — 6.86M parameters
├── Embedding: vocab × 512
├── Block 1: Mamba-2 (d_model=512, d_state=128, headdim=64, expand=2) + PostNorm(RMSNorm)
├── Block 2: Mamba-2 (d_model=512, d_state=128, headdim=64, expand=2) + PostNorm(RMSNorm)
├── Attention: MultiHeadAttention (d_model=512, n_heads=8, headdim=64) + PostNorm(RMSNorm)
├── MLP: 512 → 2048 → 512 (SwiGLU) + PostNorm(RMSNorm)
└── LM Head: 512 → vocab

Recursion: H=3 outer loops, L=4-6 inner loops
  z_L^{(t+1)} = PostNorm(z_L^{(t)} + F(z_L^{(t)}, z_H^{(t)} + embed(x)))
  z_H^{(t+1)} = PostNorm(z_H^{(t)} + F(z_H^{(t)}, z_L^{(T_L)}))

Post-norm is CRITICAL: bounds residual magnitude. Pre-norm → √t growth → NaN.
```

## Data Inventory (canonical, receipt-backed counts)

### Tier 1 — Structured Strategy (best fit for Mamba-2 causal SSM)

| Source | Rows | Features | Labels | Format |
|--------|------|----------|--------|--------|
| Ahoy strategy rows | 7,537,176 | 95 | 22 | JSONL strategy_sample |
| Ahoy first100k sample | 60,000 | 95 | 22 | JSONL (ready) |
| MTG 17lands picks (AFR/MID/VOW/WOE) | ~12M | 200+ one-hot | win_rate, game_wins | Parquet/CSV |
| River ML candidates | 160,876 | 13 binary + 3 bucket | lane, outcome_guess | JSONL |

### Tier 2 — Conversational/Instruction (fine-tuning)

| Source | Rows | Format |
|--------|------|--------|
| Krampus splits (train/val/test) | 1,701 / 212 / 214 | ChatML [system, user, assistant] |
| gen_codebase.jsonl | 1,829 | ChatML |
| gen_internal_docs.jsonl | 523 | ChatML |
| gen_ornament.jsonl | 237 | ChatML |
| gen_style_perb.jsonl | 170 | ChatML |
| gen_mtg.jsonl | 47 | ChatML |

### Tier 3 — Thick Text (reasoning extraction)

| Source | Rows | Content |
|--------|------|---------|
| BC RTB decisions | 1,685 | Full legal reasoning (issues/background/evidence/conclusion) |
| evidence_labels.csv | 120,465 | fact_id + text + label |
| heaux_sample.jsonl | 977 | Forum text excerpts with behavioral labels |
| WhatsApp chats | 6 JSONL + 6 TXT | Conversational data |
| CanLII / BCFSA | PDFs/DOCX | Legal documents (needs extraction) |

## Phase 1: Training Data Extraction Pipeline

### 1A. Ahoy → Seq2Seq Pairs

Convert strategy rows into model-training format. The Mamba-2 SSM is inherently
causal/sequential, so we frame each game state as a sequence prediction task.

Script: `scripts/trm_ahoy_extract_training.py`

```
Input: strategy_sample.jsonl (95 features + 22 labels per row)
Output: TRM_TRAINING/ahoy/ahoy_train.jsonl, ahoy_val.jsonl, ahoy_test.jsonl

Format (per row):
{
  "id": "ahoy_<game_id>_<turn>",
  "text": "<entity_mapping>\n<95 feature key:value pairs>\nPredict:",
  "labels": {
    "primary_dynamic_label": "attack/defend/positional",
    "winning_entity_role": "ENTITY_AUTHORITY/ENTITY_INSURGENT/ENTITY_OPPORTUNIST",
    "mode_authority": "lockdown/defensive/overextended/none",
    ...
  },
  "features": {...},
  "source": "ahoy_strategy"
}
```

Key design decisions:
- 95 features → natural language key:value pairs (compact, token-efficient)
- 22 labels → multi-task prediction heads (not seq2seq generation for structured labels)
- Game state → next-state prediction (Mamba-2 is causal — perfect fit)
- Train/val/test split by game_id to prevent leakage
- Target: 500K-1M training rows from the 7.5M available

### 1B. MTG 17lands → Draft Decision Sequences

Script: `scripts/trm_mtg_extract_training.py`

```
Input: Parquet chunks (WOE, MID, VOW, AFR)
Output: TRM_TRAINING/mtg/mtg_train.jsonl

Format: Each draft is a sequence of picks.
Pack N, Pick M → card choices (one-hot) + pool state → selected card → outcome

draft_sequence = [
  {"pack": 1, "pick": 1, "options": [card_ids...], "chosen": card_id, "pool": [card_ids...]},
  {"pack": 1, "pick": 2, ...},
  ...
]
```

Mamba-2 advantages for this task:
- Causal SSM naturally models "you can't see future packs"
- Selective state spaces learn to track open colors, curve, synergies
- Attention head handles rare bomb-rares and gold cards

Target: 1M training sequences from ~12M picks

### 1C. River ML Candidates → Lane Classification

Script: `scripts/trm_river_extract_training.py`

Simple classification format from 160K candidates:
- Features (13 binary term flags + file_size_bucket + extension + path_depth) → text key:value
- Labels: lane (DEV_WORK/INVESTIGATIVE_WORK/FILE_ORGANIZATION/PROMPTING)

### 1D. Krampus ChatML → Conversation Pairs

Already well-formatted. Script: `scripts/trm_krampus_extract_training.py`

- Aggregate all gen_*.jsonl + splits/*.jsonl → deduplicated training set
- Add special tokens: `<|system|>`, `<|user|>`, `<|assistant|>`
- Total: ~4,533 chat pairs after dedup

### 1E. Thick Text → Reasoning Extraction

Script: `scripts/trm_thicktext_extract_training.py`

For BC RTB decisions (1,685 records with sections):
```
Input: {"issue": "...", "background": "...", "evidence": "...", "conclusion": "..."}
Output: "Background: <background>\nEvidence: <evidence>\nQuestion: <issue>\nAnswer: <conclusion>"
```

For evidence_labels (120K rows): fact → label classification pairs.
For heaux_sample (977 rows): text → behavioral risk label pairs.

## Phase 2: Model Implementation

### 2A. PyTorch Reference Implementation

File: `01_REPOS/trm/trm_mamba2_attn.py`

Core components to implement:
1. `RMSNorm` — Root mean square layer normalization
2. `Mamba2Block` — Single Mamba-2 block with selective SSM
   - d_model=512, d_state=128, headdim=64, expand=2
   - Uses `causal_conv1d` or pure-PyTorch fallback
3. `AttentionBlock` — Multi-head attention with post-norm
4. `TRMBlock` — Mamba2 → Mamba2 → Attention → MLP, shared weights
5. `TRMRecursiveModel` — Full recursive unrolling with H=3, L=4-6
6. `MultiTaskHead` — 22 structured prediction heads for Ahoy labels

### 2B. Training Loop

File: `01_REPOS/trm/train_trm.py`

- Optimizer: AdamW with weight decay 0.1
- LR schedule: linear warmup (500 steps) → cosine decay
- Batch size: 32-64 sequences (small model, fits GTX 1650 4GB)
- Mixed precision: bf16 (if supported) or fp32
- Gradient accumulation: accumulate over 4 steps if batch < 32
- Loss: cross-entropy for primary label + auxiliary losses for sub-labels
- Recursion: unroll H=3, backprop through all unrolls

### 2C. GGUF Export

Once trained, export to GGUF format for llama.cpp inference:
File: `01_REPOS/trm/export_gguf.py`

## Phase 3: Training Strategy

### Stage 1: Ahoy Strategy (cheapest, most structured)
- 500K-1M strategy rows
- Predict primary_dynamic_label (3-class)
- Also train auxiliary heads for all 22 labels
- Expected: learns game state dynamics via Mamba-2 SSM
- Epochs: 3-5, ~2-3 hours on GTX 1650

### Stage 2: MTG Draft Decisions
- 1M draft sequences
- Fine-tune on card selection in draft context
- Expected: Mamba-2 learns set-specific synergies, color pairs, curve
- Epochs: 1-2, ~4-6 hours on GTX 1650

### Stage 3: Conversational / Instruction
- 4.5K chat pairs + 1.7K RTB reasoning pairs
- Fine-tune on instruction following and reasoning
- Expected: model learns to follow system prompts, extract reasoning chains
- Epochs: 3-5, ~1-2 hours

### Stage 4 (optional): River + Thick Text
- 160K classification + 120K labeling pairs
- Broaden to general text understanding
- Only if Stages 1-3 show positive signal

## Phase 4: Integration With LUCIDOTA

### 4A. Model Registry

Register the trained model in `gpu_model_runtime_registry.json`:
```json
{
  "name": "trm_7m_lucidota",
  "model_path": "04_RUNTIME/models/trm_7m_lucidota.gguf",
  "port": 8090,
  "required": false,
  "device_lane": "system_ram_cpu",
  "switch_group": "custom_trained_models"
}
```

### 4B. LUCI Routing

Add TRM as a bounded processor lane in the routing fabric:
- Local admission check via `aux_model_admission.py`
- Route via `lucidota_model_router.py` when strategy/game reasoning is needed
- Falls back to DeepSeek/Bonsai if TRM not admitted

### 4C. River Online Learning

Wire TRM into River bridge for continual online updates:
- Stream Ahoy game states → TRM prediction → compare with actual outcome → update
- Track drift via `river_bridge.py` OnlineTwin pattern

## Acceptance Criteria

### Phase 1 (Data Extraction)
- [ ] `scripts/trm_ahoy_extract_training.py` produces valid train/val/test splits
- [ ] `scripts/trm_mtg_extract_training.py` produces draft sequences from Parquet
- [ ] `scripts/trm_krampus_extract_training.py` produces deduplicated ChatML pairs
- [ ] `scripts/trm_thicktext_extract_training.py` extracts reasoning pairs from RTB
- [ ] All scripts write receipts to `05_OUTPUTS/trm_training/`
- [ ] Train set ≥ 500K combined examples

### Phase 2 (Model)
- [ ] `01_REPOS/trm/trm_mamba2_attn.py` passes shape checks for all components
- [ ] Forward pass through full recursive unroll (H=3, L=4) produces valid logits
- [ ] Parameter count verified at ~6.86M
- [ ] Post-norm prevents NaN on deep recursion (verified by unit test)

### Phase 3 (Training)
- [ ] Stage 1 converges: train loss < 0.5, val accuracy > baseline
- [ ] Stage 2 fine-tunes without catastrophic forgetting (Ahoy accuracy preserved)
- [ ] Stage 3 instruction following works (qualitative eval on held-out prompts)
- [ ] All training runs write receipts with loss curves

### Phase 4 (Integration)
- [ ] TRM model admitted via `aux_model_admission.py`
- [ ] LUCI can route Ahoy strategy questions to TRM
- [ ] River online learning bridge connected
- [ ] All receipts written to `05_OUTPUTS/trm_training/`

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Mamba-2 selective SSM hard to implement correctly | High | Use existing llama.cpp Mamba-2 C++ as reference; pure-PyTorch for v1, optimize later |
| GTX 1650 4GB too small for training | Medium | Use CPU + llama.cpp finetune.cpp; GGUF training with Q4_K_M quantization |
| Post-norm instability at H > 3 | Medium | Cap at H=3, L=4; add gradient clipping; monitor residual norms |
| Training data too domain-specific | Low | Multi-stage training spreads knowledge; River online learning adapts |
| 7M params too small to learn MTG draft | Medium | Accept "picks colors and curve" as success; full draft is 7B-scale task |

## File Layout

```
01_REPOS/trm/
├── __init__.py
├── trm_mamba2_attn.py        # Model architecture
├── train_trm.py               # Training loop
├── export_gguf.py             # GGUF export
└── test_trm_arch.py           # Architecture unit tests

scripts/
├── trm_ahoy_extract_training.py
├── trm_mtg_extract_training.py
├── trm_krampus_extract_training.py
├── trm_river_extract_training.py
└── trm_thicktext_extract_training.py

05_OUTPUTS/trm_training/
├── ahoy/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── mtg/
│   └── train.jsonl
├── krampus/
│   └── train.jsonl
├── thicktext/
│   └── train.jsonl
└── receipts/
    ├── ahoy_extract_receipt.json
    ├── mtg_extract_receipt.json
    └── ...
```
