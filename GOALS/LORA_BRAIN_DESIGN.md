# The 5 Brains: LoRA Adapter Architecture

Each brain is a different model with a different neural purpose. LoRAs are trained on specific books to instill specific cognitive capacities. This is the neural ecosystem.

---

## Brain 1: MAMBA 1.4B — "THE SPEED DEMON" 🏃

**Model:** `mamba-1.4b-hf-Q2_K.gguf` (130M params active, ~270 tok/s)

**Role:** Fast reaction, first-pass routing, real-time classification. The reflex arc.

**LoRA target modules:** Mamba SSM has no attention — LoRA on the input/output projections and the SSM convolution kernel. Target: `in_proj`, `out_proj`, `conv1d`, `x_proj`, `dt_proj`.

**What it learns:** Tactical pattern matching — given a text fragment, instantly classify its type, urgency, ontology category. No deep reasoning, just fast categorical judgment.

**Training data:** The Art of War, Common Sense, Communist Manifesto — short, declarative, tactical texts.

**Training objective:** Predict ABBA³ heuristic scores directly from raw text. Learn to output (urgency, category, complexity) in <50ms.

| Config | Value |
|--------|-------|
| Rank `r` | 16 |
| Alpha | 32 |
| Target modules | `in_proj`, `out_proj`, `x_proj`, `dt_proj` |
| Batch size | 4 |
| Learning rate | 2e-4 |
| Max steps | 1000 |
| Loss | MSE on ABBA³ vector (4-dim) |

---

## Brain 2: BONSAI 8B Q2 (1-bit) — "THE PHILOSOPHER" 🧠

**Model:** `Ternary-Bonsai-8B-Q2_0.gguf` (1-bit ternary weights, ~8B params)

**Role:** Deep dialectical reasoning, argument analysis, claim→evidence→conclusion tracking. The cortex.

**LoRA target modules:** All linear projections in the ternary BitNet architecture — `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

**What it learns:** Argument structure — identify premises, conclusions, logical fallacies, counterarguments. Track claim→evidence chains through dense philosophical texts.

**Training data:** The Republic (Plato), On Liberty (Mill), The Social Contract (Rousseau), Discourse on Method (Descartes) — dense philosophical dialogue.

**Training objective:** Given a philosophical passage, produce a structured argument map: `{claims: [...], evidence: [...], conclusions: [...], dialectical_opponent: {...}}`.

| Config | Value |
|--------|-------|
| Rank `r` | 8 |
| Alpha | 16 |
| Target modules | All linear layers (q/k/v/o_proj, gate/up/down_proj) |
| Batch size | 1 (4GB VRAM limit) |
| Learning rate | 1e-4 |
| Max steps | 2000 |
| Loss | Cross-entropy on structured argument tokens |

---

## Brain 3: BONSAI 8B Q1 (1.58-bit) — "THE NARRATIVE MIND" 📖

**Model:** `Bonsai-8B-Q1_0.gguf` (1.58-bit ternary, already running on port 8082)

**Role:** Story understanding, narrative arc detection, character analysis, satire recognition. The right hemisphere.

**LoRA target modules:** Same BitNet linear projections as Brain 2 (q/k/v/o_proj, gate/up/down_proj), but with a different rank distribution favoring deeper layers (for narrative-level representations).

**What it learns:** Narrative structure — character arcs, plot beats, ironic distance, satirical markers, subtext. Detect when a text is being sarcastic, ironic, or playing with genre conventions.

**Training data:** Gulliver's Travels (Swift — satire), Candide (Voltaire — satire/picaresque), Herland (Gilman — utopian satire), A Big Boy Did It and Ran Away (Brookmyre — satirical crime).

**Training objective:** Given a narrative passage, output a narrative graph: `{characters: [...], arcs: [...], genre_signals: [...], ironic_distance: float, subtext: [...]}`.

| Config | Value |
|--------|-------|
| Rank `r` | 12 (deeper layers get r=16) |
| Alpha | 24 |
| Target modules | All linear layers, deeper layers weighted 2x |
| Batch size | 1 |
| Learning rate | 8e-5 |
| Max steps | 2000 |
| Loss | CE on narrative tokens + MSE on ironic_distance scalar |

---

## Brain 4: BITVLA — "THE OBSERVER" 👁️

**Model:** `bitvla-bf16` → quantized ternary at `03_VAULT/models/bitvla-ternary/` (SigLIP + BitNet 2B)

**Role:** Visual-textual grounding — connect written descriptions to visual concepts. The eye.

**LoRA target modules:** The multimodal projector bridge (the `mm_projector` / `multi_modal_projector` layers) + the vision encoder's output projection. This is where vision meets language.

**What it learns:** To "see" through text — given a vivid descriptive passage, generate visual embeddings that match what the image encoder would produce if shown the actual scene. Learn the mapping from descriptive language to visual feature space.

**Training data:** All books with strong visual/descriptive passages — Out of Darkness (essays with visual metaphors), Blood in the Machine (industrial scenes), The Small and the Mighty (portraits of people). Generate training pairs by: (1) extract descriptive paragraphs, (2) use SigLIP to encode any available images, (3) train LoRA to map text embeddings to nearby visual embedding space.

**Training objective:** Minimize cosine distance between `SigLIP(text_desc)` and `SigLIP(related_image)` in the shared embedding space.

| Config | Value |
|--------|-------|
| Rank `r` | 8 (projector is small — don't overfit) |
| Alpha | 16 |
| Target modules | `mm_projector.linear1`, `mm_projector.linear2`, `vision_tower.output_projection` |
| Batch size | 2 |
| Learning rate | 5e-5 |
| Max steps | 500 (projector converges fast) |
| Loss | Cosine embedding loss (margin=0.3) |

---

## Brain 5: RWKV 500M — "THE PATTERN SEER" 🌀

**Model:** RWKV-5/6-world-500M (RNN with transformer-level performance, ~500M params)

**Role:** Sequential pattern detection, authorial voice analysis, rhythm and prosody understanding. The ear.

**LoRA target modules:** RWKV uses time-mix and channel-mix blocks — target the `time_mix`, `channel_mix` linear projections, and the `receptance`, `key`, `value`, `gate` projections within each. Also the output embedding layer for style analysis.

**What it learns:** Stylistic patterns — sentence rhythm, authorial voice, rhetorical devices, sentence embedding similarity across authors. Recognize who wrote something by the rhythm alone.

**Training data:** The Prince (Machiavelli — declarative/imperative), Vindication (Wollstonecraft — passionate/argumentative), One Day Everyone Will Have Always Been Against This (El Akkad — lyrical/modernist), A Death in Malta (investigative/journalistic).

**Training objective:** (1) Author attribution — given a sentence, predict the author. (2) Style embedding — produce embeddings where same-author sentences cluster. (3) Rhetorical device detection — identify anaphora, chiasmus, epistrophe in text.

| Config | Value |
|--------|-------|
| Rank `r` | 16 |
| Alpha | 32 |
| Target modules | `time_mix.*`, `channel_mix.*`, `receptance`, `key`, `value`, `gate` |
| Batch size | 4 |
| Learning rate | 2e-4 |
| Max steps | 1500 |
| Loss | CE on author attribution (n-way classification) + contrastive on style embeddings |

---

## Training Pipeline

```
Books + Odysseus Manual
  → Subtle Knife extraction (DONE: 18 books, 9.9M chars)
  → Per-book classification (which Brain gets which)
  → Training data prep (ABBA³ vectors, argument maps, narrative graphs, text-image pairs, author labels)
  → LoRA training via PEFT/HuggingFace
  → GGUF conversion for llama.cpp/Bonsai compatibility
  → Receipt in 05_OUTPUTS/receipts/
```

Each LoRA config is at `scripts/training/lora_brains/*.json` — ready for `lucidota_indy_lora_train.py` consumption.
