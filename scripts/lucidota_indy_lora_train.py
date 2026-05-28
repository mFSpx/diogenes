#!/usr/bin/env python3
"""Train an INDY_READs LoRA cartridge when a local HF base model is available.

The current resident DeepSeek file is GGUF for inference. PEFT training needs a
Hugging Face model directory/name, supplied as --base-model or
LUCIDOTA_LORA_BASE_MODEL. Without that, this script reports queued/not-trained
instead of pretending weights exist.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARTRIDGE_DIR = ROOT / "04_RUNTIME" / "lora_cartridges"


def find_manifest(adapter_id: str | None, manifest: Path | None) -> Path:
    if manifest:
        return manifest if manifest.is_absolute() else ROOT / manifest
    if not adapter_id:
        raise SystemExit("provide --adapter-id or --manifest")
    path = CARTRIDGE_DIR / adapter_id / "manifest.json"
    if not path.exists():
        raise SystemExit(f"missing cartridge manifest: {path}")
    return path


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def format_example(row: dict[str, str]) -> str:
    return (
        "<|system|>INDY_READs answers from GO primitives and cited chunks only.\n"
        f"<|user|>{row['instruction']}\n\n{row['input']}\n"
        f"<|assistant|>{row['output']}"
    )


def train(manifest_path: Path, base_model: str, epochs: int, max_steps: int, batch_size: int) -> dict[str, object]:
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    train_rows = load_jsonl(ROOT / manifest["train_jsonl"])
    val_rows = load_jsonl(ROOT / manifest["validation_jsonl"])
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto")
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )

    def tokenize(batch):
        texts = [format_example(row) for row in batch["row"]]
        return tokenizer(texts, truncation=True, max_length=1024)

    train_ds = Dataset.from_dict({"row": train_rows}).map(tokenize, batched=True, remove_columns=["row"])
    eval_ds = Dataset.from_dict({"row": val_rows}).map(tokenize, batched=True, remove_columns=["row"]) if val_rows else None
    out_dir = root / "adapter"
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    manifest["training_status"] = "trained"
    manifest["adapter_path"] = str(out_dir.relative_to(ROOT))
    manifest["base_model"] = base_model
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"ok": True, "adapter_id": manifest["adapter_id"], "adapter_path": manifest["adapter_path"], "base_model": base_model}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-id")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--base-model", default=os.environ.get("LUCIDOTA_LORA_BASE_MODEL", ""))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    manifest_path = find_manifest(args.adapter_id, args.manifest)
    if not args.base_model:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = {
            "ok": False,
            "adapter_id": manifest.get("adapter_id"),
            "training_status": "queued",
            "reason": "set LUCIDOTA_LORA_BASE_MODEL or provide --base-model to train weights",
            "manifest": str(manifest_path.relative_to(ROOT)),
        }
    else:
        result = train(manifest_path, args.base_model, args.epochs, args.max_steps, args.batch_size)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)
    return 0 if result.get("ok") or result.get("training_status") == "queued" else 1


if __name__ == "__main__":
    raise SystemExit(main())
