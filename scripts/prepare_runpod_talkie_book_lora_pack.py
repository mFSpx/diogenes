#!/usr/bin/env python3
"""Prepare portable RunPod pack for Talkie forge + BOOK_READER_LORA."""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import time
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.book_reader_lora_stage import stage_books


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_train_launcher(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('''#!/usr/bin/env python3
"""RunPod BOOK_READER_LORA QLoRA launcher.

Usage examples:

  TARGET=talkie BASE_MODEL=talkie-lm/talkie-1930-13b-it python scripts/runpod_book_reader_lora_train.py
  TARGET=bonsai8b_q1 BASE_MODEL=<compatible-hf-bonsai-base> python scripts/runpod_book_reader_lora_train.py
  TARGET=bonsai8b_q2 BASE_MODEL=<compatible-hf-bonsai-base> python scripts/runpod_book_reader_lora_train.py

This trains reading behavior cards, not raw full-book continuation. Exact text
stays in RAG/Postgres/receipts. Each target gets a separate adapter directory.
"""
from __future__ import annotations

import json
import os
import time
from hashlib import sha256
from pathlib import Path

TARGET = os.environ.get("TARGET", "talkie")
BASE_MODEL = os.environ.get("BASE_MODEL", "talkie-lm/talkie-1930-13b-it")
PACK_ROOT = Path(os.environ.get("PACK_ROOT", "/workspace/talkie_book_lora/talkie_book_lora_runpod_pack"))
BOOK_LORA = PACK_ROOT / "book_lora"
DATA_DIR = BOOK_LORA / "cards"
MANIFEST = BOOK_LORA / "adapter_targets" / TARGET / "adapter_manifest.json"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", f"/workspace/talkie_book_lora/output/{TARGET}/book_reader_lora"))
RECEIPT_DIR = Path(os.environ.get("RECEIPT_DIR", "/workspace/talkie_book_lora/receipts"))


def sha_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_receipt(status: str, **extra):
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "lucidota.runpod.book_reader_lora.train_receipt.v1",
        "status": status,
        "target": TARGET,
        "base_model": BASE_MODEL,
        "train_path": str(DATA_DIR / "reading_cards.train.jsonl"),
        "val_path": str(DATA_DIR / "reading_cards.val.jsonl"),
        "output_dir": str(OUTPUT_DIR),
        "generated_at": now_z(),
        **extra,
    }
    out = RECEIPT_DIR / f"{TARGET}_train_receipt.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    if TARGET not in {"talkie", "bonsai8b_q1", "bonsai8b_q2"}:
        raise SystemExit(f"unknown TARGET={TARGET}")
    if not MANIFEST.exists():
        raise SystemExit(f"missing target manifest: {MANIFEST}")
    train_path = DATA_DIR / "reading_cards.train.jsonl"
    val_path = DATA_DIR / "reading_cards.val.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit("missing reading card JSONL files")

    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except Exception as exc:
        write_receipt("FAIL", error=f"missing training dependency: {exc}")
        raise

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    lora = manifest.get("recommended_lora", {})
    dataset = load_dataset("json", data_files={"train": str(train_path), "validation": str(val_path)})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("lora_alpha", 32)),
        lora_dropout=float(lora.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=int(os.environ.get("BATCH_SIZE", "1")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "8")),
        learning_rate=float(os.environ.get("LR", "2e-4")),
        num_train_epochs=float(os.environ.get("EPOCHS", "2")),
        logging_steps=10,
        save_steps=int(os.environ.get("SAVE_STEPS", "100")),
        eval_steps=int(os.environ.get("EVAL_STEPS", "100")),
        eval_strategy="steps",
        save_total_limit=2,
        fp16=True,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        args=args,
    )
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    write_receipt(
        "PASS",
        train_sha256=sha_file(train_path),
        val_sha256=sha_file(val_path),
        adapter_config=str(OUTPUT_DIR / "adapter_config.json"),
        adapter_model=str(OUTPUT_DIR / "adapter_model.safetensors"),
        lora_config={"r": peft_config.r, "lora_alpha": peft_config.lora_alpha, "lora_dropout": peft_config.lora_dropout},
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
''', encoding="utf-8")


def write_grammar_assets(staging: Path) -> None:
    grammar_dir = staging / "grammar"
    profile_dir = staging / "generation_profiles"
    grammar_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    strict_json = '''root   ::= object
object ::= "{" ws "}" | "{" ws kv ( "," ws kv )* ws "}"
kv     ::= string ws ":" ws value
string ::= "\\"" char* "\\""
char   ::= [^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
value  ::= string | number | "true" | "false" | "null" | object | array
array  ::= "[" ws "]" | "[" ws value ( "," ws value )* ws "]"
ws     ::= [ \\t\\n\\r]*
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
'''
    (grammar_dir / "strict_json.gbnf").write_text(strict_json, encoding="utf-8")
    (grammar_dir / "strict_go25.gbnf").write_text(strict_json + '''
# Runtime contract: prompt must require keys compatible with GO25/GCI_O_75/O414
# packet schemas. This grammar enforces JSON structure; schema validation runs
# after generation and rejects extra/missing ontology fields.
''', encoding="utf-8")
    (profile_dir / "deterministic_json_profile.json").write_text(json.dumps({
        "schema": "lucidota.generation_profile.deterministic_json.v1",
        "temperature": 0.0,
        "top_p": 1.0,
        "repeat_penalty": 1.2,
        "grammar": "grammar/strict_go25.gbnf",
        "max_tokens": 768,
        "output_contract": "raw JSON only; no preface/postface; validate against GO packet schema after decode",
        "note": "Grammar masks structure; it does not replace schema validation or evidence checks."
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_pack(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    book_lora_root = ROOT / "04_RUNTIME" / "BOOK_READER_LORA"
    required_book_lora = [
        book_lora_root / "chunks" / "chunks_500tok.jsonl",
        book_lora_root / "cards" / "reading_cards.train.jsonl",
        book_lora_root / "embeddings" / "embedding_manifest.json",
        book_lora_root / "adapter_targets" / "talkie" / "adapter_manifest.json",
        book_lora_root / "adapter_targets" / "bonsai8b_q1" / "adapter_manifest.json",
        book_lora_root / "adapter_targets" / "bonsai8b_q2" / "adapter_manifest.json",
    ]
    if not all(p.exists() for p in required_book_lora):
        stage_books(book_lora_root, 1, 5, chunk_tokens=500, max_chunks_per_book=1, embed=True)

    staging = output_dir / "talkie_book_lora_runpod_pack"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # Scripts and instructions.
    copy_file(ROOT / "scripts" / "runpod_talkie_forge_bootstrap.sh", staging / "scripts" / "runpod_talkie_forge_bootstrap.sh")
    write_train_launcher(staging / "scripts" / "runpod_book_reader_lora_train.py")
    write_grammar_assets(staging)
    (staging / "RUNPOD_NEXT_STEPS.md").write_text("""# Talkie + BOOK_READER_LORA RunPod pack

1. Run `scripts/runpod_talkie_forge_bootstrap.sh` to install Talkie forge dependencies and download Talkie IT checkpoint on RunPod.
2. Inspect `talkie/talkie_source_custody.json` before any surgery.
3. Use `book_lora/chunks/chunks_500tok.jsonl` as the exact RAG/CAS source and `book_lora/cards/*.jsonl` for QLoRA/PEFT/TRL SFT training.
4. Train separate adapters for Talkie, Bonsai Q1, and Bonsai Q2:
   - `TARGET=talkie BASE_MODEL=talkie-lm/talkie-1930-13b-it python scripts/runpod_book_reader_lora_train.py`
   - `TARGET=bonsai8b_q1 BASE_MODEL=<compatible HF Bonsai base> python scripts/runpod_book_reader_lora_train.py`
   - `TARGET=bonsai8b_q2 BASE_MODEL=<compatible HF Bonsai base> python scripts/runpod_book_reader_lora_train.py`
5. LoRA teaches reading behavior; exact passages stay in Postgres/RAG/CAS.
6. Evaluate held-out cards before converting/loading an adapter.
7. Use `generation_profiles/deterministic_json_profile.json` + `grammar/strict_go25.gbnf` for machine JSON/GO-packet generation.
8. Dolphin/Mixtral remain untouched.
""", encoding="utf-8")

    # Talkie custody and BOOK_READER_LORA assets.
    copy_file(ROOT / "05_OUTPUTS" / "model_runtime" / "talkie_source_custody.json", staging / "talkie" / "talkie_source_custody.json")
    for rel in [
        "book_chart.json",
        "chunks/chunks_500tok.jsonl",
        "cards/reading_cards.train.jsonl",
        "cards/reading_cards.val.jsonl",
        "embeddings/embedding_manifest.json",
        "embeddings/chunk_embeddings.jsonl",
        "adapter/adapter_manifest.json",
        "ontology_adapter_feasibility_manifest.json",
        "adapter_targets/talkie/adapter_manifest.json",
        "adapter_targets/bonsai8b_q1/adapter_manifest.json",
        "adapter_targets/bonsai8b_q2/adapter_manifest.json",
        "receipts/stage_receipt.json",
    ]:
        copy_file(book_lora_root / rel, staging / "book_lora" / rel)

    chart = json.loads((book_lora_root / "book_chart.json").read_text(encoding="utf-8"))
    tarball = output_dir / "talkie_book_lora_runpod_pack.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(staging, arcname=staging.name)
    receipt = {
        "schema": "lucidota.runpod.talkie_book_lora_pack.v1",
        "status": "PASS",
        "created_at": now_z(),
        "tarball": str(tarball),
        "tarball_sha256": sha_file(tarball),
        "staging_dir": str(staging),
        "book_count": chart.get("book_count", 0),
        "actual_book_file_count": chart.get("actual_book_file_count", 0),
        "context_pack_count": chart.get("context_pack_count", 0),
        "dolphin_touched": False,
        "contains": [
            "Talkie source custody",
            "Talkie-only RunPod bootstrap",
            "BOOK_READER_LORA cards",
            "adapter manifest",
            "training launcher",
            "strict JSON/GO grammar assets",
        ],
    }
    (output_dir / "pack_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(prog="prepare-runpod-talkie-book-lora-pack")
    ap.add_argument("--output-dir", default="05_OUTPUTS/runpod/talkie_book_lora")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    receipt = build_pack(Path(args.output_dir))
    print(json.dumps(receipt, sort_keys=True) if args.json else json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
