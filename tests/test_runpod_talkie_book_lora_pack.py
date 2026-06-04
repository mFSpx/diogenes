import json
import tarfile
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/prepare_runpod_talkie_book_lora_pack.py")


def test_prepare_runpod_pack_contains_talkie_and_book_lora_assets(tmp_path):
    out = tmp_path / "pack"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(out), "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads(proc.stdout)
    assert receipt["schema"] == "lucidota.runpod.talkie_book_lora_pack.v1"
    assert receipt["status"] == "PASS"
    assert receipt["dolphin_touched"] is False
    assert receipt["book_count"] == 7
    tar_path = Path(receipt["tarball"])
    assert tar_path.exists()
    with tarfile.open(tar_path, "r:gz") as tf:
        names = set(tf.getnames())
    required_suffixes = {
        "RUNPOD_NEXT_STEPS.md",
        "scripts/runpod_talkie_forge_bootstrap.sh",
        "scripts/runpod_book_reader_lora_train.py",
        "grammar/strict_json.gbnf",
        "grammar/strict_go25.gbnf",
        "generation_profiles/deterministic_json_profile.json",
        "talkie/talkie_source_custody.json",
        "book_lora/book_chart.json",
        "book_lora/chunks/chunks_500tok.jsonl",
        "book_lora/cards/reading_cards.train.jsonl",
        "book_lora/cards/reading_cards.val.jsonl",
        "book_lora/embeddings/embedding_manifest.json",
        "book_lora/adapter/adapter_manifest.json",
        "book_lora/adapter_targets/talkie/adapter_manifest.json",
        "book_lora/adapter_targets/bonsai8b_q1/adapter_manifest.json",
        "book_lora/adapter_targets/bonsai8b_q2/adapter_manifest.json",
    }
    for suffix in required_suffixes:
        assert any(name.endswith(suffix) for name in names), suffix

    with tarfile.open(tar_path, "r:gz") as tf:
        train_member = next(n for n in tf.getnames() if n.endswith("scripts/runpod_book_reader_lora_train.py"))
        train_text = tf.extractfile(train_member).read().decode("utf-8")
        grammar_member = next(n for n in tf.getnames() if n.endswith("grammar/strict_go25.gbnf"))
        grammar_text = tf.extractfile(grammar_member).read().decode("utf-8")
    assert "TARGET=talkie" in train_text
    assert "TARGET=bonsai8b_q1" in train_text
    assert "TARGET=bonsai8b_q2" in train_text
    assert "SFTTrainer" in train_text
    assert "load_in_4bit=True" in train_text
    assert "root" in grammar_text and "object" in grammar_text
