from __future__ import annotations

from pathlib import Path


def test_api_shield_files_block_weighty_corpus_and_model_payloads() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    claudeignore = Path(".claudeignore").read_text(encoding="utf-8")

    for needle in [
        "KRAMPUSCHEWING",
        "models/",
        ".treelite",
        ".gguf",
        ".safetensors",
        ".bin",
        ".parquet",
        ".duckdb",
        ".sqlite",
        ".db",
        ".log",
    ]:
        assert needle in gitignore or needle in claudeignore

