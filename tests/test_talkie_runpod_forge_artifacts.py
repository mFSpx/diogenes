from pathlib import Path


def test_talkie_forge_doc_encodes_pressure_hull_and_no_dolphin_downloads():
    doc = Path("GOALS/TALKIE_DOUBLE_SPARSE_RUNPOD_FORGE.md")
    text = doc.read_text(encoding="utf-8")
    assert "talkie-lm/talkie-1930-13b-it" in text
    assert "Dolphin stays untouched" in text
    assert "HTTP/subprocess first; FFI later" in text
    assert "MemoryHigh" in text and "MemoryMax" in text and "zram" in text
    assert "No SELECT *" in text


def test_runpod_bootstrap_is_talkie_only_and_does_not_pull_dolphin_or_mixtral():
    script = Path("scripts/runpod_talkie_forge_bootstrap.sh")
    text = script.read_text(encoding="utf-8")
    assert "talkie-lm/talkie-1930-13b-it" in text
    forbidden = ["Dolphin-Mixtral", "dolphin-mixtral", "Mixtral-8x7B", "TheBloke/dolphin"]
    assert not any(term in text for term in forbidden)
    assert "HF_HUB_ENABLE_HF_TRANSFER" in text
    assert "mergekit" in text
