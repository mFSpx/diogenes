from pathlib import Path

BOOTSTRAP = Path("05_OUTPUTS/runpod/talkie_book_lora/RUNPOD_WEB_TERMINAL_PASTE.sh")
PUBKEY = Path.home() / ".ssh/id_ed25519.pub"


def test_web_terminal_bootstrap_authorizes_local_key_and_downloads_talkie():
    text = BOOTSTRAP.read_text(encoding="utf-8")
    pubkey = PUBKEY.read_text(encoding="utf-8").strip()
    assert pubkey in text
    assert "BEGIN OPENSSH PRIVATE KEY" not in text
    assert "talkie-lm/talkie-1930-13b-it" in text
    assert "hf_hub_download" in text
    assert "rl-refined.pt" in text
    assert "talkie_source_custody.json" in text
    assert "/workspace/talkie_forge" in text
    assert "authorized_keys" in text
    assert "Dolphin" in text and "untouched" in text
