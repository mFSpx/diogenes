from pathlib import Path

BOOTSTRAP = Path("05_OUTPUTS/runpod/talkie_book_lora/RUNPOD_WEB_TERMINAL_PASTE.sh")


def test_web_terminal_bootstrap_repairs_sshd_and_authorized_keys():
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "/root/.ssh/authorized_keys" in text
    assert "chmod 700 /root/.ssh" in text
    assert "chmod 600 /root/.ssh/authorized_keys" in text
    assert "PubkeyAuthentication yes" in text
    assert "PermitRootLogin" in text
    assert "service ssh start" in text or "/usr/sbin/sshd" in text
    assert "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHIK1ZPtc3m7+7L1vH6H1ROFItLMmd8PhruDH9dRe2oh" in text
    assert "talkie-lm/talkie-1930-13b-it" in text
    assert "nohup python" in text
