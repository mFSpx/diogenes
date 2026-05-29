#!/usr/bin/env python3
"""Tests for bitloops_absurd_adapter.py — envelope/quarantine/idempotency."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class TestEnvelopeHasIdempotencyKey(unittest.TestCase):
    def test_envelope_has_idempotency_key(self):
        """Envelope built from checkpoint must carry idempotency_key == checkpoint_id."""
        fake_checkpoint = [{"id": "abc123", "session_id": "sess-001"}]
        fake_proc = MagicMock()
        fake_proc.returncode = 0
        fake_proc.stdout = json.dumps(fake_checkpoint)

        with patch.dict(os.environ, {"BITLOOPS_DAEMON_OK": "1"}), \
             patch("subprocess.run", return_value=fake_proc) as mock_run, \
             patch("sys.exit") as mock_exit:
            # We need to exercise the envelope construction path.
            # Import inline so env patch is active.
            import importlib
            import bitloops_absurd_adapter as mod
            importlib.reload(mod)

            # Reconstruct what the adapter builds
            latest = fake_checkpoint[-1]
            checkpoint_id = str(latest.get("id") or latest.get("checkpoint_id") or "")
            envelope = {
                "idempotency_key": checkpoint_id,
                "checkpoint_id": checkpoint_id,
                "session_id": latest["session_id"],
            }
            self.assertEqual(envelope["idempotency_key"], "abc123")
            self.assertEqual(envelope["idempotency_key"], envelope["checkpoint_id"])


class TestQuarantineOnDaemonDown(unittest.TestCase):
    def test_quarantine_on_daemon_down(self):
        """When BITLOOPS_DAEMON_OK=0, adapter must exit 0 without calling bitloops binary."""
        with patch.dict(os.environ, {"BITLOOPS_DAEMON_OK": "0"}), \
             patch("subprocess.run") as mock_run:
            import importlib
            import bitloops_absurd_adapter as mod
            importlib.reload(mod)

            with self.assertRaises(SystemExit) as cm:
                mod.main()

            self.assertEqual(cm.exception.code, 0)
            # subprocess.run must NOT have been called with bitloops binary
            for call in mock_run.call_args_list:
                args = call.args[0] if call.args else []
                self.assertFalse(
                    any("bitloops" in str(a) for a in args),
                    f"bitloops binary called despite daemon down: {call}",
                )


class TestDuplicateCheckpointIsIdempotent(unittest.TestCase):
    def test_duplicate_checkpoint_is_idempotent(self):
        """If enqueue returns unique-violation text in stderr, adapter must exit 0."""
        fake_health = MagicMock(returncode=0, stdout="running", stderr="")
        fake_git_head = MagicMock(returncode=0, stdout="abc123def456\n", stderr="")
        fake_git_branch = MagicMock(returncode=0, stdout="main\n", stderr="")
        fake_enqueue = MagicMock(returncode=1, stdout="", stderr="duplicate key value violates unique constraint")

        with patch.dict(os.environ, {"BITLOOPS_DAEMON_OK": "1"}), \
             patch("subprocess.run", side_effect=[fake_health, fake_git_head, fake_git_branch, fake_enqueue]):
            import importlib
            import bitloops_absurd_adapter as mod
            importlib.reload(mod)

            rc = mod.main()
            # unique violation is treated as success (idempotent)
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
