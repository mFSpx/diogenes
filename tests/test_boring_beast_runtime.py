import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("boring_beast", ROOT / "scripts" / "boring_beast.py")
bb = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bb)


class BoringBeastRuntimeUnitTests(unittest.TestCase):
    def test_01_idempotency_normalization(self):
        self.assertEqual(bb.normalize_idempotency_key("  Hello   World!  "), "hello-world")

    def test_02_work_order_accepts_valid_target(self):
        ok, errors = bb.validate_work_order({"target_number": 1, "target_name": "ABSURD", "idempotency_key": "A"})
        self.assertTrue(ok, errors)

    def test_03_work_order_rejects_target_out_of_range(self):
        ok, errors = bb.validate_work_order({"target_number": 99, "target_name": "bad", "idempotency_key": "bad"})
        self.assertFalse(ok)
        self.assertIn("target_number_must_be_1_to_20", errors)

    def test_04_work_order_requires_target_name(self):
        ok, errors = bb.validate_work_order({"target_number": 3, "idempotency_key": "bad"})
        self.assertFalse(ok)
        self.assertIn("target_name_required", errors)

    def test_05_work_order_requires_idempotency(self):
        ok, errors = bb.validate_work_order({"target_number": 5, "target_name": "dup"})
        self.assertFalse(ok)
        self.assertIn("idempotency_key_required", errors)

    def test_06_work_order_rejects_bad_files_changed_type(self):
        ok, errors = bb.validate_work_order({"target_number": 13, "target_name": "oracle", "idempotency_key": "x", "files_changed": "nope"})
        self.assertFalse(ok)
        self.assertIn("files_changed_must_be_list", errors)

    def test_07_work_order_rejects_bad_validation_commands_type(self):
        ok, errors = bb.validate_work_order({"target_number": 7, "target_name": "records", "idempotency_key": "x", "validation_commands": "nope"})
        self.assertFalse(ok)
        self.assertIn("validation_commands_must_be_list", errors)

    def test_08_work_order_rejects_unsupported_handler(self):
        ok, errors = bb.validate_work_order({"target_number": 4, "target_name": "worker", "idempotency_key": "x", "handler": "rm_rf"})
        self.assertFalse(ok)
        self.assertIn("unsupported_handler", errors)

    def test_09_sha256_obj_is_stable(self):
        self.assertEqual(bb.sha256_obj({"b": 2, "a": 1}), bb.sha256_obj({"a": 1, "b": 2}))

    def test_10_load_payload_from_json(self):
        self.assertEqual(bb.load_payload('{"x":1}'), {"x": 1})

    def test_11_load_payload_rejects_list(self):
        with self.assertRaises(ValueError):
            bb.load_payload('[1,2,3]')

    def test_12_handle_noop_succeeds(self):
        ok, result, error, files = bb.handle_work_item({"handler": "noop", "files_changed": ["a.py"]}, "job", "idem")
        self.assertTrue(ok)
        self.assertEqual(error, "")
        self.assertEqual(files, ["a.py"])

    def test_13_handle_unsupported_fails(self):
        ok, result, error, files = bb.handle_work_item({"handler": "bad"}, "job", "idem")
        self.assertFalse(ok)
        self.assertEqual(error, "unsupported_handler")

    def test_14_handle_fail_once_fails_when_requested(self):
        ok, result, error, files = bb.handle_work_item({"handler": "fail_once", "fail": True}, "job", "idem")
        self.assertFalse(ok)
        self.assertIn("forced_failure", error)

    def test_15_parser_contains_e2e(self):
        parser = bb.build_parser()
        ns = parser.parse_args(["e2e", "--execute"])
        self.assertEqual(ns.cmd, "e2e")

    def test_16_parser_contains_transition_check(self):
        parser = bb.build_parser()
        ns = parser.parse_args(["transition-check"])
        self.assertEqual(ns.cmd, "transition-check")

    def test_17_tracer_labels_include_pfm(self):
        self.assertIn("PFM", bb.TRACER_LABELS)

    def test_18_authority_includes_operator_authored(self):
        self.assertIn("operator_authored_assertion", bb.ALLOWED_AUTHORITY)

    def test_19_legal_targets_are_1_to_20(self):
        self.assertEqual(bb.LEGAL_WORK_TARGETS, {str(i) for i in range(1, 21)})

    def test_20_redact_url_hides_credentials(self):
        self.assertTrue(bb.redact_url("postgresql://u:p@localhost/db").startswith("postgresql://<redacted>@"))

    def test_21_enqueue_adds_kernel_authorization(self):
        payload = {"target_number": 1, "target_name": "Kernel auth", "idempotency_key": "Kernel Auth"}
        ok, errors = bb.validate_work_order(payload)
        self.assertTrue(ok, errors)
        verdict = bb.ensure_kernel_authorized_work_order(payload, payload["idempotency_key"])
        self.assertTrue(verdict["required"])
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(payload["kernel_authorization"]["payload"]["queue_name"], "boring_beast")
        self.assertEqual(payload["kernel_authorization"]["payload"]["idempotency_key"], "kernel-auth")

    def test_22_worker_rejects_missing_boring_beast_kernel_authorization(self):
        verdict = bb.validate_job_kernel_authorization(
            queue_name="boring_beast",
            job_kind="boring_beast_work_item",
            payload={"target_number": 1, "target_name": "Missing auth", "idempotency_key": "missing-auth"},
        )
        self.assertTrue(verdict.required)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.error_kind, "missing_kernel_authorization")


if __name__ == "__main__":
    unittest.main()
