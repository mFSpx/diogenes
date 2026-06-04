#!/usr/bin/env python3
"""Source-backed Needle KV/shared-prefix truth probe.

Needle is a JAX encoder-decoder transformer. This probe does not start models;
it inspects the current worker and upstream model runner source and emits a
receipt distinguishing proven batching from unproven exact tensor/KV pointer
sharing.
"""
from __future__ import annotations

import argparse
import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "01_REPOS" / "needle" / "needle" / "model" / "run.py"
ARCH_PY = ROOT / "01_REPOS" / "needle" / "needle" / "model" / "architecture.py"
WORKER_PY = ROOT / "scripts" / "lucidota_needle_worker.py"
DEFAULT_RECEIPT = ROOT / "05_OUTPUTS" / "runtime" / "needle_kv_probe_latest.json"


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha_text(text: str) -> str:
    return sha256(text.encode("utf-8", errors="replace")).hexdigest()


def build_probe(*, receipt_path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    run_text = RUN_PY.read_text(encoding="utf-8")
    arch_text = ARCH_PY.read_text(encoding="utf-8")
    worker_text = WORKER_PY.read_text(encoding="utf-8")
    has_encode = "def encode_text" in arch_text and "method=\"encode\"" in run_text
    has_decode = "def decode" in arch_text and "method=\"decode\"" in run_text
    batch_available = "def generate_batch" in run_text and "generate_batch(" in worker_text and "--slots" in worker_text
    receipt = {
        "schema": "lucidota.edge_grail.needle_kv_probe.v1",
        "generated_at": now_z(),
        "status": "PARTIAL_CURRENT_RUNNER_BATCHES_PREFIX_REFACTOR_REQUIRED",
        "source_files": {
            "run_py": rel(RUN_PY),
            "architecture_py": rel(ARCH_PY),
            "worker_py": rel(WORKER_PY),
        },
        "source_hashes": {
            "run_py_sha256": sha_text(run_text),
            "architecture_py_sha256": sha_text(arch_text),
            "worker_py_sha256": sha_text(worker_text),
        },
        "architecture": {
            "kind": "jax_encoder_decoder_transformer",
            "has_separate_encode_method": bool(has_encode),
            "has_separate_decode_method": bool(has_decode),
            "encoder_input_contract": "_build_encoder_input(query, tools): [query tokens, <tools>, tools tokens]",
            "kv_cache_note": "This is not a llama.cpp-style exposed KV cache. Current reusable tensor is encoder_out/enc_mask after encode, not an external K/V pointer API.",
        },
        "current_worker": {
            "one_process_six_lane_batch": bool(batch_available),
            "worker": rel(WORKER_PY),
            "endpoint": "/generate_batch",
            "slots_default": 6,
            "truth": "Shared weights and batched JAX execution are source-proven. Exact tensor pointer sharing is not currently proven by the worker.",
        },
        "truth_flags": {
            "shared_weights_one_process_proven_by_source": "MODEL = None" in worker_text and "load_checkpoint" in worker_text,
            "batched_shared_prefix_available": bool(batch_available),
            "exact_tensor_kv_pointer_sharing_currently_proven": False,
            "prefix_reuse_possible_for_identical_full_encoder_input": bool(has_encode and has_decode),
            "prefix_reuse_for_different_lane_tools_requires_refactor": True,
        },
        "next_runner_patch": {
            "required": True,
            "reason": "Six lane tasks usually share the 500-token chunk but may differ in tool/task conditioning. Current Needle encoder mixes query and tools in one encoder input, so different tools change encoder_out; prefix-only reuse requires separating chunk encoding from lane conditioning or proving identical full encoder input.",
            "minimal_patch_shape": [
                "add shared_prefix/chunk_ref fields to /generate_batch",
                "tokenize/encode the immutable 500-token chunk once",
                "condition lane-specific task/tool data outside the shared encoder prefix or prove identical full encoder input",
                "broadcast/reuse encoder_out/enc_mask for exact identical full encoder inputs",
                "write telemetry: encode_calls_saved, prefix_hash, lane_count, output_tail_tokens, peak_memory",
            ],
        },
        "receipt_path": rel(receipt_path),
    }
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(prog="edge-grail-needle-kv-probe")
    ap.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    receipt = build_probe(receipt_path=args.receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True) if args.json else json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
