#!/usr/bin/env python3
"""Validate and query LUCI auxiliary model admission policy.

This is the pressure-hull gate for embedders/OCR/GLiNER/Whisper/Piper/etc:
move refs, not bodies; run one caged auxiliary burst at a time; never wake heavy
models just because a channel has capacity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

AUX_CLASSES = {"CPU_WARM_ONE_AT_A_TIME", "COLD_SUBPROCESS", "BATCH_OFFLINE_ONLY"}
AUX_BURST_TOOLS = {
    "embedder_onnx_cpu",
    "reranker_cpu",
    "ocr_tesseract",
    "ocr_paddle_cold",
    "gliner_local",
    "whisper_audio",
    "piper_tts",
    "vision_ocr_layout",
    "python_heavy_algo_cage",
}


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tool_by_id(manifest: dict[str, Any], tool_id: str) -> dict[str, Any] | None:
    for tool in manifest.get("tools", []):
        if tool.get("id") == tool_id:
            return tool
    return None


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    budgets = manifest.get("global_budgets", {})
    required_budgets = [
        "max_event_bytes",
        "max_stdout_bytes",
        "max_stderr_bytes",
        "max_prompt_bytes",
        "max_json_bytes",
        "max_response_tokens",
        "max_audio_buffer_ms",
        "max_db_rows",
        "max_file_read_bytes",
    ]
    for key in required_budgets:
        if int(budgets.get(key, 0) or 0) <= 0:
            errors.append(f"missing_or_invalid_budget:{key}")

    event_ref = manifest.get("event_ref_contract", {})
    if event_ref.get("mpsc_moves") != "refs_not_bodies":
        errors.append("event_ref_contract_not_refs_only")
    if int(event_ref.get("max_preview_bytes", 999999) or 999999) > 512:
        errors.append("preview_too_large")

    mutexes = {tuple(sorted(m)) for m in manifest.get("mutual_exclusion", [])}
    if tuple(sorted(("SSD_DEEP", "AUX_MODEL_BURST"))) not in mutexes:
        errors.append("missing_ssd_aux_mutex")

    seen = set()
    for tool in manifest.get("tools", []):
        tid = str(tool.get("id") or "")
        if not tid:
            errors.append("tool_missing_id")
            continue
        if tid in seen:
            errors.append(f"duplicate_tool:{tid}")
        seen.add(tid)
        limits = tool.get("limits", {})
        cgroup = tool.get("cgroup", {})
        if tool.get("class") in AUX_CLASSES and bool(tool.get("resident_default")):
            errors.append(f"aux_resident_by_default:{tid}")
        if tool.get("receipt_required") is not True:
            errors.append(f"receipt_not_required:{tid}")
        for key in ("max_input_bytes", "max_output_bytes", "timeout_ms"):
            if int(limits.get(key, 0) or 0) <= 0:
                errors.append(f"{tid}:missing_limit:{key}")
        if int(cgroup.get("MemoryHighMB", 0) or 0) <= 0:
            errors.append(f"{tid}:missing_memory_high")
        if int(cgroup.get("MemoryMaxMB", 0) or 0) < int(cgroup.get("MemoryHighMB", 0) or 0):
            errors.append(f"{tid}:memory_max_below_high")
    return errors


def active_lane_for_tool(tool_id: str) -> str:
    if tool_id == "talkie_ssd_deep_forge":
        return "SSD_DEEP"
    if tool_id in AUX_BURST_TOOLS:
        return "AUX_MODEL_BURST"
    return "LIVE_CHAT"


def decide_admission(
    manifest: dict[str, Any],
    tool_id: str,
    input_bytes: int,
    active_lanes: list[str],
    memory_pct: float,
    vram_pct: float,
) -> tuple[int, dict[str, Any]]:
    errors = validate_manifest(manifest)
    if errors:
        return 3, {"admit": False, "tool": tool_id, "reason": "manifest_invalid", "errors": errors}

    tool = tool_by_id(manifest, tool_id)
    if tool is None:
        return 2, {"admit": False, "tool": tool_id, "reason": "unknown_tool"}

    wanted_lane = active_lane_for_tool(tool_id)
    mutexes = {tuple(sorted(m)) for m in manifest.get("mutual_exclusion", [])}
    for lane in active_lanes:
        if tuple(sorted((lane, wanted_lane))) in mutexes:
            return 2, {
                "admit": False,
                "tool": tool_id,
                "reason": f"mutual_exclusion:{lane}:{wanted_lane}",
                "mode": tool.get("mode"),
            }

    limits = tool.get("limits", {})
    max_input = int(limits.get("max_input_bytes", 0) or 0)
    if input_bytes > max_input:
        return 2, {"admit": False, "tool": tool_id, "reason": "input_bytes_over_limit", "limit": max_input}

    governor = manifest.get("governor_rules", {})
    mem_rule = governor.get("memory_over_pct", {})
    if memory_pct >= float(mem_rule.get("threshold", 100)) and wanted_lane == "SSD_DEEP":
        return 2, {"admit": False, "tool": tool_id, "reason": mem_rule.get("action", "memory_pressure")}
    vram_rule = governor.get("vram_over_pct", {})
    if vram_pct >= float(vram_rule.get("threshold", 100)) and wanted_lane in {"AUX_MODEL_BURST", "SSD_DEEP"}:
        return 2, {"admit": False, "tool": tool_id, "reason": vram_rule.get("action", "vram_pressure")}

    return 0, {
        "admit": True,
        "tool": tool_id,
        "lane": wanted_lane,
        "mode": tool.get("mode"),
        "resident_default": bool(tool.get("resident_default")),
        "receipt_required": bool(tool.get("receipt_required")),
        "cgroup": tool.get("cgroup", {}),
        "limits": limits,
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="aux-model-admission")
    ap.add_argument("--manifest", default="04_RUNTIME/aux_model_admission_manifest.json")
    ap.add_argument("--tool", required=True)
    ap.add_argument("--input-bytes", type=int, default=0)
    ap.add_argument("--active-lane", action="append", default=[])
    ap.add_argument("--memory-pct", type=float, default=0.0)
    ap.add_argument("--vram-pct", type=float, default=0.0)
    args = ap.parse_args()
    manifest = load_manifest(Path(args.manifest))
    code, result = decide_admission(
        manifest,
        args.tool,
        input_bytes=max(0, args.input_bytes),
        active_lanes=list(args.active_lane or []),
        memory_pct=args.memory_pct,
        vram_pct=args.vram_pct,
    )
    print(json.dumps(result, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
