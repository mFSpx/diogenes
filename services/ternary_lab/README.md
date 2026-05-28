# LUCIDOTA Ternary Lens Lab

Ternary Lens Lab is a separate R&D workstream. It does not block or redefine Chrono-Ledger Phase C.

## Contract

A lens is FairyFuse-compatible only if runtime inference avoids reintroducing FP16/FP32 LoRA matrix multiplications in the hot path. Adapter compatibility must be proven, not assumed.

## Current architecture call

- FairyFuse v0 is a backend abstraction, not a custom kernel.
- First target is a tiny Command Envelope Router, not a general ternary LLM.
- BitNet GGUF / packed weights are the deployment path.
- BF16 BitNet weights are the training/fine-tuning path.
- Standard PEFT LoRA, QLoRA, QA-LoRA, LowRA, BitNet training adapters, packed BitNet artifacts, QVAC adapters, and BitLoRA-style candidates are distinct lens families.

## Classification vocabulary

- `usable_now`: preserves the low-bit runtime path or is a non-adapter packed deployment backend.
- `research_only`: interesting but not deployable as a FairyFuse fast-path lens yet.
- `needs_conversion`: may become usable only after a compiler/export path creates a packed hot-path artifact.
- `unsafe_for_fastpath`: expected to reintroduce FP16/FP32 adapter matmuls unless benchmarks prove otherwise.
- `unsupported`: cannot be used with the current target.

## Commands

Audit without network calls:

```bash
cd /home/mfspx/LUCIDOTA
python3 ALGOS/ternary_lens_audit.py --manifest services/ternary_lab/vendor_manifest.json --output 05_OUTPUTS/ternary_lab/lens_audit_report.json
```

Router dry-run:

```bash
cd /home/mfspx/LUCIDOTA
python3 ALGOS/ternary_lens_router.py --dry-run --raw-command "open chrono status" --normalized-intent "status_check" --context '{"surface":"terminal"}'
```

## Non-goals for v0

- No model downloads.
- No external network calls.
- No claims that ordinary PEFT LoRA is FairyFuse-compatible.
- No runtime fast-path claim without benchmark evidence.
