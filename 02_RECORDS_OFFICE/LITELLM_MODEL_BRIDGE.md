# LiteLLM Model Management Bridge

Command:

```bash
.venv/bin/python scripts/lucidota_litellm_bridge.py --json
```

Purpose: dry inventory for model routing/management. It reports configured providers/models in a LiteLLM `model_list` shape and validates only cheap local facts:

- provider API-key environment variables are present or missing;
- configured local model/artifact paths exist;
- registry rows from `lucidota_runtime.model_candidate` can be surfaced without loading weights.

It intentionally does **not** import LiteLLM, contact remote APIs, start local servers, load model weights, or run inference. Missing API keys are reported but non-fatal by default so `check_diogenes.sh` remains offline-safe. Use `--strict` when an operator wants missing configured API keys to fail the command.

Extra provider routes can be supplied without a schema change:

```bash
export LUCIDOTA_OPENAI_MODELS=gpt-4o-mini,gpt-4.1-mini
export OPENAI_API_KEY=...

export LUCIDOTA_LITELLM_MODELS_JSON='[
  {"provider":"local","model_name":"my-gguf","litellm_model":"my-gguf","local_path":"03_VAULT/models/my.gguf"}
]'
```

`check_diogenes.sh` includes the bridge as an offline validation step after model artifact readiness.
