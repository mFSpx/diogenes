#!/usr/bin/env python3
"""Load Groq credentials from .env when the shell did not export them."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_groq_env() -> None:
    env_paths = []
    secret_env = os.environ.get("LUCIDOTA_SECRET_ENV") or str(Path.home() / ".config" / "lucidota" / "secrets.env")
    env_paths.append(Path(secret_env))
    if os.environ.get("GROQ_LOAD_DOTENV", "").lower() in {"1", "true", "yes", "on"}:
        env_paths.append(ROOT / ".env")
    for env in env_paths:
        if not env.exists():
            continue
        for raw in env.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if key not in {"GROQ_API_KEY", "GROQ_BASE_URL", "GROQ_MODEL", "GROQ_GOAL_MODEL"}:
                continue
            os.environ.setdefault(key, val.strip().strip('"').strip("'"))
    key_file = Path(os.environ.get("LUCIDOTA_GROQ_KEY_FILE", "/tmp/lucidota_groq_key"))
    if key_file.exists():
        os.environ["GROQ_API_KEY"] = key_file.read_text(encoding="utf-8").strip()
