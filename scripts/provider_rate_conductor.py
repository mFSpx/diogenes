#!/usr/bin/env python3
"""Compatibility shim for the Rust provider-rate conductor.

The workflow itself now lives in the Rust binary at
`01_REPOS/lucidota_etl/crates/lucidota-workers/src/bin/provider_rate_conductor.rs`.
This shim keeps existing ABSURD allowlists and direct invocations working while
routing actual execution through the Rust-backed shell entrypoint.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "scripts" / "provider_rate_conductor.sh"

os.execv(str(SHELL), [str(SHELL), *sys.argv[1:]])
