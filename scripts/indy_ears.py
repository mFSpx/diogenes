#!/usr/bin/env python3
"""
indy_ears.py — Indy_READs Audio Capture & Transcription
=========================================================
IronClaw daemon component. Records audio from microphone, transcribes
via Groq Whisper, logs to 04_RUNTIME/audio_log.jsonl.

Ears for Indy. Groq Whisper is fast, cheap, and already on the API key.

Two modes:
  --once       Record N seconds, transcribe, print, exit
  --daemon     Continuous listen loop (VAD-triggered or interval-based)
  --file PATH  Transcribe an existing audio file

Dependencies: arecord (ALSA), openai (Groq Whisper compatible endpoint)
  Zero Python audio libraries needed. arecord handles the hardware.

Usage:
  source scripts/lucidota_safe_ops_env.sh
  python3 scripts/indy_ears.py --once              # record 5s, transcribe
  python3 scripts/indy_ears.py --once --duration 10 # record 10s
  python3 scripts/indy_ears.py --file ~/Downloads/tts.wav
  python3 scripts/indy_ears.py --daemon             # continuous listen loop
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNTIME_DIR = PROJECT_ROOT / "04_RUNTIME"
AUDIO_LOG = RUNTIME_DIR / "audio_log.jsonl"
TEMP_DIR = RUNTIME_DIR / "audio_temp"
GROQ_API_KEY_PATH = Path.home() / ".config" / "lucidota" / "secrets.env"

# ---------------------------------------------------------------------------
# Audio config
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000       # Whisper works best at 16kHz
CHANNELS = 1              # mono
SAMPLE_WIDTH = 2          # 16-bit
DEFAULT_DURATION = 5      # seconds for --once
DAEMON_INTERVAL = 5       # seconds between recordings in daemon mode
WHISPER_MODEL = "whisper-large-v3-turbo"  # fastest Groq Whisper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_groq_key() -> str:
    if GROQ_API_KEY_PATH.exists():
        for line in GROQ_API_KEY_PATH.read_text().splitlines():
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("GROQ_API_KEY not found in secrets.env")


def ts() -> str:
    return datetime.datetime.now().isoformat()


def ts_filename() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Audio capture via arecord (ALSA — zero Python deps)
# ---------------------------------------------------------------------------

def record_wav(duration: float, device: str = "default") -> Path:
    """Record audio via arecord, return path to WAV file."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TEMP_DIR / f"capture_{ts_filename()}.wav"

    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",       # 16-bit little-endian
        "-c", str(CHANNELS),
        "-r", str(SAMPLE_RATE),
        "-d", str(int(duration)),
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=int(duration) + 5)
    return out_path


# ---------------------------------------------------------------------------
# Transcription via Groq Whisper
# ---------------------------------------------------------------------------

def transcribe(file_path: Path, model: str = WHISPER_MODEL) -> str:
    """Send audio to Groq Whisper, return transcription text."""
    api_key = load_groq_key()
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

    with open(file_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=model,
            file=(file_path.name, f, "audio/wav"),
            response_format="text",
        )
    return response.strip()


def transcribe_verbose(file_path: Path, model: str = WHISPER_MODEL) -> dict:
    """Transcribe with full metadata (language, segments, etc.)."""
    api_key = load_groq_key()
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

    with open(file_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=model,
            file=(file_path.name, f, "audio/wav"),
            response_format="verbose_json",
        )
    return {
        "text": response.text.strip(),
        "language": response.language if hasattr(response, "language") else "unknown",
        "duration": response.duration if hasattr(response, "duration") else 0,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_transcription(audio_path: Path, text: str, metadata: dict | None = None):
    entry = {
        "timestamp": ts(),
        "audio_file": str(audio_path),
        "transcription": text,
        "model": WHISPER_MODEL,
    }
    if metadata:
        entry["language"] = metadata.get("language")
        entry["audio_duration_s"] = metadata.get("duration")
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIO_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def once(duration: float, keep_audio: bool = False):
    """Record, transcribe, print, exit."""
    stamp = ts_filename()
    print(f"[EARS] Recording {duration}s...", file=sys.stderr)
    wav_path = record_wav(duration)

    file_size = wav_path.stat().st_size
    print(f"[EARS] Recorded {file_size} bytes → {wav_path}", file=sys.stderr)

    print(f"[EARS] Transcribing via Groq {WHISPER_MODEL}...", file=sys.stderr)
    result = transcribe_verbose(wav_path)
    log_transcription(wav_path, result["text"], result)

    print(f"[EARS] Language: {result['language']}", file=sys.stderr)
    print(f"[EARS] Duration: {result['duration']:.1f}s", file=sys.stderr)
    print(f"[EARS] ---", file=sys.stderr)
    print(result["text"])

    if not keep_audio:
        wav_path.unlink(missing_ok=True)


def transcribe_file(file_path: str):
    """Transcribe an existing audio file."""
    path = Path(file_path)
    if not path.exists():
        print(f"[EARS] File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[EARS] Transcribing {path} via Groq {WHISPER_MODEL}...", file=sys.stderr)
    result = transcribe_verbose(path)
    log_transcription(path, result["text"], result)

    print(f"[EARS] Language: {result['language']}", file=sys.stderr)
    print(f"[EARS] Duration: {result['duration']:.1f}s", file=sys.stderr)
    print(f"[EARS] ---", file=sys.stderr)
    print(result["text"])


def daemon_mode(duration: float, keep_audio: bool = False):
    """Continuous listen loop."""
    print(f"[EARS] Daemon mode: recording {duration}s every {DAEMON_INTERVAL}s. Ctrl+C to stop.", file=sys.stderr)
    try:
        while True:
            try:
                wav_path = record_wav(duration)
                result = transcribe_verbose(wav_path)
                if result["text"]:
                    log_transcription(wav_path, result["text"], result)
                    print(f"[EARS] {ts()} | {result['language']} | {result['text'][:120]}", file=sys.stderr)
                if not keep_audio:
                    wav_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[EARS] Error: {e}", file=sys.stderr)
            time.sleep(DAEMON_INTERVAL)
    except KeyboardInterrupt:
        print("[EARS] Daemon stopped.", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Indy_READs Audio Capture & Transcription — Groq Whisper"
    )
    parser.add_argument("--once", action="store_true", help="Record and transcribe once")
    parser.add_argument("--daemon", action="store_true", help="Continuous listen loop")
    parser.add_argument("--file", type=str, help="Transcribe an existing audio file")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help=f"Recording duration in seconds (default {DEFAULT_DURATION})")
    parser.add_argument("--keep", action="store_true", help="Keep audio files after transcription")
    args = parser.parse_args()

    if args.file:
        transcribe_file(args.file)
    elif args.daemon:
        daemon_mode(args.duration, keep_audio=args.keep)
    elif args.once:
        once(args.duration, keep_audio=args.keep)
    else:
        # Default: record 5s and transcribe
        once(args.duration, keep_audio=args.keep)


if __name__ == "__main__":
    main()
