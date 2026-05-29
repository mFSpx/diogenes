from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
WHISPER_MODEL = "whisper-large-v3-turbo"
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB


def transcribe_audio(path: Path, groq_api_key: str, language: str = "en") -> dict:
    """Transcribe an audio file via Groq Whisper API."""
    if path.stat().st_size > MAX_FILE_BYTES:
        return {"error": "file_too_large_for_groq", "transcript": "", "model": WHISPER_MODEL, "language": language, "duration_s": None}

    try:
        with open(path, "rb") as fh:
            files = {"file": (path.name, fh, "audio/mpeg")}
            data = {"model": WHISPER_MODEL, "response_format": "verbose_json", "language": language}
            headers = {"Authorization": f"Bearer {groq_api_key}"}
            response = httpx.post(
                GROQ_TRANSCRIPTION_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=120.0,
            )
            response.raise_for_status()
            payload = response.json()

        return {
            "transcript": payload["text"],
            "language": language,
            "duration_s": payload.get("duration"),
            "model": WHISPER_MODEL,
            "error": None,
        }
    except Exception as exc:
        return {"error": str(exc), "transcript": "", "model": WHISPER_MODEL, "language": language, "duration_s": None}


def extract_audio_from_video(path: Path) -> Path | None:
    """Extract mono 16 kHz WAV from a video file using ffmpeg."""
    sha = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    out = Path(f"/tmp/lucidota_audio_{sha}.wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return None


if __name__ == "__main__":
    print("BUILT: artifact_audio_extractor.py")
