#!/usr/bin/env python3
"""KORPUS/KRAMPUS chronological date extraction algorithms.

Reusable math/regex layer: scripts should call this; scripts should not own it.
No LLM calls.
"""
from __future__ import annotations

import datetime as dt
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

CONTENT_DATE_PATTERNS = [
    ("frontmatter_date", re.compile(r"(?im)^\s*(?:date|created|created_at|created at|generated|timestamp|time|filed|updated|modified)\s*[:=]\s*[\"']?((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)")),
    ("iso_inline", re.compile(r"\b((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)\b")),
    ("compact_yyyymmdd", re.compile(r"\b((?:20|19)\d{2})(\d{2})(\d{2})(?:[T_-]?(\d{2})(\d{2})(\d{2})?)?Z?\b")),
]
MONTH_NAME_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+([0-3]?\d)(?:st|nd|rd|th)?,?\s+((?:20|19)\d{2})\b",
    re.I,
)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def read_text(path: Path, max_bytes: int) -> str:
    data = path.read_bytes()[:max_bytes]
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc, errors="replace")
        except (LookupError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")

def parse_loose_datetime(raw: str) -> dt.datetime | None:
    """Parse common dump/note timestamps into UTC datetimes."""
    text = normalize_ws(str(raw or "").strip().strip("'\"`[]()"))
    if not text:
        return None
    month = MONTH_NAME_RE.search(text)
    if month:
        months = {m.lower()[:3]: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
        try:
            return dt.datetime(int(month.group(3)), months[month.group(1).lower()[:3]], int(month.group(2)), tzinfo=dt.timezone.utc)
        except (KeyError, OverflowError, TypeError, ValueError):
            return None
    # Normalize YYYY_MM_DD / YYYY/MM/DD and compact HHMMSS-ish chunks.
    cleaned = text.replace("_", "-").replace("/", "-")
    cleaned = re.sub(r"^((?:20|19)\d{2}-\d{1,2}-\d{1,2})-(\d{2})(\d{2})(\d{2})$", r"\1T\2:\3:\4", cleaned)
    cleaned = re.sub(r"^((?:20|19)\d{2}-\d{1,2}-\d{1,2})-(\d{2})(\d{2})$", r"\1T\2:\3", cleaned)
    cleaned = re.sub(r"^((?:20|19)\d{2}-\d{1,2}-\d{1,2})\s+(\d{2})(\d{2})(\d{2})$", r"\1T\2:\3:\4", cleaned)
    cleaned = cleaned.replace(" ", "T")
    if re.match(r"^(?:20|19)\d{2}-\d{1,2}-\d{1,2}T\d{1,2}:?\d{2}(?::?\d{2})?", cleaned):
        cleaned = re.sub(r"T(\d{1,2})(\d{2})(\d{2})(Z|[+-]\d{2}:?\d{2})?$", r"T\1:\2:\3\4", cleaned)
        cleaned = re.sub(r"T(\d{1,2})(\d{2})(Z|[+-]\d{2}:?\d{2})?$", r"T\1:\2\3", cleaned)
    try:
        val = dt.datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        if val.tzinfo is None:
            val = val.replace(tzinfo=dt.timezone.utc)
        return val.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def chrono_candidates_for_path(path: Path, text_sample: str = "") -> list[dict[str, Any]]:
    """Find deterministic original-date candidates from frontmatter, path, text, then OS.

    Lower priority values win. Within the same priority, the earliest date wins.
    """
    candidates: list[dict[str, Any]] = []

    def add(source: str, raw: str, priority: int, confidence: int) -> None:
        parsed = parse_loose_datetime(raw)
        if parsed:
            candidates.append({
                "timestamp": parsed,
                "source": source,
                "raw": raw,
                "priority": priority,
                "confidence_bps": confidence,
            })

    # Frontmatter / explicit timestamp fields are strongest.
    for _, rx in CONTENT_DATE_PATTERNS[:1]:
        for m in rx.finditer(text_sample[:256_000]):
            add("content_frontmatter", m.group(1), 10, 9500)
            if len(candidates) >= 8:
                break

    # File/folder names are strong, but date-only filenames should not beat
    # full inline timestamps inside the document.
    path_text = str(path)
    for m in re.finditer(r"(?<!\d)((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_-]\d{1,2}:?\d{2}(?::?\d{2})?)?)(?!\d)", path_text):
        raw = m.group(1)
        has_time = bool(re.search(r"[T\s_-]\d{1,2}:?\d{2}", raw))
        add("path_iso", raw, 20 if has_time else 32, 8800 if has_time else 8200)
    for m in re.finditer(r"(?<!\d)((?:20|19)\d{2})(\d{2})(\d{2})(?:[T_-]?(\d{2})(\d{2})(\d{2})?)?Z?(?!\d)", path_text):
        y, mo, day, hh, mm, ss = m.groups()
        raw = f"{y}-{mo}-{day}" + (f"T{hh}:{mm}:{ss or '00'}" if hh and mm else "")
        add("path_compact", raw, 22 if hh and mm else 33, 8500 if hh and mm else 8000)

    # Inline text dates are useful, but can be about external events.
    for _, rx in CONTENT_DATE_PATTERNS[1:2]:
        for m in rx.finditer(text_sample[:256_000]):
            raw = m.group(1)
            has_time = bool(re.search(r"[T\s_-]\d{1,2}:?\d{2}", raw))
            add("content_inline_iso", raw, 28 if has_time else 30, 7200)
            if len(candidates) >= 64:
                break
    for m in MONTH_NAME_RE.finditer(text_sample[:256_000]):
        add("content_month_name", m.group(0), 35, 6200)
        if len(candidates) >= 96:
            break

    # Final fallback: OS metadata. Not trusted after copying, but never empty.
    try:
        st = path.stat()
        birth = float(getattr(st, "st_birthtime", 0.0) or 0.0)
        if birth > 0:
            candidates.append({"timestamp": dt.datetime.fromtimestamp(birth, tz=dt.timezone.utc), "source": "os_birthtime", "raw": str(birth), "priority": 80, "confidence_bps": 5000})
        candidates.append({"timestamp": dt.datetime.fromtimestamp(float(st.st_mtime), tz=dt.timezone.utc), "source": "os_mtime", "raw": str(st.st_mtime), "priority": 90, "confidence_bps": 3500})
    except OSError:
        return candidates
    return candidates


def archive_chrono_candidates(path: Path, max_members: int = 20_000) -> list[dict[str, Any]]:
    """Cheap archive-level chronological evidence without extracting payloads.

    ZIP central-directory metadata and member names are forensic hints. They do
    not beat explicit content timestamps, but they let delayed 200GB exports be
    re-ordered later if the raw archive was preserved in CAS.
    """
    candidates: list[dict[str, Any]] = []
    try:
        if path.suffix.lower() != ".zip" or not path.exists() or not zipfile.is_zipfile(path):
            return candidates
        with zipfile.ZipFile(path) as zf:
            for idx, info in enumerate(zf.infolist()):
                if idx >= max_members:
                    break
                if info.is_dir():
                    continue
                member = info.filename
                for cand in chrono_candidates_for_path(Path(member), ""):
                    if str(cand.get("source", "")).startswith("path_"):
                        c2 = dict(cand)
                        c2["source"] = "archive_member_" + str(c2["source"])
                        c2["priority"] = min(int(c2.get("priority", 99)) + 4, 70)
                        c2["confidence_bps"] = min(int(c2.get("confidence_bps", 0)), 7600)
                        c2["path_evidence"] = member
                        candidates.append(c2)
                try:
                    y, mo, day, hh, mm, ss = info.date_time
                    # ZIP defaults are often 1980-01-01 when no real mtime exists.
                    if y > 1980:
                        candidates.append({
                            "timestamp": dt.datetime(y, mo, day, hh, mm, ss, tzinfo=dt.timezone.utc),
                            "source": "archive_member_mtime",
                            "raw": member,
                            "priority": 48,
                            "confidence_bps": 4500,
                            "path_evidence": member,
                        })
                except (TypeError, ValueError, OverflowError):
                    continue
    except (OSError, zipfile.BadZipFile):
        return candidates
    return candidates


def chrono_file_date(path: Path, max_bytes: int = 256_000) -> dict[str, Any]:
    text_sample = ""
    try:
        if path.is_file():
            text_sample = read_text(path, max_bytes)
    except OSError:
        text_sample = ""
    candidates = chrono_candidates_for_path(path, text_sample)
    if not candidates:
        return {"timestamp": dt.datetime.fromtimestamp(0, tz=dt.timezone.utc), "iso": "1970-01-01T00:00:00Z", "source": "missing", "raw": "", "confidence_bps": 0}
    best = sorted(candidates, key=lambda c: (int(c["priority"]), c["timestamp"], str(path)))[0]
    ts: dt.datetime = best["timestamp"].astimezone(dt.timezone.utc)
    return {
        "timestamp": ts,
        "iso": ts.isoformat().replace("+00:00", "Z"),
        "source": best["source"],
        "raw": best["raw"],
        "confidence_bps": int(best["confidence_bps"]),
        "candidate_count": len(candidates),
    }


def circadian_activity_curve(timestamps: list[dt.datetime], bins: int = 48, sigma: float = 1.35) -> np.ndarray:
    """Return a normalized UTC time-of-day activity curve."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not timestamps:
        return np.zeros(bins, dtype=float)
    minutes = []
    for value in timestamps:
        ts = value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
        ts = ts.astimezone(dt.timezone.utc)
        minutes.append(ts.hour * 60 + ts.minute + ts.second / 60 + ts.microsecond / 60_000_000)
    raw = np.array(minutes, dtype=float)
    counts = np.bincount(np.floor(raw / (1440 / bins)).astype(int).clip(0, bins - 1), minlength=bins).astype(float)
    smooth = gaussian_filter1d(np.r_[counts, counts, counts], sigma=sigma, mode="wrap")[bins:2 * bins]
    span = float(smooth.max() - smooth.min())
    return (smooth - smooth.min()) / (span if span else 1.0)


def circadian_match_metrics(a_times: list[dt.datetime], b_times: list[dt.datetime], bins: int = 48, dead_quantile: float = 0.30) -> dict[str, Any]:
    """Reusable circadian biometric comparison: curve correlation + low-activity overlap."""
    a = circadian_activity_curve(a_times, bins=bins)
    b = circadian_activity_curve(b_times, bins=bins)
    corr = 0.0 if np.std(a) == 0 or np.std(b) == 0 else float(pearsonr(a, b).statistic)
    dead_a = a <= np.quantile(a, dead_quantile)
    dead_b = b <= np.quantile(b, dead_quantile)
    overlap = float(np.logical_and(dead_a, dead_b).sum() / max(1, np.logical_or(dead_a, dead_b).sum()))
    score = 100.0 * (0.68 * ((corr + 1.0) / 2.0) + 0.32 * overlap)
    return {
        "curve_a": a,
        "curve_b": b,
        "dead_mask_a": dead_a,
        "dead_mask_b": dead_b,
        "pearson_r": corr,
        "dead_zone_overlap": overlap,
        "score": max(0.0, min(100.0, score)),
    }




def chronological_key(item: tuple[Path, Path]) -> tuple[float, str]:
    """Content/path/frontmatter date first; OS time only as fallback."""
    _, path = item
    try:
        t = chrono_file_date(path)["timestamp"].timestamp()
    except (KeyError, TypeError, ValueError, OverflowError):
        t = 0.0
    return (t, str(path))
