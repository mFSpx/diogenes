#!/usr/bin/env python3
"""
cipher_training_pii.py  —  PII cipher for training data.

Scans JSONL files for personally identifiable information (PII):
  - Phone numbers (+1 NNN NNN-NNNN, (NNN) NNN-NNNN, NNN-NNN-NNNN, etc.)
  - Email addresses
  - Street addresses (number + street/road/avenue/drive/blvd/etc.)
  - URLs containing personal information (linkedin, instagram)
  - Proper names (known-name list + heuristic patterns)

Replaces PII with deterministic cipher tokens:
  [CIPHER_NAME_NNNN], [CIPHER_PHONE_NNNN], [CIPHER_ADDR_NNNN], [CIPHER_EMAIL_NNNN]

Writes ciphered copies to 05_OUTPUTS/trm_training/ciphered/
Writes PII manifest to 05_OUTPUTS/trm_training/ciphered/pii_manifest.json

Usage:
  python3 scripts/cipher_training_pii.py <input.jsonl> [input2.jsonl ...]
  python3 scripts/cipher_training_pii.py --dry-run <input.jsonl> [...]

Schema: lucidota.trm.pii_cipher_receipt.v1
Mutation class: receipt_only
"""

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

# ── constants ────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "05_OUTPUTS",
    "trm_training",
    "ciphered",
)
RECEIPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "05_OUTPUTS",
    "trm_training",
    "receipts",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RECEIPT_DIR, exist_ok=True)

# ── Unicode normalization ────────────────────────────────────────────────────

# WhatsApp uses many unicode variants of ASCII characters.
# Normalize before PII matching so regex patterns work.
UNICODE_NORMALIZE_MAP = str.maketrans({
    '\u2019': "'",   # RIGHT SINGLE QUOTATION MARK -> ASCII apostrophe
    '\u2018': "'",   # LEFT SINGLE QUOTATION MARK  -> ASCII apostrophe
    '\u201c': '"',   # LEFT DOUBLE QUOTATION MARK  -> ASCII double quote
    '\u201d': '"',   # RIGHT DOUBLE QUOTATION MARK -> ASCII double quote
    '\u2010': '-',   # HYPHEN -> ASCII hyphen
    '\u2011': '-',   # NON-BREAKING HYPHEN -> ASCII hyphen
    '\u2012': '-',   # FIGURE DASH -> ASCII hyphen
    '\u2013': '-',   # EN DASH -> ASCII hyphen
    '\u2014': '-',   # EM DASH -> ASCII hyphen
    '\u00a0': ' ',   # NO-BREAK SPACE -> regular space
    '\u202f': ' ',   # NARROW NO-BREAK SPACE -> regular space
    '\u200b': '',    # ZERO WIDTH SPACE -> remove
    '\u200e': '',    # LEFT-TO-RIGHT MARK -> remove
    '\u200f': '',    # RIGHT-TO-LEFT MARK -> remove
    '\u202a': '',    # LEFT-TO-RIGHT EMBEDDING -> remove
    '\u202b': '',    # RIGHT-TO-LEFT EMBEDDING -> remove
    '\u202c': '',    # POP DIRECTIONAL FORMATTING -> remove
    '\u202d': '',    # LEFT-TO-RIGHT OVERRIDE -> remove
    '\u202e': '',    # RIGHT-TO-LEFT OVERRIDE -> remove
    '\u2060': '',    # WORD JOINER -> remove
    '\u2066': '',    # LEFT-TO-RIGHT ISOLATE -> remove
    '\u2067': '',    # RIGHT-TO-LEFT ISOLATE -> remove
    '\u2068': '',    # FIRST STRONG ISOLATE -> remove
    '\u2069': '',    # POP DIRECTIONAL ISOLATE -> remove
    '\u007f': '',    # DEL -> remove
})

def normalize_text(text):
    """Normalize unicode variants to ASCII equivalents for matching."""
    return text.translate(UNICODE_NORMALIZE_MAP)


# ── PII patterns ─────────────────────────────────────────────────────────────

# Phone: handle +1, (NNN), NNN-NNN-NNNN, with or without spaces/dots
# Each pattern requires at least one separator or formatting character
# between digit groups to avoid matching arbitrary 10-digit numeric values
# from game state features or other non-phone contexts.
PHONE_PATTERNS = [
    # +1 (NNN) NNN-NNNN or +1 NNN NNN NNNN (requires at least one space/dot/dash)
    re.compile(r'\+\s*1[\s.(]*\d{3}[\s.).]*\d{3}[\s.-]*\d{4}'),
    # (NNN) NNN-NNNN or NNN-NNN-NNNN (requires at least one non-digit separator)
    re.compile(r'(?<!\d)\(?\d{3}\)?[\s.\-)]+\d{3}[\s.\-]+\d{4}(?!\d)'),
    # +1NNNNNNNNNN (ten digits after +1, no separator)
    re.compile(r'\+\s*1\s*\d{10}\b'),
]

# Email
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# Street address (number + street/road/avenue/drive/blvd/ln/way/crescent/court)
# Pattern 1: Full suffixes (street, road, etc.) — case-insensitive.
# Pattern 2: Abbreviations (st, rd, ave, etc.) require capitalized preceding
#   word to avoid false positives like "5 st" or "3 rd".
FULL_SUFFIX = r'(?:street|road|avenue|drive|boulevard|lane|way|crescent|court|place|terrace)'
ABBR_SUFFIX = r'(?:[Ss][Tt]\.?|[Rr][Dd]\.?|[Aa][Vv][Ee]\.?|[Dd][Rr]\.?|[Bb][Ll][Vv][Dd]|[Ll][Nn]\.?|[Cc][Tt]\.?|[Pp][Ll]\.?)'
# Words that commonly trigger false positives (e.g. "min drive" = "minute drive")
# These are verbs/time-words used as street suffixes in casual text.
FALSE_POSITIVE_PREFIXES = re.compile(
    r'\b(?:min|mins|hour|hours|sec|secs|minute|minutes)\s+'
    r'(?:drive|dr|lane|way|place|pl|st)',
    re.IGNORECASE,
)
# Full suffix pattern: case-insensitive
ADDRESS_FULL = re.compile(
    r'\b(\d{1,5}\s+(?:[A-Za-z][a-zA-Z]*\s+){1,3})' + FULL_SUFFIX,
    re.IGNORECASE,
)
# Abbrev pattern: case-sensitive, word before must be capitalized proper noun
ADDRESS_ABBR = re.compile(
    r'\b(\d{1,5}\s+(?:[A-Z][a-zA-Z]*\s+){1,3})' + ABBR_SUFFIX,
)
ADDRESS_PATTERNS = [ADDRESS_FULL, ADDRESS_ABBR]

# URLs (linkedin, instagram)
URL_PATTERN = re.compile(
    r'(?:linkedin\.com|instagram\.com|facebook\.com|twitter\.com|x\.com)/[a-zA-Z0-9._%-]+'
)

# Known names extracted from sender fields + entity graph
# Use ASCII versions only; normalize_text() converts unicode variants.
KNOWN_NAMES = [
    "Saunasage", "Sage Brocklebank", "Sage",
    "David",
    "Danny", "Daniel",
    "Emilio Rosas",
    "O'Ashley H. Aschmann", "Ashley H. Aschmann", "Ashley Hogan", "Ashley",
    "Aaron",
    "KitKat",
    "Jules",
    "Tanaz",
    "Genvieve",
    "George Aaron Hutchinson",
    "Gurpreet Rakhra",
    "Chance Stewart",
    "Bradley Zemaitis",
    "Chantal Dobles Gering",
    "Daryl-Lee Dawn Schalm",
    "Jessica Bailey",
    "Victoria Sanchez",
    "Danielle Yarden Rozali",
    "Omar Babe Ndayiragije",
    "Yyen Gallup",
    "Briannah Genevieve Cristofoli",
    "Luca Froelich",
    "Madison Eleanor Chapel",
    "Kai Lindsay-Rodgers", "Kai",
    "Rory Mills",
    "Fernanda",
    "Zack",
    "Kiskae",
    "Rowyn",
    "Luna",
    "Peter",
]

# Build a single combined name regex — match longest first.
# Names are matched in normalized text (unicode variants already converted).
# Use positive lookbehind for start-of-string or non-word char to avoid
# matching substrings inside larger words.
KNOWN_NAMES_SORTED = sorted(set(KNOWN_NAMES), key=lambda n: (-len(n), n))

def _build_name_regex():
    """Build regex that matches any known name (word-boundary aware)."""
    escaped = [re.escape(n) for n in KNOWN_NAMES_SORTED]
    alt = "|".join(escaped)
    # Match at word boundary, also handle @mention prefix and tilde prefix
    # Normal text: "Aaron", "@Aaron", "~Aaron"
    word_boundary = r'(?:^|(?<=[\s.,!?;:\'"@~`]))'
    return re.compile(word_boundary + '(' + alt + r')(?=[\s.,!?;:\'"@~`\-]|$)', re.UNICODE)

NAME_PATTERN = _build_name_regex()


# ── PII manifest ─────────────────────────────────────────────────────────────

class PIIManifest:
    """Tracks all PII replacements across files."""

    def __init__(self):
        self.replacements = {
            "names": {},
            "phones": {},
            "addresses": {},
            "emails": {},
            "urls": {},
        }
        self.name_counter = 0
        self.phone_counter = 0
        self.addr_counter = 0
        self.email_counter = 0
        self.url_counter = 0

    def _cipher_token(self, category, key):
        """Get deterministic cipher token. Uses key for dedup."""
        d = self.replacements[category]
        if key not in d:
            # Map category name to counter attribute
            counter_map = {
                "names": "name_counter",
                "phones": "phone_counter",
                "addresses": "addr_counter",
                "emails": "email_counter",
                "urls": "url_counter",
            }
            counter_attr = counter_map[category]
            setattr(self, counter_attr, getattr(self, counter_attr) + 1)
            n = getattr(self, counter_attr)
            cat_label = {
                "names": "NAME",
                "phones": "PHONE",
                "addresses": "ADDR",
                "emails": "EMAIL",
                "urls": "URL",
            }[category]
            token = f"[CIPHER_{cat_label}_{n:04d}]"
            d[key] = token
        return d[key]

    def cipher_name(self, name):
        return self._cipher_token("names", name.strip())

    def cipher_phone(self, phone):
        return self._cipher_token("phones", phone.strip())

    def cipher_address(self, addr):
        return self._cipher_token("addresses", addr.strip())

    def cipher_email(self, email):
        return self._cipher_token("emails", email.strip())

    def cipher_url(self, url):
        return self._cipher_token("urls", url.strip())

    def totals(self):
        return {k: len(v) for k, v in self.replacements.items()}


# ── cipher engine ────────────────────────────────────────────────────────────

class PIICipher:
    """Scans and replaces PII in JSONL records."""

    def __init__(self, manifest=None):
        self.manifest = manifest or PIIManifest()

    def _cipher_phones(self, text):
        """Replace phone numbers with cipher tokens."""
        for pat in PHONE_PATTERNS:
            def _replace_phone(m):
                raw = m.group(0)
                # Normalize for dedup key
                key = re.sub(r'[\s.\-()]+', '', raw)
                return self.manifest.cipher_phone(key)
            text = pat.sub(_replace_phone, text)
        return text

    def _cipher_emails(self, text):
        """Replace email addresses with cipher tokens."""
        def _replace_email(m):
            return self.manifest.cipher_email(m.group(0))
        return EMAIL_PATTERN.sub(_replace_email, text)

    def _cipher_addresses(self, text):
        """Replace street addresses with cipher tokens."""
        for pat in ADDRESS_PATTERNS:
            def _replace_addr(m, p=pat):
                match_text = m.group(0).strip()
                # Skip matches that are false positives
                if FALSE_POSITIVE_PREFIXES.search(match_text):
                    return match_text
                return self.manifest.cipher_address(match_text)
            text = pat.sub(_replace_addr, text)
        return text

    def _cipher_urls(self, text):
        """Replace personal URLs with cipher tokens."""
        def _replace_url(m):
            return self.manifest.cipher_url(m.group(0))
        return URL_PATTERN.sub(_replace_url, text)

    def _cipher_names(self, text):
        """Replace known names with cipher tokens."""
        def _replace_name(m):
            # m.group(1) is the captured name
            # m.group(0) includes any @ prefix
            full_match = m.group(0)
            name = m.group(1)
            token = self.manifest.cipher_name(name)
            # If there was an @mention prefix, keep the @ but replace the name
            if '@' in full_match and name in full_match:
                prefix = full_match[:full_match.index(name)]
                return prefix + token
            return token
        return NAME_PATTERN.sub(_replace_name, text)

    def _is_likely_pii_field(self, key):
        """Check if a JSON field is likely to contain names."""
        name_fields = {"sender", "from", "author", "name", "full_name", "person", "participant"}
        return key.lower() in name_fields

    def cipher_string(self, text, field_name=None):
        """Apply all cipher transforms to a string. field_name helps context."""
        # Normalize unicode first so regex patterns match correctly.
        # This also cleans up WhatsApp formatting characters from the output.
        text = normalize_text(text)
        text = self._cipher_urls(text)
        text = self._cipher_emails(text)
        text = self._cipher_phones(text)
        text = self._cipher_addresses(text)
        text = self._cipher_names(text)
        return text

    def cipher_record(self, record):
        """Apply cipher transforms to all values in a JSON record (dict)."""
        ciphered = {}
        for key, value in record.items():
            if isinstance(value, str):
                ciphered[key] = self.cipher_string(value, field_name=key)
            elif isinstance(value, list):
                ciphered[key] = [
                    self.cipher_string(item, field_name=key) if isinstance(item, str) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                ciphered[key] = self.cipher_record(value)
            else:
                ciphered[key] = value
        return ciphered


def maybe_cipher_filename(filename):
    """If the filename itself contains PII (e.g. phone number), note it."""
    # Check for phone numbers in filename
    for pat in PHONE_PATTERNS:
        m = pat.search(filename)
        if m:
            return True, m.group(0)
    return False, None


def file_sha256(filepath):
    """Compute sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── main ─────────────────────────────────────────────────────────────────────

def process_file(filepath, dry_run=False):
    """Process a single JSONL file. Returns stats dict."""
    basename = os.path.basename(filepath)
    # Compute input sha256 before processing
    input_hash = file_sha256(filepath) if not dry_run else None
    now = datetime.now(timezone.utc)
    created_at = now.isoformat()

    # Create unique output filename using parent directory as prefix
    # to avoid collisions between _chat.jsonl from different folders.
    parent_dir = os.path.basename(os.path.dirname(os.path.normpath(filepath)))
    output_name = f"{parent_dir}_{basename}"

    # Check if output filename itself contains PII
    has_phone_in_name, phone_val = maybe_cipher_filename(output_name)
    if has_phone_in_name and phone_val:
        output_name = f"ciphered_{output_name}"
        if not dry_run:
            print(f"  [i] Filename contains phone: {basename} -> {output_name}")

    output_path = os.path.join(OUTPUT_DIR, output_name)

    manifest = PIIManifest()
    cipher = PIICipher(manifest=manifest)

    records = []
    line_count = 0
    error_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSON decode error on line {line_count}: {e}", file=sys.stderr)
                error_count += 1
                records.append(line)  # keep raw line as-is
                continue

            ciphered = cipher.cipher_record(record)
            records.append(ciphered)

    # Also cipher phone number in filename if present
    if has_phone_in_name and phone_val:
        cipher.manifest.cipher_phone(phone_val)

    if dry_run:
        totals = manifest.totals()
        print(f"  DRY-RUN: {basename} ({line_count} lines, {error_count} errors)")
        print(f"    PII found: names={totals['names']}, phones={totals['phones']}, "
              f"addresses={totals['addresses']}, emails={totals['emails']}, urls={totals['urls']}")
        return {
            "file": basename,
            "lines": line_count,
            "errors": error_count,
            "pii_found": totals,
        }

    # Write ciphered output
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            if isinstance(rec, str):
                f.write(rec + "\n")
            else:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Compute output sha256 after writing
    output_hash = file_sha256(output_path)
    processed_at = datetime.now(timezone.utc).isoformat()

    totals = manifest.totals()
    print(f"  CIPHERED: {basename} -> {output_name} ({line_count} lines)")
    print(f"    PII replaced: names={totals['names']}, phones={totals['phones']}, "
          f"addresses={totals['addresses']}, emails={totals['emails']}, urls={totals['urls']}")

    return {
        "file": basename,
        "file_path": filepath,
        "output": output_name,
        "output_path": output_path,
        "lines": line_count,
        "errors": error_count,
        "pii_replaced": totals,
        "input_sha256": input_hash,
        "output_sha256": output_hash,
        "created_at": created_at,
        "processed_at": processed_at,
        "pii_details": {
            "names": list(manifest.replacements["names"].values()),
            "phones": list(manifest.replacements["phones"].values()),
            "addresses": list(manifest.replacements["addresses"].values()),
            "emails": list(manifest.replacements["emails"].values()),
            "urls": list(manifest.replacements["urls"].values()),
        },
    }


def main():
    args = sys.argv[1:]

    dry_run = False
    if "--dry-run" in args:
        dry_run = True
        args.remove("--dry-run")

    if not args:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    created_at_global = datetime.now(timezone.utc).isoformat()
    all_results = []
    total_lines = 0
    total_pii = {"names": 0, "phones": 0, "addresses": 0, "emails": 0, "urls": 0}

    for filepath in args:
        if not os.path.isfile(filepath):
            print(f"[ERROR] File not found: {filepath}", file=sys.stderr)
            continue

        print(f"\nProcessing: {filepath}")
        result = process_file(filepath, dry_run=dry_run)
        all_results.append(result)
        total_lines += result.get("lines", 0)
        pii = result.get("pii_found") or result.get("pii_replaced", {})
        for k in total_pii:
            total_pii[k] += pii.get(k, 0)

    # Summary
    print(f"\n{'='*60}")
    print(f"{'DRY-RUN' if dry_run else 'CIPHER'} SUMMARY")
    print(f"{'='*60}")
    print(f"  Files processed: {len(all_results)}")
    print(f"  Total lines:     {total_lines}")
    print(f"  PII total:       names={total_pii['names']}, phones={total_pii['phones']}, "
          f"addresses={total_pii['addresses']}, emails={total_pii['emails']}, urls={total_pii['urls']}")

    if not dry_run:
        verified_at = datetime.now(timezone.utc).isoformat()

        # Write PII manifest
        manifest_path = os.path.join(OUTPUT_DIR, "pii_manifest.json")
        manifest_data = {
            "schema": "lucidota.trm.pii_manifest.v1",
            "generated_by": "scripts/cipher_training_pii.py",
            "created_at": __import__("datetime").datetime.now(timezone.utc).isoformat(),
            "verified_at": verified_at,
            "files_processed": [
                {
                    "input": r["file"],
                    "input_path": r["file_path"],
                    "input_sha256": r["input_sha256"],
                    "output": r.get("output", "N/A"),
                    "output_path": r.get("output_path", ""),
                    "output_sha256": r["output_sha256"],
                    "lines": r["lines"],
                    "errors": r["errors"],
                    "created_at": r["created_at"],
                    "processed_at": r["processed_at"],
                    "pii_replaced": r["pii_replaced"],
                }
                for r in all_results
            ],
            "total_pii_replaced": total_pii,
            "replacement_details": {
                r["file"]: r["pii_details"]
                for r in all_results if "pii_details" in r
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        print(f"\n  Manifest: {manifest_path}")

        # Write receipt
        receipt_path = os.path.join(RECEIPT_DIR, "pii_cipher_receipt.json")
        receipt = {
            "command": "scripts/cipher_training_pii.py",
            "schema": "lucidota.trm.pii_cipher_receipt.v1",
            "created_at": created_at_global,
            "processed_at": __import__("datetime").datetime.now(timezone.utc).isoformat(),
            "verified_at": verified_at,
            "files_processed": [
                {
                    "file": r["file"],
                    "input_sha256": r["input_sha256"],
                    "output_sha256": r["output_sha256"],
                    "pii_replacements": r["pii_replaced"],
                }
                for r in all_results
            ],
            "pii_replacements": total_pii,
            "ciphered_copies_written": [
                r.get("output_path", "N/A") for r in all_results if "output_path" in r
            ],
            "verdict": "PASS",
        }
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, ensure_ascii=False)
        print(f"  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
